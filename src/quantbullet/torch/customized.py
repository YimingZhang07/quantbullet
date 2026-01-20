import torch
import torch.nn as nn
import numpy as np
from quantbullet.torch.hinge import ConcaveHinge, ConvexHinge
import torch
import torch.nn as nn
import torch.nn.functional as F

class EpsilonInsensitiveLoss(nn.Module):
    def __init__(self, epsilon=0.10):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        # Only penalize errors larger than epsilon
        error = torch.abs(pred - target)
        loss = torch.clamp(error - self.epsilon, min=0.0)
        return loss.mean()

class BucketAdjustedHinge(nn.Module):
    def __init__(self, 
                 bucket_idx_mapping: dict,
                 bucket_scaler_mapping: dict,
                 K_base: int = None,
                 K_bucket_adj: int = None,
                 bucket_adj_fixed_knots: np.ndarray = None,
                 base_fixed_knots: np.ndarray = None,
                 ):
        super().__init__()
        n_buckets = len(bucket_idx_mapping)
        self.base_hinge = ConcaveHinge(K=K_base, fixed_knots=base_fixed_knots, monotone_increasing=True)
        self.bucket_adj_hinges = nn.ModuleList([
            ConcaveHinge(K=K_bucket_adj, fixed_knots=bucket_adj_fixed_knots, monotone_increasing=True) for _ in range(n_buckets)
        ])

        # Initialize arrays to store the scaling params
        x_mins = torch.zeros(n_buckets)
        x_maxs = torch.zeros(n_buckets)
        clip_los = torch.full((n_buckets,), float('nan'))  # Use NaN for "no clip"
        clip_his = torch.full((n_buckets,), float('nan'))

        for k, v in bucket_idx_mapping.items():
            idx = bucket_idx_mapping[k]
            x_mins[idx] = bucket_scaler_mapping[k]['x_min']
            x_maxs[idx] = bucket_scaler_mapping[k]['x_max']
            clip_los[idx] = bucket_scaler_mapping[k]['clip_lo']
            clip_his[idx] = bucket_scaler_mapping[k]['clip_hi']

        self.register_buffer('x_mins', x_mins)
        self.register_buffer('x_maxs', x_maxs)
        self.register_buffer('clip_los', clip_los)
        self.register_buffer('clip_his', clip_his)

    def clip_and_scale(self, x, bucket_idx):
        x = x.view(-1, 1)
        
        # Gather bucket-specific params
        x_min = self.x_mins[bucket_idx].view(-1, 1)
        x_max = self.x_maxs[bucket_idx].view(-1, 1)
        clip_lo = self.clip_los[bucket_idx].view(-1, 1)
        clip_hi = self.clip_his[bucket_idx].view(-1, 1)
        
        # Clip only where finite
        x_clipped = x
        if torch.isfinite(clip_lo).any():
            x_clipped = torch.where(torch.isfinite(clip_lo), 
                                    torch.max(x_clipped, clip_lo), 
                                    x_clipped)
        if torch.isfinite(clip_hi).any():
            x_clipped = torch.where(torch.isfinite(clip_hi), 
                                    torch.min(x_clipped, clip_hi), 
                                    x_clipped)
        
        # Scale to [0,1]
        x_scaled = (x_clipped - x_min) / (x_max - x_min + 1e-12)
        return x_scaled.clamp(0.0, 1.0)

    def forward(self, x, bucket_idx):
        bucket_idx = bucket_idx.view(-1)  # (N,)
        
        # 1. Scale to [0,1]
        x01 = self.clip_and_scale(x, bucket_idx)  # (N, 1)
        
        # 2. Shared base
        base = self.base_hinge(x01)  # (N, 1)
        
        # 3. Bucket-specific adjustments - process ALL samples at once
        adjustment = torch.zeros(bucket_idx.shape[0], 1, device=x.device)
        
        unique_buckets = torch.unique(bucket_idx)
        for b in unique_buckets:
            mask = (bucket_idx == b)
            indices = torch.nonzero(mask, as_tuple=False).squeeze(1)  # Get integer indices
            
            # Apply curve to these samples
            adj_values = self.bucket_adj_hinges[b](x01[mask])
            
            # Scatter back into adjustment
            adjustment[indices] = adj_values
        
        # 4. Additive combination
        return base + adjustment


class IntegratedPricingModel(nn.Module):
    """
    Integrated pricing model with multiple features.
    
    Components are passed in (dependency injection) for clean separation:
    - MVOC model: Main driver, passed as pre-configured module
    - WAP hinge: Optional secondary effect, clamped to max absolute impact
    - Coupon hinge: Optional secondary effect, clamped to max absolute impact
    
    Architecture:
        Price = MVOC_effect(mvoc, bucket) 
                + clamp(WAP_hinge(wap), min=-max, max=+max)
                + clamp(Coupon_hinge(coupon), min=-max, max=+max)
                + bias
    
    The max impacts are HARD CAPS, not scaling factors:
    - If WAP_hinge outputs 2.0 and max_impact=0.5, contribution is clamped to 0.5
    - If WAP_hinge outputs 0.3 and max_impact=0.5, contribution is 0.3
    """
    def __init__(
        self,
        mvoc_model: nn.Module,                    # Pre-configured MVOC model
        wap_hinge: nn.Module = None,              # Optional WAP effect
        coupon_hinge: nn.Module = None,           # Optional Coupon effect
        wap_max_impact: float = 0.50,             # Max absolute WAP impact (hard cap)
        coupon_max_impact: float = 0.50,          # Max absolute Coupon impact (hard cap)
    ):
        super().__init__()
        
        # 1. MVOC effect - main driver (passed in)
        self.mvoc_model = mvoc_model
        
        # 2. WAP effect (optional, passed in)
        self.wap_hinge = wap_hinge
        self.wap_max_impact = wap_max_impact  # Hard cap value
        
        # 3. Coupon effect (optional, passed in)
        self.coupon_hinge = coupon_hinge
        self.coupon_max_impact = coupon_max_impact  # Hard cap value
        
        # Global intercept (optional offset)
        self.global_bias = nn.Parameter(torch.tensor(0.0))
    
    def forward(
        self, 
        mvoc: torch.Tensor, 
        bucket_idx: torch.Tensor,
        wap: torch.Tensor = None,
        cpnspread: torch.Tensor = None,
    ):
        """
        Args:
            mvoc: (N,) MVOC values
            bucket_idx: (N,) bucket indices for currency/rating
            wap: (N,) weighted average price (optional)
            cpnspread: (N,) coupon spread values (optional)
            
        Returns:
            (N, 1) predicted prices
        """
        # Main effect: MVOC with bucket adjustments
        price = self.mvoc_model(mvoc, bucket_idx)  # (N, 1)
        
        # Add WAP effect
        if self.wap_hinge is not None:
            wap_impact = self.wap_hinge(wap)
            wap_impact = torch.clamp(wap_impact, min=-self.wap_max_impact, max=self.wap_max_impact)
            price = price + wap_impact

        # Add Coupon effect
        if self.coupon_hinge is not None:
            coupon_impact = self.coupon_hinge(cpnspread)
            coupon_impact = torch.clamp(coupon_impact, min=-self.coupon_max_impact, max=self.coupon_max_impact)
            price = price + coupon_impact

        # Add global bias
        return price + self.global_bias
    
    # def get_feature_importance(self):
    #     """
    #     Get feature max impacts (hard caps on contributions via sigmoid).
        
    #     Returns:
    #         dict with maximum impacts for each feature (contributions in [0, max])
    #     """
    #     with torch.no_grad():
    #         result = {
    #             'mvoc': 'primary (unrestricted)',
    #             'global_bias': f'{self.global_bias.item():.3f}',
    #         }
            
    #         if self.wap_hinge is not None:
    #             if hasattr(self, 'wap_scale_raw'):
    #                 effective_max = self.wap_max_bound * torch.sigmoid(self.wap_scale_raw).item()
    #                 result['wap_max_impact'] = f'[0, {effective_max:.3f}] (learned from max={self.wap_max_bound:.3f})'
    #             else:
    #                 result['wap_max_impact'] = f'[0, {self.wap_max_bound:.3f}] (fixed)'
            
    #         if self.coupon_hinge is not None:
    #             if hasattr(self, 'coupon_scale_raw'):
    #                 effective_max = self.coupon_max_bound * torch.sigmoid(self.coupon_scale_raw).item()
    #                 result['coupon_max_impact'] = f'[0, {effective_max:.3f}] (learned from max={self.coupon_max_bound:.3f})'
    #             else:
    #                 result['coupon_max_impact'] = f'[0, {self.coupon_max_bound:.3f}] (fixed)'
            
    #         return result

