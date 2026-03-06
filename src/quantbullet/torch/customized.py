import torch
import torch.nn as nn
import numpy as np

from quantbullet.torch.hinge import Hinge


class EpsilonInsensitiveLoss(nn.Module):
    def __init__(self, epsilon=0.10):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        error = torch.abs(pred - target)
        return torch.clamp(error - self.epsilon, min=0.0).mean()


class BucketAdjustedHinge(nn.Module):
    """
    Shared base hinge + per-bucket adjustment hinges.

    Each bucket can have a different x_range so that scaling is
    handled inside every Hinge automatically.

    Args:
        bucket_idx_mapping:    {(ccy, rating): int}  index for each bucket
        bucket_scaler_mapping: {(ccy, rating): {"x_min": ..., "x_max": ...}}
        n_knots_base:          knots for the shared base hinge
        n_knots_adj:           knots for each bucket adjustment hinge
    """

    def __init__(
        self,
        bucket_idx_mapping: dict,
        bucket_scaler_mapping: dict,
        n_knots_base: int = 20,
        n_knots_adj: int = 20,
    ):
        super().__init__()
        n_buckets = len(bucket_idx_mapping)

        x_lo = min(v["x_min"] for v in bucket_scaler_mapping.values())
        x_hi = max(v["x_max"] for v in bucket_scaler_mapping.values())

        self.base_hinge = Hinge(
            n_knots=n_knots_base,
            x_range=(x_lo, x_hi),
            shape="concave",
            monotone="increasing",
        )

        adj_hinges = []
        for key, idx in sorted(bucket_idx_mapping.items(), key=lambda kv: kv[1]):
            sc = bucket_scaler_mapping[key]
            adj_hinges.append(
                Hinge(
                    n_knots=n_knots_adj,
                    x_range=(sc["x_min"], sc["x_max"]),
                    shape="concave",
                    monotone="increasing",
                )
            )
        self.bucket_adj_hinges = nn.ModuleList(adj_hinges)

    def forward(self, x: torch.Tensor, bucket_idx: torch.Tensor) -> torch.Tensor:
        bucket_idx = bucket_idx.view(-1)
        base = self.base_hinge(x)

        adjustment = torch.zeros(bucket_idx.shape[0], 1, device=x.device)
        for b in torch.unique(bucket_idx):
            mask = bucket_idx == b
            indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
            adjustment[indices] = self.bucket_adj_hinges[b](x[mask])

        return base + adjustment


class IntegratedPricingModel(nn.Module):
    """
    Additive pricing model with multiple feature components.

    Price = mvoc_model(mvoc, bucket_idx)
          + clamp(wap_hinge(wap),       -cap, +cap)
          + clamp(coupon_hinge(coupon),  -cap, +cap)
          + bias
    """

    def __init__(
        self,
        mvoc_model: nn.Module,
        wap_hinge: nn.Module | None = None,
        coupon_hinge: nn.Module | None = None,
        wap_max_impact: float = 0.50,
        coupon_max_impact: float = 0.50,
    ):
        super().__init__()
        self.mvoc_model = mvoc_model
        self.wap_hinge = wap_hinge
        self.coupon_hinge = coupon_hinge
        self.wap_max_impact = wap_max_impact
        self.coupon_max_impact = coupon_max_impact
        self.global_bias = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        mvoc: torch.Tensor,
        bucket_idx: torch.Tensor,
        wap: torch.Tensor | None = None,
        cpnspread: torch.Tensor | None = None,
    ) -> torch.Tensor:
        price = self.mvoc_model(mvoc, bucket_idx)

        if self.wap_hinge is not None and wap is not None:
            price = price + self.wap_hinge(wap).clamp(
                -self.wap_max_impact, self.wap_max_impact
            )
        if self.coupon_hinge is not None and cpnspread is not None:
            price = price + self.coupon_hinge(cpnspread).clamp(
                -self.coupon_max_impact, self.coupon_max_impact
            )
        return price + self.global_bias
