import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundedWindowBump(nn.Module):
    """
    Window bump/gate function (NON-MONOTONIC - rises then falls):
      w(x) = sigmoid((x-a)/s1) - sigmoid((x-b)/s2), with b>a
      y(x) = lower + (upper-lower) * w(x)
    
    Creates a bump: flat(0) → rise → flat(1) → fall → flat(0)

    Rule:
      - if param is provided (not None) => fixed by default
      - if param is None => trainable by default
    """
    def __init__(
        self,
        *,
        a: float | None = None,
        b: float | None = None,
        lower: float | None = None,
        upper: float | None = None,
        s1: float | None = None,
        s2: float | None = None,
        clamp_w: bool = False,
    ):
        super().__init__()
        self.clamp_w = clamp_w

        # simple defaults for omitted params
        a0 = 0.0 if a is None else float(a)
        b0 = 1.0 if b is None else float(b)
        lo0 = 0.0 if lower is None else float(lower)
        hi0 = 1.0 if upper is None else float(upper)
        s10 = 0.01 if s1 is None else float(s1)
        s20 = 0.01 if s2 is None else float(s2)

        # raw params (unconstrained)
        self.a_raw = nn.Parameter(torch.tensor(a0))
        self.delta_raw = nn.Parameter(torch.tensor(0.0))    # b = a + softplus(delta)
        self.lower_raw = nn.Parameter(torch.tensor(lo0))
        self.range_raw = nn.Parameter(torch.tensor(0.0))    # upper = lower + softplus(range)
        self.log_s1 = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(max(s10, 1e-6))))))
        self.log_s2 = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(max(s20, 1e-6))))))

        # set b, upper consistently
        self.set_params(b=b0, upper=hi0)

        # trainability: None => trainable, provided => fixed
        self.set_trainable(
            a=(a is None),
            b=(b is None),
            lower=(lower is None),
            upper=(upper is None),
            s1=(s1 is None),
            s2=(s2 is None),
        )

    # constrained views
    def a(self): return self.a_raw
    def b(self): return self.a_raw + F.softplus(self.delta_raw) + 1e-6
    def lower(self): return self.lower_raw
    def upper(self): return self.lower_raw + F.softplus(self.range_raw) + 1e-6
    def s1(self): return torch.exp(self.log_s1).clamp(1e-6, 1e3)
    def s2(self): return torch.exp(self.log_s2).clamp(1e-6, 1e3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.sigmoid((x - self.a()) / self.s1()) - torch.sigmoid((x - self.b()) / self.s2())
        if self.clamp_w:
            w = w.clamp(0.0, 1.0)
        return self.lower() + (self.upper() - self.lower()) * w

    @torch.no_grad()
    def set_params(self, *, a=None, b=None, lower=None, upper=None, s1=None, s2=None):
        if a is not None:
            self.a_raw.copy_(torch.tensor(float(a), dtype=self.a_raw.dtype))

        if b is not None:
            a_cur = float(self.a_raw.item()) if a is None else float(a)
            gap = max(float(b) - a_cur, 1e-6)
            self.delta_raw.copy_(torch.log(torch.exp(torch.tensor(gap)) - 1.0))  # inverse softplus

        if lower is not None:
            self.lower_raw.copy_(torch.tensor(float(lower), dtype=self.lower_raw.dtype))

        if upper is not None:
            lo = float(self.lower_raw.item()) if lower is None else float(lower)
            rng = max(float(upper) - lo, 1e-6)
            self.range_raw.copy_(torch.log(torch.exp(torch.tensor(rng)) - 1.0))

        if s1 is not None:
            self.log_s1.copy_(torch.log(torch.tensor(max(float(s1), 1e-6))))
        if s2 is not None:
            self.log_s2.copy_(torch.log(torch.tensor(max(float(s2), 1e-6))))

    def set_trainable(self, **flags: bool):
        mapping = {
            "a": self.a_raw,
            "b": self.delta_raw,
            "lower": self.lower_raw,
            "upper": self.range_raw,
            "s1": self.log_s1,
            "s2": self.log_s2,
        }
        for k, v in flags.items():
            mapping[k].requires_grad_(bool(v))

    def freeze_params(self, *names: str):
        self.set_trainable(**{n: False for n in names})

    def unfreeze_params(self, *names: str):
        self.set_trainable(**{n: True for n in names})


class BoundedWindowGate(nn.Module):
    """
    Monotonic sigmoid gate transitioning smoothly between a and b:
      w(x) = sigmoid((x - center) / s)
      y(x) = lower + (upper-lower) * w(x)
    
    Creates S-curve: flat(lower) → smooth rise → flat(upper)
    - center = (a+b)/2 is the inflection point
    - s controls smoothness (smaller = sharper)

    Rule:
      - if param is provided (not None) => fixed by default
      - if param is None => trainable by default
    """
    def __init__(
        self,
        *,
        a: float | None = None,
        b: float | None = None,
        lower: float | None = None,
        upper: float | None = None,
        s: float | None = None,
    ):
        super().__init__()
        
        # simple defaults for omitted params
        a0 = 0.0 if a is None else float(a)
        b0 = 1.0 if b is None else float(b)
        lo0 = 0.0 if lower is None else float(lower)
        hi0 = 1.0 if upper is None else float(upper)
        s0 = 1.0 if s is None else float(s)  # better default for learning

        # Compute initial center and width
        center0 = (a0 + b0) / 2.0
        width0 = max(abs(b0 - a0), 1e-6)

        # raw params - use center/width for better trainability
        self.center_raw = nn.Parameter(torch.tensor(center0))
        self.log_width = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(width0)))))
        self.lower_raw = nn.Parameter(torch.tensor(lo0))
        self.log_range = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(max(hi0 - lo0, 1e-6))))))
        self.log_s = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(max(s0, 1e-6))))))

        # trainability: None => trainable, provided => fixed
        self.set_trainable(
            a=(a is None),
            b=(b is None),
            lower=(lower is None),
            upper=(upper is None),
            s=(s is None),
        )

    # constrained views
    def center(self): return self.center_raw
    def width(self): return torch.exp(self.log_width).clamp(1e-6, 1e3)
    def a(self): return self.center() - self.width() / 2.0
    def b(self): return self.center() + self.width() / 2.0
    def lower(self): return self.lower_raw
    def upper(self): return self.lower_raw + torch.exp(self.log_range).clamp(1e-6, 1e3)
    def s(self): return torch.exp(self.log_s).clamp(1e-6, 1e3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.sigmoid((x - self.center()) / self.s())
        return self.lower() + (self.upper() - self.lower()) * w
    
    def compute_loss(self, x: torch.Tensor, y_target: torch.Tensor, 
                     roi_mask: torch.Tensor | None = None,
                     auto_mask: bool = False) -> torch.Tensor:
        """
        Compute MSE loss, optionally masked to a region of interest.
        
        Args:
            x: Input tensor
            y_target: Target output
            roi_mask: Boolean mask for region of interest (optional, overrides auto_mask)
            auto_mask: If True, automatically mask to [a, b] region where gate is active.
                      This focuses learning on the transition region and ignores flat regions.
        
        Returns:
            MSE loss (masked if roi_mask provided or auto_mask=True)
        
        Example:
            >>> gate = BoundedWindowGate()
            >>> # Normal loss - fit everywhere
            >>> loss = gate.compute_loss(x, y_target)
            >>> 
            >>> # Auto-mask - only fit [a, b] region
            >>> loss = gate.compute_loss(x, y_target, auto_mask=True)
            >>> 
            >>> # Custom mask - fit specific region
            >>> mask = (x >= 0) & (x <= 5)
            >>> loss = gate.compute_loss(x, y_target, roi_mask=mask)
        """
        y_pred = self.forward(x)
        errors = (y_pred - y_target) ** 2
        
        # Use provided mask, or auto-generate from [a, b] range
        if roi_mask is not None:
            errors = errors[roi_mask]
        elif auto_mask:
            # Automatically mask to the [a, b] transition region
            a_val = self.a()
            b_val = self.b()
            # Handle both 1D and multi-dimensional x
            if x.ndim == 1:
                mask = (x >= a_val) & (x <= b_val)
            else:
                # For multi-dim, assume last dimension or use broadcasting
                mask = (x >= a_val) & (x <= b_val)
                if mask.ndim > 1:
                    mask = mask.squeeze()
            errors = errors[mask]
        
        return torch.mean(errors)

    @torch.no_grad()
    def set_params(self, *, a=None, b=None, lower=None, upper=None, s=None):
        # Handle a/b → center/width conversion
        if a is not None or b is not None:
            a_val = float(a) if a is not None else float(self.a())
            b_val = float(b) if b is not None else float(self.b())
            center = (a_val + b_val) / 2.0
            width = max(abs(b_val - a_val), 1e-6)
            self.center_raw.copy_(torch.tensor(center, dtype=self.center_raw.dtype))
            self.log_width.copy_(torch.log(torch.tensor(width)))

        if lower is not None:
            self.lower_raw.copy_(torch.tensor(float(lower), dtype=self.lower_raw.dtype))

        if upper is not None:
            lo = float(self.lower_raw.item()) if lower is None else float(lower)
            rng = max(float(upper) - lo, 1e-6)
            self.log_range.copy_(torch.log(torch.tensor(rng)))

        if s is not None:
            self.log_s.copy_(torch.log(torch.tensor(max(float(s), 1e-6))))

    def set_trainable(self, **flags: bool):
        mapping = {
            "a": self.center_raw,      # training 'a' affects center
            "b": self.log_width,       # training 'b' affects width
            "lower": self.lower_raw,
            "upper": self.log_range,
            "s": self.log_s,
        }
        for k, v in flags.items():
            mapping[k].requires_grad_(bool(v))

    def freeze_params(self, *names: str):
        self.set_trainable(**{n: False for n in names})

    def unfreeze_params(self, *names: str):
        self.set_trainable(**{n: True for n in names})
