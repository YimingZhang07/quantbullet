"""Model diagnostics that need only a model's predictions, not the model; plotting loads only to draw."""
import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "two_way": ["TwoWayDiagnostic", "TwoWayAE", "TwoWayCorrection"],
    },
)
