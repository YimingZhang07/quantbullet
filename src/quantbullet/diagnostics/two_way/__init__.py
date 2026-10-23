"""Two-way diagnostics of a model's predictions over two variables x and z.

``TwoWayDiagnostic`` reads x, z, y, the prediction and an optional weight once. Its two methods answer different
questions about the same rows:

- ``ae`` bins both variables and reports actual / expected per cell (``TwoWayAE``);
- ``correction`` fits a multiplier surface D(x, z) on the prediction (``TwoWayCorrection``). With a reference on
  both axes, D holds only the interaction that the model's separable main effects miss.

Both results draw themselves with ``plot()``: a heatmap above one slice panel per row.
"""
from .ae import TwoWayAE
from .correction import TwoWayCorrection
from .diagnostic import TwoWayDiagnostic

__all__ = ["TwoWayAE", "TwoWayCorrection", "TwoWayDiagnostic"]
