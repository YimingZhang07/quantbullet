"""Plotting utilities for Hinge models."""
import torch
import torch.nn as nn


def plot_hinge(hinge: nn.Module, n_points: int = 200, ax=None, show_knots: bool = True):
    """
    Plot a Hinge curve in its original feature space.

    Args:
        hinge: A ``Hinge`` instance (has ``x_min``, ``x_max``, ``knots()``).
        n_points: Number of evaluation points.
        ax: Optional matplotlib axis.  Creates a new figure if *None*.
        show_knots: Mark knot positions on the curve.

    Returns:
        (fig, ax)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    x_min = float(hinge.x_min.detach().cpu())
    x_max = float(hinge.x_max.detach().cpu())
    device = next(hinge.parameters()).device

    x_np = np.linspace(x_min, x_max, n_points)
    x_t = torch.tensor(x_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        y_np = hinge(x_t).detach().cpu().numpy().squeeze()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    ax.plot(x_np, y_np, "b-", linewidth=2, label="Hinge function")

    if show_knots:
        knots_np = hinge.knots(original=True).detach().cpu().numpy()
        knots_t = torch.tensor(knots_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            y_knots = hinge(knots_t).detach().cpu().numpy().squeeze()
        for k in knots_np:
            ax.axvline(k, color="gray", linestyle="--", alpha=0.3, linewidth=1)
        ax.plot(knots_np, y_knots, "ro", markersize=6,
                label=f"Knots (n={len(knots_np)})")

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(f"{hinge!r}", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig, ax
