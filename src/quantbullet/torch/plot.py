"""Plotting utilities for PyTorch models."""
import torch
import torch.nn as nn


def plot_hinge(hinge: nn.Module, n_points: int = 200, ax=None, show_knots: bool = True):
    """
    Generic plotting function for all hinge models.
    
    Args:
        hinge: Any hinge model (Generic/Concave/Convex, scaled or unscaled)
        n_points: Number of points to plot
        ax: Optional matplotlib axis. If None, creates new figure
        show_knots: If True, mark knot positions with vertical lines
        
    Returns:
        fig, ax: matplotlib figure and axis objects
        
    Example:
        >>> from quantbullet.torch.hinge import MinMaxScaledConvexHinge
        >>> from quantbullet.torch.plot import plot_hinge
        >>> hinge = MinMaxScaledConvexHinge(K=10, x_min=0.5, x_max=1.5)
        >>> fig, ax = plot_hinge(hinge)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Determine if this is a scaled hinge
    is_scaled = hasattr(hinge, 'scale') and hasattr(hinge, 'x_min')
    
    # Get the underlying hinge model
    inner_hinge = hinge.hinge if is_scaled else hinge
    
    # Determine x range
    if is_scaled:
        x_min = float(hinge.x_min.detach().cpu().item())
        x_max = float(hinge.x_max.detach().cpu().item())
        x_np = np.linspace(x_min, x_max, n_points)
    else:
        x_np = np.linspace(0, 1, n_points)
    
    # Convert to tensor on the same device as the model
    device = next(hinge.parameters()).device
    x_tensor = torch.tensor(x_np, dtype=torch.float32, device=device)
    
    # Forward pass
    with torch.no_grad():
        y_tensor = hinge(x_tensor)
    
    # Convert to numpy
    y_np = y_tensor.detach().cpu().numpy().squeeze()
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    # Plot the curve
    ax.plot(x_np, y_np, 'b-', linewidth=2, label='Hinge function')
    
    # Mark knots if requested
    if show_knots and hasattr(inner_hinge, 'knots'):
        knots_01 = inner_hinge.knots().detach().cpu().numpy()
        
        # Convert knots to original scale if needed
        if is_scaled:
            knots_original = x_min + knots_01 * (x_max - x_min)
        else:
            knots_original = knots_01
        
        # Plot vertical lines at knots
        y_min, y_max = ax.get_ylim()
        for knot in knots_original:
            ax.axvline(knot, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        
        # Add knot markers on the curve
        knots_tensor = torch.tensor(knots_original, dtype=torch.float32, device=device)
        with torch.no_grad():
            y_knots = hinge(knots_tensor).detach().cpu().numpy().squeeze()
        ax.plot(knots_original, y_knots, 'ro', markersize=6, label=f'Knots (n={len(knots_original)})')
    
    # Labels and formatting
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    
    # Add title with model info
    model_name = type(hinge).__name__
    if is_scaled:
        title = f'{model_name}\n[{x_min:.3f}, {x_max:.3f}]'
    else:
        title = f'{model_name}\n[0, 1]'
    ax.set_title(title, fontsize=14)
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    return fig, ax
