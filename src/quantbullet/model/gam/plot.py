import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Union, Tuple, Optional

from quantbullet.plot import (
    EconomistBrandColor,
    get_grid_fig_axes,
)
from quantbullet.plot.utils import close_unused_axes
from quantbullet.plot.cycles import use_economist_cycle

from .terms import (
    GAMTermData,
    SplineTermData,
    SplineByGroupTermData,
    TensorTermData,
    FactorTermData,
)


def plot_tensor(
    ax,
    X1,
    X2,
    Z,
    style="contour",
    levels=10,
    cmap="viridis",
    alpha=0.6,
    show_labels=True,
    colorbar=False,
    colorbar_label=None,
):
    """
    Plot a tensor term (2D smooth) on the given axes.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes.
    X1, X2 : 2D arrays
        Meshgrid arrays for the two dimensions.
    Z : 2D array
        Partial dependence values.
    style : {"contour", "contourf", "heatmap"}
        Plotting style.
    levels : int
        Number of contour levels.
    cmap : str or Colormap
        Colormap to use. 'viridis' is good for screen, 'Greys' for print.
    alpha : float
        Transparency for filled plots.
    show_labels : bool
        Whether to label contour lines.
    colorbar : bool
        Whether to add a colorbar.
    colorbar_label : str
        Label for the colorbar.
    """

    if style not in {"contour", "contourf", "heatmap"}:
        raise ValueError(f"Unknown style: {style}")

    mappable = None  # for optional colorbar

    # -------------------------
    # 1) Contour only (best for print)
    # -------------------------
    if style == "contour":
        cs = ax.contour(
            X1, X2, Z,
            levels=levels,
            colors="black",
            linewidths=1.0,
        )
        if show_labels:
            ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

        mappable = cs

    # -------------------------
    # 2) Contour + light fill
    # -------------------------
    elif style == "contourf":
        cf = ax.contourf(
            X1, X2, Z,
            levels=levels,
            cmap=cmap,
            alpha=alpha,
        )
        cs = ax.contour(
            X1, X2, Z,
            levels=levels,
            colors="black",
            linewidths=0.8,
        )
        if show_labels:
            ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

        mappable = cf

    # -------------------------
    # 3) Heatmap (screen only)
    # -------------------------
    elif style == "heatmap":
        mappable = ax.pcolormesh(
            X1, X2, Z,
            shading="auto",
            cmap=cmap,
        )

    # -------------------------
    # Optional colorbar
    # -------------------------
    if colorbar and mappable is not None:
        cb = plt.colorbar(mappable, ax=ax)
        cb.set_label(colorbar_label)

    return ax


def plot_partial_dependence(
    pdep_data: Dict[Union[str, "Tuple[str, str]"], "GAMTermData"],
    *,
    n_cols: int = 3,
    suptitle: Optional[str] = None,
    scale_y_axis: bool = True,
    te_plot_style: str = "heatmap",
    width: float = 5,
    height: float = 4,
):
    """Plot partial dependence from a term-data dict (shared by WrapperGAM and GAMReplayModel).

    Parameters
    ----------
    pdep_data : dict
        Mapping returned by ``get_partial_dependence_data()``.
    n_cols, suptitle, scale_y_axis, te_plot_style, width, height
        Layout and style options.

    Returns
    -------
    fig, axes
    """
    keys = list(pdep_data.keys())

    with use_economist_cycle():
        fig, axes = get_grid_fig_axes(n_charts=len(keys), n_cols=n_cols, width=width, height=height)
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    continuous_axes: list = []

    for i, key in enumerate(keys):
        ax = axes.flat[i]
        td = pdep_data[key]
        feature_name = key if isinstance(key, str) else key[0]

        if isinstance(td, SplineByGroupTermData):
            for label, curves in td.group_curves.items():
                ax.plot(curves["x"], curves["y"], label=label)
                if curves.get("conf_lower") is not None and curves.get("conf_upper") is not None:
                    ax.fill_between(curves["x"], curves["conf_lower"], curves["conf_upper"], alpha=0.15)
            ax.set_xlabel(f"{td.feature} (by {td.by_feature})", fontdict={"fontsize": 12})
            ax.set_ylabel("Partial Dependence", fontdict={"fontsize": 12})
            ax.legend(title=td.by_feature)
            continuous_axes.append(ax)

        elif isinstance(td, SplineTermData):
            ax.plot(td.x, td.y, color=EconomistBrandColor.CHICAGO_45)
            if td.conf_lower is not None and td.conf_upper is not None:
                ax.fill_between(td.x, td.conf_lower, td.conf_upper,
                                alpha=0.2, color=EconomistBrandColor.CHICAGO_45)
            ax.set_xlabel(feature_name, fontdict={"fontsize": 12})
            ax.set_ylabel("Partial Dependence", fontdict={"fontsize": 12})
            continuous_axes.append(ax)

        elif isinstance(td, TensorTermData):
            X1, X2 = np.meshgrid(td.x, td.y, indexing="ij")
            plot_tensor(ax, X1, X2, td.z, style=te_plot_style)
            ax.set_xlabel(td.feature_x, fontsize=12)
            ax.set_ylabel(td.feature_y, fontsize=12)
            ax.set_title(f"{td.feature_x} x {td.feature_y} (tensor surface)", fontsize=12)

        elif isinstance(td, FactorTermData):
            if td.conf_lower is not None and td.conf_upper is not None:
                yerr = [td.values - td.conf_lower, td.conf_upper - td.values]
                ax.errorbar(td.categories, td.values, yerr=yerr,
                            fmt="o", capsize=5, color=EconomistBrandColor.CHICAGO_45)
            else:
                ax.plot(td.categories, td.values, "o", color=EconomistBrandColor.CHICAGO_45)
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)
            ax.set_xlabel(feature_name, fontdict={"fontsize": 12})
            ax.set_ylabel("Partial Dependence", fontdict={"fontsize": 12})

        else:
            ax.set_title(f"{feature_name} (unknown type)")
            ax.axis("off")

    if scale_y_axis and continuous_axes:
        y_min = min(a.get_ylim()[0] for a in continuous_axes)
        y_max = max(a.get_ylim()[1] for a in continuous_axes)
        for a in continuous_axes:
            a.set_ylim(y_min, y_max)

    if suptitle:
        plt.suptitle(suptitle, fontsize=14)

    close_unused_axes(axes)
    return fig, axes
