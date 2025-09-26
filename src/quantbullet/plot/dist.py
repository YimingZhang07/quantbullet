import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from quantbullet.plot.utils import get_grid_fig_axes, close_unused_axes

def plot_distributions(
    df: pd.DataFrame,
    columns: list[str],
    sample: int | float | None = None,
    max_unique_cats: int = 20,
    bins: int = 30,
    figsize: tuple[int, int] | None = None,
):
    """
    Plot distribution plots for a list of columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : list[str]
        List of column names to plot
    sample : int | float | None
        If int -> sample n rows; if float -> sample frac (0-1); if None -> use all data
    max_unique_cats : int
        If a non-numeric column has fewer than this many unique values, treat it as categorical
    bins : int
        Number of bins for histograms
    figsize : tuple[int, int] | None
        Size of the overall figure (default scales by number of subplots)
    """

    # Sampling
    if sample is not None:
        df = df.sample(n=sample if isinstance(sample, int) else None,
                       frac=sample if isinstance(sample, float) else None,
                       random_state=42)

    fig, axes = get_grid_fig_axes( n_charts=len(columns)

    for i, col in enumerate(columns):
        ax = axes[i]
        series = df[col].dropna()

        # Decide categorical vs continuous
        if not is_numeric_dtype(series):
            is_categorical = True
        else:
            # numeric but few unique values → treat as categorical
            is_categorical = series.nunique() <= max_unique_cats

        if is_categorical:
            counts = series.astype(str).value_counts()
            sns.barplot(x=counts.index, y=counts.values, ax=ax, color="skyblue")
            ax.set_title(f"{col} (categorical)")
            ax.set_ylabel("Count")
            ax.set_xlabel(col)
            ax.tick_params(axis="x", rotation=45)
        else:
            sns.histplot(series, bins=bins, kde=True, ax=ax, color="steelblue")
            ax.set_title(f"{col} (continuous)")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")

    # Remove empty subplots if any
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
