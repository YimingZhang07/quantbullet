import matplotlib.pyplot as plt

class PlotStyle:
    def __init__(self, xlabel='', ylabel='', title='', legend_title='', cmap='viridis', fig_size=(8, 6), label_size=12, legend_title_size=12, title_size=14):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.legend_title = legend_title
        self.cmap = cmap
        self.fig_size = fig_size
        self.label_size = label_size
        self.legend_title_size = legend_title_size
        self.title_size = max(label_size, title_size)

    def apply_labels(self, ax):
        if self.xlabel:
            ax.set_xlabel(self.xlabel, fontsize=self.label_size)
        if self.ylabel:
            ax.set_ylabel(self.ylabel, fontsize=self.label_size)
        if self.title:
            ax.set_title(self.title, fontsize=self.title_size)

    def get(self, attr_name):
        return getattr(self, attr_name, None)

def scatter_with_hue(df, x_col, y_col, hue_col, style: PlotStyle = None):
    style = style or PlotStyle(xlabel=x_col, ylabel=y_col, title=f'{y_col} vs {x_col}', legend_title=hue_col)

    fig, ax = plt.subplots(figsize=style.fig_size)
    
    scatter = ax.scatter(
        df[x_col], df[y_col],
        c=df[hue_col],
        cmap=style.cmap,
        edgecolor='k',
        alpha=0.7
    )

    style.apply_labels(ax)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(style.legend_title, fontsize=style.legend_title_size)

    plt.tight_layout()
    plt.show()

def scatter(df, x_col, y_col, style: PlotStyle = None):
    style = style or PlotStyle(xlabel=x_col, ylabel=y_col, title=f'{y_col} vs {x_col}')
    edgecolor = 'none' if style.get( 'edgecolor' ) is None else style.get( 'edgecolor' )

    fig, ax = plt.subplots(figsize=style.fig_size)

    ax.scatter(
        df[x_col], df[y_col],
        color='tab:blue',
        edgecolor=edgecolor,
        alpha=0.7
    )

    style.apply_labels(ax)

    plt.tight_layout()
    plt.show()
