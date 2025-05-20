import matplotlib.pyplot as plt

class LineStyler:
    def __init__(self):
        self.color = 'tab:blue'
        self.linestyle = '-'
        self.linewidth = 2
        self.alpha = 0.8
        self.marker = None

    def set_color(self, c): self.color = c; return self
    def set_linestyle(self, ls): self.linestyle = ls; return self
    def set_linewidth(self, lw): self.linewidth = lw; return self
    def set_alpha(self, a): self.alpha = a; return self
    def set_marker(self, m): self.marker = m; return self

    def plot(self, ax, x, y, **kwargs):
        return ax.plot(
            x, y,
            color=self.color,
            linestyle=self.linestyle,
            linewidth=self.linewidth,
            alpha=self.alpha,
            marker=self.marker,
            **kwargs
        )
        
class ScatterStyler:
    def __init__(self):
        self.color = 'tab:blue'
        self.marker = 'o'
        self.alpha = 0.8
        self.edgecolor = 'k'
        self.size = 40
        self.cmap = 'viridis'

    def set_color(self, c): self.color = c; return self
    def set_marker(self, m): self.marker = m; return self
    def set_alpha(self, a): self.alpha = a; return self
    def set_edgecolor(self, ec): self.edgecolor = ec; return self
    def set_size(self, s): self.size = s; return self

    def plot(self, ax, x, y, **kwargs):
        return ax.scatter(
            x, y,
            c=self.color,
            marker=self.marker,
            alpha=self.alpha,
            edgecolors=self.edgecolor,
            s=self.size,
            **kwargs
        )
        
    def plot_with_hue(self, ax, x, y, hue, cmap=None, **kwargs):
        cmap = cmap or self.cmap
        return ax.scatter(
            x, y,
            c=hue,
            cmap=cmap,
            marker=self.marker,
            alpha=self.alpha,
            edgecolors=self.edgecolor,
            s=self.size,
            **kwargs
        )

class PlotStyler:
    def __init__(self):
        self.title = None
        self.title_size = 14
        self.xlabel = None
        self.ylabel = None
        self.label_size = 12
        self.x_date_format = None
        self.y_percent = False
        self.grid = False
        
    @property
    def default_x_date_format(self):
        return '%Y-%m-%d'
        
    # --- Fluent setters ---
    def set_title(self, title, size=None): self.title = title; self.title_size = size or self.title_size; return self
    def set_xlabel(self, label): self.xlabel = label; return self
    def set_ylabel(self, label): self.ylabel = label; return self
    def set_label_size(self, size): self.label_size = size; return self
    def set_x_date_format(self, fmt = None):
        self.x_date_format = fmt or self.default_x_date_format
        return self
    def set_y_percent(self, on=True): self.y_percent = on; return self
    def set_grid(self, on=True): self.grid = on; return self

    # --- Apply to ax ---
    def apply(self, ax):
        if self.title: ax.set_title(self.title, fontsize=self.title_size)
        if self.xlabel: ax.set_xlabel(self.xlabel, fontsize=self.label_size)
        if self.ylabel: ax.set_ylabel(self.ylabel, fontsize=self.label_size)
        if self.x_date_format:
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter(self.x_date_format))
        if self.y_percent:
            from matplotlib.ticker import PercentFormatter
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        if self.grid:
            ax.grid(True)
        return ax

    # --- Built-in plot methods ---
    def plot_line(self, ax, x, y, line_styler=None, **kwargs):
        if line_styler is None: line_styler = LineStyler()
        line_styler.plot(ax, x, y, **kwargs)
        return self.apply(ax)

    def plot_scatter(self, ax, x, y, scatter_styler=None, **kwargs):
        if scatter_styler is None: scatter_styler = ScatterStyler()
        scatter_styler.plot(ax, x, y, **kwargs)
        return self.apply(ax)
    
    def plot_scatter_with_hue(self, ax, x, y, hue, cmap=None, scatter_styler=None, **kwargs):
        if scatter_styler is None: scatter_styler = ScatterStyler()
        scatter = scatter_styler.plot_with_hue(ax, x, y, hue, cmap, **kwargs)
        plt.colorbar(scatter, ax=ax)
        return self.apply(ax)
