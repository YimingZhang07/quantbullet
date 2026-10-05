# Binned means

`plot_binned_means` draws the weighted mean of one or more numeric metrics per
bin of an x dimension, with optional line groups, facets, and background bars
that show row counts or summed weights. It accepts pandas and eager Polars DataFrames without requiring PyArrow.

Three functions share one summary object, `BinnedMeans`:

| Function | Does |
| --- | --- |
| `summarize_binned_means(df, ...)` | Aggregates raw rows once and returns `BinnedMeans` |
| `draw_binned_means(data, ...)` | Draws a `BinnedMeans`, or any subset of it, without re-aggregating |
| `plot_binned_means(df, ...)` | Both in one call |

## Architecture

这套接口按职责分层：

| Layer | Responsibility |
| --- | --- |
| `utils/grouped_stats.py` | 通用 Polars group-by：count、总 weight_sum，以及每个指标的 valid count、weighted sum、weight sum、mean |
| `plot/binned_means/binning.py` | `BinSpec` 与 Polars key 表达式：分箱边界、rounding、类别 codes |
| `plot/binned_means/summary.py` | `summarize_binned_means`、`BinnedMeans`：类别顺序、空 bins、x 位置及 bin-level 变换 |
| `plot/binned_means/draw.py` | `draw_binned_means`：曲线、背景柱（行数或权重）、双轴、legend 与 layout |
| `plot/panels.py` | `PanelSet`、`panel_grid`、`label_outer_panels`：报告分页与外侧标签 |
| `MortgageDiagnostics` | Mortgage 字段映射、`plot()` 与各业务方法、bin-level SMM → CPR |

`MortgageDiagnostics.plot → summarize_binned_means → grouped_weighted_summary →
draw_binned_means` 是独立调用链，不调用旧 `binned_plots`。旧模块保持原有的点大小
样式，`plot_binned_actual_vs_pred` 会发出 `DeprecationWarning`。Numeric implied actual
的 loss-specific 公式由 toolkit 计算，小型结果通过 `BinnedMeans.from_summary()`
交给 renderer；不改算成普通 weighted mean。

统计函数也可独立用于数据检查：

```python
from quantbullet.utils.grouped_stats import grouped_weighted_summary

stats = grouped_weighted_summary(
    df, by=["month", "purpose"], metrics=["actual", "prediction"], weight="upb",
)
```

这个接口只接受 eager Polars，不负责分箱或丢弃 null group keys，也不加载 plotting
dependencies。`utils` 与 `plot` 的 exports 使用 lazy imports，独立加载新体系时不会
顺带加载旧 `binned_plots`。Plot 数据准备则排除无效 dimension keys，
保留 `excluded_count`。两种 DataFrame 输入最终使用同一个统计实现。

```python
from quantbullet.plot import BinSpec, plot_binned_means

result = plot_binned_means(
    df,
    x="incentive",
    y=["historical_cpr", "model_cpr"],
    weight="upb",
    bins={"incentive": BinSpec.step(0.25)},
    group="vintage",
    bar_mode="total",
    y_format=".0%",
    y_label="CPR (UPB weighted)",
    labels={
        "incentive": "Refinance incentive (pp)",
        "historical_cpr": "Historical CPR",
        "model_cpr": "Model CPR",
    },
)
result.fig.savefig("cpr.png", dpi=150)
result.summary  # exact statistics used in the figure
```

## Dimensions and layout

`x` is mandatory; the other roles are optional and can be combined:

| Role | Meaning |
| --- | --- |
| `group` | Overlapped curves on the same axes |
| `row` | One subplot row per value/bin |
| `col` | One subplot column per value/bin |

- Use `col="vintage", wrap=3` for a wrapped collection of panels.
- Use `row="channel", col="occupancy"` for a matrix.
- Use `group="vintage", col="fico"` for grouped curves within FICO panels.
- Use `group="purpose", row="occupancy", col="vintage"` for grouped curves
  within a matrix. Every role splits the rows further, so coarser x bins and
  `min_count` keep sparse cells readable.
- `wrap` applies only to `col` without `row`. With fewer panels than `wrap`, the
  grid narrows to fit; `pad_to_wrap=True` keeps all `wrap` columns.
- `panel_size=(5.2, 3.5)` is the size in inches **per subplot**.
- All panels share the x scale and show their x tick labels.

## Y scales, tick labels and axis titles

Each figure has two y axes: the left axis for means (`y_*`) and the right axis
for bars (`bar_*`). Each has one argument for its range, one for its tick
labels and one for its title; `x_titles` places the x title. `y_label` and
`labels` set the title text:

| Argument | Values (default first) | Controls |
| --- | --- | --- |
| `y_scale` | `"shared"`, `"free"`, `(low, high)` | Left-axis range: one for all panels, one per panel, or fixed |
| `bar_scale` | `"free"`, `"shared"`, top value | Bar-axis range, same meaning |
| `y_ticks` | `"all"`, `"outer"` | Left tick labels on every panel, or only the first panel of each row |
| `bar_ticks` | `"all"`, `"outer"` | Bar tick labels on every panel, or only the last panel of each row |
| `y_titles` | `"all"`, `"outer"` | Left-axis title on every panel, or only the first panel of each row |
| `bar_titles` | `"all"`, `"outer"` | Bar-axis title on every panel, or only the last panel of each row |
| `x_titles` | `"all"`, `"outer"` | x title on every panel, or only the lowest panel of each column |

A scale only sets ranges. Tick labels appear on every panel until `y_ticks` or
`bar_ticks` says `"outer"`. That setting needs a shared or fixed scale and
raises an error with `"free"`, because each free panel needs its own numbers.
The two axes are independent, so any left setting combines with any bar
setting, and titles never depend on the scale:

| Want | Arguments |
| --- | --- |
| Left axis shared, tick labels on every panel | `y_scale="shared"` |
| Left axis shared, tick labels on the outer panels | `y_scale="shared", y_ticks="outer"` |
| Left axis per panel | `y_scale="free"` |
| Bar axis shared, tick labels on every panel | `bar_scale="shared"` |
| Bar axis shared, tick labels on the outer panels | `bar_scale="shared", bar_ticks="outer"` |
| Bar axis per panel | `bar_scale="free"` |
| Remove repeated titles (any scale) | `y_titles="outer"`, `bar_titles="outer"`, `x_titles="outer"`, each on its own |

Fixed values (`y_scale=(0, 0.3)`, `bar_scale=50_000`) keep separate figures,
such as pages drawn from one aggregation, on the same scale.
`MortgageDiagnostics` takes the same arguments. Its `facet_panels` reads
`"shared"` as one scale across every page. It defaults every axis title to
`"outer"`, and tick labels to `"outer"` wherever the scale is not free.

Without `group`, colors identify metrics and numeric/binned curves are solid.
With `group`, colors identify groups and line styles identify metrics.
Categorical x values are shown as unconnected points by default; marker shapes
identify metrics, including when colors identify groups. Encodings are
consistent across all panels.
Empty matrix combinations display "No observations"; unused wrapped slots are
hidden. The result's `axes` and `bar_axes` always have two dimensions.

## Binning

`bins` maps source dimension names to `BinSpec` instances. Omit a dimension to
group by its exact values. Specs apply to x, group, row, or col alike:

```python
bins = {
    "incentive": BinSpec.quantile(12),
    "fico": BinSpec.edges([600, 700, 740, 780, 850]),
}
```

- `BinSpec.edges(...)`: finite, strictly increasing edges.
- `BinSpec.quantile(n)`: global quantiles, with duplicate edges removed. Constant
  data produces a single bin.
- `BinSpec.step(width)`: intervals with edges at multiples of width, covering
  the data range. This bins values; it does **not** round them to nearest steps.
- `BinSpec.round(width)`: round to nearest width multiple, with ties to even.
  Mortgage numeric bin configuration uses this rule. For width 0.5, 0.25
  maps to 0 and 0.75 maps to 1; these are centers, not interval boundaries.

Bins are fitted once using each dimension's finite values in the entire input,
before filtering missing group keys or metrics. Intervals are right-closed,
with the lowest edge included. Explicit out-of-range values and missing keys
are excluded; `result.data.excluded_count` reports their combined row count.
The actual edges and levels are available through `result.bin_info`.

Binned x uses numeric interval midpoints, raw numeric x uses its values, and
categorical x uses equal spacing. Pandas categoricals and Polars Enums preserve
their declared order (including empty categories); other raw levels sort by value.
Date dimensions use real date positions and a date formatter.
Categorical values remain unconnected even when their declared order is known;
set `style.connect_categorical=True` when connecting them has meaning.
Empty x bins within observed contexts retain missing means, breaking curves.

## Means and counts

For each metric independently, the mean is `sum(weight * y) / sum(weight)` over
finite y and finite weights. Without `weight`, all rows have unit weight.
Negative weights raise an error. Zero weights contribute no weight; a zero
denominator produces a missing mean. All-null metric columns are supported.

`summary` contains role columns `x`, and optionally `group`, `row`, `col`, plus:

- `count`: all records with valid grouping keys, even if their y/weight is missing.
- `weight_sum`: the finite weights of those records, whatever their y (equals
  `count` without `weight`). `data.weight` names the weight column.
- `<metric>__mean`: the weighted mean.
- `<metric>__valid_count`: records with finite y and weight, including zero weights.
- `<metric>__weight_sum`: the denominator used for that metric.
- `<metric>__weighted_sum`: the numerator used for that metric.

No rate or unit transformations occur implicitly. `y_format=".0%"` only formats
tick labels and expects proportions (0.12 means 12%).

Transformations and support thresholds apply to the aggregated table only:

- `data.transform_means(fn)` transforms bin-level means, e.g. SMM → CPR after
  weighting. Weighted sums keep their original units.
- `data.mask_sparse(n)` hides means where `count < n` and keeps the bars,
  so curves break at sparse bins.
- `plot_binned_means(..., y_transform=fn, min_count=n)` applies both in that order.

Both methods return a new object; `result.summary` holds the values that were drawn.

The left axis shows means; the right axis shows bars. `bar_value` sets what a
bar measures and `bar_mode` how bars are drawn; any value combines with any mode:

- `bar_value="count"` (default): rows at each x. The axis title is "Count".
- `bar_value="weight"`: the summed `weight` (e.g. UPB) at each x; needs a
  weighted summary. The axis title is the weight's label from `labels`.
- `bar_mode="total"` (default): one bar per x for the whole panel.
- `bar_mode="stacked"`: bars stacked by `group` (requires `group`).
- `bar_mode="none"`: omit bars and secondary axes.

`bar_label` overrides the axis title, e.g. `bar_label="Loan-months"`. Adding
metrics never multiplies bars. `min_count` and "No observations" always use row
counts, since they judge sample size. Bin widths can vary with quantiles: bar
**height**, not area, represents the value.

## Aggregate once, render again

```python
from quantbullet.plot import summarize_binned_means, draw_binned_means

data = summarize_binned_means(
    df, x="incentive", y=["historical_cpr", "model_cpr"], weight="upb",
    group="vintage", bins={"incentive": BinSpec.step(0.25)},
)
first = draw_binned_means(data, bar_mode="total", y_format=".0%")
second = draw_binned_means(data, bar_mode="stacked", y_format=".0%")
```

`PlotTheme` controls shared colors, axes, titles, labels, and legend appearance;
`BinnedMeansStyle` controls the mean curves or points, bars, and categorical ticks.
Both are immutable, so derive a variant with `dataclasses.replace`:

```python
from dataclasses import replace
import pandas as pd
from quantbullet.plot import (
    MINIMAL_THEME, DEFAULT_BINNED_MEANS_STYLE, plot_binned_means,
)

custom_theme = replace(
    MINIMAL_THEME,
    palette=("#16697A", "#DB6400"),
    facecolor="#F4F8F7",
    muted_text_color="#5D6D70",
    figure_title_fontsize=16,
)
custom_style = replace(
    DEFAULT_BINNED_MEANS_STYLE,
    metric_linestyles=("-", ":"),
    marker="s",
    stacked_bar_alpha=0.25,
)
result = plot_binned_means(
    df, x="incentive", y=["historical_cpr", "model_cpr"],
    weight="upb", group="vintage", bar_mode="stacked",
    theme=custom_theme, style=custom_style,
)

# Connect categorical values only when their order has meaning.
ordered_df = df.assign(vintage_label=pd.Categorical(
    df["vintage"].astype(str), categories=["2019", "2020", "2021"], ordered=True,
))
connected_style = replace(DEFAULT_BINNED_MEANS_STYLE, connect_categorical=True)
categorical_result = plot_binned_means(
    ordered_df, x="vintage_label", y="historical_cpr", weight="upb",
    style=connected_style,
)
```

All data processing is separate from rendering. 全量行只处理一次：

1. 只取用到的列组成 Polars frame。Pandas 列经 NumPy 转换（不需要 PyArrow），
   categorical 转成 codes 并保留类别顺序。
2. 每个维度变成一个 Polars key 表达式：离散值直接作 key；`round` 为
   `(x / w).round() * w`；`edges` / `step` / `quantile` 先用一次小 select 求出
   edges，再由 `search_sorted` 得到区间编号。无效 key 为 null。
3. `grouped_weighted_summary` 做唯一一次 group-by。
4. 只在聚合后的小表上排序 levels、补全空 bins、计算 x 位置；null key 组的行数计入
   `excluded_count`。

这是 eager、非 streaming 的接口；1000 万行时聚合耗时与旧 `prepare_binned_data_polars`
同一量级。

Mortgage 的公开业务方法继续返回 `(fig, primary_axes)`，都委托给
`MortgageDiagnostics.plot(x, bins=...)`。`x` 可以是 `MortgageColnames` 的 role，
也可以是任意源列名；`bins` 覆盖 `bin_config`，可为 `'discrete'`、rounding 单位或
`BinSpec`。背景柱的坐标轴留在 figure 内；facets 共用柱子的尺度；柱子默认是行数，`bar_value='weight'` 时改为权重之和。`min_count` 只隐藏
曲线，保留背景柱；SMM 聚合后才转为 CPR，不对逐行 binary target 转换。`figsize` 为
每个 panel 的尺寸，`n_cols` 控制分面布局。

```python
diagnostics.plot("incentive", facet_col="purpose", min_count=200)
diagnostics.plot("orig_balance", bins=50_000, x_label="Original balance")
```

`draw_binned_means(..., ax=existing_axes, legend="none")` 可把单个 panel 画进调用方
自己的网格，由调用方只保留一个 legend。

## Reproducible examples

Run the tests for the package, including the gallery:

```shell
python -m pytest tests/plot/binned_means -q
```

`make_fake_mortgage_data()` in `tests/plot/binned_means/test_gallery.py` creates 40,000 deterministic
synthetic records. Ten visual cases are grouped by layout:

| Layout | Cases |
| --- | --- |
| Single panel | 01 multiple metrics · 02 groups overlaid · 03 groups + stacked counts · 04 groups + stacked UPB |
| Facets | 05 wrapped facets · 06 groups with quantile and edge bins · 07 groups, one metric, SMM → CPR |
| Row x col matrix | 08 multiple metrics with shared scales · 09 groups overlaid |
| Categorical x | 10 unconnected points with column facets |

Set `QB_TEST_KEEP_ARTIFACTS=1` in your `.env` (see `.env.example`) or process
environment before running the tests to retain the gallery. Then open
`tests/_cache_dir/binned_means/gallery.html`. With the setting off, the gallery
is generated in a temporary directory and cleaned up after the unittest class.
The index groups the cases by layout, with bilingual titles. Each case lists
parameter tags read from its call, so they always match the code. Each image
shows the exact `plot_binned_means(...)` call used to generate it.
Replace `self.df` in those unittest calls with your own DataFrame.
Individual PNGs are saved beside it. Generated artifacts are ignored by Git.
