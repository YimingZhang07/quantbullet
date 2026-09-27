# Grouped weighted means

`plot_grouped_means` compares one or more numeric metrics over an x dimension,
with optional line groups, facets, and background count bars. It accepts pandas
and eager Polars DataFrames without requiring PyArrow.

```python
from quantbullet.plot import BinSpec, plot_grouped_means

result = plot_grouped_means(
    df,
    x="incentive",
    y=["historical_cpr", "model_cpr"],
    weight="upb",
    bins={"incentive": BinSpec.step(0.25)},
    group="vintage",
    count_mode="total",
    y_format=".0%",
    ylabel="CPR (UPB weighted)",
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

`x` is mandatory; specify at most two additional roles:

| Role | Meaning |
| --- | --- |
| `group` | Overlapped curves on the same axes |
| `row` | One subplot row per value/bin |
| `col` | One subplot column per value/bin |

- Use `col="vintage", wrap=3` for a wrapped collection of panels.
- Use `row="channel", col="occupancy"` for a matrix.
- Use `group="vintage", col="fico"` for grouped curves within FICO panels.
- `wrap` applies only to `col` without `row`.
- `panel_size=(5.2, 3.5)` is the size in inches **per subplot**.
- `share_y=True` shares the metric scale. Count scales are independent unless
  `share_count_y=True`. All panels share the x scale.

Without `group`, colors identify metrics. With `group`, colors identify groups
and line styles identify metrics. Encodings are consistent across all panels.
Empty matrix combinations display "No observations"; unused wrapped slots are
hidden. The result's `axes` and `count_axes` always have two dimensions.

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

Bins are fitted once using each dimension's finite values in the entire input,
before filtering missing group keys or metrics. Intervals are right-closed,
with the lowest edge included. Explicit out-of-range values and missing keys
are excluded; `result.data.excluded_count` reports their combined row count.
The actual edges and levels are available through `result.bin_info`.

Binned x uses numeric interval midpoints, raw numeric x uses its values, and
categorical x uses equal spacing. Ordered pandas categoricals preserve their
declared order (including empty categories); other raw levels sort by value.
Empty x bins within observed contexts retain missing means, breaking curves.

## Means and counts

For each metric independently, the mean is `sum(weight * y) / sum(weight)` over
finite y and finite weights. Without `weight`, all rows have unit weight.
Negative weights raise an error. Zero weights contribute no weight; a zero
denominator produces a missing mean. All-null metric columns are supported.

`summary` contains role columns `x`, and optionally `group`, `row`, `col`, plus:

- `count`: all records with valid grouping keys, even if their y/weight is missing.
- `<metric>__mean`: the weighted mean.
- `<metric>__valid_count`: records with finite y and weight, including zero weights.
- `<metric>__weight_sum`: the denominator used for that metric.

No rate or unit transformations occur implicitly. `y_format=".0%"` only formats
tick labels and expects proportions (0.12 means 12%).

The left axis shows means; the right axis shows counts:

- `count_mode="total"` (default): total records in the panel at each x.
- `count_mode="stacked"`: count bars stacked by `group` (requires `group`).
- `count_mode="none"`: omit bars and secondary axes.

Adding metrics never multiplies counts. Counts are row counts, not weight sums.
Bin widths can vary with quantiles: bar **height**, not area, represents count.

## Aggregate once, render again

```python
from quantbullet.plot import summarize_grouped_means, draw_grouped_means

data = summarize_grouped_means(
    df, x="incentive", y=["historical_cpr", "model_cpr"], weight="upb",
    group="vintage", bins={"incentive": BinSpec.step(0.25)},
)
first = draw_grouped_means(data, count_mode="total", y_format=".0%")
second = draw_grouped_means(data, count_mode="stacked", y_format=".0%")
```

Pass an existing `PlotTheme` to `theme` for axes, title, labels and legend
appearance. All data processing is separate from rendering. The implementation
extracts selected columns and uses vectorized NumPy aggregation for identical
pandas/Polars semantics; it is not a streaming/LazyFrame implementation.

## Reproducible examples

Run the unittest cases with either runner:

```shell
python -m unittest discover -s tests/plot -p test_grouped_means.py
python -m pytest tests/plot/test_grouped_means.py -q
```

`make_fake_mortgage_data()` in that test module creates 12,000 deterministic
synthetic records. Seven visual cases cover basic comparison, overlapping
groups, stacked counts, wrapped facets, a two-dimensional matrix, binned facets
with groups, and a single metric on categorical x.

Open `tests/_cache_dir/grouped_means/gallery.html` after running the tests.
Individual PNGs are saved beside it. Generated artifacts are ignored by Git.
