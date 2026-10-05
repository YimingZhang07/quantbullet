"""Stage 3: compose saved-model diagnostics using the shared plotting interfaces."""

import argparse
import gc
import pickle
import time
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")

import polars as pl

from quantbullet.linear_product_model.mortgage_diagnostics import MortgageColnames, MortgageDiagnostics
from quantbullet.plot.formatter import compact_number
from quantbullet.plot.binned_means import PRINT_BINNED_MEANS_STYLE
from quantbullet.plot.theme import PRINT_THEME
from quantbullet.preprocessing.transformers import FlatRampTransformer
from quantbullet.reporting import PdfTextReport
from quantbullet.utils.files import file_sha256, temporary_output

from .config import Config, read_config
from .fit_turnover import FIT_NUMERIC, make_container, to_model_data


MIN_COUNT = 500
MIN_COUNT_FACET = 200
TARGET = "y_full_prepay"
KEYS = ("row_id", "loan_identifier", "d_reporting_month")
IMPLIED_BIN_CONFIG = {
    "c_age_fit": "discrete",
    "c_incentive_fit": .25,
    "c_sato_fit": .125,
    "c_orig_fico_fit": 20,
    "c_orig_ltv_fit": 5,
    "c_updated_ltv_fit": 5,
    "c_orig_balance_real_fit": 50000,
    "c_factor_fit": .05,
    "c_hpi_ratio_fit": .1,
}
MORTGAGE_COLUMNS = MortgageColnames(
    response=TARGET, model_preds={"Model": "pred_turnover"},
    incentive=("c_incentive_fit", .25), age=("c_age_fit", "discrete"), sato=("c_sato_fit", .125),
    cltv=("c_updated_ltv_fit", 5), current_factor=("c_factor_fit", .05),
    fico=("c_orig_fico_fit", 20), orig_balance=("c_orig_balance_real_fit", 50000),
    current_balance=("c_prev_balance", 50000), orig_dt="d_origination_month",
    factor_dt=("d_reporting_month", "discrete"), weight="c_prev_balance",
)
CURRENT_BALANCE_BREAKS = (100000, 150000, 250000, 350000, 500000, 700000)
# Right-closed buckets, as pl.cut makes them: (-inf, 100K], (100K, 150K], ..., (700K, inf).
CURRENT_BALANCE_LABELS = (
    f"≤{compact_number(CURRENT_BALANCE_BREAKS[0])}",
    *(f"{compact_number(low)}–{compact_number(high)}"
      for low, high in zip(CURRENT_BALANCE_BREAKS, CURRENT_BALANCE_BREAKS[1:])),
    f">{compact_number(CURRENT_BALANCE_BREAKS[-1])}",
)
# Categorical features with too many levels for a grid column get a full-width row.
WIDE_CATEGORICAL = ("f_state",)


def load_artifacts(config: Config):
    root = config.output_root
    with (root / "turnover_model.pkl").open("rb") as file:
        bundle = pickle.load(file)
    meta = bundle["meta"]
    if file_sha256(root / "turnover_frame.parquet") != meta["frame_sha256"]:
        raise ValueError("Prepared frame changed; explicitly rerun fit before reporting")
    if file_sha256(root / "turnover_predictions.parquet") != meta["predictions_sha256"]:
        raise ValueError("Saved predictions do not belong to this model")
    frame = pl.read_parquet(root / "turnover_frame.parquet").join(
        pl.read_parquet(root / "turnover_predictions.parquet"), on=list(KEYS), how="inner", validate="1:1",
    )
    if frame.height != meta["rows"]:
        raise ValueError("Prediction keys do not align with the fitted rows")
    return bundle, frame


def report(config: Config) -> dict:
    bundle, frame = load_artifacts(config)
    model, toolkit, meta = bundle["model"], bundle["toolkit"], bundle["meta"]
    model_data = to_model_data(frame)
    container = make_container(model_data, toolkit)
    weights = model_data["weight"].to_numpy()
    print(f"[report] rebuilt saved toolkit design for {frame.height:,} rows; no refit", flush=True)

    with temporary_output(config.output_root / "turnover_report.pdf") as path:
        pdf = PdfTextReport(path, report_title="Freddie Turnover | Multiplicative Diagnostics", page_numbering=True)
        _, frame_h = pdf.content_size_inches()
        half_page = frame_h / 2 - .4  # two single charts, each under its heading, per page

        def timed(title, draw):
            """Figures are drawn while the PDF is built, at the size the page gives them."""
            def run(*args):
                started = time.perf_counter()
                fig = draw(*args)
                print(f"[report] {title}: drawn in {time.perf_counter()-started:.1f}s", flush=True)
                return fig
            return run

        def chart(title, draw, *, height=None, level=1, new_page=True):
            """Report assembly only; ``draw(width, height)`` returns the shared interface's figure."""
            if new_page:
                pdf.add_page_break()
            pdf.add_heading(title, level=level)
            pdf.add_figure(timed(title, draw), height=height)

        def panels(title, make, *, level=2, new_page=False):
            """Aggregate once now; the page layout sizes, splits and draws the panels."""
            if new_page:
                pdf.add_page_break()
            pdf.add_heading(title, level=level)
            started = time.perf_counter()
            result = make()
            print(f"[report] {title}: aggregated in {time.perf_counter()-started:.1f}s", flush=True)
            return replace(result, render=timed(title, result.render))

        pdf.add_heading("1. Fit summary")
        pdf.add_body("IN-SAMPLE ONLY. Negative-incentive full-prepayment baseline, not directly observed moving-only turnover. Cohort concentrated in 2022 onward; no train/test split.")
        pdf.add_kv_line({"Observations": f"{meta['rows']:,}", "Loans": f"{frame['loan_identifier'].n_unique():,}",
                         "Events": f"{frame[TARGET].sum():,.0f}", "Expanded columns": meta["expanded_columns"]})
        pdf.add_kv_line({"Loss": meta["loss"], "Fit seconds": f"{meta['fit_seconds']:.1f}",
                         "Best iteration": meta["best_iteration"], "Global scalar": f"{meta['global_scalar']:.6f}"})
        for kind in ("balance_weighted", "equal_weight"):
            metric = meta["metrics"].get(kind, {})
            ratio = metric.get("actual_over_expected")
            pdf.add_kv_line({"Weighting": kind, "Actual SMM": f"{metric.get('actual',float('nan')):.4%}",
                             "Predicted SMM": f"{metric.get('predicted',float('nan')):.4%}",
                             "A/E": "undefined" if ratio is None else f"{ratio:.4f}"})
        pdf.add_body("Raw prediction range counts: " + str(meta["metrics"]["prediction_range"]), font_size=9)
        pdf.add_body("Background bars show loan-month row counts on the right axis. Curves use balance weighting; bins below 500 rows (200 per purpose facet) retain Count bars but have no displayed curve value. Marker size is fixed.", font_size=9)
        pdf.add_body(f"Raw min/max: {meta['metrics'].get('min_prediction',float('nan')):.3g} / {meta['metrics'].get('max_prediction',float('nan')):.3g}. Predictions are not clipped.", font_size=9)
        pdf.add_body("Summary rates are monthly full-payoff SMM proxies. Actual-vs-predicted charts follow the reference report: bin-level SMM is converted to equivalent CPR using 1-(1-SMM)^12. Partial curtailments and scheduled-amortization corrections are not included.", font_size=9)
        if meta["smoke_rows"] is not None:
            pdf.add_body("SMOKE RUN: an explicit subset was fitted, not the final cohort.")

        pdf.add_page_break()
        pdf.add_heading("2. Feature configuration")
        config_lines = []
        for name, columns in toolkit.feature_groups.items():
            transformer = toolkit.preprocess_config[name]
            detail = f"FlatRamp knots={transformer.knots.tolist()}" if isinstance(transformer, FlatRampTransformer) else type(transformer).__name__
            config_lines.append(f"<b>{name}</b> ({len(columns)} terms): {detail}")
        pdf.add_list(config_lines)
        pdf.add_body("Model-numeric actual-vs-predicted charts and implied actuals use clipped _fit fields. Reporting month and current balance are diagnostics without a fit column. SATO is the original rate minus the origination-month national PMMS, in percentage points (no LLPA adjustment). Original balance is in January-2025 dollars (CPIAUCNS at the origination month; October 2025 interpolated). Balance factor is previous balance / original balance. The age ramp is estimated separately by f_purpose. Burnout is excluded. Numeric missing values are dropped; categorical missing values become MISSING.", font_size=9)
        pdf.add_body("HPI ratio is lag1 ZHVI / origination ZHVI: 1.0 means unchanged, 1.1 means a cumulative 10% increase. Ratio axes show multiples, not percentages.", font_size=9)

        chart("3. Convergence", lambda w, h: toolkit.plot_convergence_diagnostics(model, figsize=(w, h))[0])
        pdf.add_figure_grid(panels("4. Numeric implied actuals", lambda: toolkit.implied_actual_panels(
            model=model, dcontainer=container, sample_weights=weights, bin_config=IMPLIED_BIN_CONFIG,
            min_count=MIN_COUNT, n_cols=3, theme=PRINT_THEME, style=PRINT_BINNED_MEANS_STYLE),
            level=1, new_page=True))
        categorical = panels("5. Categorical implied actuals", lambda: toolkit.categorical_panels(
            model=model, dcontainer=container, sample_weights=weights, theme=PRINT_THEME),
            level=1, new_page=True)
        narrow = [name for name in categorical.panels if name not in WIDE_CATEGORICAL]
        wide = [name for name in categorical.panels if name in WIDE_CATEGORICAL]
        if narrow:
            pdf.add_figure_grid(categorical.subset(narrow), max_panel_aspect=.6)
        if wide:
            pdf.add_figure_grid(categorical.subset(wide, n_cols=1), max_panel_aspect=.3)
        frame = frame.with_columns(
            *(pl.Series(name, model_data[name].to_numpy()) for name in FIT_NUMERIC),
            pl.col("c_prev_balance").cut(list(CURRENT_BALANCE_BREAKS), labels=list(CURRENT_BALANCE_LABELS))
            .cast(pl.Enum(CURRENT_BALANCE_LABELS)).alias("current_balance_bucket"),
        )
        del container, model_data, weights
        gc.collect()

        diagnostics = MortgageDiagnostics(
            df=frame, colnames=MORTGAGE_COLUMNS, y_transform="smm_to_cpr", y_as_percent=True,
            theme=PRINT_THEME, style=PRINT_BINNED_MEANS_STYLE,
        )
        chart("6. Reporting month", lambda w, h: diagnostics.factor_date_plot(
            min_count=MIN_COUNT, figsize=(w, h), x_label=MORTGAGE_COLUMNS.factor_dt, title=None)[0], height=half_page)
        for number, name, role, role_plot, x_label in (
            (7, "Incentive", "incentive", diagnostics.incentive_plot, MORTGAGE_COLUMNS.incentive),
            (8, "Age", "age", diagnostics.age_plot, MORTGAGE_COLUMNS.age),
        ):
            pdf.add_page_break()
            pdf.add_heading(f"{number}. {name}", level=1)
            chart(f"{number}a. Overall", lambda w, h, role_plot=role_plot, x_label=x_label: role_plot(
                min_count=MIN_COUNT, figsize=(w, h), x_label=x_label, title=None)[0],
                height=frame_h / 2, level=2, new_page=False)
            for title, facet, facet_label in (
                (f"{number}b. By purpose", "f_purpose", "Purpose"),
                (f"{number}c. By current balance", "current_balance_bucket", "Balance"),
            ):
                pdf.add_figure_grid(panels(title, lambda role=role, facet=facet, facet_label=facet_label, x_label=x_label:
                    diagnostics.facet_panels(role, facet, facet_label=facet_label, min_count=MIN_COUNT_FACET,
                                             n_cols=3, x_label=x_label, y_scale='shared',
                                             y_ticks='all', count_ticks='all')))

        singles = [
            ("9. Updated first-lien LTV", diagnostics.cltv_plot, MORTGAGE_COLUMNS.cltv),
            ("10. Balance factor (previous / original)", diagnostics.current_factor_plot, MORTGAGE_COLUMNS.current_factor),
            ("11. Original FICO", diagnostics.fico_plot, MORTGAGE_COLUMNS.fico),
            ("12. Original balance (Jan-2025 USD)", diagnostics.orig_balance_plot, MORTGAGE_COLUMNS.orig_balance),
            ("13. Current balance", diagnostics.current_balance_plot, MORTGAGE_COLUMNS.current_balance),
        ]
        for index, (title, role_plot, x_label) in enumerate(singles):
            chart(title, lambda w, h, role_plot=role_plot, x_label=x_label: role_plot(
                min_count=MIN_COUNT, figsize=(w, h), x_label=x_label, title=None)[0],
                height=half_page, new_page=index == 0)
        for title, column, step in (
            ("14. ZHVI ratio since origination", "c_hpi_ratio_fit", .1),
            ("15. Original LTV", "c_orig_ltv_fit", 5),
        ):
            label = "ZHVI ratio since origination (1.0 = unchanged)" if column == "c_hpi_ratio_fit" else column
            chart(title, lambda w, h, c=column, s=step, label=label: diagnostics.plot(
                c, bins=s, min_count=MIN_COUNT, figsize=(w, h), x_label=label,
                y_label="Full-payoff CPR proxy (%)", title=None)[0], height=half_page, new_page=False)
        chart("16. SATO (origination rate - PMMS)", lambda w, h: diagnostics.sato_plot(
            min_count=MIN_COUNT, figsize=(w, h), x_label=MORTGAGE_COLUMNS.sato, title=None)[0],
            height=half_page, new_page=False)
        pdf.save()
    print("[report] shared-interface report saved; preparation/model/predictions unchanged", flush=True)
    return {"rows":frame.height,"path":"turnover_report.pdf"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    report(read_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
