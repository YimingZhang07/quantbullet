"""Stage 3: compose saved-model diagnostics using the shared plotting interfaces."""

import argparse
import gc
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import polars as pl

from quantbullet.linear_product_model.mortgage_diagnostics import MortgageColnames, MortgageDiagnostics
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
    "c_orig_fico_fit": 20,
    "c_updated_ltv_fit": 5,
    "c_orig_balance_fit": 50000,
    "c_hpi_ratio_fit": .1,
}
MORTGAGE_COLUMNS = MortgageColnames(
    response=TARGET, model_preds={"Model": "pred_turnover"},
    incentive=("c_incentive_fit", .25), age=("c_age_fit", "discrete"),
    cltv=("c_updated_ltv_fit", 5), current_factor=("c_factor", .1),
    fico=("c_orig_fico_fit", 20), orig_dt="d_origination_month",
    factor_dt=("d_reporting_month", "discrete"), weight="c_prev_balance",
)


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

        def chart(title, draw, *, paged_cols=None):
            """Report assembly only; the supplied shared interface owns the figure."""
            started = time.perf_counter()
            pdf.add_page_break()
            pdf.add_heading(title)
            fig, _ = draw()
            if paged_cols is None:
                pdf.add_matplotlib_figure(fig, dpi=150, reserve_height=55)
            else:
                pdf.add_matplotlib_figure_paged(fig, n_cols=paged_cols, title=title, dpi=150)
            plt.close(fig)
            print(f"[report] {title}: {time.perf_counter()-started:.1f}s", flush=True)

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
        pdf.add_body("Model-numeric actual-vs-predicted charts and implied actuals use clipped _fit fields. Reporting month, previous factor, and original LTV have no fit column. Burnout and interactions are excluded. Numeric missing values are dropped; categorical missing values become MISSING.", font_size=9)
        pdf.add_body("HPI ratio is lag1 ZHVI / origination ZHVI: 1.0 means unchanged, 1.1 means a cumulative 10% increase. Ratio axes show multiples, not percentages.", font_size=9)

        chart("3. Convergence", lambda: toolkit.plot_convergence_diagnostics(model, figsize=(14,9)))
        chart("4. Numeric implied actuals", lambda: toolkit.plot_implied_actuals(
            model=model, dcontainer=container, sample_weights=weights, bin_config=IMPLIED_BIN_CONFIG,
            min_count=MIN_COUNT, n_cols=3))
        def categorical_figure():
            fig, axes = toolkit.plot_categorical_plots(
                model=model, dcontainer=container, sample_weights=weights, hspace=.45, wspace=.35)
            # Formatting only: use the shared PDF paginator to give dense state
            # labels more width; the toolkit still computes/draws every bar.
            fig.set_size_inches(12,12)
            for ax in axes:
                ax.tick_params(axis="x", labelsize=8)
            return fig, axes
        chart("5. Categorical implied actuals", categorical_figure, paged_cols=2)
        frame = frame.with_columns(*(pl.Series(name, model_data[name].to_numpy()) for name in FIT_NUMERIC))
        del container, model_data, weights
        gc.collect()

        diagnostics = MortgageDiagnostics(
            df=frame, colnames=MORTGAGE_COLUMNS,
            y_transform="smm_to_cpr", y_as_percent=True,
        )
        chart("6. Reporting month", lambda: diagnostics.factor_date_plot(
            min_count=MIN_COUNT, figsize=(14,5), x_label=MORTGAGE_COLUMNS.factor_dt))
        chart("7. Incentive", lambda: diagnostics.incentive_plot(
            min_count=MIN_COUNT, figsize=(12,5), x_label=MORTGAGE_COLUMNS.incentive))
        chart("8. Incentive by purpose", lambda: diagnostics.incentive_plot(
            facet_col="f_purpose", min_count=MIN_COUNT_FACET, n_cols=3, x_label=MORTGAGE_COLUMNS.incentive))
        chart("9. Age", lambda: diagnostics.age_plot(
            min_count=MIN_COUNT, figsize=(12,5), x_label=MORTGAGE_COLUMNS.age))
        chart("10. Age by purpose", lambda: diagnostics.age_plot(
            facet_col="f_purpose", min_count=MIN_COUNT_FACET, n_cols=3, x_label=MORTGAGE_COLUMNS.age))
        chart("11. Updated first-lien LTV", lambda: diagnostics.cltv_plot(
            min_count=MIN_COUNT, figsize=(12,5), x_label=MORTGAGE_COLUMNS.cltv))
        chart("12. Previous balance factor", lambda: diagnostics.current_factor_plot(
            min_count=MIN_COUNT, figsize=(12,5), x_label=MORTGAGE_COLUMNS.current_factor))
        chart("13. Original FICO", lambda: diagnostics.fico_plot(
            min_count=MIN_COUNT, figsize=(12,5), x_label=MORTGAGE_COLUMNS.fico))

        # Fields without a mortgage role use the same MortgageDiagnostics interface by source column.
        for title, column, step in (
            ("14. Original balance", "c_orig_balance_fit", 50000),
            ("15. ZHVI ratio since origination", "c_hpi_ratio_fit", .1),
            ("16. Original LTV", "c_orig_ltv", 5),
        ):
            label = "ZHVI ratio since origination (1.0 = unchanged)" if column == "c_hpi_ratio_fit" else column
            chart(title, lambda c=column,s=step,label=label: diagnostics.plot(
                c, bins=s, min_count=MIN_COUNT, figsize=(12,5), x_label=label, y_label="Full-payoff CPR proxy (%)"))
        pdf.save()
    print("[report] shared-interface report saved; preparation/model/predictions unchanged", flush=True)
    return {"rows":frame.height,"path":"turnover_report.pdf"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    report(read_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
