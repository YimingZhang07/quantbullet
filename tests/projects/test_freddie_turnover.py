from dataclasses import replace
from datetime import date

import numpy as np
import polars as pl
import pytest

from quantbullet.utils.files import file_sha256
from projects.freddie_prepayment.config import Config, read_config
from projects.freddie_prepayment.prepare_turnover import CPI_BASE, CPI_BASE_MONTH, QUALITY_EXCLUSIONS, prepare, prepare_frame
from projects.freddie_prepayment.fit_turnover import fit, prediction_metrics, to_model_data
from projects.freddie_prepayment import report_turnover


def row(identifier="A", **updates):
    value = dict(loan_identifier=identifier, d_reporting_month=date(2025,6,1),
                 d_origination_month=date(2020,1,1), d_exit_month=None, d_maturity_month=date(2050,1,1),
                 vintage="2020Q1", c_factor=.8, c_orig_ltv=80., c_prev_balance=100000.,
                 c_age=65, c_incentive=-1., c_orig_fico=740., c_updated_ltv=60.,
                 c_orig_balance=125000., c_orig_cpi=257.971, c_hpi_ratio=1.25, zero_balance_code=None,
                 f_pre_status="CURRENT", f_status="CURRENT", is_consecutive_month=True,
                 is_ever_modified=False, f_purpose="P", f_occupancy="P", f_property_type="SF",
                 f_first_time_buyer="N", f_month="06", f_state="CA")
    value.update(dict.fromkeys(QUALITY_EXCLUSIONS,False))
    value.update(updates)
    return value


def test_target_risk_set_modified_and_incentive_boundary():
    records = [row("payoff",zero_balance_code="01",d_exit_month=date(2025,6,1)),
               row("dq",f_status="DQ30"), row("boundary",c_incentive=-.5),
               row("near_zero",c_incentive=-.49), row("modified",is_ever_modified=True),
               row("not_current",f_pre_status="DQ30"), row("gap",is_consecutive_month=False),
               row("no_balance",c_prev_balance=0.), row("bad_age",c_age=-1),
               row("matured",zero_balance_code="01",d_exit_month=date(2025,6,1),d_maturity_month=date(2025,6,1)),
               row("missing_maturity",zero_balance_code="01",d_exit_month=date(2025,6,1),d_maturity_month=None),
               row("mismatch",zero_balance_code="01",d_exit_month=date(2025,5,1),is_event_month_mismatch=True),
               row("other_exit",zero_balance_code="15",d_exit_month=date(2025,6,1),f_status="WHOLE_LOAN_SALE")]
    result=prepare_frame(pl.DataFrame(records).lazy()).collect()
    assert result["loan_identifier"].to_list()==["payoff","dq","boundary","matured","other_exit"]
    assert result["y_full_prepay"].to_list()==[1.,0.,0.,0.,0.]
    assert result["weight"].to_list()==[1.]*5
    assert result["row_id"].to_list()==list(range(5))


@pytest.mark.parametrize("field,value",[("c_orig_fico",None),("c_updated_ltv",float("inf")),("c_orig_balance",float("nan")),
                                         ("c_hpi_ratio",None),("c_hpi_ratio",float("inf")),("c_hpi_ratio",float("nan")),
                                         ("c_orig_cpi",None),("c_factor",None),("c_factor",float("nan"))])
def test_numeric_missing_and_nonfinite_excluded(field,value):
    result=prepare_frame(pl.DataFrame([row("valid"),row("bad",**{field:value})]).lazy()).collect()
    assert result["loan_identifier"].to_list()==["valid"]


def test_caps_raw_values_categories_and_previous_balance_weights():
    frame=prepare_frame(pl.DataFrame([
        row("A",c_age=134,c_orig_balance=2000000.,c_incentive=-9.,c_prev_balance=200000.,f_first_time_buyer=None),
        row("B",zero_balance_code="01",d_exit_month=date(2025,6,1),current_balance=0.,c_factor=1.05),
    ]).lazy()).collect()
    assert frame["c_age"].to_list()==[134,65]
    assert "c_age_fit" not in frame.columns
    assert frame["f_first_time_buyer"][0]=="MISSING"
    assert frame["weight"].to_list()==pytest.approx([4/3,2/3])
    model=to_model_data(frame)
    assert model.index.to_list()==[0,1]
    assert model["c_age_fit"].to_list()==pytest.approx([120.,65.])
    assert frame["c_orig_balance_real"].to_list()==pytest.approx([2e6*CPI_BASE/257.971, 125000.*CPI_BASE/257.971])
    assert model["c_orig_balance_real_fit"][0]==1_000_000.
    assert model["c_factor_fit"].to_list()==pytest.approx([.8,1.])
    assert model["c_incentive_fit"][0]==-5.
    assert model["y_full_prepay"].to_list()==[0.,1.]


def test_prepare_rejects_a_base_month_cpi_that_differs_from_cpi_base(tmp_path):
    source = tmp_path / "panel.parquet"
    pl.DataFrame([row("base", d_origination_month=CPI_BASE_MONTH, c_orig_cpi=CPI_BASE + 1)]).write_parquet(source)
    with pytest.raises(ValueError, match="CPI_BASE"):
        prepare(Config(source, tmp_path / "turnover"))


def test_hpi_ratio_raw_values_and_fit_caps():
    frame = prepare_frame(pl.DataFrame([
        row("low", c_hpi_ratio=.6), row("high", c_hpi_ratio=3.), row("unchanged", c_hpi_ratio=1.),
    ]).lazy()).collect()
    assert frame["c_hpi_ratio"].to_list() == [.6, 3., 1.]
    assert "c_hpi_growth" not in frame.columns and "c_hpi_ratio_fit" not in frame.columns
    assert to_model_data(frame)["c_hpi_ratio_fit"].to_list() == pytest.approx([.8, 2., 1.])


@pytest.fixture
def synthetic_config(tmp_path):
    rng=np.random.default_rng(42)
    n=1800
    events=rng.random(n)<.06
    records=[]
    for index in range(n):
        records.append(row(str(index),c_age=int(rng.integers(1,135)),c_incentive=-float(rng.uniform(.5,5.8)),
            c_orig_fico=float(rng.uniform(621,839)),c_updated_ltv=float(rng.uniform(6,119)),
            c_orig_balance=float(rng.uniform(26000,1490000)),c_hpi_ratio=float(rng.uniform(.85,2.4)),
            c_prev_balance=float(rng.uniform(10000,790000)),c_factor=float(rng.uniform(.15,1.05)),
            f_purpose=("P","C","N")[index%3],f_month=f"{index%12+1:02d}",
            f_state=("CA","NY")[index%2],f_property_type=("SF","CO")[index%2],
            zero_balance_code="01" if events[index] else None,
            d_exit_month=date(2025,6,1) if events[index] else None))
    source=tmp_path/"panel.parquet"
    pl.DataFrame(records).write_parquet(source)
    return Config(source,tmp_path/"turnover",n_iterations=6)


def test_fit_roundtrip_alignment_and_report_independence(synthetic_config,monkeypatch):
    config=synthetic_config
    stats=prepare(config)
    assert stats["rows"]==1800
    meta=fit(config)
    assert meta["rows"]==1800 and meta["interactions"]=={"c_age_fit":"f_purpose"}
    assert "c_hpi_ratio_fit" in meta["model_inputs"]
    assert "c_hpi_growth_fit" not in meta["model_inputs"]
    assert meta["clips"]["c_hpi_ratio"] == [.8, 2.]
    assert meta["knots"]["c_hpi_ratio"] == [.95, 1, 1.1, 1.25, 1.5, 1.75]
    bundle,frame=report_turnover.load_artifacts(config)
    assert meta['timing_seconds']['fit'] == meta['fit_seconds']
    assert set(meta['timing_seconds']) == {'read_frame', 'model_data', 'toolkit', 'container',
                                         'fit', 'predict', 'metrics', 'metadata'}
    assert all(value >= 0. for value in meta['timing_seconds'].values())
    assert meta['actual_sweeps'] == len(bundle['model'].loss_history_)
    assert len(bundle['model'].fit_timing_['sweep_seconds']) == meta['actual_sweeps']
    assert bundle['meta']['timing_seconds'] == meta['timing_seconds']
    assert frame.height==1800 and frame["row_id"].n_unique()==1800
    assert not any("burnout" in name for name in meta["model_inputs"])
    files=[config.output_root/name for name in ("turnover_frame.parquet","turnover_model.pkl","turnover_predictions.parquet")]
    hashes=[file_sha256(path) for path in files]
    import projects.freddie_prepayment.fit_turnover as fitting
    import projects.freddie_prepayment.prepare_turnover as preparation
    def forbidden(*args,**kwargs):
        raise AssertionError("Report must not refit or prepare")
    monkeypatch.setattr(fitting,"fit",forbidden)
    monkeypatch.setattr(preparation,"prepare",forbidden)
    # This small fixture needs lower support thresholds; production keeps 500/200.
    monkeypatch.setattr(report_turnover,"MIN_COUNT",5)
    monkeypatch.setattr(report_turnover,"MIN_COUNT_FACET",2)
    from quantbullet.linear_product_model import LinearProductModelToolkit
    from quantbullet.linear_product_model.mortgage_diagnostics import MortgageDiagnostics
    calls=[]
    def spy(cls, name):
        original=getattr(cls,name)
        def wrapped(*args,**kwargs):
            calls.append(name)
            if name == "plot" and len(args) > 1 and args[1] == "c_hpi_ratio_fit":
                calls.append("hpi_ratio_plot")
                assert kwargs["bins"] == .1
                assert kwargs["x_label"] == "ZHVI ratio since origination (1.0 = unchanged)"
            result=original(*args,**kwargs)
            if name=="plot_convergence_diagnostics":
                assert "Poisson" in result[1].flat[0].get_ylabel()
            return result
        if name=="plot_convergence_diagnostics":
            monkeypatch.setattr(cls,name,staticmethod(wrapped))
        else:
            monkeypatch.setattr(cls,name,wrapped)
    for name in ("plot_convergence_diagnostics","implied_actual_panels","categorical_panels"):
        spy(LinearProductModelToolkit,name)
    for name in ("factor_date_plot","incentive_plot","age_plot","cltv_plot","current_factor_plot","fico_plot","plot","facet_panels"):
        spy(MortgageDiagnostics,name)
    import quantbullet.plot.binned_plots as legacy
    def legacy_called(*args,**kwargs):
        raise AssertionError("Report charts must use the grouped-means system")
    monkeypatch.setattr(legacy,"plot_binned_actual_vs_pred",legacy_called)
    report_turnover.report(config)
    monkeypatch.setattr(report_turnover,"MIN_COUNT",10)
    report_turnover.report(config)
    assert all(name in calls for name in ("plot_convergence_diagnostics","implied_actual_panels","categorical_panels",
                                         "factor_date_plot","incentive_plot","age_plot","cltv_plot","current_factor_plot","fico_plot","plot",
                                         "facet_panels"))
    assert "hpi_ratio_plot" in calls
    assert [file_sha256(path) for path in files]==hashes
    assert (config.output_root/"turnover_report.pdf").stat().st_size>10000
    # Changed preparation is detected instead of silently mixing old predictions.
    source=config.output_root/"turnover_frame.parquet"
    pl.read_parquet(source).with_columns((pl.col("weight")*2).alias("weight")).write_parquet(source)
    with pytest.raises(ValueError,match="frame changed"):
        report_turnover.load_artifacts(config)


def test_failed_fit_preserves_artifacts(synthetic_config,monkeypatch):
    config=synthetic_config
    prepare(config); fit(config)
    files=[config.output_root/name for name in ("turnover_model.pkl","turnover_predictions.parquet")]
    before=[file_sha256(path) for path in files]
    from quantbullet.linear_product_model import LinearProductRegressorBCD
    def fail(*args,**kwargs):
        raise RuntimeError("synthetic fit failure")
    monkeypatch.setattr(LinearProductRegressorBCD,"fit",fail)
    with pytest.raises(RuntimeError,match="synthetic fit failure"):
        fit(config)
    assert [file_sha256(path) for path in files]==before


def test_missing_artifacts_do_not_run_upstream(synthetic_config):
    with pytest.raises(FileNotFoundError):
        report_turnover.report(synthetic_config)


def test_config_environment_and_paths(tmp_path,monkeypatch):
    path=tmp_path/"model.toml"
    path.write_text('[data]\npanel_path="${TURNOVER_TEST_ROOT}/panel.parquet"\noutput_root="model"\n')
    monkeypatch.delenv("TURNOVER_TEST_ROOT",raising=False)
    with pytest.raises(ValueError,match="environment variable"):
        read_config(path)
    monkeypatch.setenv("TURNOVER_TEST_ROOT",str(tmp_path))
    config=read_config(path)
    assert config.incentive_max==-.5 and config.n_iterations==60
    assert config.output_root==tmp_path/"model"
    with pytest.raises(ValueError,match="outside the repository"):
        replace(config,output_root=__import__('pathlib').Path(__file__).resolve().parents[2]/"model-output")


def test_metrics_preserve_negative_predictions():
    metrics=prediction_metrics([0,1],[-.1,.9],[1,1])
    assert metrics["prediction_range"]["negative"]==1
    assert metrics["min_prediction"]==-.1
    assert metrics["balance_weighted"]["poisson_deviance"] is None


def test_shared_polars_binning_does_not_require_pyarrow(monkeypatch):
    from quantbullet.plot.grouped_data import summarize_grouped_means
    def forbidden(*args,**kwargs):
        raise AssertionError("Optional PyArrow conversion must not be needed for aggregates")
    monkeypatch.setattr(pl.DataFrame,"to_pandas",forbidden)
    monkeypatch.setattr(pl.Series,"to_pandas",forbidden)
    frame=pl.DataFrame({"month":[date(2025,1,1)]*2+[date(2025,2,1)],
                        "y":[0.,1.,0.],"pred":[.1,.3,.2],"w":[1.,3.,2.]})
    summary=summarize_grouped_means(frame,x="month",y=["y","pred"],weight="w").summary
    assert summary["y__mean"].to_list()==[.75,0.]
    assert summary["pred__mean"].to_list()==pytest.approx([.25,.2])
    assert summary["count"].to_list()==[2,1]
