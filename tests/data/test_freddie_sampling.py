from dataclasses import replace
from datetime import date
import json
from pathlib import Path

import polars as pl
import pytest

from procs.freddie_sflld import build_sample_panel as process
from quantbullet.data.freddie_sflld import (
    allocate_vintage_counts, filter_orig_loans, sample_loan_ids, scan_loan_panel,
)
from quantbullet.data.freddie_sflld.schema import ORIG_COLUMNS, PERF_COLUMNS


def _write(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema={name: pl.String for name in columns}).write_parquet(path)


@pytest.fixture
def config(tmp_path):
    root = tmp_path / "source"
    quarters = {}
    for vintage, identifiers, performance in (
        ("2015Q1", ["A", "B", "C", "D"], [
            {"loan_identifier": "A", "period": "201501"},
            {"loan_identifier": "A", "period": "201502", "zero_balance_code": "01"},
            {"loan_identifier": "A", "period": "201503", "actual_loss": "70"},
            {"loan_identifier": "B", "period": "201503"},
            {"loan_identifier": "C", "period": "201503"},
        ]),
        ("2015Q2", ["E", "F"], [
            {"loan_identifier": "E", "period": "201506"},
            {"loan_identifier": "F", "period": "201507"},
        ]),
    ):
        orig = f"parquet/orig/vintage={vintage}/current.parquet"
        perf = f"parquet/perf/vintage={vintage}/current.parquet"
        _write(root / orig, ORIG_COLUMNS, [
            {"loan_identifier": identifier, "classic_fico": "750", "postal_code": "001",
             "amortization_type": "FRM", "original_loan_term": "360"}
            for identifier in identifiers
        ])
        _write(root / perf, PERF_COLUMNS, performance)
        quarters[vintage] = {"orig": {"path": orig}, "perf": {"path": perf}}
    (root / "manifests").mkdir()
    (root / "manifests/conversion.json").write_text(json.dumps({"manifest_version": 1, "quarters": quarters}))
    return process.SampleConfig(root, tmp_path / "output", "2015Q1", "2015Q2", 6, 42)


def _source(config, vintage="2015Q1", kind="orig"):
    return config.freddie_root / f"parquet/{kind}/vintage={vintage}/current.parquet"


def _panel(config):
    return pl.read_parquet(str(config.output_root / "panel/vintage=*/panel.parquet"), hive_partitioning=False)


def test_largest_remainders_and_tie_order():
    assert allocate_vintage_counts({"2015Q2": 1, "2015Q1": 1, "2015Q3": 1}, 2) == {
        "2015Q1": 1, "2015Q2": 1, "2015Q3": 0,
    }
    assert allocate_vintage_counts({"2015Q1": 4, "2015Q2": 2}, 3) == {"2015Q1": 2, "2015Q2": 1}
    assert allocate_vintage_counts({"2015Q1": 0, "2015Q2": 2}, 2) == {"2015Q1": 0, "2015Q2": 2}


@pytest.mark.parametrize("populations,n", [({}, 1), ({"Q": -1}, 1), ({"Q": 0}, 1), ({"Q": 1}, 2), ({"Q": 2}, 0)])
def test_invalid_allocation(populations, n):
    with pytest.raises(ValueError):
        allocate_vintage_counts(populations, n)


def test_reproducible_sample_without_replacement():
    ids = pl.DataFrame({"loan_identifier": [f"loan-{i:03d}" for i in range(100)]})
    kwargs = {"n_loans": 20, "seed": 42, "vintage": "2015Q1"}
    first = sample_loan_ids(ids, **kwargs)
    assert first.equals(sample_loan_ids(ids.reverse(), **kwargs))
    assert first.height == first["loan_identifier"].n_unique() == 20
    assert not first.equals(sample_loan_ids(ids, **{**kwargs, "seed": 43}))
    assert not first.equals(sample_loan_ids(ids, **{**kwargs, "vintage": "2015Q2"}))


@pytest.mark.parametrize("ids", [["A", "A"], ["A", None], ["A", " "]])
def test_bad_orig_identifiers(ids):
    with pytest.raises(ValueError, match="nonempty and unique"):
        sample_loan_ids(pl.DataFrame({"loan_identifier": ids}), n_loans=1, seed=42, vintage="2015Q1")


def test_complete_panel_and_orig_only_loan(config):
    summary = process.build_sample_panel(config)
    loans = pl.read_parquet(config.output_root / "sampled_loans.parquet")
    panel = _panel(config)
    assert loans.height == loans["loan_identifier"].n_unique() == 6
    assert panel.height == 7 and panel["loan_identifier"].n_unique() == 5
    assert summary["panel_loans"] + summary["loans_without_perf"] == summary["sampled_loans"] == 6
    assert loans.width == 32 and panel.width == 67
    assert panel.schema["month"] == pl.Date
    assert all(dtype == pl.String for name, dtype in panel.schema.items() if name != "month")
    assert loans["postal_code"].to_list() == ["001"] * 6
    terminal = panel.filter((pl.col("loan_identifier") == "A") & (pl.col("period") == "201502"))
    assert terminal["zero_balance_code"].item() == "01"
    assert terminal["month"].item() == date(2015, 2, 1)
    assert panel.filter((pl.col("loan_identifier") == "A") & (pl.col("period") == "201503"))["actual_loss"].item() == "70"
    assert loans.filter(pl.col("loan_identifier") == "D").height == 1
    assert panel.filter(pl.col("loan_identifier") == "D").height == 0
    assert [row["quota"] for row in summary["vintages"]] == [4, 2]
    saved = (config.output_root / "sampling_summary.json").read_text()
    assert str(config.freddie_root) not in saved and str(config.output_root) not in saved
    assert json.loads(saved) == summary


def test_rerun_reproducibility_and_no_stale_partitions(config):
    process.build_sample_panel(replace(config, n_loans=3))
    first = pl.read_parquet(config.output_root / "sampled_loans.parquet")
    process.build_sample_panel(replace(config, n_loans=3))
    assert first.equals(pl.read_parquet(config.output_root / "sampled_loans.parquet"))
    note = config.output_root / "notes.txt"
    note.write_text("keep")
    process.build_sample_panel(replace(config, start_vintage="2015Q2", n_loans=2))
    assert not (config.output_root / "panel/vintage=2015Q1").exists()
    assert set(_panel(config)["loan_identifier"]) == {"E", "F"}
    assert note.read_text() == "keep"


@pytest.mark.parametrize("period", ["invalid", "201513", "20151", None])
def test_invalid_month_aborts_without_summary(config, period):
    path = _source(config, kind="perf")
    _write(path, PERF_COLUMNS, [{"loan_identifier": "A", "period": period}])
    with pytest.raises(pl.exceptions.PolarsError):
        process.build_sample_panel(config)
    assert not (config.output_root / "sampling_summary.json").exists()


def test_duplicate_loan_month_fails(config):
    path = _source(config, kind="perf")
    _write(path, PERF_COLUMNS, [{"loan_identifier": "A", "period": "201501"}] * 2)
    with pytest.raises(ValueError, match="duplicate loan-month"):
        process.build_sample_panel(config)
    assert not (config.output_root / "sampling_summary.json").exists()


def test_duplicate_static_join_fails(config):
    path = _source(config)
    _write(path, ORIG_COLUMNS, [{"loan_identifier": "A"}] * 2)
    with pytest.raises(pl.exceptions.PolarsError):
        scan_loan_panel(path, _source(config, kind="perf"), pl.DataFrame({"loan_identifier": ["A"]}), vintage="2015Q1").collect()
    with pytest.raises(ValueError, match="nonempty and unique"):
        process.build_sample_panel(replace(config, n_loans=4))


def test_zero_quota_and_all_orig_only_sample(config):
    _write(_source(config, kind="perf"), PERF_COLUMNS, [{"loan_identifier": "unsampled", "period": "invalid"}])
    summary = process.build_sample_panel(replace(config, n_loans=1))
    assert summary["panel_rows"] == summary["panel_loans"] == 0
    assert summary["loans_without_perf"] == 1
    assert _panel(config).height == 0
    assert not (config.output_root / "panel/vintage=2015Q2").exists()


def test_cross_vintage_sample_identifiers_must_be_unique(config):
    _write(_source(config, vintage="2015Q2"), ORIG_COLUMNS, [{"loan_identifier": "A"}, {"loan_identifier": "F"}])
    with pytest.raises(ValueError, match="globally unique"):
        process.build_sample_panel(config)


def test_missing_quarter_and_file_fail_before_cleanup(config):
    with pytest.raises(ValueError, match="Missing vintages"):
        process.build_sample_panel(replace(config, end_vintage="2015Q3"))
    _source(config, kind="perf").unlink()
    with pytest.raises(ValueError, match="missing or invalid perf"):
        process.build_sample_panel(config)


def test_output_cannot_overlap_sources_or_repository(config):
    with pytest.raises(ValueError, match="dedicated directory"):
        process.build_sample_panel(replace(config, output_root=config.freddie_root))
    with pytest.raises(ValueError, match="outside the repository"):
        process.build_sample_panel(replace(config, output_root=Path(__file__).resolve().parents[2] / "local-data"))
    assert _source(config).exists()


def test_optional_orig_filters_and_numeric_term():
    orig = pl.DataFrame({
        "loan_identifier": ["A", "B", "C", "D", "E", "F"],
        "amortization_type": ["FRM", "FRM", "ARM", "FRM", "FRM", None],
        "original_loan_term": ["360", "180", "360", "0360", "invalid", None],
    }).lazy()
    assert filter_orig_loans(orig).collect().height == 6
    assert filter_orig_loans(orig, amortization_type="FRM").collect().height == 4
    assert filter_orig_loans(orig, original_loan_term=360).collect()["loan_identifier"].to_list() == ["A", "C", "D"]
    assert filter_orig_loans(orig, amortization_type="FRM", original_loan_term=360).collect()["loan_identifier"].to_list() == ["A", "D"]


@pytest.mark.parametrize("filters", [
    {"amortization_type": "fixed"}, {"original_loan_term": "360"},
    {"original_loan_term": 0}, {"original_loan_term": True},
])
def test_invalid_product_filters_fail_before_cleanup(config, filters):
    process.build_sample_panel(config)
    saved = (config.output_root / "sampling_summary.json").read_text()
    with pytest.raises(ValueError):
        process.build_sample_panel(replace(config, **filters))
    assert (config.output_root / "sampling_summary.json").read_text() == saved


def test_filter_before_quota_allocation_and_sampling(config):
    for vintage, products in (
        ("2015Q1", [("FRM", "360"), ("FRM", "180"), ("ARM", "360"), ("FRM", "360")]),
        ("2015Q2", [("FRM", "360"), ("FRM", "180")]),
    ):
        path = _source(config, vintage=vintage)
        orig = pl.read_parquet(path).with_columns(
            pl.Series("amortization_type", [p[0] for p in products]),
            pl.Series("original_loan_term", [p[1] for p in products]),
        )
        orig.write_parquet(path)
    filtered = replace(config, n_loans=3, amortization_type="FRM", original_loan_term=360)
    summary = process.build_sample_panel(filtered)
    loans = pl.read_parquet(config.output_root / "sampled_loans.parquet")
    assert set(loans["loan_identifier"]) == {"A", "D", "E"}
    assert summary["raw_population_loans"] == 6 and summary["population_loans"] == 3
    assert summary["sampling"]["population"] == "filtered_orig"
    assert summary["sampling"]["amortization_type"] == "FRM"
    assert summary["sampling"]["original_loan_term"] == 360
    assert [r["quota"] for r in summary["vintages"]] == [2, 1]
    assert summary["sampled_loans"] == 3 and summary["loans_without_perf"] == 1
    assert _panel(config).filter(pl.col("loan_identifier") == "A").height == 3
    saved = (config.output_root / "sampling_summary.json").read_text()
    with pytest.raises(ValueError, match="exceeds"):
        process.build_sample_panel(replace(filtered, n_loans=4))
    assert (config.output_root / "sampling_summary.json").read_text() == saved
    summary = process.build_sample_panel(replace(filtered, n_loans=2))
    assert [r["quota"] for r in summary["vintages"]] == [1, 1]
    assert summary["sampled_loans"] == 2
    first = pl.read_parquet(config.output_root / "sampled_loans.parquet")
    process.build_sample_panel(replace(filtered, n_loans=2))
    assert first.equals(pl.read_parquet(config.output_root / "sampled_loans.parquet"))
    summary = process.build_sample_panel(replace(filtered, amortization_type="ARM", n_loans=1))
    assert [r["quota"] for r in summary["vintages"]] == [1, 0]
    assert not (config.output_root / "panel/vintage=2015Q2").exists()


def test_toml_paths_environment_and_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_FREDDIE_ROOT", str(tmp_path / "source"))
    path = tmp_path / "sample.toml"
    path.write_text('''[data]
freddie_root = "${TEST_FREDDIE_ROOT}"
output_root = "output"
[sample]
start_vintage = "2015Q1"
end_vintage = "2015Q2"
n_loans = 6
seed = 42
''')
    config = process.read_config(path)
    assert config.freddie_root == tmp_path / "source"
    assert config.output_root == tmp_path / "output"
    assert config.amortization_type is None and config.original_loan_term is None
    path.write_text(path.read_text() + 'amortization_type = "FRM"\noriginal_loan_term = 360\n')
    filtered = process.read_config(path)
    assert filtered.amortization_type == "FRM" and filtered.original_loan_term == 360
    monkeypatch.delenv("TEST_FREDDIE_ROOT")
    with pytest.raises(ValueError, match="TEST_FREDDIE_ROOT"):
        process.read_config(path)
    assert process.main(["--config", str(path)]) == 1
    with pytest.raises(ValueError):
        process._vintages("2015Q5", "2015Q2")
    with pytest.raises(ValueError):
        process._vintages("2015Q2", "2015Q1")
