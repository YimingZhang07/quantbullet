import subprocess
import sys

import pytest

from projects.freddie_prepayment import refresh


def write_configs(tmp_path, *, sample_root="sample", panel_path="prepared/panel.parquet"):
    # Relative paths resolve against the TOML directory, so no environment variables are needed.
    sample, panel, turnover = (tmp_path / name for name in ("sample.toml", "panel.toml", "turnover.toml"))
    sample.write_text('[data]\nfreddie_root="freddie"\noutput_root="sample"\n'
                      '[sample]\nstart_vintage="2020Q1"\nend_vintage="2021Q4"\nn_loans=10\nseed=1\n')
    panel.write_text(f'[data]\nsample_root="{sample_root}"\nmacro_root="macro"\noutput_root="prepared"\n')
    turnover.write_text(f'[data]\npanel_path="{panel_path}"\noutput_root="model"\n')
    return sample, panel, turnover


def test_plan_derives_every_data_root_from_the_configs(tmp_path):
    sample, panel, turnover = write_configs(tmp_path)
    stages = refresh.plan(sample, panel, turnover)
    root = tmp_path.resolve()
    assert list(stages) == list(refresh.STAGES)
    assert stages["macro"] == [
        ["procs.housing_macro.download", "--data-root", str(root / "macro"), "--refresh"],
        ["procs.housing_macro.build_parquet", "--data-root", str(root / "macro")],
    ]
    assert stages["convert"] == [["procs.freddie_sflld.build_parquet", "--data-root", str(root / "freddie"),
                                  "--start", "2020Q1", "--end", "2021Q4"]]
    assert stages["sample"] == [["procs.freddie_sflld.build_sample_panel", "--config", str(sample)]]
    assert stages["panel"] == [["procs.freddie_sflld.prepare_panel", "--config", str(panel)]]
    assert stages["report"] == [["projects.freddie_prepayment.report_turnover", "--config", str(turnover)]]


@pytest.mark.parametrize("updates,message", [
    (dict(sample_root="other"), "sample_root"),
    (dict(panel_path="other/panel.parquet"), "panel_path"),
])
def test_plan_rejects_a_broken_chain(tmp_path, updates, message):
    with pytest.raises(ValueError, match=message):
        refresh.plan(*write_configs(tmp_path, **updates))


def test_run_selects_the_range_and_stops_at_the_first_failure(monkeypatch):
    stages = {stage: [[f"module.{stage}"]] for stage in refresh.STAGES}
    calls = []

    def fake_run(command, cwd):
        assert command[:2] == [sys.executable, "-m"] and cwd == refresh.REPOSITORY
        calls.append(command[2])
        return subprocess.CompletedProcess(command, 3 if command[2] == "module.frame" else 0)

    monkeypatch.setattr(refresh.subprocess, "run", fake_run)
    assert refresh.run(stages, "sample", "fit") == 3
    assert calls == ["module.sample", "module.panel", "module.frame"]
    calls.clear()
    assert refresh.run(stages, "panel", "panel") == 0
    assert calls == ["module.panel"]
    calls.clear()
    assert refresh.run(stages, "macro", "report", dry_run=True) == 0
    assert calls == []


def test_from_must_not_come_after_to():
    with pytest.raises(SystemExit):
        refresh.main(["--from", "report", "--to", "panel"])
