"""Shared configuration and small artifact helpers for the three independent stages."""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import tomllib


@dataclass(frozen=True)
class NumericFeature:
    lower: float
    upper: float
    knots: tuple[float, ...]
    bin_width: float
    label: str


# Raw values stay in the modeling frame; _fit columns carry only these caps.
NUMERIC = {
    "c_age": NumericFeature(1, 120, (3, 6, 12, 18, 24, 36, 60, 84, 108), 6, "Loan age (months)"),
    "c_incentive": NumericFeature(-6, -.5, (-4, -3, -2, -1.5, -1, -.75), .25, "Incentive (percentage points)"),
    "c_orig_fico": NumericFeature(620, 840, (660, 700, 740, 780), 20, "Original FICO"),
    "c_updated_ltv": NumericFeature(5, 120, (20, 40, 60, 80, 95), 5, "Updated first-lien LTV (%)"),
    "c_orig_balance": NumericFeature(25000, 1500000, (100000, 150000, 250000, 350000, 500000, 700000, 950000), 50000, "Original balance (nominal USD)"),
    "c_hpi_growth": NumericFeature(-.2, 1.5, (-.05, 0, .1, .25, .5, .75, 1), .1, "ZHVI growth since origination (ratio)"),
}
CATEGORICAL = ("f_purpose", "f_occupancy", "f_property_type", "f_first_time_buyer", "f_month", "f_state")
TARGET = "y_full_prepay"
FIT_NUMERIC = tuple(name + "_fit" for name in NUMERIC)
MODEL_INPUTS = (*FIT_NUMERIC, *CATEGORICAL)
KEYS = ("row_id", "loan_identifier", "d_reporting_month")


@dataclass(frozen=True)
class Config:
    panel_path: Path
    output_root: Path
    incentive_max: float = -.5
    n_iterations: int = 60
    early_stopping_rounds: int = 10
    ftol: float = 1e-8

    def __post_init__(self):
        if not math.isfinite(self.incentive_max) or self.incentive_max > -.5:
            raise ValueError("incentive_max must be finite and <= -0.5 for this turnover specification")
        if self.n_iterations < 1 or self.early_stopping_rounds < 1 or not math.isfinite(self.ftol) or self.ftol <= 0:
            raise ValueError("Fit iterations, stopping rounds and ftol must be positive")
        repository = Path(__file__).resolve().parents[2]
        for path in (self.panel_path, self.output_root):
            if path.resolve().is_relative_to(repository):
                raise ValueError("Data and artifacts must be outside the repository")
        if self.panel_path.resolve().is_relative_to(self.output_root.resolve()):
            raise ValueError("Output must be separate from the input panel")


def read_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    with path.open("rb") as file:
        config = tomllib.load(file)

    def expand(value):
        def variable(match):
            if not os.environ.get(match[1]):
                raise ValueError(f"Set environment variable {match[1]}")
            return os.environ[match[1]]
        expanded = Path(re.sub(r"\$\{([A-Za-z_]\w*)\}", variable, value)).expanduser()
        return (expanded if expanded.is_absolute() else path.parent / expanded).resolve()

    return Config(
        expand(config["data"]["panel_path"]), expand(config["data"]["output_root"]),
        **config.get("cohort", {}), **config.get("fit", {}),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def temporary_output(target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{target.stem}-", suffix=target.suffix, dir=target.parent)
    os.close(handle)
    path = Path(name)
    try:
        yield path
        path.replace(target)
    finally:
        path.unlink(missing_ok=True)


def read_summary(root: Path) -> dict:
    return json.loads((root / "turnover_summary.json").read_text(encoding="utf-8"))


def write_summary(root: Path, summary: dict):
    with temporary_output(root / "turnover_summary.json") as path:
        path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
