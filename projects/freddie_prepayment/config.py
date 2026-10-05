"""Run configuration for the turnover workflow."""

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib

from dotenv import load_dotenv

from quantbullet.utils.files import expand_env_path


@dataclass(frozen=True)
class Config:
    panel_path: Path
    output_root: Path
    incentive_max: float = -.5
    n_iterations: int = 60
    early_stopping_rounds: int = 10
    ftol: float = 1e-5

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
    # Process environment wins over the gitignored repo .env.
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    path = Path(path).resolve()
    with path.open("rb") as file:
        config = tomllib.load(file)
    return Config(
        expand_env_path(config["data"]["panel_path"], base=path.parent),
        expand_env_path(config["data"]["output_root"], base=path.parent),
        **config.get("cohort", {}), **config.get("fit", {}),
    )
