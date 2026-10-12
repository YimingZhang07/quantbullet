"""Run configuration for the turnover workflow."""

from dataclasses import dataclass, field, fields
import math
from pathlib import Path
import tomllib

from dotenv import load_dotenv

from quantbullet.utils.files import expand_env_path

REPOSITORY = Path(__file__).resolve().parents[2]
BCD_SETTINGS = ("n_iterations", "early_stopping_rounds", "ftol")


def load_repo_env() -> None:
    # Process environment wins over the gitignored repo .env.
    load_dotenv(REPOSITORY / ".env", override=False)


@dataclass(frozen=True)
class Config:
    panel_path: Path
    output_root: Path
    incentive_max: float = -.5
    block_link: str = "identity"   # "identity": BCD (settings below); "log": log-link GLM by Newton
    n_iterations: int = 60
    early_stopping_rounds: int = 10
    ftol: float = 1e-5
    newton: dict = field(default_factory=dict)   # [fit.newton]: Newton solver_options for block_link = "log"

    def __post_init__(self):
        if not math.isfinite(self.incentive_max) or self.incentive_max > -.5:
            raise ValueError("incentive_max must be finite and <= -0.5 for this turnover specification")
        if self.block_link not in ("identity", "log"):
            raise ValueError("block_link must be 'identity' (BCD) or 'log' (Newton)")
        if self.n_iterations < 1 or self.early_stopping_rounds < 1 or not math.isfinite(self.ftol) or self.ftol <= 0:
            raise ValueError("Fit iterations, stopping rounds and ftol must be positive")
        if self.block_link == "log":
            changed = [item.name for item in fields(self)
                       if item.name in BCD_SETTINGS and getattr(self, item.name) != item.default]
            if changed:
                raise ValueError(f"{', '.join(changed)} apply to block_link = 'identity'; "
                                 "Newton settings go in [fit.newton]")
        elif self.newton:
            raise ValueError("[fit.newton] settings apply to block_link = 'log'")
        for path in (self.panel_path, self.output_root):
            if path.resolve().is_relative_to(REPOSITORY):
                raise ValueError("Data and artifacts must be outside the repository")
        if self.panel_path.resolve().is_relative_to(self.output_root.resolve()):
            raise ValueError("Output must be separate from the input panel")


def read_config(path: str | Path) -> Config:
    load_repo_env()
    path = Path(path).resolve()
    with path.open("rb") as file:
        config = tomllib.load(file)
    return Config(
        expand_env_path(config["data"]["panel_path"], base=path.parent),
        expand_env_path(config["data"]["output_root"], base=path.parent),
        **config.get("cohort", {}), **config.get("fit", {}),
    )
