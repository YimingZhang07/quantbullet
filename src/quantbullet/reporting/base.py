from dataclasses import dataclass, field
from typing import Any, Callable, Optional

@dataclass
class BaseColumnFormat:
    decimals    : int           = 2
    comma       : bool          = False
    transformer : Optional[ Callable[ [ Any ], Any ] ] = None
    percent     : bool          = False

@dataclass
class BaseColumnMeta:
    name        : str
    display_name: Optional[ str ] = None
    format      : BaseColumnFormat = field( default_factory=BaseColumnFormat )

    def __post_init__(self):
        if self.display_name is None:
            self.display_name = self.raw_name