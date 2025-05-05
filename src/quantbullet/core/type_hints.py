import pandas as pd
import numpy as np
from datetime import date, datetime
from typing import Union
from typing import List, Tuple

DateAndDatetimeLike = Union[date, datetime, pd.Timestamp, np.datetime64]
ArrayLike = Union[ List, Tuple, pd.Index, pd.Series, np.ndarray ]