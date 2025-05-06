import pandas as pd
import matplotlib.pyplot as plt
from quantbullet.core.type_hints import ArrayLike
from scipy.signal import savgol_filter
from quantbullet.plot.colors import EconomistBrandColor
from ..utils.validation import Validator

class SavgolFilterSmoother:
    def __init__( self, window_length: int = 11, polyorder: int = 2 ):
        if window_length % 2 == 0:
            raise ValueError( "Window length must be odd." )
        
        self.window_length = window_length
        self.polyorder = polyorder

    def apply( self, x: ArrayLike ):
        """Apply Savitzky-Golay filter to smooth the input series."""
        if isinstance( x, pd.Series ):
            x_smooth = pd.Series(
                savgol_filter( x, window_length=self.window_length, polyorder=self.polyorder ),
                index=x.index
            )
        else:
            x_smooth = savgol_filter( x, window_length=self.window_length, polyorder=self.polyorder )
        return x_smooth
    
    def plot( self, x: ArrayLike ):
        """Plot the original and smoothed series."""
        x_smooth = self.smooth( x )
        
        plt.plot( x, label='x', color=EconomistBrandColor.CHICAGO_30.value )
        plt.plot( x_smooth, label='Smoothed x', color=EconomistBrandColor.HONG_KONG_45.value )
        plt.legend()
        plt.show()
        
        
class MultiSeriesConstraintSmoother:
    """A class to enforce constraints on multiple time series.
    
    This class allows for the enforcement of monotonicity and band constraints on a set of time series.
    """
    def __init__(self, monotonic_pairs=None, band_constraints=None, series_order=None):
        """
        Initialize the MultiSeriesConstraintSmoother.
        
        Parameters
        ----------
        monotonic_pairs : list of tuples
            Pairs of series names that should be monotonic. E.g. [('A', 'B'), ('C', 'D')] means A <= B and C <= D.
        band_constraints : list of tuples
            Pairs of series names with their min and max spread constraints. E.g. [('A', 'B', 0, 10)] means A - B should be between 0 and 10.
        series_order : list of str
            The order of series to enforce the constraints. E.g. ['A', 'B', 'C'] means A is the most senior and C is the least.
        """
        self.monotonic_pairs = monotonic_pairs or []
        self.band_constraints = band_constraints or []
        self.order_rank = {name: i for i, name in enumerate(series_order or [])}
        self.logs = []  # logs of adjustments

    def _prefer_adjust(self, a, b):
        return a if self.order_rank.get(a, 999) > self.order_rank.get(b, 999) else b

    def _log(self, date, typ, series, amount, details):
        self.logs.append({
            "date": date,
            "type": typ,
            "series": series,
            "adjustment": amount,
            "details": details
        })

    def _enforce_monotonic(self, row, a, b, date):
        """
        Enforce monotonicity a <= b, adjusting only the junior one based on series_order.
        """
        if pd.notna(row[a]) and pd.notna(row[b]) and row[a] > row[b]:
            target = self._prefer_adjust(a, b)
            original = row[target]
            if target == a:
                row[a] = row[b] - 0.01
            elif target == b:
                row[b] = row[a] + 0.01
            self._log(date, "monotonic", target, row[target] - original, f"{a} <= {b}")


    def _enforce_band(self, row, a, b, min_spread, max_spread, date):
        """
        Enforce constraint: a - b ∈ [min_spread, max_spread], but only adjust junior one.
        """
        if pd.notna(row[a]) and pd.notna(row[b]):
            spread = row[a] - row[b]
            if spread < min_spread:
                target = self._prefer_adjust(a, b)
                original = row[target]
                if target == a:
                    row[a] = row[b] + min_spread
                elif target == b:
                    row[b] = row[a] - min_spread
                self._log(date, "band_too_narrow", target, row[target] - original, f"{a} - {b} < {min_spread}")
            elif spread > max_spread:
                target = self._prefer_adjust(a, b)
                original = row[target]
                if target == a:
                    row[a] = row[b] + max_spread
                elif target == b:
                    row[b] = row[a] - max_spread
                self._log(date, "band_too_wide", target, row[target] - original, f"{a} - {b} > {max_spread}")


    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logs.clear()
        df_out = df.copy()

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        if not Validator.is_index_datetime(df):
            raise ValueError("DataFrame index must be of datetime type.")

        for date in df.index:
            row = df.loc[date].copy()

            # 从最 senior 的 series 依次处理
            for series in self.order_rank.keys():
                # 所有关联 monotonic constraints：series 是左边或右边
                for a, b in self.monotonic_pairs:
                    if series in (a, b):
                        self._enforce_monotonic(row, a, b, date)

                # 所有关联 band constraints：series 是左边或右边
                for a, b, min_spread, max_spread in self.band_constraints:
                    if series in (a, b):
                        self._enforce_band(row, a, b, min_spread, max_spread, date)

            df_out.loc[date] = row

        return df_out
    
    def get_logs(self):
        return pd.DataFrame(self.logs)
