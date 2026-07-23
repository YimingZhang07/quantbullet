import pandas as pd
import pytest

from quantbullet.loan_simulation import DataFrameMacroFeatureProvider


def test_dataframe_macro_provider_matches_by_calendar_month():
    features = pd.DataFrame(
        {"hpi": [100.0, 101.0], "rate": [0.06, 0.061]},
        index=pd.to_datetime(["2026-01-31", "2026-02-28"]),
    )
    provider = DataFrameMacroFeatureProvider(features)

    assert provider.features_for_date("2026-02-01") == {
        "hpi": 101.0,
        "rate": 0.061,
    }


def test_dataframe_macro_provider_can_forward_fill():
    features = pd.DataFrame(
        {"hpi": [100.0, 103.0]},
        index=pd.to_datetime(["2026-01-31", "2026-04-30"]),
    )
    provider = DataFrameMacroFeatureProvider(features, method="ffill")

    assert provider.features_for_date("2026-03-31") == {"hpi": 100.0}


def test_dataframe_macro_provider_rejects_duplicate_months():
    features = pd.DataFrame(
        {"hpi": [100.0, 101.0]},
        index=pd.to_datetime(["2026-01-01", "2026-01-31"]),
    )

    with pytest.raises(ValueError, match="duplicate periods"):
        DataFrameMacroFeatureProvider(features)
