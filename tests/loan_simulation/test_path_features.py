from quantbullet.loan_simulation import PathFeatureTracker, PeriodCashflow, StatusConfig


def _status_config():
    return StatusConfig(
        valid_statuses={"C", "D1M"},
        terminal_statuses=set(),
        prepay_statuses=set(),
        default_statuses=set(),
        delinquency_buckets={"D1M": "dq30_balance"},
    )


def test_path_feature_tracker_updates_delinquency_history():
    tracker = PathFeatureTracker()
    config = _status_config()

    assert tracker.features() == {
        "ever_delinquent": False,
        "months_since_last_delinquency": None,
        "consecutive_delinquent_months": 0,
        "times_delinquent": 0,
    }

    tracker.update(
        PeriodCashflow("L1", 0, 1, 0, 1, 100.0, 100.0, "C", "D1M"),
        config,
    )
    assert tracker.features() == {
        "ever_delinquent": True,
        "months_since_last_delinquency": 0,
        "consecutive_delinquent_months": 1,
        "times_delinquent": 1,
    }

    tracker.update(
        PeriodCashflow("L1", 0, 2, 1, 2, 100.0, 90.0, "D1M", "C"),
        config,
    )
    assert tracker.features() == {
        "ever_delinquent": True,
        "months_since_last_delinquency": 1,
        "consecutive_delinquent_months": 0,
        "times_delinquent": 1,
    }
