import pytest

from quantbullet.loan_simulation import (
    DEFAULT_STATUS_CONFIG,
    MatrixPaymentPolicy,
    StatusConfig,
)


def _status_config(dq_statuses):
    return StatusConfig(
        valid_statuses={"C", *dq_statuses, "PIF", "LIQ"},
        terminal_statuses={"PIF", "LIQ"},
        prepay_statuses={"PIF"},
        default_statuses={"LIQ"},
        delinquency_buckets={
            status: f"dq{30 * (index + 1)}_balance"
            for index, status in enumerate(dq_statuses)
        },
    )


def test_from_delinquency_chain_matches_hand_built_one_bucket_matrix():
    config = _status_config(["D1M"])
    policy = MatrixPaymentPolicy.from_delinquency_chain(
        ["C", "D1M"], status_config=config
    )
    assert policy.payment_matrix == {
        "C": {"C": 1, "D1M": 0, "PIF": 0, "LIQ": 0},
        "D1M": {"C": 2, "D1M": 1, "PIF": 0, "LIQ": 0},
        "PIF": {"C": 0, "D1M": 0, "PIF": 0, "LIQ": 0},
        "LIQ": {"C": 0, "D1M": 0, "PIF": 0, "LIQ": 0},
    }


def test_from_delinquency_chain_matches_hand_built_four_bucket_matrix():
    config = _status_config(["D1M", "D2M", "D3M", "D4M"])
    policy = MatrixPaymentPolicy.from_delinquency_chain(
        ["C", "D1M", "D2M", "D3M", "D4M"], status_config=config
    )
    assert policy.payment_matrix == {
        "C": {"C": 1, "D1M": 0, "D2M": 0, "D3M": 0, "D4M": 0, "PIF": 0, "LIQ": 0},
        "D1M": {"C": 2, "D1M": 1, "D2M": 0, "D3M": 0, "D4M": 0, "PIF": 0, "LIQ": 0},
        "D2M": {"C": 3, "D1M": 2, "D2M": 1, "D3M": 0, "D4M": 0, "PIF": 0, "LIQ": 0},
        "D3M": {"C": 4, "D1M": 3, "D2M": 2, "D3M": 1, "D4M": 0, "PIF": 0, "LIQ": 0},
        "D4M": {"C": 5, "D1M": 4, "D2M": 3, "D3M": 2, "D4M": 1, "PIF": 0, "LIQ": 0},
        "PIF": {"C": 0, "D1M": 0, "D2M": 0, "D3M": 0, "D4M": 0, "PIF": 0, "LIQ": 0},
        "LIQ": {"C": 0, "D1M": 0, "D2M": 0, "D3M": 0, "D4M": 0, "PIF": 0, "LIQ": 0},
    }


def test_from_delinquency_chain_matches_hand_built_synthetic_matrix():
    config = StatusConfig(
        valid_statuses={"C", "D1", "D2", "PIF", "CO"},
        terminal_statuses={"PIF", "CO"},
        prepay_statuses={"PIF"},
        default_statuses={"CO"},
        delinquency_buckets={"D1": "dq30_balance", "D2": "dq60_balance"},
    )
    policy = MatrixPaymentPolicy.from_delinquency_chain(
        ["C", "D1", "D2"], status_config=config
    )
    assert policy.payment_matrix == {
        "C": {"C": 1, "D1": 0, "D2": 0, "PIF": 0, "CO": 0},
        "D1": {"C": 2, "D1": 1, "D2": 0, "PIF": 0, "CO": 0},
        "D2": {"C": 3, "D1": 2, "D2": 1, "PIF": 0, "CO": 0},
        "PIF": {"C": 0, "D1": 0, "D2": 0, "PIF": 0, "CO": 0},
        "CO": {"C": 0, "D1": 0, "D2": 0, "PIF": 0, "CO": 0},
    }


def test_from_delinquency_chain_works_with_default_status_config():
    policy = MatrixPaymentPolicy.from_delinquency_chain(
        ["CURRENT", "DQ30", "DQ60", "DQ90"],
        status_config=DEFAULT_STATUS_CONFIG,
    )
    assert policy.payment_matrix["DQ90"]["CURRENT"] == 4
    assert policy.payment_matrix["CURRENT"]["CURRENT"] == 1
    assert policy.payment_matrix["CURRENT"]["PAID_OFF"] == 0
    assert policy.payment_matrix["PAID_OFF"]["CURRENT"] == 0


def test_from_delinquency_chain_rejects_empty_chain():
    with pytest.raises(ValueError, match="non-empty"):
        MatrixPaymentPolicy.from_delinquency_chain(
            [], status_config=_status_config(["D1M"])
        )


def test_from_delinquency_chain_rejects_duplicate_statuses():
    with pytest.raises(ValueError, match="duplicate"):
        MatrixPaymentPolicy.from_delinquency_chain(
            ["C", "D1M", "D1M"], status_config=_status_config(["D1M"])
        )


def test_from_delinquency_chain_rejects_terminal_statuses():
    with pytest.raises(ValueError, match="terminal"):
        MatrixPaymentPolicy.from_delinquency_chain(
            ["C", "D1M", "PIF"], status_config=_status_config(["D1M"])
        )


def test_from_delinquency_chain_rejects_unknown_statuses():
    with pytest.raises(ValueError, match="Unknown status"):
        MatrixPaymentPolicy.from_delinquency_chain(
            ["C", "D9M"], status_config=_status_config(["D1M"])
        )


def test_from_delinquency_chain_rejects_uncovered_non_terminal_statuses():
    with pytest.raises(ValueError, match="neither"):
        MatrixPaymentPolicy.from_delinquency_chain(
            ["C", "D1M"], status_config=_status_config(["D1M", "D2M"])
        )
