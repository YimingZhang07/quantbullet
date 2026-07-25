import math

import pandas as pd
import pytest

from quantbullet.loan_simulation import Loan, SoftmaxTransitionModel, StatusConfig
from quantbullet.loan_simulation.adapters import parse_rollrate_coefficients
from quantbullet.model.gam.utils import dump_partial_dependence_json
from quantbullet.model.gam_replay import GAMReplayModel


def _write_coefficients(tmp_path, rows):
    path = tmp_path / "fromC.txt"
    header = "model\tvar_name1\tvar_val1\tvar_name2\tvar_val2\tvalue"
    body = ["\t".join(map(str, row)) for row in rows]
    path.write_text("\n".join([header, *body]) + "\n", encoding="utf-8")
    return path


def _supported_rows():
    rows = [
        ("PIF", "intercept", "intercept", "", "", 1.0),
        ("PIF", "purpose", "Business", "", "", 0.5),
    ]
    rows.extend(("PIF", "opti", x, "", "", x * x) for x in range(6))
    rows.extend(
        ("PIF", "credit_age", x, "v_credit_age", "", 2 * x)
        for x in range(6)
    )
    for term in (36, 60):
        offset = 10 if term == 36 else 20
        rows.extend(
            ("PIF", "c_age_pct", x, "oterm_f", term, offset + x)
            for x in range(6)
        )
    return rows


def test_parse_rollrate_coefficients_replays_supported_term_shapes(tmp_path):
    """Factors, linear smooths, factor-by, and numeric-by terms all replay."""
    models = parse_rollrate_coefficients(
        _write_coefficients(tmp_path, _supported_rows())
    )

    replay = models["PIF"]
    features = pd.DataFrame(
        {
            "purpose": ["Business", "Other"],
            "opti": [1.5, -1.0],
            "credit_age": [2.5, 10.0],
            "v_credit_age": [0.4, 0.0],
            "c_age_pct": [1.5, 10.0],
            "oterm_f": ["36", "60"],
        }
    )

    # Row 0: intercept + factor + linear opti + numeric-by + factor-by.
    # Linear interpolation gives opti(1.5) == 2.5 rather than a PCHIP value.
    assert replay.predict(features) == pytest.approx([17.5, 26.0])


def test_rollrate_replay_terms_round_trip_through_shared_gam_json(tmp_path):
    """The adapter's replay terms use the standard GAM serialization contract."""
    replay = parse_rollrate_coefficients(
        _write_coefficients(tmp_path, _supported_rows())
    )["PIF"]
    payload_path = tmp_path / "replay.json"
    dump_partial_dependence_json(
        replay.term_data,
        str(payload_path),
        intercept=replay.intercept,
    )

    restored = GAMReplayModel.from_partial_dependence_json(str(payload_path))
    features = pd.DataFrame(
        {
            "purpose": ["Business"],
            "opti": [1.5],
            "credit_age": [2.5],
            "v_credit_age": [0.4],
            "c_age_pct": [1.5],
            "oterm_f": ["36"],
        }
    )

    assert restored.predict(features) == pytest.approx([17.5])


def test_parsed_rollrate_logit_can_drive_softmax_transition(tmp_path):
    """A parsed edge replay model plugs into the existing softmax layer."""
    replay = parse_rollrate_coefficients(
        _write_coefficients(tmp_path, _supported_rows())
    )["PIF"]
    status_config = StatusConfig(
        valid_statuses={"C", "PIF"},
        terminal_statuses={"PIF"},
        prepay_statuses={"PIF"},
        default_statuses=set(),
        delinquency_buckets={},
    )

    def pif_logit(context):
        return float(
            replay.predict(
                pd.DataFrame(
                    [
                        {
                            "purpose": context.loan.metadata["purpose"],
                            "opti": 1.5,
                            "credit_age": 2.5,
                            "v_credit_age": 0.4,
                            "c_age_pct": 1.5,
                            "oterm_f": context.loan.metadata["term_bucket"],
                        }
                    ]
                )
            )[0]
        )

    model = SoftmaxTransitionModel(
        {"C": {"PIF": pif_logit}},
        status_config=status_config,
    )
    loan = Loan(
        "L1",
        balance=1_000.0,
        annual_rate=0.12,
        term_months=36,
        status="C",
        metadata={"purpose": "Business", "term_bucket": "36"},
    )

    probabilities = model.predict(loan, loan.initial_state())
    expected = math.exp(17.5) / (1.0 + math.exp(17.5))
    assert probabilities["PIF"] == pytest.approx(expected)
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_parse_rollrate_coefficients_rejects_categorical_interactions(tmp_path):
    """Unsupported TSV shapes fail instead of being silently misinterpreted."""
    rows = [
        ("PIF", "intercept", "intercept", "", "", 1.0),
        ("PIF", "purpose", "Business", "segment", "Prime", 0.5),
    ]

    with pytest.raises(ValueError, match="categorical interaction"):
        parse_rollrate_coefficients(_write_coefficients(tmp_path, rows))


def test_parse_rollrate_coefficients_rejects_single_by_level_terms(tmp_path):
    """One by-level is ambiguous: factor-by for us, numeric-by for roll-rate."""
    rows = [("PIF", "intercept", "intercept", "", "", 1.0)]
    rows.extend(
        ("PIF", "c_age_pct", x, "oterm_f", 36, 10 + x)
        for x in range(6)
    )

    with pytest.raises(ValueError, match="single by-level '36'"):
        parse_rollrate_coefficients(_write_coefficients(tmp_path, rows))


def test_parse_rollrate_coefficients_requires_intercept(tmp_path):
    """Every parsed edge model must have an explicit intercept row."""
    rows = [("PIF", "purpose", "Business", "", "", 0.5)]

    with pytest.raises(ValueError, match="does not contain an intercept"):
        parse_rollrate_coefficients(_write_coefficients(tmp_path, rows))
