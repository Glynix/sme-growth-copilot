"""Temporal ML, private audience, anomaly, and opportunity scenario checks."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, replace
from datetime import date, timedelta

import pandas as pd
import pytest

from sme_growth.analytics import get_merchant_facts, get_peer_benchmark
from sme_growth.analytics.common import settled_transactions
from sme_growth.audiences.builder import PrivateAudienceStore, build_eligible_audience, build_model_risk_audience
from sme_growth.audiences.private_rules import select_private_lapsed_lunch
from sme_growth.ml.anomaly import get_merchant_anomaly_signal
from sme_growth.ml.retention import get_retention_cohort_summary, select_private_top_risk_regulars, train_retention_model
from sme_growth.recommendations.economics import ImpactAssumptions, calculate_impact
from sme_growth.recommendations.engine import generate_opportunities
from sme_growth.synthetic.generator import (
    DEMO_MERCHANT_ID, RETENTION_PURPOSE, SIMULATED_CHANNEL,
    PrivateSyntheticDataset, generate_dataset,
)


AS_OF = date(2026, 10, 1)


@pytest.fixture(scope="module")
def dataset() -> PrivateSyntheticDataset:
    return generate_dataset()


@pytest.fixture(scope="module")
def model(dataset: PrivateSyntheticDataset):
    return train_retention_model(dataset.transactions, AS_OF)


def _facts_and_anomaly(dataset: PrivateSyntheticDataset):
    facts = get_merchant_facts(
        dataset.transactions, dataset.merchants, dataset.program_eligibility,
        DEMO_MERCHANT_ID, AS_OF,
    )
    peers = get_peer_benchmark(dataset.transactions, dataset.merchants, DEMO_MERCHANT_ID, AS_OF)
    anomaly = get_merchant_anomaly_signal(dataset.transactions, DEMO_MERCHANT_ID, AS_OF, peers)
    return facts, anomaly


def test_retention_pipeline_temporal_label_baseline_and_calibration(model) -> None:
    result = model.evaluation
    assert result.future_label == "No settled purchase at the same merchant in the 30 days after snapshot cutoff"
    assert result.train_examples > 1_000
    assert result.validation_examples > 1_000
    assert result.test_examples > 1_000
    assert date.fromisoformat(result.train_cutoffs[-1]) + timedelta(days=30) < date.fromisoformat(result.validation_cutoff)
    assert date.fromisoformat(result.validation_cutoff) + timedelta(days=30) < date.fromisoformat(result.test_cutoff)
    assert date.fromisoformat(result.test_cutoff) + timedelta(days=30) <= AS_OF
    assert result.selected_signal == "behavioral_logistic_regression"
    assert result.model_average_precision > result.recency_average_precision
    assert result.model_brier_score < result.recency_brier_score
    assert 0 <= result.model_calibration_error <= 1
    assert 0 <= result.recency_calibration_error <= 1
    assert "not real-world accuracy" in result.limitation


def test_retention_training_and_safe_cohort_output_are_reproducible(
    dataset: PrivateSyntheticDataset, model,
) -> None:
    again = train_retention_model(dataset.transactions, AS_OF)
    assert asdict(model.evaluation) == asdict(again.evaluation)
    first = get_retention_cohort_summary(model, dataset.transactions, DEMO_MERCHANT_ID, AS_OF)
    second = get_retention_cohort_summary(again, dataset.transactions, DEMO_MERCHANT_ID, AS_OF)
    assert asdict(first) == asdict(second)
    assert first.regular_customers is not None and first.regular_customers >= 50
    public = json.dumps({"evaluation": asdict(model.evaluation), "cohort": asdict(first)})
    assert "syn_c_" not in public
    assert "synthetic_customer_id" not in public
    assert "individual_score" not in public
    assert "member" not in public.lower()


def test_model_top_quintile_audience_is_disjoint_private_and_versioned(dataset, model, tmp_path) -> None:
    path = tmp_path / "audiences.sqlite"
    store = PrivateAudienceStore(path)
    lapsed = build_eligible_audience(dataset.transactions, dataset.program_eligibility, DEMO_MERCHANT_ID, AS_OF, store)
    first = build_model_risk_audience(
        model, dataset.transactions, dataset.program_eligibility, DEMO_MERCHANT_ID, AS_OF, store,
    )
    assert first.activation_eligible and first.eligible_count == 92
    assert first == build_model_risk_audience(
        model, dataset.transactions, dataset.program_eligibility, DEMO_MERCHANT_ID, AS_OF, store,
    )
    assert first.audience_id != lapsed.audience_id
    top = select_private_top_risk_regulars(model, dataset.transactions, DEMO_MERCHANT_ID, AS_OF)
    assert len(top) == 113
    with sqlite3.connect(path) as connection:
        model_members = {row[0] for row in connection.execute(
            "SELECT synthetic_customer_id FROM audience_members WHERE audience_id = ?", (first.audience_id,),
        )}
        lapsed_members = {row[0] for row in connection.execute(
            "SELECT synthetic_customer_id FROM audience_members WHERE audience_id = ?", (lapsed.audience_id,),
        )}
    assert model_members <= top and not model_members & lapsed_members
    settled = settled_transactions(dataset.transactions)
    historical = settled.loc[
        settled["merchant_id"].eq(DEMO_MERCHANT_ID)
        & settled["synthetic_customer_id"].isin(model_members)
        & settled["transaction_datetime"].ge(pd.Timestamp(AS_OF - timedelta(days=90)))
        & settled["transaction_datetime"].lt(pd.Timestamp(AS_OF))
    ]
    assert historical["synthetic_customer_id"].nunique() == first.eligible_count
    assert first.historical_spend_aed_per_30d_equivalent == round(float(historical["net_sales_aed"].sum()) / 3, 2)
    public = json.dumps(asdict(first), default=str)
    assert "syn_c_" not in public and "synthetic_customer_id" not in public

    revoked = dataset.program_eligibility.copy()
    member_row = revoked["synthetic_customer_id"].isin(model_members).idxmax()
    revoked.loc[member_row, "revoked_at"] = pd.Timestamp(AS_OF)
    revised = build_model_risk_audience(model, dataset.transactions, revoked, DEMO_MERCHANT_ID, AS_OF, store)
    assert revised.audience_version == first.audience_version + 1
    assert revised.eligible_count == first.eligible_count - 1

    baseline = replace(model.evaluation, selected_signal="recency_baseline")
    blocked = build_model_risk_audience(
        replace(model, evaluation=baseline), dataset.transactions, dataset.program_eligibility,
        DEMO_MERCHANT_ID, AS_OF, store,
    )
    assert not blocked.activation_eligible
    assert blocked.historical_spend_aed_per_30d_equivalent is None


def test_anomaly_signal_is_transparent_and_requires_evidence(dataset: PrivateSyntheticDataset) -> None:
    _, anomaly = _facts_and_anomaly(dataset)
    assert anomaly.status == "available"
    assert anomaly.unusually_weak is True
    assert anomaly.observed_sales_change_pct == -9.69
    assert anomaly.peer_gap_percentage_points == pytest.approx(-10.0, abs=.1)
    tiny = dataset.transactions.iloc[:0].copy()
    unavailable = get_merchant_anomaly_signal(tiny, DEMO_MERCHANT_ID, AS_OF)
    assert unavailable.status == "insufficient_evidence"
    assert unavailable.unusually_weak is None


def test_private_audience_snapshot_is_scoped_versioned_and_aggregate_only(
    dataset: PrivateSyntheticDataset, tmp_path,
) -> None:
    store = PrivateAudienceStore(tmp_path / "runtime" / "audiences.sqlite")
    first = build_eligible_audience(dataset.transactions, dataset.program_eligibility, DEMO_MERCHANT_ID, AS_OF, store)
    repeated = build_eligible_audience(dataset.transactions, dataset.program_eligibility, DEMO_MERCHANT_ID, AS_OF, store)
    assert first == repeated
    assert first.activation_eligible
    assert first.eligible_count == 126
    assert first.audience_version == 1
    assert first.audience_id.startswith("aud_")
    assert DEMO_MERCHANT_ID not in first.audience_id
    assert first.expires_at.date() == AS_OF + timedelta(days=7)
    public = json.dumps(asdict(first), default=str)
    for forbidden in ("syn_c_", "synthetic_customer_id", "membership", "individual_score", "email", "phone"):
        assert forbidden not in public

    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM audience_members").fetchone()[0] == 126
        assert connection.execute("SELECT COUNT(*) FROM audience_snapshots").fetchone()[0] == 1

    revoked = dataset.program_eligibility.copy()
    revoked.loc[0, "revoked_at"] = pd.Timestamp(AS_OF)
    revised = build_eligible_audience(dataset.transactions, revoked, DEMO_MERCHANT_ID, AS_OF, store)
    assert revised.audience_id == first.audience_id
    assert revised.audience_version == 2
    assert revised.eligible_count == 125
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT eligible_count FROM audience_snapshots WHERE audience_version = 2").fetchone()[0] == 125


@pytest.mark.parametrize("failed_check", [
    "unknown", "expired", "revoked", "frequency_cap", "suppressed", "unenrolled", "wrong_purpose", "wrong_channel",
])
def test_audience_fails_closed_below_minimum(dataset: PrivateSyntheticDataset, tmp_path, failed_check: str) -> None:
    program = dataset.program_eligibility.copy()
    rows = program.index[:77]
    if failed_check == "unknown":
        program.loc[rows, "permission_state"] = "unknown"
    elif failed_check == "expired":
        program.loc[rows, "valid_until"] = pd.Timestamp(AS_OF - timedelta(days=1))
    elif failed_check == "revoked":
        program.loc[rows, "revoked_at"] = pd.Timestamp(AS_OF - timedelta(days=1))
    elif failed_check == "frequency_cap":
        program.loc[rows, "frequency_cap_ok"] = False
    elif failed_check == "suppressed":
        program.loc[rows, "suppressed"] = True
    elif failed_check == "unenrolled":
        program.loc[rows, "enrolled"] = False
    elif failed_check == "wrong_purpose":
        program.loc[rows, "purpose"] = "unavailable"
    else:
        program.loc[rows, "channel"] = "unavailable"
    store = PrivateAudienceStore(tmp_path / "audiences.sqlite")
    blocked = build_eligible_audience(dataset.transactions, program, DEMO_MERCHANT_ID, AS_OF, store)
    assert not blocked.activation_eligible
    assert blocked.audience_id is None
    assert blocked.audience_version is None
    assert blocked.eligible_count is None
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM audience_snapshots").fetchone()[0] == 0


def test_audience_missing_or_ambiguous_program_data_fails_closed(
    dataset: PrivateSyntheticDataset, tmp_path,
) -> None:
    store = PrivateAudienceStore(tmp_path / "audiences.sqlite")
    assert not build_eligible_audience(dataset.transactions, None, DEMO_MERCHANT_ID, AS_OF, store).activation_eligible
    # A duplicate record for one eligible token cannot add a second member.
    duplicated = pd.concat([dataset.program_eligibility, dataset.program_eligibility.iloc[[0]]], ignore_index=True)
    result = build_eligible_audience(dataset.transactions, duplicated, DEMO_MERCHANT_ID, AS_OF, store)
    assert result.eligible_count == 125


def test_audience_suppresses_count_when_ineligible_complement_is_too_small(
    dataset: PrivateSyntheticDataset, tmp_path,
) -> None:
    cohort = select_private_lapsed_lunch(
        dataset.transactions, dataset.program_eligibility, DEMO_MERCHANT_ID, AS_OF,
    )
    candidates = sorted(cohort.candidates)
    assert len(candidates) == 180
    program = dataset.program_eligibility.copy()
    member_rows = program["synthetic_customer_id"].isin(candidates)
    program.loc[member_rows, "enrolled"] = True
    program.loc[member_rows, "permission_state"] = "granted"
    program.loc[member_rows, "purpose"] = RETENTION_PURPOSE
    program.loc[member_rows, "channel"] = SIMULATED_CHANNEL
    program.loc[member_rows, "valid_until"] = pd.Timestamp(AS_OF + timedelta(days=30))
    program.loc[member_rows, "revoked_at"] = pd.NaT
    program.loc[member_rows, "frequency_cap_ok"] = True
    program.loc[member_rows, "suppressed"] = False
    program.loc[program["synthetic_customer_id"].isin(candidates[:9]), "permission_state"] = "unknown"
    store = PrivateAudienceStore(tmp_path / "audiences.sqlite")
    result = build_eligible_audience(dataset.transactions, program, DEMO_MERCHANT_ID, AS_OF, store)
    assert result.activation_eligible is False
    assert result.eligible_count is None
    assert result.audience_id is None
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM audience_snapshots").fetchone()[0] == 0


def test_impact_arithmetic_uses_assumptions_including_all_redemptions() -> None:
    assumptions = ImpactAssumptions(
        addressable_count=100, reach_rate=.8,
        response_rate_low=.05, response_rate_high=.10,
        visits_per_response=2, spend_per_visit_aed=70,
        incentive_rate=.10,
        all_qualifying_redemption_rate_low=.12,
        all_qualifying_redemption_rate_high=.20,
        response_meaning="Assumed incremental return share",
    )
    impact = calculate_impact(assumptions)
    assert impact.incremental_sales_low_aed == 560
    assert impact.incremental_sales_high_aed == 1120
    assert impact.campaign_cost_low_aed == 134.4
    assert impact.campaign_cost_high_aed == 224
    assert impact.net_additional_sales_low_aed == 336
    assert impact.net_additional_sales_high_aed == 985.6
    doubled = calculate_impact(replace(assumptions, addressable_count=200))
    assert doubled.incremental_sales_low_aed == 1120
    assert doubled.incremental_sales_high_aed == 2240
    assert doubled.incremental_sales_low_aed != impact.incremental_sales_low_aed
    zero_cost = calculate_impact(replace(assumptions, incentive_rate=0))
    assert zero_cost.net_additional_sales_low_aed == zero_cost.incremental_sales_low_aed
    assert zero_cost.net_additional_sales_high_aed == zero_cost.incremental_sales_high_aed
    with pytest.raises(ValueError):
        calculate_impact(replace(assumptions, response_rate_high=1.2))


def test_demo_opportunity_ranking_is_stable_and_private(dataset: PrivateSyntheticDataset, model, tmp_path) -> None:
    facts, anomaly = _facts_and_anomaly(dataset)
    audience = build_eligible_audience(
        dataset.transactions, dataset.program_eligibility, DEMO_MERCHANT_ID, AS_OF,
        PrivateAudienceStore(tmp_path / "audiences.sqlite"),
    )
    retention = get_retention_cohort_summary(model, dataset.transactions, DEMO_MERCHANT_ID, AS_OF)
    model_audience = build_model_risk_audience(
        model, dataset.transactions, dataset.program_eligibility, DEMO_MERCHANT_ID, AS_OF,
        PrivateAudienceStore(tmp_path / "audiences.sqlite"),
    )
    first = generate_opportunities(facts, audience, anomaly, retention, model_audience)
    second = generate_opportunities(facts, audience, anomaly, retention, model_audience)
    assert first == second
    assert len(first) >= 2
    assert first[0].opportunity_id == "weekday_lunch_winback"
    assert first[1].opportunity_id == "model_retention_offer"
    assert first[1].eligible_count == 92
    assert first[1].audience_id == model_audience.audience_id
    assert first[0].opportunity_score > first[1].opportunity_score
    assert first[0].audience_id == audience.audience_id
    assert first[0].audience_version == audience.audience_version
    assert first[0].eligible_count == 126
    assert first[0].historical_value_aed_per_month == 21_000
    assert first[0].estimated_impact_range.incremental_sales_low_aed == 1209.6
    assert first[0].estimated_impact_range.incremental_sales_high_aed == 1814.4
    assert first[0].estimated_impact_range.campaign_cost_low_aed == 317.52
    assert first[0].estimated_impact_range.campaign_cost_high_aed == 498.96
    assert first[0].estimated_impact_range.net_additional_sales_low_aed == 710.64
    assert first[0].estimated_impact_range.net_additional_sales_high_aed == 1496.88
    assert first[1].estimated_impact_range.campaign_cost_low_aed == 110.4
    assert first[1].estimated_impact_range.campaign_cost_high_aed == 198.72
    assert first[1].estimated_impact_range.net_additional_sales_low_aed == 463.68
    assert first[1].estimated_impact_range.net_additional_sales_high_aed == 993.6
    assert first[0].value_type == "estimated_scenario"
    assert all(item.opportunity_score >= next_item.opportunity_score for item, next_item in zip(first, first[1:]))
    public = json.dumps([asdict(item) for item in first])
    for forbidden in ("syn_c_", "synthetic_customer_id", "membership", "individual_score", "email", "phone"):
        assert forbidden not in public


def test_no_recommendation_without_sufficient_observed_evidence(dataset: PrivateSyntheticDataset, tmp_path) -> None:
    facts, anomaly = _facts_and_anomaly(dataset)
    audience = build_eligible_audience(
        dataset.transactions, dataset.program_eligibility, DEMO_MERCHANT_ID, AS_OF,
        PrivateAudienceStore(tmp_path / "audiences.sqlite"),
    )
    assert generate_opportunities(replace(facts, peer_change_pct=None, peer_count=None), audience, anomaly) == ()
    assert generate_opportunities(facts, audience, replace(anomaly, status="insufficient_evidence", unusually_weak=None)) == ()
    assert generate_opportunities(replace(facts, weekday_lunch_change_pct=None), audience, anomaly) == ()
    blocked = replace(audience, audience_id=None, audience_version=None, eligible_count=None, activation_eligible=False)
    remaining = generate_opportunities(facts, blocked, anomaly)
    assert all(item.audience_id is None for item in remaining)
