"""Generator and synthetic-story checks; these frames remain private."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import pytest

from sme_growth.synthetic.generator import (
    DEMO_MERCHANT_ID,
    GenerationConfig,
    PrivateSyntheticDataset,
    generate_dataset,
    safe_run_summary,
    write_dataset,
)


@pytest.fixture(scope="module")
def dataset() -> PrivateSyntheticDataset:
    return generate_dataset()


def _windows(dataset: PrivateSyntheticDataset) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    as_of = dataset.config.as_of_date
    return pd.Timestamp(as_of - timedelta(days=56)), pd.Timestamp(as_of - timedelta(days=28)), pd.Timestamp(as_of)


def _completed(dataset: PrivateSyntheticDataset) -> pd.DataFrame:
    return dataset.transactions[dataset.transactions["transaction_status"] == "completed"]


def test_generation_is_reproducible() -> None:
    config = GenerationConfig(seed=701, as_of_date=date(2026, 10, 1), merchant_count=30, customer_count=3_000)
    first = generate_dataset(config)
    second = generate_dataset(config)
    pd.testing.assert_frame_equal(first.merchants, second.merchants)
    pd.testing.assert_frame_equal(first.transactions, second.transactions)
    pd.testing.assert_frame_equal(first.program_eligibility, second.program_eligibility)


def test_scale_schema_and_dates(dataset: PrivateSyntheticDataset) -> None:
    merchants = dataset.merchants
    transactions = dataset.transactions
    program = dataset.program_eligibility
    assert len(merchants) == 300
    assert len(program) == 20_000
    assert 300_000 <= len(transactions) <= 500_000
    assert transactions["synthetic_customer_id"].nunique() == 20_000
    assert transactions["transaction_id"].is_unique
    assert merchants["merchant_id"].is_unique
    assert {
        "merchant_id", "merchant_name", "merchant_category", "merchant_location",
        "merchant_size", "opening_date",
    } <= set(merchants)
    assert {
        "transaction_id", "transaction_datetime", "amount", "currency",
        "transaction_status", "refund_amount", "merchant_id", "merchant_category",
        "synthetic_customer_id", "domestic_or_international", "merchant_location",
    } <= set(transactions)
    assert {
        "synthetic_customer_id", "enrolled", "permission_state", "purpose",
        "channel", "valid_until", "revoked_at", "frequency_cap_ok", "suppressed",
    } <= set(program)
    assert transactions["transaction_datetime"].min() >= pd.Timestamp(dataset.config.as_of_date - timedelta(days=365))
    assert transactions["transaction_datetime"].max() < pd.Timestamp(dataset.config.as_of_date)
    assert set(transactions["currency"]) == {"AED"}
    assert set(transactions["domestic_or_international"]) == {"domestic", "international"}
    assert set(transactions["transaction_status"]) == {"completed", "refunded", "declined"}
    assert (transactions["amount"] > 0).all()
    assert (transactions["refund_amount"] >= 0).all()
    assert (transactions["refund_amount"] <= transactions["amount"]).all()


def test_background_merchants_cover_continuous_trading_hours(dataset: PrivateSyntheticDataset) -> None:
    background = dataset.transactions[dataset.transactions["merchant_id"].ne(DEMO_MERCHANT_ID)]
    hours = background["transaction_datetime"].dt.hour.value_counts()
    assert set(range(8, 22)) <= set(hours.index)
    assert hours.loc[15] > len(background) * .03
    assert hours.loc[17] > len(background) * .03


def test_no_personal_fields_or_private_identifiers_in_public_summary(dataset: PrivateSyntheticDataset, tmp_path) -> None:
    private_fields = {
        "customer_name", "first_name", "last_name", "email", "phone",
        "pan", "card_number", "customer_location", "member_ids",
    }
    for frame in (dataset.merchants, dataset.transactions, dataset.program_eligibility):
        assert private_fields.isdisjoint(frame.columns)
    summary = safe_run_summary(dataset)
    assert set(summary) == {
        "data_label", "as_of_date", "seed", "merchant_count",
        "private_synthetic_customer_count", "transaction_count",
    }
    assert all(isinstance(value, (str, int)) for value in summary.values())
    written = write_dataset(dataset, tmp_path / "generated")
    manifest = (tmp_path / "generated" / "run_summary.json").read_text(encoding="utf-8")
    assert json.loads(manifest) == written
    assert "syn_c_" not in manifest
    assert "transaction_id" not in manifest
    assert "merchant_name" not in manifest
    assert (tmp_path / "generated" / f"transactions.{written['format']}").exists()
    assert (tmp_path / "generated" / f"program_eligibility.{written['format']}").exists()


def test_eligibility_has_invalid_and_unknown_cases(dataset: PrivateSyntheticDataset) -> None:
    program = dataset.program_eligibility
    as_of = pd.Timestamp(dataset.config.as_of_date)
    assert program["permission_state"].eq("unknown").any()
    assert program["permission_state"].eq("denied").any()
    assert program["revoked_at"].notna().any()
    assert program["valid_until"].lt(as_of).any()
    assert (~program["enrolled"]).any()
    assert (~program["frequency_cap_ok"]).any()
    assert program["suppressed"].any()


def test_visa_versa_cafe_has_decline_and_weaker_repeat_activity(dataset: PrivateSyntheticDataset) -> None:
    start, middle, end = _windows(dataset)
    cafe = _completed(dataset)
    cafe = cafe[cafe["merchant_id"] == DEMO_MERCHANT_ID]
    baseline = cafe[cafe["transaction_datetime"].between(start, middle, inclusive="left")]
    current = cafe[cafe["transaction_datetime"].between(middle, end, inclusive="left")]
    sales_change = current["amount"].sum() / baseline["amount"].sum() - 1
    assert -.10 <= sales_change <= -.08

    def lunch_count(frame: pd.DataFrame) -> int:
        times = frame["transaction_datetime"]
        return int(((times.dt.weekday < 5) & times.dt.hour.between(11, 14)).sum())

    lunch_change = lunch_count(current) / lunch_count(baseline) - 1
    assert -.21 <= lunch_change <= -.20

    historical_tokens = set(cafe[cafe["transaction_datetime"] < start]["synthetic_customer_id"])
    baseline_repeat = set(baseline["synthetic_customer_id"]) & historical_tokens
    current_repeat = set(current["synthetic_customer_id"]) & historical_tokens
    assert len(current_repeat) < len(baseline_repeat)


def test_similar_peer_cafes_are_broadly_stable(dataset: PrivateSyntheticDataset) -> None:
    start, middle, end = _windows(dataset)
    merchants = dataset.merchants
    peer_ids = set(merchants.loc[
        (merchants["merchant_category"] == "Cafe")
        & (merchants["merchant_location"] == "Dubai")
        & (merchants["merchant_size"] == "medium")
        & (merchants["merchant_id"] != DEMO_MERCHANT_ID),
        "merchant_id",
    ])
    assert len(peer_ids) >= 10
    tx = _completed(dataset)
    tx = tx[tx["merchant_id"].isin(peer_ids)]
    baseline = tx[tx["transaction_datetime"].between(start, middle, inclusive="left")].groupby("merchant_id")["amount"].sum()
    current = tx[tx["transaction_datetime"].between(middle, end, inclusive="left")].groupby("merchant_id")["amount"].sum()
    changes = current / baseline - 1
    assert changes.notna().sum() >= 10
    assert abs(changes.median()) <= .05


def test_lapsed_candidates_and_eligible_subset(dataset: PrivateSyntheticDataset) -> None:
    as_of = pd.Timestamp(dataset.config.as_of_date)
    qualification_start = pd.Timestamp(dataset.config.as_of_date - timedelta(days=135))
    lapse_cutoff = pd.Timestamp(dataset.config.as_of_date - timedelta(days=45))
    cafe = _completed(dataset)
    cafe = cafe[cafe["merchant_id"] == DEMO_MERCHANT_ID]
    qualification = cafe[cafe["transaction_datetime"].between(qualification_start, lapse_cutoff, inclusive="left")]
    visits = qualification.groupby("synthetic_customer_id").size()
    lunch = qualification[
        (qualification["transaction_datetime"].dt.weekday < 5)
        & qualification["transaction_datetime"].dt.hour.between(11, 14)
    ].groupby("synthetic_customer_id").size()
    last_visit = cafe.groupby("synthetic_customer_id")["transaction_datetime"].max()
    candidates = set(visits[visits >= 3].index) & set(lunch[lunch >= 2].index) & set(last_visit[last_visit < lapse_cutoff].index)
    assert len(candidates) == 180

    program = dataset.program_eligibility.set_index("synthetic_customer_id").loc[sorted(candidates)]
    eligible = (
        program["enrolled"]
        & program["permission_state"].eq("granted")
        & program["purpose"].eq("merchant_retention")
        & program["channel"].eq("issuer_app")
        & program["valid_until"].ge(as_of)
        & program["revoked_at"].isna()
        & program["frequency_cap_ok"]
        & ~program["suppressed"]
    )
    eligible_tokens = set(program.index[eligible])
    assert len(eligible_tokens) == 126
    assert len(eligible_tokens) < len(candidates)
    historical_monthly_spend = qualification[
        qualification["synthetic_customer_id"].isin(eligible_tokens)
    ]["amount"].sum() / 3
    assert historical_monthly_spend == 21_000
