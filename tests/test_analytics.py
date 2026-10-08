"""Deterministic calculations and aggregate release rules."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from sme_growth.analytics import (
    get_day_of_week_analysis, get_daypart_analysis,
    get_lapsed_customer_summary, get_merchant_facts, get_peer_benchmark,
    get_repeat_customer_analysis, get_sales_overview, get_sales_performance, get_weekly_sales_pulse,
)
from sme_growth.synthetic.generator import DEMO_MERCHANT_ID, PrivateSyntheticDataset, generate_dataset
import sme_growth.ui.backend as ui_backend


AS_OF = date(2026, 10, 1)


@pytest.fixture(scope="module")
def dataset() -> PrivateSyntheticDataset:
    return generate_dataset()


def _row(token: str, timestamp: str, amount: float = 100, refund: float = 0,
         status: str = "completed", currency: str = "AED") -> dict:
    return {
        "merchant_id": "M1", "synthetic_customer_id": token,
        "transaction_datetime": timestamp, "amount": amount,
        "refund_amount": refund, "transaction_status": status, "currency": currency,
    }


def test_sales_refunds_ticket_and_complete_window_boundaries() -> None:
    rows = [_row(f"private_{i}", "2026-08-06 12:00") for i in range(10)]
    rows[0]["refund_amount"] = 20
    rows += [_row(f"private_{i}", "2026-09-03 12:00", 90) for i in range(10)]
    rows += [
        _row("fully_refunded", "2026-08-10 12:00", 100, 100, "refunded"),
        _row("declined", "2026-08-10 12:00", 100, 0, "declined"),
        _row("missing", "2026-08-10 12:00", float("nan")),
        _row("wrong_currency", "2026-08-10 12:00", currency="USD"),
        _row("outside", "2026-10-01 00:00"),
    ]
    result = get_sales_performance(pd.DataFrame(rows), "M1", AS_OF)
    assert result.status == "available"
    assert (result.baseline_sales_aed, result.current_sales_aed) == (980, 900)
    assert (result.baseline_transactions, result.current_transactions) == (10, 10)
    assert (result.baseline_average_ticket_aed, result.current_average_ticket_aed) == (98, 90)
    assert result.sales_change_pct == -8.16
    assert result.transaction_count_change_pct == 0
    assert result.average_ticket_change_pct == -8.16
    assert result.meta.baseline_window.start == "2026-08-06"
    assert result.meta.baseline_window.end_exclusive == result.meta.current_window.start == "2026-09-03"
    assert result.meta.current_window.end_exclusive == result.meta.as_of_date == "2026-10-01"
    assert result.units["sales_change_pct"] == "percent change"


def test_weekly_pulse_uses_safe_full_weeks_and_weekly_average(dataset: PrivateSyntheticDataset) -> None:
    pulse = get_weekly_sales_pulse(dataset.transactions, DEMO_MERCHANT_ID, AS_OF)
    assert pulse.status == "available"
    assert (pulse.current_window.start, pulse.current_window.end_exclusive) == ("2026-09-24", "2026-10-01")
    assert (pulse.previous_week_window.start, pulse.previous_week_window.end_exclusive) == ("2026-09-17", "2026-09-24")
    assert (pulse.previous_four_weeks_window.start, pulse.previous_four_weeks_window.end_exclusive) == ("2026-08-27", "2026-09-24")
    assert pulse.current_sales_aed == 17_234
    assert pulse.change_vs_previous_week_pct == pytest.approx((17_234 / 16_270 - 1) * 100, abs=.01)
    assert pulse.change_vs_previous_four_week_average_pct == pytest.approx(
        (17_234 / ((16_270 + 15_130 + 17_620 + 17_446) / 4) - 1) * 100, abs=.01,
    )
    serialized = json.dumps(asdict(pulse))
    assert "synthetic_customer_id" not in serialized and "syn_c_" not in serialized


def test_weekly_pulse_suppresses_all_values_when_one_source_week_is_small() -> None:
    rows = [
        _row(f"private_{person}", (AS_OF - timedelta(days=7 * week + 1)).isoformat() + " 12:00")
        for week in range(5) for person in range(10 if week != 3 else 9)
    ]
    pulse = get_weekly_sales_pulse(pd.DataFrame(rows), "M1", AS_OF)
    assert pulse.status == "suppressed"
    assert pulse.current_sales_aed is None
    assert pulse.change_vs_previous_week_pct is None
    assert pulse.change_vs_previous_four_week_average_pct is None


def test_overview_releases_fixed_periods_and_complete_daily_trend(dataset: PrivateSyntheticDataset) -> None:
    overview = get_sales_overview(dataset.transactions, dataset.merchants, DEMO_MERCHANT_ID, AS_OF)
    assert overview.day.current_window.start == "2026-09-30"
    assert overview.day.baseline_window.start == "2026-09-23"
    assert overview.week.current_window.start == "2026-09-24"
    assert overview.month.current_window.start == "2026-09-01"
    assert overview.month.baseline_window.start == "2026-08-01"
    assert [period.net_sales_aed for period in (overview.day, overview.week, overview.month)] == [
        2844, 17234, 71720,
    ]
    assert all(period.peer_status == "available" and period.peer_count >= 10
               for period in (overview.day, overview.week, overview.month))
    assert overview.daily_trend_status == "available"
    assert overview.daily_trend is not None and len(overview.daily_trend) == 30
    assert sum(point.net_sales_aed for point in overview.daily_trend) == pytest.approx(overview.month.net_sales_aed)
    assert overview.peer_daily_trend_status == "available" and overview.peer_daily_trend_count == 36
    assert overview.peer_daily_trend is not None and len(overview.peer_daily_trend) == 30
    assert overview.peer_daily_trend[0].median_net_sales_aed == 508.84
    assert [point.date for point in overview.peer_daily_trend] == [point.date for point in overview.daily_trend]
    serialized = json.dumps(asdict(overview))
    assert "synthetic_customer_id" not in serialized and "syn_c_" not in serialized
    assert "M000" not in serialized and "Synthetic Cafe" not in serialized


def test_overview_suppresses_daily_trend_and_small_peer_group(dataset: PrivateSyntheticDataset) -> None:
    transactions = dataset.transactions
    day_rows = transactions.index[
        transactions["merchant_id"].eq(DEMO_MERCHANT_ID)
        & transactions["transaction_datetime"].ge(pd.Timestamp("2026-09-01"))
        & transactions["transaction_datetime"].lt(pd.Timestamp("2026-09-02"))
    ]
    limited = transactions.drop(day_rows[9:])
    focal = dataset.merchants.loc[dataset.merchants["merchant_id"].eq(DEMO_MERCHANT_ID)].iloc[0]
    matching = dataset.merchants.loc[
        dataset.merchants["merchant_category"].eq(focal["merchant_category"])
        & dataset.merchants["merchant_location"].eq(focal["merchant_location"])
        & dataset.merchants["merchant_size"].eq(focal["merchant_size"])
        & dataset.merchants["merchant_id"].ne(DEMO_MERCHANT_ID)
    ].iloc[:9]
    small_group = pd.concat([dataset.merchants.iloc[[0]], matching])
    overview = get_sales_overview(limited, small_group, DEMO_MERCHANT_ID, AS_OF)
    assert overview.daily_trend_status == "suppressed" and overview.daily_trend is None
    assert overview.peer_daily_trend_status == "suppressed" and overview.peer_daily_trend is None
    assert overview.peer_daily_trend_count is None
    assert all(period.peer_status == "suppressed" and period.peer_group_change_pct is None
               and period.peer_count is None for period in (overview.day, overview.week, overview.month))


def test_peer_daily_trend_suppresses_all_days_when_one_peer_cell_is_small(dataset: PrivateSyntheticDataset) -> None:
    merchants = dataset.merchants
    focal = merchants.loc[merchants["merchant_id"].eq(DEMO_MERCHANT_ID)].iloc[0]
    matching = merchants.loc[
        merchants["merchant_category"].eq(focal["merchant_category"])
        & merchants["merchant_location"].eq(focal["merchant_location"])
        & merchants["merchant_size"].eq(focal["merchant_size"])
        & merchants["merchant_id"].ne(DEMO_MERCHANT_ID),
        "merchant_id",
    ]
    keep_on_day = set(matching.iloc[:9])
    transactions = dataset.transactions
    remove = (
        transactions["merchant_id"].isin(set(matching) - keep_on_day)
        & transactions["transaction_datetime"].ge(pd.Timestamp("2026-09-01"))
        & transactions["transaction_datetime"].lt(pd.Timestamp("2026-09-02"))
    )
    overview = get_sales_overview(transactions.loc[~remove], merchants, DEMO_MERCHANT_ID, AS_OF)
    assert overview.daily_trend_status == "available"
    assert overview.peer_daily_trend_status == "suppressed"
    assert overview.peer_daily_trend is None and overview.peer_daily_trend_count is None


def test_historical_choices_are_fixed_and_do_not_contain_campaign_data(dataset: PrivateSyntheticDataset, monkeypatch) -> None:
    monkeypatch.setattr(ui_backend, "_private_dataset", lambda: dataset)
    first = ui_backend.get_historical_comparison(0)
    second = ui_backend.get_historical_comparison(1)
    assert (first.sales.meta.current_window.start, first.sales.meta.current_window.end_exclusive) == (
        "2026-07-09", "2026-08-06",
    )
    assert (second.sales.meta.current_window.start, second.sales.meta.current_window.end_exclusive) == (
        "2026-05-14", "2026-06-11",
    )
    assert first.sales.status == second.sales.status == "available"
    assert not hasattr(first, "audience") and not hasattr(first, "opportunities")
    assert "synthetic_customer_id" not in json.dumps(asdict(first))
    for invalid in (-1, 2, True):
        with pytest.raises(ValueError, match="unavailable"):
            ui_backend.get_historical_comparison(invalid)


def test_zero_baseline_and_missing_data_do_not_produce_infinite_changes() -> None:
    current_only = pd.DataFrame([_row(f"private_{i}", "2026-09-10 12:00") for i in range(10)])
    result = get_sales_performance(current_only, "M1", AS_OF)
    assert result.status == "suppressed"
    assert result.sales_change_pct is None
    assert result.transaction_count_change_pct is None
    assert result.average_ticket_change_pct is None

    empty = current_only.iloc[0:0].copy()
    assert get_sales_performance(empty, "M1", AS_OF).status == "suppressed"
    invalid = current_only.drop(columns=["refund_amount"])
    with pytest.raises(ValueError, match="schema is incomplete"):
        get_sales_performance(invalid, "M1", AS_OF)


def test_fixed_breakdowns_suppress_small_cells_and_related_totals() -> None:
    rows = [_row(f"private_{i}", "2026-08-10 12:00" if i == 0 else "2026-08-10 18:00") for i in range(10)]
    rows += [_row(f"private_{i}", "2026-09-07 12:00" if i == 0 else "2026-09-07 18:00") for i in range(10)]
    transactions = pd.DataFrame(rows)
    assert get_sales_performance(transactions, "M1", AS_OF).status == "available"
    for result in (
        get_daypart_analysis(transactions, "M1", AS_OF),
        get_day_of_week_analysis(transactions, "M1", AS_OF),
    ):
        assert result.status == "suppressed"
        assert result.segments is None
        assert "total" not in asdict(result)


def test_new_and_repeat_customer_cohorts_are_distinct() -> None:
    rows = [_row(f"prior_{i}", "2026-07-01 12:00") for i in range(10)]
    rows += [_row(f"prior_{i}", "2026-08-10 12:00") for i in range(10)]
    rows += [_row(f"baseline_new_{i}", "2026-08-10 12:00") for i in range(10)]
    rows += [_row(f"prior_{i}", "2026-09-10 12:00") for i in range(10)]
    rows += [_row(f"baseline_new_{i}", "2026-09-10 12:00") for i in range(10)]
    rows += [_row(f"current_new_{i}", "2026-09-10 12:00") for i in range(10)]
    result = get_repeat_customer_analysis(pd.DataFrame(rows), "M1", AS_OF)
    assert result.new_customers.status == "available"
    assert (result.new_customers.baseline_customers, result.new_customers.current_customers) == (10, 10)
    assert (result.repeat_customers.baseline_customers, result.repeat_customers.current_customers) == (10, 20)
    assert result.repeat_customers.change_pct == 100


def test_demo_breakdowns_reconcile_and_lunch_is_largest_observed_contributor(dataset: PrivateSyntheticDataset) -> None:
    sales = get_sales_performance(dataset.transactions, DEMO_MERCHANT_ID, AS_OF)
    dayparts = get_daypart_analysis(dataset.transactions, DEMO_MERCHANT_ID, AS_OF)
    weekdays = get_day_of_week_analysis(dataset.transactions, DEMO_MERCHANT_ID, AS_OF)
    assert sales.status == dayparts.status == weekdays.status == "available"
    assert len(dayparts.segments) == 5
    assert len(weekdays.segments) == 7
    for breakdown in (dayparts, weekdays):
        assert sum(item.baseline_transactions for item in breakdown.segments) == sales.baseline_transactions
        assert sum(item.current_transactions for item in breakdown.segments) == sales.current_transactions
        assert sum(item.baseline_sales_aed for item in breakdown.segments) == pytest.approx(sales.baseline_sales_aed, abs=.02)
        assert sum(item.current_sales_aed for item in breakdown.segments) == pytest.approx(sales.current_sales_aed, abs=.02)
    lunch = next(item for item in dayparts.segments if item.segment == "weekday_lunch")
    assert lunch.transaction_count_change_pct == pytest.approx(-20.86, abs=.01)
    assert lunch.sales_delta_aed == min(item.sales_delta_aed for item in dayparts.segments)
    assert sales.sales_change_pct == pytest.approx(-9.69, abs=.01)


def test_lapsed_summary_separates_candidates_and_eligible_spend(dataset: PrivateSyntheticDataset) -> None:
    summary = get_lapsed_customer_summary(dataset.transactions, dataset.program_eligibility, DEMO_MERCHANT_ID, AS_OF)
    assert summary.status == "available"
    assert summary.lapsed_candidate_customers == 180
    assert summary.eligible_lapsed_customers == 126
    assert summary.historical_monthly_spend_aed == 21_000
    assert summary.qualification_window.start == "2026-05-19"
    assert summary.qualification_window.end_exclusive == "2026-08-17"
    assert summary.lapse_days == 45
    assert summary.regular_min_visits == 3
    assert summary.weekday_lunch_min_visits == 2


def test_lapsed_summary_ignores_transactions_after_as_of(dataset: PrivateSyntheticDataset) -> None:
    future = dataset.transactions.iloc[[0]].copy()
    future.loc[:, "merchant_id"] = DEMO_MERCHANT_ID
    future.loc[:, "synthetic_customer_id"] = dataset.program_eligibility.iloc[0]["synthetic_customer_id"]
    future.loc[:, "transaction_datetime"] = pd.Timestamp("2026-10-02 12:00")
    future.loc[:, "amount"] = 60.0
    future.loc[:, "refund_amount"] = 0.0
    future.loc[:, "transaction_status"] = "completed"
    future.loc[:, "currency"] = "AED"
    transactions = pd.concat([dataset.transactions, future], ignore_index=True)
    summary = get_lapsed_customer_summary(transactions, dataset.program_eligibility, DEMO_MERCHANT_ID, AS_OF)
    assert summary.lapsed_candidate_customers == 180
    assert summary.eligible_lapsed_customers == 126


def test_lapsed_summary_fails_closed_and_suppresses_related_totals(dataset: PrivateSyntheticDataset) -> None:
    program = dataset.program_eligibility.copy()
    program.loc[:99, "permission_state"] = "unknown"
    result = get_lapsed_customer_summary(dataset.transactions, program, DEMO_MERCHANT_ID, AS_OF)
    assert result.status == "suppressed"
    assert result.lapsed_candidate_customers is None
    assert result.eligible_lapsed_customers is None
    assert result.historical_monthly_spend_aed is None
    missing = get_lapsed_customer_summary(dataset.transactions, program.drop(columns=["purpose"]), DEMO_MERCHANT_ID, AS_OF)
    assert missing.status == "suppressed"


def test_peer_threshold_and_no_individual_peer_results(dataset: PrivateSyntheticDataset) -> None:
    merchants = dataset.merchants
    focal = merchants[merchants["merchant_id"].eq(DEMO_MERCHANT_ID)].iloc[0]
    peers = merchants[
        merchants["merchant_category"].eq(focal["merchant_category"])
        & merchants["merchant_location"].eq(focal["merchant_location"])
        & merchants["merchant_size"].eq(focal["merchant_size"])
        & merchants["merchant_id"].ne(DEMO_MERCHANT_ID)
    ]
    assert len(peers) >= 10
    for count, expected_status in ((9, "suppressed"), (10, "available")):
        subset = pd.concat([merchants.iloc[[0]], peers.iloc[:count]])
        result = get_peer_benchmark(dataset.transactions, subset, DEMO_MERCHANT_ID, AS_OF)
        assert result.status == expected_status
        assert result.peer_count == (count if count == 10 else None)
        assert result.median_sales_change_pct is None if count == 9 else result.median_sales_change_pct is not None
        serialized = json.dumps(asdict(result))
        assert "M000" not in serialized
        assert "Synthetic Cafe" not in serialized


def test_structured_demo_facts_are_safe_and_observed(dataset: PrivateSyntheticDataset) -> None:
    facts = get_merchant_facts(dataset.transactions, dataset.merchants, dataset.program_eligibility, DEMO_MERCHANT_ID, AS_OF)
    assert facts.merchant_name == "Visa Versa Cafe"
    assert facts.largest_driver == "weekday_lunch"
    assert facts.weekday_lunch_change_pct == -20.86
    assert facts.repeat_customer_change_pct is not None and facts.repeat_customer_change_pct < 0
    assert facts.peer_count is not None and facts.peer_count >= 10
    assert abs(facts.peer_change_pct) < 5
    assert (facts.lapsed_candidate_customers, facts.lapsed_regular_customers) == (180, 126)
    assert facts.historical_lapsed_spend == 21_000
    assert facts.units["historical_lapsed_spend"] == "AED per historical 30-day equivalent"
    serialized = json.dumps(asdict(facts))
    assert "syn_c_" not in serialized
    assert "synthetic_customer_id" not in serialized
    assert "transaction_id" not in serialized
    assert "membership" not in serialized
    assert "score" not in serialized
