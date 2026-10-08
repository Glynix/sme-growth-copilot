"""Allowlisted aggregate analytics objects for the merchant, UI, and future AI.

These immutable shapes contain no customer token, membership, individual
score, peer identity, or row-level transaction. All windows are local Dubai
calendar dates with an exclusive end.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Window:
    start: str
    end_exclusive: str


@dataclass(frozen=True, slots=True)
class ComparisonMeta:
    as_of_date: str
    baseline_window: Window
    current_window: Window
    evidence_type: str = "observed"


@dataclass(frozen=True, slots=True)
class SalesPerformance:
    meta: ComparisonMeta
    baseline_sales_aed: float | None
    current_sales_aed: float | None
    sales_change_pct: float | None
    baseline_transactions: int | None
    current_transactions: int | None
    transaction_count_change_pct: float | None
    baseline_average_ticket_aed: float | None
    current_average_ticket_aed: float | None
    average_ticket_change_pct: float | None
    status: str
    note: str
    units: dict[str, str] = field(default_factory=lambda: {
        "baseline_sales_aed": "AED net sales", "current_sales_aed": "AED net sales",
        "sales_change_pct": "percent change", "baseline_transactions": "settled purchases",
        "current_transactions": "settled purchases", "transaction_count_change_pct": "percent change",
        "baseline_average_ticket_aed": "AED per settled purchase",
        "current_average_ticket_aed": "AED per settled purchase",
        "average_ticket_change_pct": "percent change",
    })


@dataclass(frozen=True, slots=True)
class WeeklySalesPulse:
    """One fixed, merchant-only seven-day comparison with safe source weeks."""

    as_of_date: str
    current_window: Window
    previous_week_window: Window
    previous_four_weeks_window: Window
    current_sales_aed: float | None
    change_vs_previous_week_pct: float | None
    change_vs_previous_four_week_average_pct: float | None
    status: str
    note: str
    evidence_type: str = "observed_synthetic_transactions"
    units: dict[str, str] = field(default_factory=lambda: {
        "current_sales_aed": "AED net sales per seven days",
        "change_vs_previous_week_pct": "percent change",
        "change_vs_previous_four_week_average_pct": "percent change versus AED net sales per seven days",
    })


@dataclass(frozen=True, slots=True)
class OverviewPeriod:
    current_window: Window
    baseline_window: Window
    net_sales_aed: float | None
    sales_change_pct: float | None
    peer_group_change_pct: float | None
    peer_count: int | None
    status: str
    peer_status: str
    unit: str = "AED net sales"
    evidence_type: str = "observed_synthetic_transactions"


@dataclass(frozen=True, slots=True)
class DailySalesPoint:
    date: str
    net_sales_aed: float


@dataclass(frozen=True, slots=True)
class PeerDailySalesPoint:
    date: str
    median_net_sales_aed: float


@dataclass(frozen=True, slots=True)
class SalesOverview:
    as_of_date: str
    day: OverviewPeriod
    week: OverviewPeriod
    month: OverviewPeriod
    daily_trend_window: Window
    daily_trend: tuple[DailySalesPoint, ...] | None
    daily_trend_status: str
    peer_daily_trend: tuple[PeerDailySalesPoint, ...] | None
    peer_daily_trend_count: int | None
    peer_daily_trend_status: str
    evidence_type: str = "observed_synthetic_transactions"


@dataclass(frozen=True, slots=True)
class SegmentComparison:
    segment: str
    baseline_sales_aed: float
    current_sales_aed: float
    sales_delta_aed: float
    baseline_transactions: int
    current_transactions: int
    transaction_count_change_pct: float | None


@dataclass(frozen=True, slots=True)
class BreakdownResult:
    meta: ComparisonMeta
    segments: tuple[SegmentComparison, ...] | None
    status: str
    note: str
    units: dict[str, str] = field(default_factory=lambda: {
        "baseline_sales_aed": "AED net sales", "current_sales_aed": "AED net sales",
        "sales_delta_aed": "AED observed contribution",
        "baseline_transactions": "settled purchases", "current_transactions": "settled purchases",
        "transaction_count_change_pct": "percent change",
    })


@dataclass(frozen=True, slots=True)
class CustomerTrend:
    baseline_customers: int | None
    current_customers: int | None
    change_pct: float | None
    status: str


@dataclass(frozen=True, slots=True)
class RepeatCustomerAnalysis:
    meta: ComparisonMeta
    new_customers: CustomerTrend
    repeat_customers: CustomerTrend
    evidence_type: str = "observed"
    unit: str = "distinct customers"
    note: str = "New means first observed settled purchase at this merchant in that window; repeat means observed before it."


@dataclass(frozen=True, slots=True)
class LapsedCustomerSummary:
    as_of_date: str
    qualification_window: Window
    lapse_days: int
    regular_min_visits: int
    weekday_lunch_min_visits: int
    lapsed_candidate_customers: int | None
    eligible_lapsed_customers: int | None
    historical_monthly_spend_aed: float | None
    status: str
    note: str
    evidence_type: str = "observed_history_and_simulated_program_eligibility"
    units: dict[str, str] = field(default_factory=lambda: {
        "lapsed_candidate_customers": "distinct customers",
        "eligible_lapsed_customers": "distinct customers",
        "historical_monthly_spend_aed": "AED per historical 30-day equivalent",
    })


@dataclass(frozen=True, slots=True)
class PeerBenchmark:
    meta: ComparisonMeta
    peer_group: str
    peer_count: int | None
    median_sales_change_pct: float | None
    lower_quartile_pct: float | None
    upper_quartile_pct: float | None
    status: str
    note: str
    unit: str = "percent change in net sales"


@dataclass(frozen=True, slots=True)
class MerchantFacts:
    """One safe, structured response for the demo merchant."""

    merchant_name: str
    as_of_date: str
    baseline_window: Window
    current_window: Window
    qualification_window: Window
    evidence_type: str
    sales_change_pct: float | None
    transaction_count_change_pct: float | None
    average_ticket_change_pct: float | None
    largest_driver: str | None
    largest_driver_sales_delta_aed: float | None
    weekday_lunch_change_pct: float | None
    weekday_lunch_transactions_lost: int | None
    repeat_customer_change_pct: float | None
    peer_change_pct: float | None
    peer_count: int | None
    lapsed_candidate_customers: int | None
    lapsed_regular_customers: int | None
    historical_lapsed_spend: float | None
    units: dict[str, str] = field(default_factory=lambda: {
        "sales_change_pct": "percent change", "transaction_count_change_pct": "percent change",
        "average_ticket_change_pct": "percent change",
        "largest_driver_sales_delta_aed": "AED observed sales contribution",
        "weekday_lunch_change_pct": "percent change",
        "weekday_lunch_transactions_lost": "settled purchases",
        "repeat_customer_change_pct": "percent change",
        "peer_change_pct": "percent change", "peer_count": "merchants",
        "lapsed_candidate_customers": "distinct customers",
        "lapsed_regular_customers": "distinct customers",
        "historical_lapsed_spend": "AED per historical 30-day equivalent",
    })
    interpretation: str = "Observed contributors and peer comparisons do not establish cause or incremental impact."
