"""Private data orchestration; Streamlit receives only safe aggregate objects."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd

from sme_growth.activation.campaign import (
    ALL_DAYS, ALL_PURCHASES_WINDOW, CampaignDefinition, CampaignService,
    CampaignState, WEEKDAY_LUNCH_DAYS, WEEKDAY_LUNCH_WINDOW,
)
from sme_growth.analytics import (
    get_daypart_analysis, get_merchant_facts, get_peer_benchmark,
    get_sales_overview, get_sales_performance, get_weekly_sales_pulse,
)
from sme_growth.analytics.contracts import MerchantFacts, PeerBenchmark, SalesOverview, SalesPerformance, WeeklySalesPulse
from sme_growth.audiences.builder import PrivateAudienceStore, build_eligible_audience, build_model_risk_audience
from sme_growth.contracts import AudienceSummary
from sme_growth.ml.anomaly import MerchantAnomalySignal, get_merchant_anomaly_signal
from sme_growth.ml.retention import RetentionCohortSummary, get_retention_cohort_summary, train_retention_model
from sme_growth.recommendations.engine import Opportunity, generate_opportunities
from sme_growth.synthetic.generator import (
    DEMO_MERCHANT_ID, GenerationConfig, PrivateSyntheticDataset,
    generate_dataset, write_dataset,
)


ROOT = Path(__file__).resolve().parents[3]
AS_OF_DATE = GenerationConfig().as_of_date
DATA_DIR = ROOT / "data" / "generated"
STATE_PATH = ROOT / "runtime" / "demo.sqlite"
AUDIT_PATH = ROOT / "runtime" / "copilot_audit.sqlite"
HISTORICAL_COMPARISON_ENDS = tuple(AS_OF_DATE - timedelta(days=56 * index) for index in (1, 2))


@dataclass(frozen=True, slots=True)
class DashboardView:
    merchant_name: str
    merchant_category: str
    merchant_location: str
    merchant_size: str
    sales: SalesPerformance
    overview: SalesOverview
    facts: MerchantFacts
    peers: PeerBenchmark
    audience: AudienceSummary
    model_audience: AudienceSummary | None
    anomaly: MerchantAnomalySignal
    retention: RetentionCohortSummary | None
    opportunities: tuple[Opportunity, ...]


@dataclass(frozen=True, slots=True)
class HistoricalComparison:
    sales: SalesPerformance
    weekday_lunch_change_pct: float | None
    peers: PeerBenchmark


def get_historical_comparison(period_index: int) -> HistoricalComparison:
    """Release only preapproved, disjoint 28-day comparison pairs."""

    if type(period_index) is not int or not 0 <= period_index < len(HISTORICAL_COMPARISON_ENDS):
        raise ValueError("Historical comparison is unavailable")
    private = _private_dataset()
    end = HISTORICAL_COMPARISON_ENDS[period_index]
    sales = get_sales_performance(private.transactions, DEMO_MERCHANT_ID, end)
    breakdown = get_daypart_analysis(private.transactions, DEMO_MERCHANT_ID, end)
    lunch_change = None
    if breakdown.segments is not None:
        lunch_change = next(
            segment.transaction_count_change_pct
            for segment in breakdown.segments if segment.segment == "weekday_lunch"
        )
    peers = get_peer_benchmark(private.transactions, private.merchants, DEMO_MERCHANT_ID, end)
    return HistoricalComparison(sales, lunch_change, peers)


def get_last_week_pulse() -> WeeklySalesPulse:
    private = _private_dataset()
    return get_weekly_sales_pulse(private.transactions, DEMO_MERCHANT_ID, AS_OF_DATE)


def _read_table(stem: str) -> pd.DataFrame:
    parquet = DATA_DIR / f"{stem}.parquet"
    csv = DATA_DIR / f"{stem}.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        frame = pd.read_csv(csv)
        for column in ("transaction_datetime", "valid_until", "revoked_at", "opening_date"):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
        return frame
    raise FileNotFoundError(stem)


def _private_dataset() -> PrivateSyntheticDataset:
    if not all(
        any((DATA_DIR / f"{stem}.{extension}").exists() for extension in ("parquet", "csv"))
        for stem in ("merchants", "transactions", "program_eligibility")
    ):
        write_dataset(generate_dataset(), DATA_DIR)
    return PrivateSyntheticDataset(
        _read_table("merchants"), _read_table("transactions"),
        _read_table("program_eligibility"), GenerationConfig(),
    )


def _service() -> CampaignService:
    private = _private_dataset()
    return CampaignService(
        STATE_PATH, private.transactions, private.program_eligibility,
        DEMO_MERCHANT_ID, AS_OF_DATE,
    )


def get_dashboard() -> DashboardView:
    private = _private_dataset()
    row = private.merchants.loc[private.merchants["merchant_id"].eq(DEMO_MERCHANT_ID)].iloc[0]
    sales = get_sales_performance(private.transactions, DEMO_MERCHANT_ID, AS_OF_DATE)
    overview = get_sales_overview(private.transactions, private.merchants, DEMO_MERCHANT_ID, AS_OF_DATE)
    facts = get_merchant_facts(
        private.transactions, private.merchants, private.program_eligibility,
        DEMO_MERCHANT_ID, AS_OF_DATE,
    )
    peers = get_peer_benchmark(private.transactions, private.merchants, DEMO_MERCHANT_ID, AS_OF_DATE)
    audience = build_eligible_audience(
        private.transactions, private.program_eligibility, DEMO_MERCHANT_ID,
        AS_OF_DATE, PrivateAudienceStore(STATE_PATH),
    )
    anomaly = get_merchant_anomaly_signal(private.transactions, DEMO_MERCHANT_ID, AS_OF_DATE, peers)
    try:
        model = train_retention_model(private.transactions, AS_OF_DATE)
        retention = get_retention_cohort_summary(model, private.transactions, DEMO_MERCHANT_ID, AS_OF_DATE)
        model_audience = build_model_risk_audience(
            model, private.transactions, private.program_eligibility,
            DEMO_MERCHANT_ID, AS_OF_DATE, PrivateAudienceStore(STATE_PATH),
        )
    except ValueError:
        retention = None
        model_audience = None
    opportunities = generate_opportunities(facts, audience, anomaly, retention, model_audience)
    return DashboardView(
        merchant_name=str(row["merchant_name"]), merchant_category=str(row["merchant_category"]),
        merchant_location=str(row["merchant_location"]), merchant_size=str(row["merchant_size"]),
        sales=sales, overview=overview, facts=facts, peers=peers, audience=audience, anomaly=anomaly,
        model_audience=model_audience, retention=retention, opportunities=opportunities,
    )


def default_campaign_definition(view: DashboardView, opportunity_id: str = "weekday_lunch_winback") -> CampaignDefinition | None:
    opportunity = next((item for item in view.opportunities if item.opportunity_id == opportunity_id), None)
    audience = view.audience if opportunity_id == "weekday_lunch_winback" else view.model_audience
    if opportunity is None or audience is None or opportunity_id not in ("weekday_lunch_winback", "model_retention_offer"):
        return None
    if opportunity.audience_id is None or opportunity.audience_version is None:
        return None
    latest_end = min(AS_OF_DATE + timedelta(days=6), audience.expires_at.date() - timedelta(days=1))
    if latest_end < AS_OF_DATE + timedelta(days=1):
        return None
    high_cost = Decimal(str(opportunity.estimated_impact_range.campaign_cost_high_aed))
    default_budget = max(Decimal("700"), high_cost.quantize(Decimal("1")) + 1)
    return CampaignDefinition(
        DEMO_MERCHANT_ID, opportunity.opportunity_id,
        opportunity.audience_id, opportunity.audience_version,
        15 if opportunity_id == "weekday_lunch_winback" else 10,
        AS_OF_DATE + timedelta(days=1), latest_end,
        "issuer_app", default_budget,
        offer_days=WEEKDAY_LUNCH_DAYS if opportunity_id == "weekday_lunch_winback" else ALL_DAYS,
        offer_window=WEEKDAY_LUNCH_WINDOW if opportunity_id == "weekday_lunch_winback" else ALL_PURCHASES_WINDOW,
    )


def review_campaign(definition: CampaignDefinition, opportunity: Opportunity) -> CampaignState:
    return _service().review(definition, opportunity)


def approve_campaign(ref: str, version: int, definition: CampaignDefinition, *, explicit: bool) -> CampaignState:
    return _service().approve(ref, version, definition, explicit=explicit)


def submit_campaign(ref: str, version: int, definition: CampaignDefinition) -> CampaignState:
    return _service().submit(ref, version, definition)


def advance_campaign(ref: str, version: int, definition: CampaignDefinition) -> CampaignState:
    return _service().advance(ref, version, definition)


def get_campaign_state(ref: str, version: int) -> CampaignState:
    return _service().get_state(ref, version)


def get_latest_campaign_state(opportunity_id: str) -> CampaignState | None:
    return _service().latest_state(opportunity_id)


def ask_copilot(question: str, view: DashboardView, *, previous_intent: str | None = None):
    from sme_growth.copilot.service import CopilotService, build_copilot_context

    return CopilotService(AUDIT_PATH).answer(question, build_copilot_context(view), previous_intent=previous_intent)
