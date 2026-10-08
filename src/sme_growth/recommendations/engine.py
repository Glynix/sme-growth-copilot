"""Rank safe, structured opportunity proposals from observed evidence."""

from __future__ import annotations

from dataclasses import dataclass

from sme_growth.analytics.contracts import MerchantFacts
from sme_growth.contracts import AudienceSummary, MIN_ELIGIBLE_AUDIENCE
from sme_growth.ml.anomaly import MerchantAnomalySignal
from sme_growth.ml.retention import RetentionCohortSummary
from sme_growth.recommendations.economics import (
    EstimatedImpactRange, ImpactAssumptions, calculate_impact,
)


@dataclass(frozen=True, slots=True)
class OpportunityEvidence:
    metric: str
    value: float | int | str
    unit: str
    evidence_type: str


@dataclass(frozen=True, slots=True)
class Opportunity:
    opportunity_id: str
    title: str
    as_of_date: str
    evidence: tuple[OpportunityEvidence, ...]
    audience_id: str | None
    audience_version: int | None
    eligible_count: int | None
    historical_value_aed_per_month: float | None
    historical_value_evidence_type: str | None
    estimated_impact_range: EstimatedImpactRange
    confidence: float
    actionability: float
    opportunity_score: float
    value_type: str = "estimated_scenario"
    status: str = "proposal_requires_approval"
    scoring_note: str = "Score = midpoint estimated incremental sales in AED × evidence confidence × actionability; it is a ranking aid, not a net sales forecast."


def _score(impact: EstimatedImpactRange, confidence: float, actionability: float) -> float:
    midpoint = (impact.incremental_sales_low_aed + impact.incremental_sales_high_aed) / 2
    return round(midpoint * confidence * actionability, 2)


def generate_opportunities(
    facts: MerchantFacts,
    audience: AudienceSummary,
    anomaly: MerchantAnomalySignal,
    retention: RetentionCohortSummary | None = None,
    model_audience: AudienceSummary | None = None,
) -> tuple[Opportunity, ...]:
    """Generate proposals only when the core observed evidence is sufficient."""

    if (
        facts.sales_change_pct is None or facts.sales_change_pct >= -3
        or facts.weekday_lunch_change_pct is None or facts.weekday_lunch_change_pct >= -5
        or facts.weekday_lunch_transactions_lost is None or facts.weekday_lunch_transactions_lost < 10
        or facts.peer_change_pct is None or facts.peer_count is None or facts.peer_count < 10
        or anomaly.status != "available" or anomaly.unusually_weak is not True
    ):
        return ()
    evidence = (
        OpportunityEvidence("sales_change_pct", facts.sales_change_pct, "percent change", "observed"),
        OpportunityEvidence("weekday_lunch_change_pct", facts.weekday_lunch_change_pct, "percent change", "observed"),
        OpportunityEvidence("peer_change_pct", facts.peer_change_pct, "median peer percent change", "observed_aggregate"),
        OpportunityEvidence("anomaly_peer_gap_pp", anomaly.peer_gap_percentage_points or 0.0, "percentage points", "observed_comparison"),
    )
    confidence = .60 + (.08 if facts.repeat_customer_change_pct is not None and facts.repeat_customer_change_pct < 0 else 0)
    if retention and retention.status in ("available", "partly_suppressed") and retention.model_signal == "behavioral_logistic_regression":
        confidence += .05
        evidence += (
            OpportunityEvidence("retention_model_association", retention.top_model_associations[0], "cohort-level association", "estimated_model"),
        )
    confidence = round(min(confidence, .90), 2)
    opportunities: list[Opportunity] = []

    if (
        audience.activation_eligible
        and audience.audience_id is not None
        and audience.audience_version is not None
        and audience.eligible_count is not None
        and audience.eligible_count >= MIN_ELIGIBLE_AUDIENCE
        and facts.historical_lapsed_spend is not None
    ):
        winback = calculate_impact(ImpactAssumptions(
            addressable_count=audience.eligible_count,
            reach_rate=1.0,
            response_rate_low=.08, response_rate_high=.12,
            visits_per_response=2, spend_per_visit_aed=60,
            incentive_rate=.15,
            all_qualifying_redemption_rate_low=.14,
            all_qualifying_redemption_rate_high=.22,
            response_meaning="Assumed incremental return share of reached eligible customers",
        ))
        opportunities.append(Opportunity(
            "weekday_lunch_winback", "Review a 15% weekday lunch win-back offer",
            facts.as_of_date,
            evidence + (OpportunityEvidence("eligible_lapsed_customers", audience.eligible_count, "distinct customers", "observed_behavior_and_simulated_eligibility"),),
            audience.audience_id, audience.audience_version, audience.eligible_count,
            facts.historical_lapsed_spend, "observed_history",
            winback, confidence, .95, _score(winback, confidence, .95),
        ))
        lighter = calculate_impact(ImpactAssumptions(
            addressable_count=audience.eligible_count,
            reach_rate=.85,
            response_rate_low=.04, response_rate_high=.06,
            visits_per_response=1, spend_per_visit_aed=60,
            incentive_rate=.10,
            all_qualifying_redemption_rate_low=.08,
            all_qualifying_redemption_rate_high=.12,
            response_meaning="Assumed incremental return share of reached eligible customers",
        ))
        opportunities.append(Opportunity(
            "lighter_lunch_offer", "Review a lighter 10% lunch offer",
            facts.as_of_date, evidence,
            audience.audience_id, audience.audience_version, audience.eligible_count,
            facts.historical_lapsed_spend, "observed_history",
            lighter, round(confidence * .8, 2), .85,
            _score(lighter, round(confidence * .8, 2), .85),
        ))

    if (
        retention is not None and retention.model_signal == "behavioral_logistic_regression"
        and model_audience is not None and model_audience.activation_eligible
        and model_audience.audience_id is not None and model_audience.audience_version is not None
        and model_audience.eligible_count is not None
        and model_audience.eligible_count >= MIN_ELIGIBLE_AUDIENCE
    ):
        prevention = calculate_impact(ImpactAssumptions(
            addressable_count=model_audience.eligible_count,
            reach_rate=1.0, response_rate_low=.06, response_rate_high=.10,
            visits_per_response=2, spend_per_visit_aed=60,
            incentive_rate=.10,
            all_qualifying_redemption_rate_low=.10,
            all_qualifying_redemption_rate_high=.18,
            response_meaning="Assumed incremental retained share of reached at-risk regulars; not predicted by the model",
        ))
        opportunities.append(Opportunity(
            "model_retention_offer", "Review 10% off all purchases for at-risk regulars",
            facts.as_of_date,
            evidence + (
                OpportunityEvidence("model_targeting", "Top 20% predicted 30-day non-return risk among active regulars", "private cohort rule", "estimated_model"),
                OpportunityEvidence("eligible_at_risk_regulars", model_audience.eligible_count, "distinct customers", "estimated_model_and_simulated_eligibility"),
            ),
            model_audience.audience_id, model_audience.audience_version,
            model_audience.eligible_count, None, None,
            prevention, confidence, .90, _score(prevention, confidence, .90),
        ))

    operations = calculate_impact(ImpactAssumptions(
        addressable_count=facts.weekday_lunch_transactions_lost,
        reach_rate=1.0,
        response_rate_low=.10, response_rate_high=.20,
        visits_per_response=1, spend_per_visit_aed=62,
        incentive_rate=0,
        all_qualifying_redemption_rate_low=0,
        all_qualifying_redemption_rate_high=0,
        response_meaning="Assumed share of lost weekday-lunch visits recovered through operations",
    ))
    opportunities.append(Opportunity(
        "weekday_lunch_operations", "Improve weekday lunch service and availability",
        facts.as_of_date, evidence,
        None, None, None, None, None,
        operations, round(confidence * .75, 2), .70,
        _score(operations, round(confidence * .75, 2), .70),
    ))
    return tuple(sorted(opportunities, key=lambda item: (-item.opportunity_score, item.opportunity_id)))
