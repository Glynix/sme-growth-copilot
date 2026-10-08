"""Allowlisted merchant-facing response shapes.

Only safe aggregate summaries may reach the UI, exports, or a future LLM.
Opaque audience and campaign references are permitted; customer PII, customer
IDs or tokens, membership, contact details, and individual scores are not.
Analytics, audience and measurement release checks live in their respective
modules. Local Python boundaries are not production permission isolation.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


MIN_PEER_MERCHANTS = 10
MIN_ELIGIBLE_AUDIENCE = 50
MIN_OUTCOME_CONTRIBUTORS = 10


@dataclass(frozen=True, slots=True)
class AudienceSummary:
    """Safe audience metadata, with counts omitted when release is suppressed."""

    audience_id: str | None
    audience_version: int | None
    safe_definition: str
    eligible_count: int | None
    snapshot_at: datetime
    expires_at: datetime
    activation_eligible: bool
    suppression_reason: str | None = None
    evidence_type: str = "observed_behavior_and_simulated_program_eligibility"
    historical_spend_aed_per_30d_equivalent: float | None = None


@dataclass(frozen=True, slots=True)
class AggregateOutcome:
    """Aggregate simulated results; None means unavailable or suppressed."""

    campaign_ref: str
    reporting_start: datetime
    reporting_end: datetime
    purchasers: int | None
    campaign_associated_sales_aed: Decimal | None
    campaign_cost_aed: Decimal | None
    suppressed: bool
    attribution_note: str
    simulation: bool = True
