"""Generate private synthetic payment and program fixtures.

The returned frames contain private synthetic customer tokens. They must not
be passed to Streamlit, exports, logs, or an LLM. Only `safe_run_summary`
produces an allowlisted aggregate report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


DEMO_MERCHANT_ID = "M0001"
DEMO_MERCHANT_NAME = "Visa Versa Cafe"
RETENTION_PURPOSE = "merchant_retention"
SIMULATED_CHANNEL = "issuer_app"
ELIGIBLE_DEMO_COUNT = 126
LAPSED_DEMO_CANDIDATES = 180

CATEGORIES = ("Cafe", "Restaurant", "Retail", "Grocery", "Services")
LOCATIONS = ("Dubai", "Abu Dhabi", "Sharjah", "Ajman")
BASE_AMOUNTS = {
    "Cafe": 61.0,
    "Restaurant": 92.0,
    "Retail": 125.0,
    "Grocery": 78.0,
    "Services": 145.0,
}


@dataclass(frozen=True)
class GenerationConfig:
    seed: int = 20261001
    as_of_date: date = date(2026, 10, 1)
    merchant_count: int = 300
    customer_count: int = 20_000

    def __post_init__(self) -> None:
        if self.merchant_count < 2:
            raise ValueError("At least two merchants are required")
        if self.customer_count < LAPSED_DEMO_CANDIDATES + 600:
            raise ValueError("Customer count is too small for the demo cohort")


@dataclass(repr=False)
class PrivateSyntheticDataset:
    """Internal frames; repr is disabled to avoid accidental token logging."""

    merchants: pd.DataFrame = field(repr=False)
    transactions: pd.DataFrame = field(repr=False)
    program_eligibility: pd.DataFrame = field(repr=False)
    config: GenerationConfig = field(repr=False)


def _merchant_records(config: GenerationConfig, rng: np.random.Generator) -> pd.DataFrame:
    records = [{
        "merchant_id": DEMO_MERCHANT_ID,
        "merchant_name": DEMO_MERCHANT_NAME,
        "merchant_category": "Cafe",
        "merchant_location": "Dubai",
        "merchant_size": "medium",
        "opening_date": date(2019, 4, 1),
    }]
    for number in range(2, config.merchant_count + 1):
        if number <= min(75, config.merchant_count):
            category = "Cafe"
        else:
            category = CATEGORIES[1 + ((number - 76) % 4)]
        location = "Dubai" if 2 <= number <= 31 else str(rng.choice(LOCATIONS, p=[0.48, 0.27, 0.17, 0.08]))
        size = "medium" if 2 <= number <= 31 else str(rng.choice(["small", "medium"], p=[0.65, 0.35]))
        opened = date(2016, 1, 1) + timedelta(days=int(rng.integers(0, 3300)))
        records.append({
            "merchant_id": f"M{number:04d}",
            "merchant_name": f"Synthetic {category} {number:03d}",
            "merchant_category": category,
            "merchant_location": location,
            "merchant_size": size,
            "opening_date": opened,
        })
    return pd.DataFrame.from_records(records)


def _normal_merchant_transactions(
    merchant: pd.Series,
    customer_tokens: np.ndarray,
    config: GenerationConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    category = str(merchant["merchant_category"])
    size = str(merchant["merchant_size"])
    if category == "Cafe":
        count = int(rng.integers(2350, 3051)) if size == "medium" else int(rng.integers(1800, 2501))
    else:
        count = int(rng.integers(650, 1001)) if size == "medium" else int(rng.integers(500, 851))
    start = config.as_of_date - timedelta(days=365)
    days = np.arange(365)
    weekdays = np.array([(start + timedelta(days=int(day))).weekday() for day in days])
    weekend_weight = 0.85 if category == "Cafe" else 1.12
    weights = np.where(weekdays >= 5, weekend_weight, 1.0)
    weights /= weights.sum()
    sampled_days = rng.choice(days, size=count, p=weights)
    if category == "Cafe":
        hours = rng.choice(
            np.arange(8, 21), size=count,
            p=[.07, .09, .08, .05, .17, .17, .11, .06, .07, .04, .04, .03, .02],
        )
    else:
        hours = rng.choice(
            np.arange(9, 22), size=count,
            p=[.06, .04, .08, .12, .11, .13, .08, .10, .08, .09, .05, .04, .02],
        )
    minutes = rng.integers(0, 60, size=count)
    timestamps = (
        np.datetime64(start, "ns")
        + sampled_days.astype("timedelta64[D]")
        + hours.astype("timedelta64[h]")
        + minutes.astype("timedelta64[m]")
    )
    # Assign every private token at least once before sampling repeat visits.
    customers = np.concatenate([customer_tokens, rng.choice(customer_tokens, size=count - len(customer_tokens))])
    rng.shuffle(customers)
    statuses = rng.choice(["completed", "refunded", "declined"], size=count, p=[.965, .015, .020])
    amounts = np.round(np.clip(rng.lognormal(np.log(BASE_AMOUNTS[category]) - .06, .35, count), 12, 450), 2)
    return pd.DataFrame({
        "transaction_datetime": pd.to_datetime(timestamps),
        "amount": amounts,
        "currency": "AED",
        "transaction_status": statuses,
        "refund_amount": np.where(statuses == "refunded", amounts, 0.0),
        "merchant_id": merchant["merchant_id"],
        "merchant_category": category,
        "synthetic_customer_id": customers,
        "domestic_or_international": rng.choice(["domestic", "international"], size=count, p=[.88, .12]),
        "merchant_location": merchant["merchant_location"],
    })


def _special_rows(
    timestamps: list[datetime],
    amounts: list[float],
    customers: list[str],
    statuses: list[str] | None = None,
) -> pd.DataFrame:
    count = len(customers)
    state = statuses or ["completed"] * count
    return pd.DataFrame({
        "transaction_datetime": pd.to_datetime(timestamps),
        "amount": amounts,
        "currency": "AED",
        "transaction_status": state,
        "refund_amount": [amount if status == "refunded" else 0.0 for amount, status in zip(amounts, state)],
        "merchant_id": DEMO_MERCHANT_ID,
        "merchant_category": "Cafe",
        "synthetic_customer_id": customers,
        "domestic_or_international": "domestic",
        "merchant_location": "Dubai",
    })


def _weekday(day: date) -> date:
    return day if day.weekday() < 5 else day + timedelta(days=7 - day.weekday())


def _special_transactions(config: GenerationConfig, tokens: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    as_of = config.as_of_date
    baseline_start = as_of - timedelta(days=56)
    current_start = as_of - timedelta(days=28)
    qualification_start = as_of - timedelta(days=135)
    candidate_tokens = tokens[:LAPSED_DEMO_CANDIDATES]
    active_tokens = tokens[LAPSED_DEMO_CANDIDATES:LAPSED_DEMO_CANDIDATES + 600]
    frames: list[pd.DataFrame] = []

    # Eight settled purchases worth AED 500 per candidate in the 90-day
    # qualification period. The first two visits are weekday lunches.
    candidate_times: list[datetime] = []
    candidate_amounts: list[float] = []
    candidate_customers: list[str] = []
    values = (55.0, 60.0, 65.0, 70.0, 55.0, 60.0, 65.0, 70.0)
    for index, token in enumerate(candidate_tokens):
        for visit in range(8):
            if visit < 7:
                day = qualification_start + timedelta(days=visit * 8 + index % 5)
            else:
                day = as_of - timedelta(days=int(rng.integers(46, 74)))
            if visit < 2:
                day = _weekday(day)
                hour = 12
            else:
                hour = int(rng.choice([9, 12, 13, 17, 19]))
            candidate_times.append(datetime.combine(day, datetime.min.time()).replace(hour=hour, minute=index % 60))
            candidate_amounts.append(values[visit])
            candidate_customers.append(str(token))
    candidates = _special_rows(candidate_times, candidate_amounts, candidate_customers)
    frames.append(candidates)

    # Existing regulars have historical purchases. Exactly 500 of them also
    # appear in the current window; the remaining 100 visited recently enough
    # to avoid the lapsed definition, but not in the current 28 days.
    history_start = as_of - timedelta(days=365)
    history_count = 8_000
    history_days = rng.integers(0, (baseline_start - history_start).days, size=history_count)
    history_tokens = np.concatenate([active_tokens, rng.choice(active_tokens, size=history_count - len(active_tokens))])
    rng.shuffle(history_tokens)
    history_hours = rng.choice([8, 10, 12, 13, 16, 19], size=history_count)
    history_times = [
        datetime.combine(history_start + timedelta(days=int(day)), datetime.min.time()).replace(hour=int(hour), minute=int(rng.integers(0, 60)))
        for day, hour in zip(history_days, history_hours)
    ]
    history_amounts = np.round(rng.uniform(45, 85, size=history_count), 2).tolist()
    frames.append(_special_rows(history_times, history_amounts, history_tokens.astype(str).tolist()))

    baseline_candidates = candidates[
        (candidates["transaction_datetime"] >= pd.Timestamp(baseline_start))
        & (candidates["transaction_datetime"] < pd.Timestamp(current_start))
    ]
    candidate_lunch = int((
        (baseline_candidates["transaction_datetime"].dt.weekday < 5)
        & baseline_candidates["transaction_datetime"].dt.hour.between(11, 14)
    ).sum())
    candidate_other = len(baseline_candidates) - candidate_lunch
    baseline_lunch = 350 - candidate_lunch
    baseline_other = 850 - candidate_other
    if baseline_lunch < 0 or baseline_other < 100:
        raise ValueError("Candidate visits exceed the controlled baseline")

    def fill_window(start: date, lunch_count: int, other_count: int, *, baseline: bool) -> pd.DataFrame:
        lunch_days = [start + timedelta(days=day) for day in range(28) if (start + timedelta(days=day)).weekday() < 5]
        all_days = [start + timedelta(days=day) for day in range(28)]
        lunch_dates = rng.choice(lunch_days, size=lunch_count)
        other_dates = rng.choice(all_days, size=other_count)
        dates = list(lunch_dates) + list(other_dates)
        hours = [int(rng.choice([11, 12, 13, 14])) for _ in range(lunch_count)]
        hours += [int(rng.choice([8, 9, 16, 18, 19, 20])) for _ in range(other_count)]
        times = [datetime.combine(day, datetime.min.time()).replace(hour=hour, minute=int(rng.integers(0, 60))) for day, hour in zip(dates, hours)]
        if baseline:
            guaranteed = active_tokens[500:]
            # These visits are all after the 45-day lapse cutoff, within
            # the baseline window. Keep them paired with guaranteed tokens.
            late_lunch_days = [day for day in lunch_days if day > as_of - timedelta(days=45)]
            for index in range(len(guaranteed)):
                times[index] = times[index].replace(
                    year=late_lunch_days[index % len(late_lunch_days)].year,
                    month=late_lunch_days[index % len(late_lunch_days)].month,
                    day=late_lunch_days[index % len(late_lunch_days)].day,
                )
            customers = np.concatenate([guaranteed, rng.choice(active_tokens, size=len(dates) - len(guaranteed))])
        else:
            current_active = active_tokens[:500]
            customers = np.concatenate([current_active, rng.choice(current_active, size=len(dates) - len(current_active))])
        amounts = [62.0] * lunch_count + [60.0] * other_count
        return _special_rows(times, amounts, customers.astype(str).tolist())

    frames.append(fill_window(baseline_start, baseline_lunch, baseline_other, baseline=True))
    frames.append(fill_window(current_start, 277, 818, baseline=False))
    return pd.concat(frames, ignore_index=True)


def _program_eligibility(config: GenerationConfig, tokens: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    count = len(tokens)
    as_of = config.as_of_date
    enrolled = rng.random(count) < .80
    permission = rng.choice(["granted", "denied", "unknown"], size=count, p=[.80, .12, .08])
    purpose = rng.choice([RETENTION_PURPOSE, "unavailable"], size=count, p=[.92, .08])
    channel = rng.choice([SIMULATED_CHANNEL, "unavailable"], size=count, p=[.90, .10])
    expiry = np.where(rng.random(count) < .10, as_of - timedelta(days=5), as_of + timedelta(days=90))
    revoked = np.full(count, pd.NaT, dtype=object)
    revoked[rng.random(count) < .05] = as_of - timedelta(days=10)
    frequency_cap_ok = rng.random(count) < .92
    suppressed = rng.random(count) < .06

    enrolled[:ELIGIBLE_DEMO_COUNT] = True
    permission[:ELIGIBLE_DEMO_COUNT] = "granted"
    purpose[:ELIGIBLE_DEMO_COUNT] = RETENTION_PURPOSE
    channel[:ELIGIBLE_DEMO_COUNT] = SIMULATED_CHANNEL
    expiry[:ELIGIBLE_DEMO_COUNT] = as_of + timedelta(days=90)
    revoked[:ELIGIBLE_DEMO_COUNT] = pd.NaT
    frequency_cap_ok[:ELIGIBLE_DEMO_COUNT] = True
    suppressed[:ELIGIBLE_DEMO_COUNT] = False

    # Ensure the remaining 54 lapsed candidates fail at least one distinct
    # simulated program check; this is fixture construction, not activation.
    for index in range(ELIGIBLE_DEMO_COUNT, LAPSED_DEMO_CANDIDATES):
        enrolled[index] = True
        permission[index] = "granted"
        purpose[index] = RETENTION_PURPOSE
        channel[index] = SIMULATED_CHANNEL
        expiry[index] = as_of + timedelta(days=90)
        revoked[index] = pd.NaT
        frequency_cap_ok[index] = True
        suppressed[index] = False
        reason = (index - ELIGIBLE_DEMO_COUNT) % 7
        if reason == 0:
            enrolled[index] = False
        elif reason == 1:
            permission[index] = "denied"
        elif reason == 2:
            permission[index] = "unknown"
        elif reason == 3:
            expiry[index] = as_of - timedelta(days=1)
        elif reason == 4:
            revoked[index] = as_of - timedelta(days=1)
        elif reason == 5:
            frequency_cap_ok[index] = False
        else:
            suppressed[index] = True

    return pd.DataFrame({
        "synthetic_customer_id": tokens,
        "enrolled": enrolled,
        "permission_state": permission,
        "purpose": purpose,
        "channel": channel,
        "valid_until": pd.to_datetime(expiry),
        "revoked_at": pd.to_datetime(revoked),
        "frequency_cap_ok": frequency_cap_ok,
        "suppressed": suppressed,
    })


def generate_dataset(config: GenerationConfig = GenerationConfig()) -> PrivateSyntheticDataset:
    """Build one deterministic internal dataset without reading wall-clock time."""

    rng = np.random.default_rng(config.seed)
    merchants = _merchant_records(config, rng)
    tokens = np.array([f"syn_c_{number:06d}" for number in range(1, config.customer_count + 1)])
    special_tokens = tokens[:LAPSED_DEMO_CANDIDATES + 600]
    other_pools = np.array_split(tokens[len(special_tokens):], config.merchant_count - 1)
    frames = [_special_transactions(config, special_tokens, rng)]
    for (_, merchant), pool in zip(merchants.iloc[1:].iterrows(), other_pools):
        if len(pool) == 0:
            raise ValueError("Every merchant requires at least one customer token")
        frames.append(_normal_merchant_transactions(merchant, pool, config, rng))
    transactions = pd.concat(frames, ignore_index=True)
    transactions.insert(0, "transaction_id", [f"T{number:09d}" for number in range(1, len(transactions) + 1)])
    transactions = transactions.sort_values(["transaction_datetime", "transaction_id"], kind="stable").reset_index(drop=True)
    program = _program_eligibility(config, tokens, rng)
    # Give the active-regular demo cohort enough valid simulated permissions
    # for a top-quintile model audience, while retaining a safe ineligible complement.
    boundary = pd.Timestamp(config.as_of_date)
    recent = transactions.loc[
        transactions["merchant_id"].eq(DEMO_MERCHANT_ID)
        & transactions["transaction_datetime"].ge(boundary - timedelta(days=90))
        & transactions["transaction_datetime"].lt(boundary)
        & transactions["transaction_status"].eq("completed")
    ]
    regular = recent.groupby("synthetic_customer_id")["transaction_datetime"].agg(["size", "max"])
    active_tokens = sorted(regular.index[
        regular["size"].ge(3) & regular["max"].ge(boundary - timedelta(days=45))
    ])
    active_valid = set(token for index, token in enumerate(active_tokens) if index % 4 != 0)
    active_invalid = set(active_tokens) - active_valid
    valid_rows = program["synthetic_customer_id"].isin(active_valid)
    program.loc[valid_rows, ["enrolled", "frequency_cap_ok"]] = True
    program.loc[valid_rows, ["permission_state", "purpose", "channel"]] = [
        "granted", RETENTION_PURPOSE, SIMULATED_CHANNEL,
    ]
    program.loc[valid_rows, "valid_until"] = boundary + timedelta(days=90)
    program.loc[valid_rows, "revoked_at"] = pd.NaT
    program.loc[valid_rows, "suppressed"] = False
    program.loc[program["synthetic_customer_id"].isin(active_invalid), "permission_state"] = "denied"
    return PrivateSyntheticDataset(merchants, transactions, program, config)


def safe_run_summary(dataset: PrivateSyntheticDataset) -> dict[str, str | int]:
    """Allowlisted aggregate run metadata, with no row-level records or IDs."""

    return {
        "data_label": "Synthetic data only",
        "as_of_date": dataset.config.as_of_date.isoformat(),
        "seed": dataset.config.seed,
        "merchant_count": len(dataset.merchants),
        "private_synthetic_customer_count": len(dataset.program_eligibility),
        "transaction_count": len(dataset.transactions),
    }


def write_dataset(dataset: PrivateSyntheticDataset, destination: Path) -> dict[str, str | int]:
    """Store private raw fixtures in a Git-ignored directory."""

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        extension = "csv"
    else:
        extension = "parquet"
    for name, frame in (
        ("merchants", dataset.merchants),
        ("transactions", dataset.transactions),
        ("program_eligibility", dataset.program_eligibility),
    ):
        target = destination / f"{name}.{extension}"
        if extension == "parquet":
            frame.to_parquet(target, index=False)
        else:
            frame.to_csv(target, index=False)
    summary = safe_run_summary(dataset) | {"format": extension}
    (destination / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
