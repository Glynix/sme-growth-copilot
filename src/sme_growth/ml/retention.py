"""Private, explainable retention model with temporal evaluation.

Individual probabilities and customer tokens never leave this module's
private computation boundary. Public outputs are cohort aggregates only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from math import ceil

import numpy as np
import pandas as pd

from sme_growth.analytics.common import settled_transactions


FEATURES = (
    "recency_days", "frequency_90d", "monetary_90d_aed",
    "visit_trend_30d", "weekday_lunch_affinity",
)
MIN_REGULAR_VISITS = 3
LABEL_HORIZON_DAYS = 30


@dataclass(frozen=True, slots=True)
class RetentionEvaluation:
    as_of_date: str
    train_cutoffs: tuple[str, ...]
    validation_cutoff: str
    test_cutoff: str
    future_label: str
    train_examples: int
    validation_examples: int
    test_examples: int
    test_nonreturn_rate_pct: float
    model_average_precision: float
    recency_average_precision: float
    model_brier_score: float
    recency_brier_score: float
    model_calibration_error: float
    recency_calibration_error: float
    selected_signal: str
    limitation: str = (
        "Synthetic temporal evaluation measures pipeline behavior, not real-world accuracy, "
        "campaign response, or incremental impact."
    )


@dataclass(frozen=True, slots=True)
class RetentionCohortSummary:
    as_of_date: str
    regular_customers: int | None
    elevated_risk_share_pct: float | None
    top_model_associations: tuple[str, ...]
    model_signal: str
    status: str
    evidence_type: str = "estimated_model_cohort_aggregate"
    note: str = (
        "Cohort risk describes predicted non-return, not campaign response or proven causes. "
        "Individual probabilities stay private."
    )


@dataclass(repr=False)
class PrivateRetentionModel:
    model: object = field(repr=False)
    recency_baseline: object = field(repr=False)
    training_feature_means: np.ndarray = field(repr=False)
    elevated_risk_threshold: float = field(repr=False)
    evaluation: RetentionEvaluation


@dataclass(repr=False)
class _PrivateLogisticRegression:
    """Small deterministic L2 logistic regression using the existing NumPy stack."""

    mean: np.ndarray = field(repr=False)
    scale: np.ndarray = field(repr=False)
    weights: np.ndarray = field(repr=False)
    intercept: float

    @classmethod
    def fit(cls, features: pd.DataFrame, labels: np.ndarray) -> _PrivateLogisticRegression:
        raw = features.to_numpy(dtype=float)
        mean = raw.mean(axis=0)
        scale = raw.std(axis=0)
        scale[scale < 1e-8] = 1.0
        standardized = (raw - mean) / scale
        weights = np.zeros(standardized.shape[1], dtype=float)
        prevalence = float(np.clip(labels.mean(), 1e-5, 1 - 1e-5))
        intercept = float(np.log(prevalence / (1 - prevalence)))
        for _ in range(350):
            logits = np.clip(standardized @ weights + intercept, -35, 35)
            probabilities = 1 / (1 + np.exp(-logits))
            residual = probabilities - labels
            gradient = standardized.T @ residual / len(labels) + .0001 * weights
            intercept_gradient = float(residual.mean())
            next_weights = weights - .45 * gradient
            next_intercept = intercept - .45 * intercept_gradient
            if max(float(np.max(np.abs(next_weights - weights))), abs(next_intercept - intercept)) < 1e-7:
                weights, intercept = next_weights, next_intercept
                break
            weights, intercept = next_weights, next_intercept
        return cls(mean, scale, weights, intercept)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        standardized = (features.to_numpy(dtype=float) - self.mean) / self.scale
        logits = np.clip(standardized @ self.weights + self.intercept, -35, 35)
        positive = 1 / (1 + np.exp(-logits))
        return np.column_stack((1 - positive, positive))


def _average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    order = np.argsort(-probabilities, kind="stable")
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels)
    precision = true_positives / (np.arange(len(labels)) + 1)
    return float(precision[sorted_labels == 1].sum() / sorted_labels.sum())


def _brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((probabilities - labels) ** 2))


def _snapshot_features(settled: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    """Only the 90 days strictly before cutoff may enter the features."""

    boundary = pd.Timestamp(cutoff)
    start = boundary - timedelta(days=90)
    past = settled[
        settled["transaction_datetime"].ge(start)
        & settled["transaction_datetime"].lt(boundary)
    ].copy()
    if past.empty:
        return pd.DataFrame(columns=FEATURES)
    timestamps = past["transaction_datetime"]
    past["recent30"] = timestamps.ge(boundary - timedelta(days=30)).astype(int)
    past["previous30"] = (
        timestamps.ge(boundary - timedelta(days=60))
        & timestamps.lt(boundary - timedelta(days=30))
    ).astype(int)
    past["lunch"] = ((timestamps.dt.weekday < 5) & timestamps.dt.hour.ge(11) & timestamps.dt.hour.lt(15)).astype(int)
    grouped = past.groupby(["merchant_id", "synthetic_customer_id"]).agg(
        last_visit=("transaction_datetime", "max"),
        frequency_90d=("transaction_datetime", "size"),
        monetary_90d_aed=("net_sales_aed", "sum"),
        recent30=("recent30", "sum"),
        previous30=("previous30", "sum"),
        lunch=("lunch", "sum"),
    )
    regular = grouped[grouped["frequency_90d"].ge(MIN_REGULAR_VISITS)].copy()
    if regular.empty:
        return pd.DataFrame(columns=FEATURES)
    regular["recency_days"] = (boundary - regular["last_visit"]).dt.total_seconds() / 86_400
    regular["visit_trend_30d"] = regular["recent30"] - regular["previous30"]
    regular["weekday_lunch_affinity"] = regular["lunch"] / regular["frequency_90d"]
    return regular.loc[:, FEATURES].astype(float)


def _labeled_snapshot(settled: pd.DataFrame, cutoff: date, as_of_date: date) -> tuple[pd.DataFrame, np.ndarray]:
    if cutoff + timedelta(days=LABEL_HORIZON_DAYS) > as_of_date:
        raise ValueError("Future non-return label has not matured by the as-of date")
    features = _snapshot_features(settled, cutoff)
    boundary = pd.Timestamp(cutoff)
    future = settled[
        settled["transaction_datetime"].ge(boundary)
        & settled["transaction_datetime"].lt(boundary + timedelta(days=LABEL_HORIZON_DAYS))
    ]
    future_pairs = pd.MultiIndex.from_frame(future[["merchant_id", "synthetic_customer_id"]].drop_duplicates())
    labels = (~features.index.isin(future_pairs)).astype(int)
    return features, np.asarray(labels)


def _calibration_error(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if not len(labels):
        return float("nan")
    bins = np.minimum((probabilities * 10).astype(int), 9)
    error = 0.0
    for number in range(10):
        selected = bins == number
        if selected.any():
            error += float(selected.mean()) * abs(float(labels[selected].mean() - probabilities[selected].mean()))
    return error


def train_retention_model(transactions: pd.DataFrame, as_of_date: date) -> PrivateRetentionModel:
    """Fit on earlier snapshots; validate and test on later matured labels."""

    settled = settled_transactions(transactions)
    train_cutoffs = tuple(as_of_date - timedelta(days=days) for days in (273, 243, 213, 183))
    validation_cutoff = as_of_date - timedelta(days=122)
    test_cutoff = as_of_date - timedelta(days=61)
    train_parts = [_labeled_snapshot(settled, cutoff, as_of_date) for cutoff in train_cutoffs]
    train_x = pd.concat([part[0] for part in train_parts], ignore_index=True)
    train_y = np.concatenate([part[1] for part in train_parts])
    validation_x, validation_y = _labeled_snapshot(settled, validation_cutoff, as_of_date)
    test_x, test_y = _labeled_snapshot(settled, test_cutoff, as_of_date)
    if min(len(train_x), len(validation_x), len(test_x)) < 100:
        raise ValueError("Insufficient previously regular customers for temporal evaluation")
    if any(len(np.unique(labels)) < 2 for labels in (train_y, validation_y, test_y)):
        raise ValueError("Both future-return classes are required in each temporal split")
    model = _PrivateLogisticRegression.fit(train_x, train_y)
    recency = _PrivateLogisticRegression.fit(train_x.loc[:, ["recency_days"]], train_y)

    # The validation period is used for model choice; the final test period
    # remains untouched until the choice has been made.
    val_model = model.predict_proba(validation_x)[:, 1]
    val_recency = recency.predict_proba(validation_x.loc[:, ["recency_days"]])[:, 1]
    use_model = (
        _average_precision(validation_y, val_model) >= _average_precision(validation_y, val_recency)
        and _brier_score(validation_y, val_model) <= _brier_score(validation_y, val_recency) + .01
    )
    selected = "behavioral_logistic_regression" if use_model else "recency_baseline"
    test_model = model.predict_proba(test_x)[:, 1]
    test_recency = recency.predict_proba(test_x.loc[:, ["recency_days"]])[:, 1]
    train_selected = (
        model.predict_proba(train_x)[:, 1] if use_model
        else recency.predict_proba(train_x.loc[:, ["recency_days"]])[:, 1]
    )
    evaluation = RetentionEvaluation(
        as_of_date=as_of_date.isoformat(),
        train_cutoffs=tuple(cutoff.isoformat() for cutoff in train_cutoffs),
        validation_cutoff=validation_cutoff.isoformat(),
        test_cutoff=test_cutoff.isoformat(),
        future_label="No settled purchase at the same merchant in the 30 days after snapshot cutoff",
        train_examples=len(train_x), validation_examples=len(validation_x), test_examples=len(test_x),
        test_nonreturn_rate_pct=round(float(test_y.mean()) * 100, 2),
        model_average_precision=round(_average_precision(test_y, test_model), 4),
        recency_average_precision=round(_average_precision(test_y, test_recency), 4),
        model_brier_score=round(_brier_score(test_y, test_model), 4),
        recency_brier_score=round(_brier_score(test_y, test_recency), 4),
        model_calibration_error=round(_calibration_error(test_y, test_model), 4),
        recency_calibration_error=round(_calibration_error(test_y, test_recency), 4),
        selected_signal=selected,
    )
    return PrivateRetentionModel(
        model=model, recency_baseline=recency,
        training_feature_means=train_x.loc[:, FEATURES].mean().to_numpy(),
        elevated_risk_threshold=float(np.quantile(train_selected, .75)),
        evaluation=evaluation,
    )


def _association_labels(bundle: PrivateRetentionModel, features: pd.DataFrame) -> tuple[str, ...]:
    if bundle.evaluation.selected_signal == "recency_baseline":
        return ("Time since last purchase is the selected baseline signal.",)
    coefficients = bundle.model.weights
    mean_features = features.loc[:, FEATURES].mean().to_numpy()
    standardized_difference = (mean_features - bundle.training_feature_means) / bundle.model.scale
    contributions = standardized_difference * coefficients
    descriptions = {
        "recency_days": ("Longer time since last purchase", "Shorter time since last purchase"),
        "frequency_90d": ("More purchases in 90 days", "Fewer purchases in 90 days"),
        "monetary_90d_aed": ("Higher historical spend", "Lower historical spend"),
        "visit_trend_30d": ("Rising visit frequency", "Falling visit frequency"),
        "weekday_lunch_affinity": ("Higher weekday-lunch share", "Lower weekday-lunch share"),
    }
    ordered = np.argsort(-contributions)
    labels = []
    for index in ordered:
        if contributions[index] <= 0:
            continue
        feature = FEATURES[index]
        labels.append(descriptions[feature][0 if standardized_difference[index] >= 0 else 1])
        if len(labels) == 2:
            break
    return tuple(labels) or ("No elevated cohort factor relative to the training reference.",)


def get_retention_cohort_summary(
    bundle: PrivateRetentionModel, transactions: pd.DataFrame,
    merchant_id: str, as_of_date: date,
) -> RetentionCohortSummary:
    """Release only an adequately sized merchant cohort risk summary."""

    if bundle.evaluation.as_of_date != as_of_date.isoformat():
        raise ValueError("Model evaluation and cohort as-of dates differ")
    features = _snapshot_features(settled_transactions(transactions), as_of_date)
    if features.empty or not isinstance(features.index, pd.MultiIndex):
        return RetentionCohortSummary(as_of_date.isoformat(), None, None, (), bundle.evaluation.selected_signal, "suppressed")
    cohort = features.loc[features.index.get_level_values("merchant_id") == merchant_id]
    if len(cohort) < 50:
        return RetentionCohortSummary(as_of_date.isoformat(), None, None, (), bundle.evaluation.selected_signal, "suppressed")
    if bundle.evaluation.selected_signal == "behavioral_logistic_regression":
        private_probabilities = bundle.model.predict_proba(cohort)[:, 1]
    else:
        private_probabilities = bundle.recency_baseline.predict_proba(cohort.loc[:, ["recency_days"]])[:, 1]
    elevated = int((private_probabilities >= bundle.elevated_risk_threshold).sum())
    share = round(elevated / len(cohort) * 100, 2) if elevated >= 10 and len(cohort) - elevated >= 10 else None
    return RetentionCohortSummary(
        as_of_date.isoformat(), len(cohort), share, _association_labels(bundle, cohort),
        bundle.evaluation.selected_signal, "available" if share is not None else "partly_suppressed",
    )


def select_private_top_risk_regulars(
    bundle: PrivateRetentionModel, transactions: pd.DataFrame,
    merchant_id: str, as_of_date: date,
) -> frozenset[str]:
    """Top quintile by predicted non-return among currently active regulars."""

    if bundle.evaluation.as_of_date != as_of_date.isoformat():
        raise ValueError("Model evaluation and audience as-of dates differ")
    if bundle.evaluation.selected_signal != "behavioral_logistic_regression":
        return frozenset()
    features = _snapshot_features(settled_transactions(transactions), as_of_date)
    if features.empty or not isinstance(features.index, pd.MultiIndex):
        return frozenset()
    cohort = features.loc[features.index.get_level_values("merchant_id") == merchant_id]
    cohort = cohort.loc[cohort["recency_days"].lt(45)]
    if len(cohort) < 50:
        return frozenset()
    probabilities = bundle.model.predict_proba(cohort)[:, 1]
    ranked = sorted(
        zip(cohort.index.get_level_values("synthetic_customer_id"), probabilities),
        key=lambda item: (-item[1], item[0]),
    )
    return frozenset(token for token, _ in ranked[:ceil(len(ranked) * .20)])
