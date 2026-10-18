"""Score groups and a stratified, equal-allocation outreach experiment."""

from statistics import NormalDist

import numpy as np
import pandas as pd

from .ranking import validate_scores


GROUPS = ["Top 10%", "Next 10%", "Remaining 80%"]


def score_groups(scores, seed=42):
    validate_scores(scores["returned"], scores["score"])
    if not scores["user_id"].is_unique or scores["user_id"].isna().any():
        raise ValueError("Visitor IDs must be present and unique.")
    first, second = int(len(scores) * 0.10), int(len(scores) * 0.20)
    if first < 2:
        raise ValueError("At least 20 visitors are needed for the experiment groups.")
    # Canonical ID order makes seeded tie-breaking independent of parquet row order.
    grouped = scores.sort_values("user_id").sample(frac=1, random_state=seed).sort_values(
        "score", ascending=False, kind="stable",
    ).reset_index(drop=True)
    grouped["score range"] = np.select(
        [grouped.index < first, grouped.index < second], GROUPS[:2], default=GROUPS[2],
    )
    return grouped


def simulate_experiments(reference_rates, visitors, effects, repeats=1, seed=42):
    """Draw binomial arm totals; fixed allocation keeps odd visitors in control."""
    rates, counts, effects = map(np.asarray, (reference_rates, visitors, effects))
    if any(array.shape != (3,) for array in (rates, counts, effects)):
        raise ValueError("Supply one rate, visitor count, and effect per score group.")
    if not np.isfinite(counts).all() or (counts < 2).any() or (counts != counts.astype(int)).any():
        raise ValueError("Each group needs at least two visitors, with integer counts.")
    probabilities = np.r_[rates, rates + effects]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Return probabilities must be finite and in [0, 1].")
    if repeats < 1:
        raise ValueError("At least one experiment is required.")
    contacted = counts.astype(int) // 2
    control = counts.astype(int) - contacted
    rng = np.random.default_rng(seed)
    control_rates = rng.binomial(control, rates, size=(repeats, 3)) / control
    treatment_rates = rng.binomial(contacted, rates + effects, size=(repeats, 3)) / contacted
    estimates = treatment_rates - control_rates
    errors = np.sqrt(
        treatment_rates * (1 - treatment_rates) / contacted
        + control_rates * (1 - control_rates) / control
    )
    return pd.DataFrame({
        "repeat": np.repeat(np.arange(repeats), 3),
        "score range": np.tile(GROUPS, repeats),
        "contacted visitors": np.tile(contacted, repeats),
        "control visitors": np.tile(control, repeats),
        "contacted return rate": treatment_rates.ravel(),
        "control return rate": control_rates.ravel(),
        "estimated effect": estimates.ravel(),
        "standard error": errors.ravel(),
        "lower 95% bound": (estimates - 1.96 * errors).ravel(),
        "upper 95% bound": (estimates + 1.96 * errors).ravel(),
        "assumed true effect": np.tile(effects, repeats),
    })


def compare_groups(results):
    """Next-minus-top interaction; groups contain independent visitors."""
    indexed = results.set_index(["repeat", "score range"])
    top = indexed.xs(GROUPS[0], level="score range")
    next_group = indexed.xs(GROUPS[1], level="score range")
    difference = next_group["estimated effect"] - top["estimated effect"]
    error = np.sqrt(next_group["standard error"] ** 2 + top["standard error"] ** 2)
    return pd.DataFrame({
        "estimated next-minus-top effect": difference,
        "lower 95% bound": difference - 1.96 * error,
        "upper 95% bound": difference + 1.96 * error,
        "assumed true difference": next_group["assumed true effect"] - top["assumed true effect"],
    })


def interaction_sample_size(top_rate, next_rate, difference=0.01, power=0.80, alpha=0.05):
    """Normal approximation: four equal arms, top effect zero, next effect difference."""
    probabilities = np.array([top_rate, top_rate, next_rate, next_rate + difference])
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Planning probabilities must be in [0, 1].")
    if not 0 < difference < 1 or not 0 < power < 1 or not 0 < alpha < 1:
        raise ValueError("Difference, power, and alpha must be in (0, 1).")
    normal = NormalDist()
    z = normal.inv_cdf(1 - alpha / 2) + normal.inv_cdf(power)
    per_arm = int(np.ceil(z ** 2 * np.sum(probabilities * (1 - probabilities)) / difference ** 2))
    return {"per arm": per_arm, "per score group": 2 * per_arm, "total top and next": 4 * per_arm}
