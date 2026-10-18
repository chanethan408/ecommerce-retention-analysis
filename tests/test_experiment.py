import numpy as np
import pandas as pd
import pytest

from src.experiment import (
    GROUPS, compare_groups, interaction_sample_size, score_groups, simulate_experiments,
)


def test_group_boundaries_keep_all_visitors_and_break_ties_without_outcomes():
    scores = pd.DataFrame({"user_id": range(29), "score": [1] * 29, "returned": [0] * 29})
    grouped = score_groups(scores)
    assert grouped.groupby("score range").size().reindex(GROUPS).tolist() == [2, 3, 24]
    changed = scores.sample(frac=1, random_state=9).assign(returned=1)
    pd.testing.assert_series_equal(grouped["user_id"], score_groups(changed)["user_id"])
    assert set(grouped["user_id"]) == set(scores["user_id"])


def test_score_groups_sort_higher_scores_first():
    scores = pd.DataFrame({"user_id": range(20), "score": range(20), "returned": [0] * 20})
    grouped = score_groups(scores)
    assert grouped.loc[grouped["score range"] == "Top 10%", "user_id"].tolist() == [19, 18]


def test_simulation_preserves_odd_counts_and_recovers_deterministic_effects():
    result = simulate_experiments([0, 0, 1], [5, 7, 9], [1, 0, -1], repeats=2)
    assert result["contacted visitors"].tolist() == [2, 3, 4] * 2
    assert result["control visitors"].tolist() == [3, 4, 5] * 2
    assert result["estimated effect"].tolist() == [1, 0, -1] * 2
    comparison = compare_groups(result)
    assert comparison["estimated next-minus-top effect"].tolist() == [-1, -1]
    assert comparison["lower 95% bound"].tolist() == [-1, -1]


def test_interaction_interval_adds_independent_group_variances():
    result = pd.DataFrame({
        "repeat": [0, 0], "score range": GROUPS[:2],
        "estimated effect": [0.01, 0.04], "standard error": [0.003, 0.004],
        "assumed true effect": [0.0, 0.02],
    })
    comparison = compare_groups(result).iloc[0]
    assert comparison["estimated next-minus-top effect"] == pytest.approx(0.03)
    assert comparison["lower 95% bound"] == pytest.approx(0.03 - 1.96 * 0.005)


def test_smaller_interaction_requires_more_visitors_and_reaches_planned_power():
    plan = interaction_sample_size(0.10, 0.05, difference=0.01)
    assert plan["total top and next"] == 4 * plan["per arm"]
    assert interaction_sample_size(0.10, 0.05, difference=0.005)["per arm"] > plan["per arm"]
    n = plan["per score group"]
    results = simulate_experiments([0.10, 0.05, 0.02], [n, n, 20], [0, 0.01, 0], repeats=3000)
    power = (compare_groups(results)["lower 95% bound"] > 0).mean()
    assert 0.76 < power < 0.84


@pytest.mark.parametrize("rates,counts,effects", [
    ([0.1] * 3, [1, 4, 4], [0] * 3),
    ([0.1] * 3, [2.5, 4, 4], [0] * 3),
    ([0.1] * 3, [4] * 3, [1] * 3),
    ([np.nan] * 3, [4] * 3, [0] * 3),
])
def test_simulation_rejects_invalid_group_inputs(rates, counts, effects):
    with pytest.raises(ValueError):
        simulate_experiments(rates, counts, effects)
