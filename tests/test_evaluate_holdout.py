import numpy as np
import pandas as pd
import pytest

from src import evaluate_holdout as holdout_module
from src.train_model import ALL_FEATURES, FEATURE_SETS, TARGET


@pytest.mark.parametrize("feature_set", ["full", "without recency"])
def test_holdout_refits_only_the_chosen_model_with_available_outcomes(monkeypatch, feature_set):
    cohort = pd.DataFrame({
        "first_event_time": pd.to_datetime([
            "2015-06-01", "2015-06-02", "2015-07-01",
            "2015-07-25", "2015-07-26",
        ], utc=True),
        TARGET: [0, 1, 1, 0, 1],
        **{column: [1, 2, 3, 4, 5] for column in ALL_FEATURES},
    })
    calls = []

    class Model:
        def fit(self, features, labels):
            assert features.columns.tolist() == FEATURE_SETS[feature_set]
            # The July visitor's outcome is not ready by August 8.
            assert features["events_first_14_days"].tolist() == [1, 2]
            assert labels.tolist() == [0, 1]

        def predict_proba(self, features):
            assert features.columns.tolist() == FEATURE_SETS[feature_set]
            assert features["events_first_14_days"].tolist() == [4, 5]
            return np.array([[0.8, 0.2], [0.2, 0.8]])

    def make_model(name, params):
        calls.append(name)
        assert params == {"C": 0.1}
        return Model()

    monkeypatch.setattr(holdout_module, "_make_model", make_model)
    results = holdout_module.evaluate_holdout(
        cohort, "logistic regression", feature_set, {"C": 0.1},
    )
    assert calls == ["logistic regression"]
    assert results["final_train_users"] == 2
    assert results["test_users"] == 2
    assert set(results["metrics"].index) == {
        f"logistic regression / {feature_set}", "event count", "active days", "high intent", "recency",
    }
    assert results["params"] == {"C": 0.1}
