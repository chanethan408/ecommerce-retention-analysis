import json

import pandas as pd
import pytest

from src.provenance import load_validation_scores, save_validation_scores
from src.ranking import FEATURE_SETS


@pytest.mark.parametrize("model,params", [
    ("recency", {}), ("XGBoost", {"max_depth": 4}), ("logistic regression", {"C": 0.1}),
])
def test_export_preserves_scores_and_selected_configuration(tmp_path, model, params):
    scores = pd.DataFrame({
        "user_id": [1, 2], "returned": [0, 1], "score": [0.1, 0.9], "ranking": [model] * 2,
    })
    path = tmp_path / "processed/validation_scores.parquet"
    spec = {"model": model, "feature_set": "full", "params": params}
    save_validation_scores(scores, path, spec)
    loaded, metadata = load_validation_scores(path)
    pd.testing.assert_frame_equal(loaded, scores)
    assert metadata["selection"] == spec
    assert metadata["features"] == ([] if model == "recency" else FEATURE_SETS["full"])


@pytest.mark.parametrize("problem", ["missing", "rows", "ranking", "scores", "ids"])
def test_loading_rejects_missing_or_invalid_exports(tmp_path, problem):
    scores = pd.DataFrame({
        "user_id": [1, 2], "returned": [0, 1], "score": [0.1, 0.9], "ranking": ["recency"] * 2,
    })
    path = tmp_path / "scores.parquet"
    save_validation_scores(scores, path, {"model": "recency", "feature_set": "full", "params": {}})
    if problem == "missing":
        path.with_suffix(".json").unlink()
    elif problem in ("rows", "ranking"):
        metadata = json.loads(path.with_suffix(".json").read_text())
        metadata[problem] = 3 if problem == "rows" else "another ranking"
        path.with_suffix(".json").write_text(json.dumps(metadata))
    else:
        scores["score" if problem == "scores" else "user_id"] = float("nan")
        scores.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="rerun notebook 02"):
        load_validation_scores(path)
