"""Save validation scores with the configuration used to produce them."""

import json
from pathlib import Path

import pandas as pd

from .ranking import FEATURE_SETS, RULES, validate_scores


def _check_scores(scores):
    validate_scores(scores["returned"], scores["score"])
    if scores["user_id"].isna().any() or not scores["user_id"].is_unique:
        raise ValueError("Visitor IDs must be present and unique.")
    if scores["ranking"].isna().any() or scores["ranking"].nunique() != 1:
        raise ValueError("The export must contain one named ranking.")


def save_validation_scores(scores, path, spec):
    path = Path(path)
    _check_scores(scores)
    metadata = {
        "rows": len(scores),
        "ranking": scores["ranking"].iloc[0],
        "selection": spec,
        "features": FEATURE_SETS[spec["feature_set"]] if spec["model"] not in RULES else [],
    }
    text = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(path, index=False)
    path.with_suffix(".json").write_text(text, encoding="utf-8")


def load_validation_scores(path):
    path = Path(path)
    message = "Validation scores are missing or invalid; rerun notebook 02 through its export step."
    try:
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        scores = pd.read_parquet(path)
        _check_scores(scores)
        if len(scores) != metadata["rows"] or scores["ranking"].iloc[0] != metadata["ranking"]:
            raise ValueError(message)
    except (OSError, KeyError, ValueError) as error:
        raise ValueError(message) from error
    return scores, metadata
