from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

from src.config import FEATURES_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, SPLITS_DIR, TRAIN_SPLIT
from src.pipeline.features import FEATURE_COLUMNS


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    baseline = np.var(y_true)
    if baseline == 0:
        return 0.0
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - (ss_res / ss_tot))


def train_models(input_path: Optional[Path] = None) -> Dict[str, float]:
    if input_path is None:
        input_path = FEATURES_DIR / "matches_features.csv"

    df = pd.read_csv(input_path, parse_dates=["match_date"])
    df = df.sort_values(["match_date", "match_number"]).reset_index(drop=True)

    split_index = int(len(df) * TRAIN_SPLIT)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]

    y_train_home_goals = train_df["home_goals"]
    y_test_home_goals = test_df["home_goals"]
    y_train_away_goals = train_df["away_goals"]
    y_test_away_goals = test_df["away_goals"]

    home_goals_model = RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    away_goals_model = RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    home_goals_model.fit(X_train, y_train_home_goals)
    away_goals_model.fit(X_train, y_train_away_goals)

    train_df["home_goals_pred"] = home_goals_model.predict(X_train)
    train_df["away_goals_pred"] = away_goals_model.predict(X_train)
    test_df["home_goals_pred"] = home_goals_model.predict(X_test)
    test_df["away_goals_pred"] = away_goals_model.predict(X_test)

    # Time-based validation favored this setup over stacked predicted-goal features.
    outcome_features = FEATURE_COLUMNS.copy()

    outcome_model = HistGradientBoostingClassifier(
        learning_rate=0.07,
        max_depth=6,
        max_iter=500,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )
    outcome_model.fit(train_df[outcome_features], train_df["outcome"])

    outcome_preds = outcome_model.predict(test_df[outcome_features])
    home_goal_preds = test_df["home_goals_pred"].to_numpy()
    away_goal_preds = test_df["away_goals_pred"].to_numpy()

    metrics = {
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "outcome_accuracy": float(accuracy_score(test_df["outcome"], outcome_preds)),
        "home_goals_mae": float(mean_absolute_error(y_test_home_goals, home_goal_preds)),
        "away_goals_mae": float(mean_absolute_error(y_test_away_goals, away_goal_preds)),
        "home_goals_r2": _safe_r2(y_test_home_goals.to_numpy(), home_goal_preds),
        "away_goals_r2": _safe_r2(y_test_away_goals.to_numpy(), away_goal_preds),
    }

    joblib.dump(home_goals_model, MODELS_DIR / "home_goals_model.pkl", compress=3)
    joblib.dump(away_goals_model, MODELS_DIR / "away_goals_model.pkl", compress=3)
    joblib.dump(outcome_model, MODELS_DIR / "outcome_model.pkl", compress=3)
    joblib.dump(outcome_features, MODELS_DIR / "outcome_features.pkl", compress=3)
    joblib.dump(FEATURE_COLUMNS, MODELS_DIR / "goal_features.pkl", compress=3)

    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    test_with_preds = test_df.copy()
    test_with_preds["outcome_pred"] = outcome_preds
    test_with_preds.to_csv(REPORTS_DIR / "test_predictions.csv", index=False)

    return metrics


if __name__ == "__main__":
    results = train_models()
    print(json.dumps(results, indent=2))
