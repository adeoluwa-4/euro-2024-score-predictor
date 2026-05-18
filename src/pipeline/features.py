from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import FEATURES_DIR, PROCESSED_DIR


FEATURE_COLUMNS = [
    "elo_diff",
    "fifa_rank_diff",
    "fifa_points_diff",
    "tournament_importance_score",
    "h2h_matches_prior",
    "h2h_goal_diff_prior",
    "home_points_per_match_last_5",
    "away_points_per_match_last_5",
    "home_points_per_match_last_10",
    "away_points_per_match_last_10",
    "home_goals_for_last_5",
    "away_goals_for_last_5",
    "home_goals_against_last_5",
    "away_goals_against_last_5",
]


def build_features(input_path: Optional[Path] = None, output_path: Optional[Path] = None) -> Path:
    if input_path is None:
        input_path = PROCESSED_DIR / "matches_preprocessed.csv"
    if output_path is None:
        output_path = FEATURES_DIR / "matches_features.csv"

    df = pd.read_csv(input_path, parse_dates=["match_date"])
    df = df.sort_values(["match_date", "match_number"]).reset_index(drop=True)

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())

    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    saved = build_features()
    print(f"Feature set saved to {saved}")
