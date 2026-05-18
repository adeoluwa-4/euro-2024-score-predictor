from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import PROCESSED_DIR


NUMERIC_COLUMNS = [
    "home_goals",
    "away_goals",
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


def add_outcome_label(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["outcome"] = 0
    result.loc[result["home_goals"] > result["away_goals"], "outcome"] = 1
    result.loc[result["home_goals"] < result["away_goals"], "outcome"] = 2
    return result


def preprocess_data(input_path: Optional[Path] = None, output_path: Optional[Path] = None) -> Path:
    if input_path is None:
        input_path = PROCESSED_DIR / "matches_cleaned.csv"
    if output_path is None:
        output_path = PROCESSED_DIR / "matches_preprocessed.csv"

    df = pd.read_csv(input_path, parse_dates=["match_date"])
    df = df.sort_values(["match_date", "match_number"]).reset_index(drop=True)

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df = add_outcome_label(df)
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    saved = preprocess_data()
    print(f"Preprocessed data saved to {saved}")
