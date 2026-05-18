from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import DATA_CUTOFF, PROCESSED_DIR, RAW_DATA_PATH, TARGET_MATCH_COUNT


REQUIRED_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
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


def clean_data(input_path: Path = RAW_DATA_PATH, output_path: Optional[Path] = None) -> Path:
    if output_path is None:
        output_path = PROCESSED_DIR / "matches_cleaned.csv"

    df = pd.read_csv(input_path, parse_dates=["date"])

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in world-cup dataset: {missing}")

    df = df[df["date"] <= pd.Timestamp(DATA_CUTOFF)].copy()
    df = df.sort_values("date").reset_index(drop=True)
    if TARGET_MATCH_COUNT is not None:
        df = df.tail(TARGET_MATCH_COUNT).reset_index(drop=True)

    cleaned = pd.DataFrame(
        {
            "match_date": df["date"],
            "home_team": df["home_team"].astype(str).str.strip(),
            "away_team": df["away_team"].astype(str).str.strip(),
            "home_goals": pd.to_numeric(df["home_score"], errors="coerce"),
            "away_goals": pd.to_numeric(df["away_score"], errors="coerce"),
            "elo_diff": pd.to_numeric(df["elo_diff"], errors="coerce"),
            "fifa_rank_diff": pd.to_numeric(df["fifa_rank_diff"], errors="coerce"),
            "fifa_points_diff": pd.to_numeric(df["fifa_points_diff"], errors="coerce"),
            "tournament_importance_score": pd.to_numeric(df["tournament_importance_score"], errors="coerce"),
            "h2h_matches_prior": pd.to_numeric(df["h2h_matches_prior"], errors="coerce"),
            "h2h_goal_diff_prior": pd.to_numeric(df["h2h_goal_diff_prior"], errors="coerce"),
            "home_points_per_match_last_5": pd.to_numeric(df["home_points_per_match_last_5"], errors="coerce"),
            "away_points_per_match_last_5": pd.to_numeric(df["away_points_per_match_last_5"], errors="coerce"),
            "home_points_per_match_last_10": pd.to_numeric(df["home_points_per_match_last_10"], errors="coerce"),
            "away_points_per_match_last_10": pd.to_numeric(df["away_points_per_match_last_10"], errors="coerce"),
            "home_goals_for_last_5": pd.to_numeric(df["home_goals_for_last_5"], errors="coerce"),
            "away_goals_for_last_5": pd.to_numeric(df["away_goals_for_last_5"], errors="coerce"),
            "home_goals_against_last_5": pd.to_numeric(df["home_goals_against_last_5"], errors="coerce"),
            "away_goals_against_last_5": pd.to_numeric(df["away_goals_against_last_5"], errors="coerce"),
        }
    )

    cleaned["data_cutoff"] = DATA_CUTOFF
    cleaned["match_number"] = range(1, len(cleaned) + 1)

    cleaned.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    saved = clean_data()
    print(f"Cleaned data saved to {saved}")
