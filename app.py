from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from PIL import Image

from src.pipeline.predictor import MatchPredictor

st.set_page_config(page_title="Euro 2024 Match Predictor", layout="wide")

ASSET_IMAGE = Path("assets/euro-2024.jpeg")
TEAM_PHOTOS_DIR = Path("assets/team_photos")
FEATURE_DATA = Path("data/features/matches_features.csv")

EURO_2024_TEAMS = [
    "Albania", "Austria", "Belgium", "Croatia", "Czech Republic", "Denmark",
    "England", "France", "Georgia", "Germany", "Hungary", "Italy",
    "Netherlands", "Poland", "Portugal", "Romania", "Scotland", "Serbia",
    "Slovakia", "Slovenia", "Spain", "Switzerland", "Turkey", "Ukraine",
]

st.title("Euro 2024 Match Predictor")
st.caption("Pick two teams and get a projected score + winner.")

if ASSET_IMAGE.exists():
    st.image(Image.open(ASSET_IMAGE), use_column_width=True)

if not FEATURE_DATA.exists():
    st.warning("Data file missing. Run `python3 run_pipeline.py` first.")
    st.stop()

features_df = pd.read_csv(FEATURE_DATA)
teams_in_data = set(features_df["home_team"]).union(set(features_df["away_team"]))
teams = [team for team in EURO_2024_TEAMS if team in teams_in_data]


def find_team_photo(team: str) -> Optional[Path]:
    if not TEAM_PHOTOS_DIR.exists():
        return None
    for ext in [".avif", ".webp", ".png", ".jpg", ".jpeg", ".JPG"]:
        p = TEAM_PHOTOS_DIR / f"{team}{ext}"
        if p.exists():
            return p
    return None


def latest_for_pair(home: str, away: str) -> pd.Series:
    pair = features_df[(features_df["home_team"] == home) & (features_df["away_team"] == away)]
    if pair.empty:
        pair = features_df[(features_df["home_team"] == away) & (features_df["away_team"] == home)]
    if pair.empty:
        home_rows = features_df[features_df["home_team"] == home]
        away_rows = features_df[features_df["away_team"] == away]
        if not home_rows.empty and not away_rows.empty:
            mixed = pd.concat([home_rows.tail(1), away_rows.tail(1)])
            return mixed.mean(numeric_only=True)
        return features_df.iloc[-1]
    return pair.iloc[-1]


left, right = st.columns(2)
with left:
    home_team = st.selectbox("Home Team", teams)
with right:
    away_team = st.selectbox("Away Team", [t for t in teams if t != home_team])

photo_col_1, photo_col_2 = st.columns(2)
with photo_col_1:
    st.markdown(f"### {home_team}")
    home_photo = find_team_photo(home_team)
    if home_photo:
        st.image(str(home_photo), width=230)
with photo_col_2:
    st.markdown(f"### {away_team}")
    away_photo = find_team_photo(away_team)
    if away_photo:
        st.image(str(away_photo), width=230)

if st.button("Predict Match"):
    base = latest_for_pair(home_team, away_team)
    predictor = MatchPredictor()

    payload = {
        "elo_diff": float(base.get("elo_diff", 0.0)),
        "fifa_rank_diff": float(base.get("fifa_rank_diff", 0.0)),
        "fifa_points_diff": float(base.get("fifa_points_diff", 0.0)),
        "tournament_importance_score": float(base.get("tournament_importance_score", 1.0)),
        "h2h_matches_prior": float(base.get("h2h_matches_prior", 0.0)),
        "h2h_goal_diff_prior": float(base.get("h2h_goal_diff_prior", 0.0)),
        "home_points_per_match_last_5": float(base.get("home_points_per_match_last_5", 1.0)),
        "away_points_per_match_last_5": float(base.get("away_points_per_match_last_5", 1.0)),
        "home_points_per_match_last_10": float(base.get("home_points_per_match_last_10", 1.0)),
        "away_points_per_match_last_10": float(base.get("away_points_per_match_last_10", 1.0)),
        "home_goals_for_last_5": float(base.get("home_goals_for_last_5", 1.0)),
        "away_goals_for_last_5": float(base.get("away_goals_for_last_5", 1.0)),
        "home_goals_against_last_5": float(base.get("home_goals_against_last_5", 1.0)),
        "away_goals_against_last_5": float(base.get("away_goals_against_last_5", 1.0)),
    }

    pred = predictor.predict(payload)
    st.success(f"Projected score: {home_team} {pred['home_goals']} - {pred['away_goals']} {away_team}")

    if pred["predicted_outcome"] == "Home Win":
        st.subheader(f"Predicted winner: {home_team}")
    elif pred["predicted_outcome"] == "Away Win":
        st.subheader(f"Predicted winner: {away_team}")
    else:
        st.subheader("Predicted result: Draw")
