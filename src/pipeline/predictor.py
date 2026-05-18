from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import joblib
import pandas as pd

from src.config import MODELS_DIR


class MatchPredictor:
    def __init__(self, model_dir: Optional[Path] = None) -> None:
        model_dir = model_dir or MODELS_DIR
        self.home_goals_model = joblib.load(model_dir / "home_goals_model.pkl")
        self.away_goals_model = joblib.load(model_dir / "away_goals_model.pkl")
        self.outcome_model = joblib.load(model_dir / "outcome_model.pkl")
        self.outcome_features = joblib.load(model_dir / "outcome_features.pkl")
        self.goal_features = joblib.load(model_dir / "goal_features.pkl")

    def predict(self, features: Dict[str, float]) -> Dict[str, Union[float, int, str]]:
        goal_frame = pd.DataFrame([features]).reindex(columns=self.goal_features, fill_value=0.0)
        home_goals = float(self.home_goals_model.predict(goal_frame)[0])
        away_goals = float(self.away_goals_model.predict(goal_frame)[0])

        outcome_input = goal_frame.copy()
        outcome_input["home_goals_pred"] = home_goals
        outcome_input["away_goals_pred"] = away_goals
        outcome_input = outcome_input.reindex(columns=self.outcome_features, fill_value=0.0)

        outcome = int(self.outcome_model.predict(outcome_input)[0])
        outcome_map = {0: "Draw", 1: "Home Win", 2: "Away Win"}

        return {
            "home_goals": round(home_goals, 3),
            "away_goals": round(away_goals, 3),
            "predicted_outcome_code": outcome,
            "predicted_outcome": outcome_map.get(outcome, "Draw"),
        }
