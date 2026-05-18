import argparse

from src.pipeline.predictor import MatchPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Euro 2024 match outcome from engineered features")
    parser.add_argument("--home-attack-strength", type=float, required=True)
    parser.add_argument("--away-attack-strength", type=float, required=True)
    parser.add_argument("--home-defense-weakness", type=float, required=True)
    parser.add_argument("--away-defense-weakness", type=float, required=True)
    parser.add_argument("--home-form-points", type=float, required=True)
    parser.add_argument("--away-form-points", type=float, required=True)
    parser.add_argument("--home-avg-shots", type=float, required=True)
    parser.add_argument("--away-avg-shots", type=float, required=True)
    parser.add_argument("--home-avg-shots-on-target", type=float, required=True)
    parser.add_argument("--away-avg-shots-on-target", type=float, required=True)
    parser.add_argument("--home-expected-goals-xg", type=float, required=True)
    parser.add_argument("--away-expected-goals-xg", type=float, required=True)
    args = parser.parse_args()

    features = {
        "home_attack_strength": args.home_attack_strength,
        "away_attack_strength": args.away_attack_strength,
        "home_defense_weakness": args.home_defense_weakness,
        "away_defense_weakness": args.away_defense_weakness,
        "home_form_points": args.home_form_points,
        "away_form_points": args.away_form_points,
        "home_avg_shots": args.home_avg_shots,
        "away_avg_shots": args.away_avg_shots,
        "home_avg_shots_on_target": args.home_avg_shots_on_target,
        "away_avg_shots_on_target": args.away_avg_shots_on_target,
        "home_expected_goals_xg": args.home_expected_goals_xg,
        "away_expected_goals_xg": args.away_expected_goals_xg,
    }

    predictor = MatchPredictor()
    result = predictor.predict(features)
    print(result)


if __name__ == "__main__":
    main()
