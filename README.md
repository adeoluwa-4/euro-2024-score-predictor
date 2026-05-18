# Euro 2024 Score Predictor (Full Rebuild)

This project was rebuilt using the **World Cup Score Predictor** workflow as the reference architecture.

It now includes a reproducible end-to-end pipeline:
1. data cleaning
2. preprocessing
3. feature engineering
4. model training
5. evaluation
6. prediction interface
7. dashboard
8. Streamlit deployment

## Data cutoff policy
- Cutoff is fixed at **2024-06-14** (`src/config.py`).
- Pipeline only uses records at or before this summer-2024 cutoff.
- Source dataset is tournament-scoped (Euro 2024), and all rows are processed under that boundary.

## Project structure

- `data/matches.csv`: raw source dataset
- `data/processed/matches_cleaned.csv`: cleaned raw data output
- `data/processed/matches_preprocessed.csv`: preprocessed dataset with target labels
- `data/features/matches_features.csv`: engineered feature dataset
- `data/splits/train.csv`: time-ordered training split
- `data/splits/test.csv`: time-ordered test split

- `src/config.py`: global configuration (paths, cutoff, split, random state)
- `src/pipeline/data_cleaning.py`: column normalization, numeric parsing, metadata, cutoff filtering
- `src/pipeline/preprocessing.py`: essential-column validation, missing-value handling, outcome label creation
- `src/pipeline/features.py`: pre-match team strength/form/shot feature engineering
- `src/pipeline/train.py`: xG regressors + outcome classifier, evaluation, artifact export
- `src/pipeline/predictor.py`: reusable inference interface for app/CLI

- `run_pipeline.py`: full rebuild entrypoint from raw data to trained artifacts
- `model.py`: compatibility training launcher (calls pipeline trainer)
- `predict.py`: CLI prediction interface from engineered inputs
- `app.py`: Streamlit dashboard + interactive prediction UI

- `models/home_goals_model.pkl`: trained home xG regressor
- `models/away_goals_model.pkl`: trained away xG regressor
- `models/outcome_model.pkl`: trained match-outcome classifier
- `models/xg_features.pkl`: ordered xG model features
- `models/outcome_features.pkl`: ordered outcome model features

- `reports/metrics.json`: evaluation summary
- `reports/test_predictions.csv`: test-set predictions for review

## Rebuild instructions

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run full pipeline:

```bash
python3 run_pipeline.py
```

Start Streamlit app:

```bash
python3 -m streamlit run app.py
```

## Latest training outputs
From the current rebuild:
- rows_total: 51
- rows_train: 40
- rows_test: 11
- outcome_accuracy: 0.1818
- home_goals_mae: 0.0653
- away_goals_mae: 0.0828

## File-by-file commit/push flow
To push incrementally file-by-file:

```bash
git add <file>
git commit -m "rebuild: <what changed in that file>"
git push origin main
```

Recommended order:
1. `src/config.py`
2. `src/pipeline/data_cleaning.py`
3. `src/pipeline/preprocessing.py`
4. `src/pipeline/features.py`
5. `src/pipeline/train.py`
6. `src/pipeline/predictor.py`
7. `run_pipeline.py`
8. `model.py`
9. `predict.py`
10. `app.py`
11. `README.md`
12. `requirements.txt`
13. generated artifacts under `data/processed`, `data/features`, `data/splits`, `reports`, and `models`

