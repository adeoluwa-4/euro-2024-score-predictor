from src.pipeline.data_cleaning import clean_data
from src.pipeline.features import build_features
from src.pipeline.preprocessing import preprocess_data
from src.pipeline.train import train_models


if __name__ == "__main__":
    clean_data()
    preprocess_data()
    build_features()
    metrics = train_models()
    print("Pipeline complete")
    for key, value in metrics.items():
        print(f"{key}: {value}")
