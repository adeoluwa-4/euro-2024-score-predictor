from src.pipeline.train import train_models


if __name__ == "__main__":
    metrics = train_models()
    print(metrics)
