from src.preprocessing import load_data, cleaning
from src.feature_engineering import prepare_features
from src.train import train_models
from src.evaluate import evaluate_models


def main():
    df = load_data("data/NYC.csv")
    df = cleaning(df)
    df = prepare_features(df)

    models, X_test, y_test = train_models(df)

    evaluate_models(models, X_test, y_test)


if __name__ == "__main__":
    main()