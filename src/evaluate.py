from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def evaluate_models(models, X_test, y_test):
    """
    Evaluate multiple models
    """

    for name, model in models.items():
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\nModel: {name}")
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"R²: {r2}")