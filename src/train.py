from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import numpy as np


def train_models(df):
    """
    Train multiple models:
    - Linear Regression (baseline)
    - XGBoost (with hyperparameter tuning)

    Returns:
    --------
    models_dict : dict
    X_test, y_test
    """

    features = [
        'passenger_count',
        'is_weekend',
        'distance_km',
        'store_and_fwd_flag',
        'hour_sin',
        'hour_cos',
        'day_sin',
        'day_cos',
        'pickup_latitude',
        'pickup_longitude',
        'dropoff_latitude',
        'dropoff_longitude'
    ]

    X = df[features]
    y = np.log1p(df['trip_duration'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {}

    # -----------------------------
    # 1. Linear Regression (Baseline)
    # -----------------------------
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    models['LinearRegression'] = lr

    # -----------------------------
    # 2. XGBoost + GridSearch 
    # -----------------------------
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1]
    }

    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_xgb = grid.best_estimator_

    print("Best XGBoost Params:", grid.best_params_)

    models['XGBoost'] = best_xgb

    return models, X_test, y_test