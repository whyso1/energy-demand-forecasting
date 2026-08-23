"""Model construction and fit/predict, ported from the legacy ml_model().

Dead code removed: the legacy notebook assigned `n_estimators = 1000`
in a loop right before calling ml_model() but never passed it through,
so it silently did nothing and every model used the function's default
(500). Here n_estimators is just a normal config value
(config.yaml: models.xgboost.n_estimators).
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

MODEL_BUILDERS = {
    "linear_regression": lambda params: LinearRegression(**params),
    "random_forest": lambda params: RandomForestRegressor(**params),
    "xgboost": lambda params: xgb.XGBRegressor(**params),
}


def build_model(name: str, params: dict):
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model {name!r}; expected one of {list(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[name](params or {})


def fit_predict(name: str, params: dict, X_train, y_train, X_test) -> np.ndarray:
    model = build_model(name, params)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def feature_importance(name: str, params: dict, X_train, y_train) -> pd.Series:
    """Fits a model and returns a feature-importance-style ranking:
    |coefficient| for linear regression, feature_importances_ for the
    tree models. Useful for sanity-checking that the top features make
    physical sense (e.g. demand lags and temperature, not something
    that shouldn't be predictive).
    """
    model = build_model(name, params)
    model.fit(X_train, y_train)
    if hasattr(model, "coef_"):
        values = np.abs(model.coef_)
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        raise ValueError(f"Model {name!r} exposes neither coef_ nor feature_importances_")
    return pd.Series(values, index=X_train.columns).sort_values(ascending=False)
