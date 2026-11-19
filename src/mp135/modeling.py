"""One preprocessing pipeline shared by training, the API and batch scoring."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_model(config, settings):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, config["features"]["numeric"]),
        ("categorical", categorical, config["features"]["categorical"]),
    ], remainder="drop")
    options = dict(settings)
    kind = options.pop("kind")
    if kind == "ridge":
        regressor = Ridge(**options)
    elif kind == "hist_gradient_boosting":
        regressor = HistGradientBoostingRegressor(
            random_state=config["training"]["random_state"], **options,
        )
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    return Pipeline([("preprocess", preprocess), ("regressor", regressor)])


def predict(model, frame):
    # Counts cannot be negative. Use the same postprocessing everywhere.
    return np.maximum(0.0, model.predict(frame))
