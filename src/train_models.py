"""
=============================================================
  train_models.py — Model Training, Tuning & Evaluation
=============================================================
Models trained:
  1. Linear Regression       (baseline)
  2. Ridge Regression        (L2 regularisation)
  3. Lasso Regression        (L1 regularisation)
  4. Decision Tree Regressor
  5. Random Forest Regressor
  6. Gradient Boosting       (GBM)
  7. XGBoost Regressor       (optional, falls back gracefully)

Outputs
-------
  • reports/model_comparison.csv
  • models/<model_name>.joblib
  • models/best_model.joblib
  • models/preprocessor.joblib
"""

import warnings
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
MODELS  = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)


# ── Metric helper ────────────────────────────────────────
def compute_metrics(y_true, y_pred, model_name: str, train_time: float) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        "Model":       model_name,
        "MAE":         round(mae,  2),
        "MSE":         round(mse,  2),
        "RMSE":        round(rmse, 2),
        "R²":          round(r2,   4),
        "MAPE (%)":    round(mape, 2),
        "Train Time":  round(train_time, 3),
    }


# ── Model catalogue ──────────────────────────────────────
def get_model_catalogue():
    """Return list of (name, estimator, param_grid) tuples."""

    catalogue = [
        (
            "Linear Regression",
            LinearRegression(),
            {},               # no hyperparams to tune
        ),
        (
            "Ridge Regression",
            Ridge(),
            {"alpha": [0.1, 1.0, 10.0, 100.0]},
        ),
        (
            "Lasso Regression",
            Lasso(max_iter=5000),
            {"alpha": [0.01, 0.1, 1.0, 10.0]},
        ),
        (
            "Decision Tree",
            DecisionTreeRegressor(random_state=42),
            {
                "max_depth":        [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
            },
        ),
        (
            "Random Forest",
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {
                "n_estimators": [100, 200],
                "max_depth":    [10, 20, None],
                "max_features": ["sqrt", "log2"],
            },
        ),
        (
            "Gradient Boosting",
            GradientBoostingRegressor(random_state=42),
            {
                "n_estimators":   [100, 200],
                "learning_rate":  [0.05, 0.10],
                "max_depth":      [3, 5],
            },
        ),
    ]

    # XGBoost (optional dependency)
    try:
        from xgboost import XGBRegressor
        catalogue.append((
            "XGBoost",
            XGBRegressor(random_state=42, verbosity=0, n_jobs=-1),
            {
                "n_estimators":  [100, 200],
                "learning_rate": [0.05, 0.10],
                "max_depth":     [4, 6],
            },
        ))
        print("  XGBoost detected — included in training run.")
    except ImportError:
        print("  XGBoost not installed — skipping.")

    return catalogue


# ── Core training function ───────────────────────────────
def train_and_evaluate(
    X_train, X_test,
    y_train, y_test,
    feature_names: list[str],
    cv: int = 5,
) -> pd.DataFrame:
    """
    Train all models, tune hyperparams with GridSearchCV,
    persist each model, and return a comparison DataFrame.
    """
    catalogue = get_model_catalogue()
    results   = []
    best_r2   = -np.inf
    best_name = None

    print("\n" + "="*60)
    print("  MODEL TRAINING & EVALUATION")
    print("="*60)

    for name, estimator, param_grid in catalogue:
        print(f"\n→ {name}")

        t0 = time.time()

        if param_grid:
            # GridSearchCV for hyperparameter tuning
            grid = GridSearchCV(
                estimator,
                param_grid,
                cv=cv,
                scoring="r2",
                n_jobs=-1,
                refit=True,
            )
            grid.fit(X_train, y_train)
            best_est = grid.best_estimator_
            print(f"  Best params: {grid.best_params_}")
        else:
            best_est = estimator
            best_est.fit(X_train, y_train)

        train_time = time.time() - t0

        # Cross-validation R² on training set
        cv_scores = cross_val_score(best_est, X_train, y_train,
                                    cv=cv, scoring="r2")
        print(f"  CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Test-set evaluation
        y_pred = best_est.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, name, train_time)
        metrics["CV R² Mean"] = round(cv_scores.mean(), 4)
        metrics["CV R² Std"]  = round(cv_scores.std(),  4)
        results.append(metrics)

        print(f"  Test  R² = {metrics['R²']:.4f}   RMSE = ${metrics['RMSE']:,.0f}")

        # Persist model
        model_path = MODELS / f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(best_est, model_path)

        # Track best model
        if metrics["R²"] > best_r2:
            best_r2   = metrics["R²"]
            best_name = name
            joblib.dump(best_est, MODELS / "best_model.joblib")

        # Save feature importances if available
        _save_importances(best_est, feature_names, name)

    # ── Summary ──────────────────────────────────────────
    df_results = pd.DataFrame(results).sort_values("R²", ascending=False)
    df_results.to_csv(REPORTS / "model_comparison.csv", index=False)

    print("\n" + "="*60)
    print(f"  🏆  Best model: {best_name}  (R² = {best_r2:.4f})")
    print(f"  Results saved → {REPORTS / 'model_comparison.csv'}")
    print("="*60 + "\n")

    return df_results


def _save_importances(model, feature_names: list[str], model_name: str):
    """Persist feature importances as JSON if available."""
    attr = None
    if hasattr(model, "feature_importances_"):
        attr = model.feature_importances_
    elif hasattr(model, "coef_"):
        attr = np.abs(model.coef_)

    if attr is None:
        return

    importances = dict(zip(feature_names, attr.tolist()))
    fname = REPORTS / f"{model_name.lower().replace(' ', '_')}_importances.json"
    with open(fname, "w") as f:
        json.dump(importances, f, indent=2)


def load_best_model():
    """Load the persisted best model."""
    path = MODELS / "best_model.joblib"
    if not path.exists():
        raise FileNotFoundError("No best_model.joblib found. Run training first.")
    return joblib.load(path)


def load_model(name: str):
    """Load a specific model by name (e.g. 'random_forest')."""
    path = MODELS / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


# ── Entry point ──────────────────────────────────────────
if __name__ == "__main__":
    from preprocess import load_and_preprocess

    DATA = Path(__file__).parent.parent / "data" / "employees.csv"

    (X_train, X_test,
     y_train, y_test,
     feature_names,
     preprocessor,
     df) = load_and_preprocess(DATA)

    # Persist preprocessor for inference
    joblib.dump(preprocessor, MODELS / "preprocessor.joblib")
    joblib.dump(feature_names, MODELS / "feature_names.joblib")
    print(f"✅  Preprocessor saved → {MODELS / 'preprocessor.joblib'}")

    results = train_and_evaluate(
        X_train, X_test,
        y_train, y_test,
        feature_names,
    )
    print(results[["Model", "R²", "RMSE", "MAE"]].to_string(index=False))
