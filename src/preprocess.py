"""
=============================================================
  preprocess.py — Data Preprocessing & Feature Engineering
=============================================================
Handles:
  • Missing-value imputation
  • Ordinal & one-hot encoding
  • Feature engineering (derived columns)
  • Train / test split
  • Feature scaling

Public API
----------
  load_and_preprocess(path) → (X_train, X_test, y_train, y_test,
                                feature_names, preprocessor)
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer


# ── Ordinal maps ─────────────────────────────────────────
EDUCATION_ORDER = [
    ["High School", "Bachelor's", "Master's", "PhD"]
]
JOB_TITLE_ORDER = [
    ["Junior", "Mid-Level", "Senior", "Lead", "Manager", "Director"]
]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the CSV and perform basic type fixes."""
    df = pd.read_csv(path)
    # Drop non-predictive ID column
    df = df.drop(columns=["employee_id"], errors="ignore")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features that capture domain knowledge."""
    df = df.copy()

    # Seniority ratio (experience vs. age)
    df["experience_ratio"] = df["years_experience"] / (df["age"] + 1)

    # Loyalty index (how long they've stayed at this company)
    df["loyalty_index"] = df["years_at_company"] / (df["years_experience"] + 1)

    # Productivity proxy (projects / year at company)
    df["productivity"] = df["num_projects"] / (df["years_at_company"] + 1)

    # Skills composite (certifications × perf + training / 100)
    df["skills_composite"] = (
        df["certifications"].fillna(0) * df["performance_score"].fillna(3.2)
        + df["training_hours"].fillna(0) / 100
    )

    return df


def build_preprocessor(
    numeric_cols: list[str],
    ordinal_cols: list[str],
    ordinal_categories: list[list[str]],
    nominal_cols: list[str],
) -> ColumnTransformer:
    """Construct the sklearn ColumnTransformer pipeline."""

    # --- Numeric pipeline ---
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    # --- Ordinal pipeline ---
    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    # --- Nominal pipeline (one-hot) ---
    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("numeric",  numeric_pipeline,  numeric_cols),
        ("ordinal",  ordinal_pipeline,  ordinal_cols),
        ("nominal",  nominal_pipeline,  nominal_cols),
    ], remainder="drop")

    return preprocessor


def load_and_preprocess(
    path: str | Path,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """
    Full preprocessing pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    feature_names                     : list[str]
    preprocessor                      : fitted ColumnTransformer
    df_engineered                     : pd.DataFrame (for EDA)
    """
    # 1. Load & engineer features
    df = load_raw(path)
    df = engineer_features(df)

    # 2. Separate target
    y = df["salary"].values
    X = df.drop(columns=["salary"])

    # 3. Column groups
    numeric_cols = [
        "age", "years_experience", "years_at_company",
        "performance_score", "num_projects", "training_hours",
        "certifications", "team_size",
        "experience_ratio", "loyalty_index", "productivity", "skills_composite",
    ]
    ordinal_cols       = ["education_level", "job_title"]
    ordinal_categories = [EDUCATION_ORDER[0], JOB_TITLE_ORDER[0]]
    nominal_cols       = ["gender", "department", "location"]

    # 4. Build & fit preprocessor
    preprocessor = build_preprocessor(
        numeric_cols, ordinal_cols, ordinal_categories, nominal_cols
    )

    # 5. Train / test split (before fitting → no leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    # 6. Recover feature names
    ohe_cols = (
        preprocessor
        .named_transformers_["nominal"]
        .named_steps["encoder"]
        .get_feature_names_out(nominal_cols)
        .tolist()
    )
    feature_names = numeric_cols + ordinal_cols + ohe_cols

    print(f"✅  Preprocessing complete")
    print(f"    Train: {X_train_t.shape}   Test: {X_test_t.shape}")
    print(f"    Features: {len(feature_names)}")

    return (
        X_train_t, X_test_t,
        y_train,   y_test,
        feature_names,
        preprocessor,
        df,           # full engineered df (for EDA)
    )


if __name__ == "__main__":
    DATA = Path(__file__).parent.parent / "data" / "employees.csv"
    load_and_preprocess(DATA)
