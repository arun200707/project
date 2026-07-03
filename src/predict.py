"""
=============================================================
  predict.py — Salary Prediction for New Employees
=============================================================
Loads the best model + preprocessor and provides a clean
predict() function for single or batch inference.

Usage (CLI):
    python src/predict.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib

ROOT   = Path(__file__).parent.parent
MODELS = ROOT / "models"


def load_artifacts():
    """Load preprocessor and best model from disk."""
    try:
        preprocessor   = joblib.load(MODELS / "preprocessor.joblib")
        model          = joblib.load(MODELS / "best_model.joblib")
        feature_names  = joblib.load(MODELS / "feature_names.joblib")
        return preprocessor, model, feature_names
    except (AttributeError, ImportError, ModuleNotFoundError) as e:
        raise RuntimeError(f"Model compatibility error (sklearn version mismatch): {e}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model files not found: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading models: {e}")


def predict_salary(employee: dict, preprocessor=None, model=None) -> dict:
    """
    Predict salary for a single employee record.

    Parameters
    ----------
    employee : dict
        Keys should match the raw dataset columns (excluding salary).
    preprocessor, model : optional
        Pre-loaded artifacts. If None, they are loaded from disk.

    Returns
    -------
    dict with 'predicted_salary', 'lower_bound', 'upper_bound'
    """
    if preprocessor is None or model is None:
        preprocessor, model, _ = load_artifacts()

    # Build a one-row DataFrame and add engineered features
    from preprocess import engineer_features
    df = pd.DataFrame([employee])
    df = engineer_features(df)

    X = preprocessor.transform(df)
    pred = model.predict(X)[0]

    # Simple ± 10 % confidence interval
    return {
        "predicted_salary": round(pred, 2),
        "lower_bound":      round(pred * 0.90, 2),
        "upper_bound":      round(pred * 1.10, 2),
    }


def predict_batch(employees_df: pd.DataFrame,
                  preprocessor=None, model=None) -> pd.DataFrame:
    """
    Predict salaries for a batch of employees.

    Parameters
    ----------
    employees_df : pd.DataFrame
        Must NOT include the 'salary' column.

    Returns
    -------
    Copy of employees_df with a 'predicted_salary' column appended.
    """
    if preprocessor is None or model is None:
        preprocessor, model, _ = load_artifacts()

    from preprocess import engineer_features
    df = engineer_features(employees_df.copy())
    X  = preprocessor.transform(df)
    employees_df = employees_df.copy()
    employees_df["predicted_salary"] = model.predict(X).round(2)
    return employees_df


# ── Demo ─────────────────────────────────────────────────
if __name__ == "__main__":
    sample_employees = [
        {
            "age": 30, "gender": "Female",
            "education_level": "Master's",
            "department": "Data Science", "job_title": "Senior",
            "location": "San Francisco",
            "years_experience": 7, "years_at_company": 3,
            "performance_score": 4.2, "num_projects": 12,
            "training_hours": 80, "certifications": 3, "team_size": 8,
        },
        {
            "age": 45, "gender": "Male",
            "education_level": "Bachelor's",
            "department": "Sales", "job_title": "Manager",
            "location": "Chicago",
            "years_experience": 20, "years_at_company": 8,
            "performance_score": 3.8, "num_projects": 6,
            "training_hours": 40, "certifications": 1, "team_size": 15,
        },
        {
            "age": 25, "gender": "Non-binary",
            "education_level": "Bachelor's",
            "department": "Engineering", "job_title": "Junior",
            "location": "Austin",
            "years_experience": 2, "years_at_company": 1,
            "performance_score": 3.5, "num_projects": 3,
            "training_hours": 60, "certifications": 0, "team_size": 5,
        },
    ]

    preprocessor, model, _ = load_artifacts()

    print("\n" + "="*55)
    print("  SALARY PREDICTIONS FOR SAMPLE EMPLOYEES")
    print("="*55)
    for i, emp in enumerate(sample_employees, 1):
        result = predict_salary(emp, preprocessor, model)
        print(f"\nEmployee #{i}")
        print(f"  Dept/Title : {emp['department']} / {emp['job_title']}")
        print(f"  Education  : {emp['education_level']} | Exp: {emp['years_experience']} yrs")
        print(f"  Location   : {emp['location']}")
        print(f"  Prediction : ${result['predicted_salary']:>10,.2f}")
        print(f"  95% Range  : ${result['lower_bound']:>10,.2f}  –  ${result['upper_bound']:,.2f}")
