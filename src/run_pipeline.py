"""
=============================================================
  run_pipeline.py — Full End-to-End Pipeline Orchestrator
=============================================================
Runs in order:
  1. Dataset generation
  2. Preprocessing + feature engineering
  3. EDA figures
  4. Model training + hyperparameter tuning
  5. Evaluation figures
  6. Prediction demo

Usage:
    cd salary_prediction
    python src/run_pipeline.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import joblib

# ── Step 1: Generate dataset ─────────────────────────────
print("\n" + "━"*60)
print("  STEP 1/5 — Dataset Generation")
print("━"*60)
from generate_dataset import main as gen_data
gen_data()

# ── Step 2: Preprocess ───────────────────────────────────
print("\n" + "━"*60)
print("  STEP 2/5 — Preprocessing & Feature Engineering")
print("━"*60)
from preprocess import load_and_preprocess
DATA = ROOT / "data" / "employees.csv"
(X_train, X_test,
 y_train, y_test,
 feature_names,
 preprocessor,
 df) = load_and_preprocess(DATA)

# Persist preprocessor
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
joblib.dump(preprocessor,  MODELS / "preprocessor.joblib")
joblib.dump(feature_names, MODELS / "feature_names.joblib")
print(f"✅  Preprocessor + feature names saved → {MODELS}")

# ── Step 3: EDA ──────────────────────────────────────────
print("\n" + "━"*60)
print("  STEP 3/5 — Exploratory Data Analysis")
print("━"*60)
from eda import run_all_eda
run_all_eda(df)

# ── Step 4: Train ────────────────────────────────────────
print("\n" + "━"*60)
print("  STEP 4/5 — Model Training & Tuning")
print("━"*60)
from train_models import train_and_evaluate
results = train_and_evaluate(X_train, X_test, y_train, y_test, feature_names)
print("\n── Model Leaderboard ──")
print(results[["Model", "R²", "RMSE", "MAE", "MAPE (%)"]].to_string(index=False))

# ── Step 5: Evaluate ─────────────────────────────────────
print("\n" + "━"*60)
print("  STEP 5/5 — Evaluation Figures")
print("━"*60)
from evaluate import run_all_evaluation
run_all_evaluation(X_test, y_test, feature_names)

# ── Demo predictions ──────────────────────────────────────
print("\n" + "━"*60)
print("  DEMO — Salary Predictions")
print("━"*60)
from predict import predict_salary, load_artifacts
preprocessor_loaded, model_loaded, _ = load_artifacts()

sample = {
    "age": 32, "gender": "Female",
    "education_level": "Master's",
    "department": "Data Science", "job_title": "Senior",
    "location": "San Francisco",
    "years_experience": 8, "years_at_company": 4,
    "performance_score": 4.5, "num_projects": 14,
    "training_hours": 90, "certifications": 4, "team_size": 10,
}
result = predict_salary(sample, preprocessor_loaded, model_loaded)
print(f"\n  Sample employee → Predicted salary: ${result['predicted_salary']:,.2f}")
print(f"  Confidence range: ${result['lower_bound']:,.2f} – ${result['upper_bound']:,.2f}")

print("\n" + "="*60)
print("  ✅  PIPELINE COMPLETE!")
print("  Launch the dashboard with:")
print("      streamlit run src/app.py")
print("="*60 + "\n")
