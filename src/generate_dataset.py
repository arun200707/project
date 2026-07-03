"""
=============================================================
  generate_dataset.py — Synthetic Employee Dataset Generator
=============================================================
Generates a realistic employee dataset for salary prediction.
Run this FIRST before training or launching the dashboard.

Usage:
    python src/generate_dataset.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Reproducibility ──────────────────────────────────────
np.random.seed(42)
N = 2000  # number of employees

# ── Categorical pools ─────────────────────────────────────
DEPARTMENTS   = ["Engineering", "Sales", "Marketing", "Finance",
                 "HR", "Operations", "Data Science", "Product"]
EDUCATION     = ["High School", "Bachelor's", "Master's", "PhD"]
JOB_TITLES    = ["Junior", "Mid-Level", "Senior", "Lead", "Manager", "Director"]
LOCATIONS     = ["New York", "San Francisco", "Chicago", "Austin",
                 "Seattle", "Boston", "Remote"]
GENDERS       = ["Male", "Female", "Non-binary"]

# ── Weights / base salaries ───────────────────────────────
DEPT_PREMIUM  = {
    "Engineering": 1.30, "Data Science": 1.35, "Finance": 1.20,
    "Product": 1.25, "Marketing": 1.00, "Sales": 1.05,
    "Operations": 0.95, "HR": 0.90
}
EDU_PREMIUM   = {
    "High School": 0.85, "Bachelor's": 1.00,
    "Master's": 1.18, "PhD": 1.35
}
LOC_PREMIUM   = {
    "San Francisco": 1.40, "New York": 1.35, "Seattle": 1.25,
    "Boston": 1.20, "Austin": 1.10, "Chicago": 1.05, "Remote": 1.00
}
TITLE_PREMIUM = {
    "Junior": 0.70, "Mid-Level": 0.90, "Senior": 1.10,
    "Lead": 1.25, "Manager": 1.40, "Director": 1.65
}

BASE_SALARY = 55_000  # USD starting point


def build_dataset(n: int = N) -> pd.DataFrame:
    """Build and return the synthetic employee DataFrame."""

    # ── Core categorical features ─────────────────────────
    departments = np.random.choice(DEPARTMENTS, n,
                                   p=[0.25, 0.12, 0.10, 0.10,
                                      0.08, 0.12, 0.15, 0.08])
    education   = np.random.choice(EDUCATION,   n, p=[0.10, 0.50, 0.30, 0.10])
    job_titles  = np.random.choice(JOB_TITLES,  n, p=[0.20, 0.25, 0.25, 0.12, 0.12, 0.06])
    locations   = np.random.choice(LOCATIONS,   n, p=[0.18, 0.18, 0.12, 0.12, 0.12, 0.10, 0.18])
    genders     = np.random.choice(GENDERS,     n, p=[0.52, 0.44, 0.04])

    # ── Numeric features ──────────────────────────────────
    age              = np.random.randint(22, 62, n)
    years_experience = np.clip(np.random.normal(8, 5, n), 0, 35).astype(int)

    # Ensure experience is age-consistent
    years_experience = np.minimum(years_experience, age - 18)
    years_experience = np.clip(years_experience, 0, 35)

    years_at_company = np.clip(
        np.random.exponential(3, n), 0, years_experience + 1
    ).astype(int)

    performance_score = np.clip(np.random.normal(3.2, 0.7, n), 1.0, 5.0).round(1)
    num_projects      = np.random.randint(1, 20, n)
    training_hours    = np.random.randint(0, 120, n)
    certifications    = np.random.randint(0, 8, n)
    team_size         = np.random.randint(1, 50, n)

    # ── Salary computation ────────────────────────────────
    salaries = []
    for i in range(n):
        salary = BASE_SALARY
        salary *= DEPT_PREMIUM[departments[i]]
        salary *= EDU_PREMIUM[education[i]]
        salary *= LOC_PREMIUM[locations[i]]
        salary *= TITLE_PREMIUM[job_titles[i]]

        # Experience bump (5% per year, diminishing returns after 10y)
        exp = years_experience[i]
        exp_factor = 1 + min(exp, 10) * 0.05 + max(exp - 10, 0) * 0.02
        salary *= exp_factor

        # Performance bonus (up to +25 %)
        salary *= (1 + (performance_score[i] - 3.0) * 0.08)

        # Small boosts for certifications and training
        salary += certifications[i] * 800
        salary += training_hours[i] * 30

        # Realistic noise (± 8 %)
        salary *= np.random.uniform(0.92, 1.08)

        salaries.append(round(salary, 2))

    # ── Assemble DataFrame ────────────────────────────────
    df = pd.DataFrame({
        "employee_id":       [f"EMP{str(i+1).zfill(4)}" for i in range(n)],
        "age":               age,
        "gender":            genders,
        "education_level":   education,
        "department":        departments,
        "job_title":         job_titles,
        "location":          locations,
        "years_experience":  years_experience,
        "years_at_company":  years_at_company,
        "performance_score": performance_score,
        "num_projects":      num_projects,
        "training_hours":    training_hours,
        "certifications":    certifications,
        "team_size":         team_size,
        "salary":            salaries,
    })

    # ── Introduce ~3 % missing values (realistic) ─────────
    for col in ["performance_score", "training_hours", "certifications"]:
        mask = np.random.random(n) < 0.03
        df.loc[mask, col] = np.nan

    return df


def main():
    out_path = Path(__file__).parent.parent / "data" / "employees.csv"
    out_path.parent.mkdir(exist_ok=True)

    df = build_dataset()
    df.to_csv(out_path, index=False)

    print(f"✅  Dataset saved → {out_path}")
    print(f"    Rows: {len(df):,}   Columns: {df.shape[1]}")
    print(f"    Salary range: ${df['salary'].min():,.0f} – ${df['salary'].max():,.0f}")
    print(f"    Missing values:\n{df.isnull().sum()[df.isnull().sum()>0]}")


if __name__ == "__main__":
    main()
