# 💰 Employee Salary Prediction & Analytics System

> An end-to-end Machine Learning project for salary prediction, featuring exploratory data
> analysis, multiple regression models with hyperparameter tuning, interactive analytics,
> and a production-ready Streamlit web application.

---

## 🏆 Results Summary

| Model | R² | RMSE | MAE | MAPE |
|---|---|---|---|---|
| **Gradient Boosting** 🏆 | **0.9487** | **$10,023** | **$7,586** | **6.2%** |
| XGBoost | 0.9460 | $10,285 | $7,753 | 6.3% |
| Linear Regression | 0.9253 | $12,100 | $9,401 | 8.3% |
| Ridge Regression | 0.9252 | $12,109 | $9,400 | 8.3% |
| Lasso Regression | 0.9253 | $12,102 | $9,396 | 8.3% |
| Random Forest | 0.7783 | $20,843 | $15,565 | 13.5% |
| Decision Tree | 0.6318 | $26,858 | $20,421 | 16.8% |

---

## 📁 Project Structure

```
salary_prediction/
├── data/
│   └── employees.csv              ← Generated dataset (2,000 records)
├── models/
│   ├── best_model.joblib          ← Best trained model (Gradient Boosting)
│   ├── preprocessor.joblib        ← Fitted ColumnTransformer
│   ├── feature_names.joblib       ← Feature name list
│   ├── gradient_boosting.joblib
│   ├── random_forest.joblib
│   ├── linear_regression.joblib
│   ├── ridge_regression.joblib
│   ├── lasso_regression.joblib
│   ├── decision_tree.joblib
│   └── xgboost.joblib
├── reports/
│   ├── model_comparison.csv       ← Metrics for all models
│   ├── *_importances.json         ← Feature importance per model
│   └── figures/                   ← 20+ PNG charts
│       ├── 01_salary_distribution.png
│       ├── 02_salary_by_department.png
│       ├── 03_experience_salary.png
│       ├── 04_correlation_heatmap.png
│       ├── 05_performance_salary.png
│       ├── 06_location_salary.png
│       ├── 07_gender_salary.png
│       ├── 08_feature_correlations.png
│       ├── eval_*_actual_vs_pred.png
│       ├── eval_*_residuals.png
│       ├── feat_imp_*.png
│       └── model_comparison.png
├── src/
│   ├── generate_dataset.py        ← Synthetic data generator
│   ├── preprocess.py              ← Preprocessing + feature engineering
│   ├── eda.py                     ← EDA charts (Matplotlib/Seaborn)
│   ├── train_models.py            ← Training + GridSearchCV + joblib
│   ├── evaluate.py                ← Evaluation plots
│   ├── predict.py                 ← Inference module
│   ├── run_pipeline.py            ← Full pipeline orchestrator
│   └── app.py                     ← Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone / Set Up

```bash
git clone <your-repo>
cd salary_prediction
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
python src/run_pipeline.py
```

This single command:
- Generates 2,000 synthetic employee records
- Runs data preprocessing + feature engineering
- Produces 8 EDA charts
- Trains and tunes 7 regression models
- Saves all models with joblib
- Generates evaluation plots for every model
- Prints a live salary prediction demo

### 3. Launch the Dashboard

```bash
streamlit run src/app.py
```

Open `http://localhost:8501` in your browser.

---

## 📊 Dataset Description

### Synthetic Generation

The dataset is programmatically generated (`generate_dataset.py`) to simulate realistic
employment data. Salary is computed using a multiplicative model with domain-specific
premiums and controlled random noise, ensuring the data is learnable but not trivial.

### Features (14 input + 1 target)

| Feature | Type | Description |
|---|---|---|
| age | Numeric | Employee age (22–62) |
| gender | Nominal | Male / Female / Non-binary |
| education_level | Ordinal | High School → Bachelor's → Master's → PhD |
| department | Nominal | 8 departments |
| job_title | Ordinal | Junior → Mid-Level → Senior → Lead → Manager → Director |
| location | Nominal | 7 US cities + Remote |
| years_experience | Numeric | Total work experience |
| years_at_company | Numeric | Tenure at current employer |
| performance_score | Numeric | 1.0–5.0 rating |
| num_projects | Numeric | Projects completed |
| training_hours | Numeric | Annual training hours |
| certifications | Numeric | Professional certifications held |
| team_size | Numeric | Direct team size |
| **salary** | **Numeric** | **Target variable (USD/year)** |

### Engineered Features (derived)

| Feature | Formula | Rationale |
|---|---|---|
| experience_ratio | years_experience / age | Seniority relative to age |
| loyalty_index | years_at_company / years_experience | Retention signal |
| productivity | num_projects / years_at_company | Output rate |
| skills_composite | certs × perf_score + training/100 | Combined skills proxy |

---

## 🧪 Methodology

### Preprocessing Pipeline

```
Raw CSV
  → Drop employee_id
  → Feature Engineering (4 derived columns)
  → Separate Target (salary)
  → Train/Test Split (80 / 20, stratified by random_state=42)
  → ColumnTransformer:
      ├── Numeric: Median Imputer → StandardScaler
      ├── Ordinal: MostFrequent Imputer → OrdinalEncoder
      └── Nominal: MostFrequent Imputer → OneHotEncoder
```

### Hyperparameter Tuning

All non-trivial models use `GridSearchCV` with 5-fold cross-validation scored on R²:

```python
GridSearchCV(estimator, param_grid, cv=5, scoring="r2", n_jobs=-1)
```

### Evaluation Metrics

| Metric | Formula | Meaning |
|---|---|---|
| MAE | mean(\|y - ŷ\|) | Average absolute error in dollars |
| MSE | mean((y - ŷ)²) | Penalises large errors heavily |
| RMSE | √MSE | Same unit as salary |
| R² | 1 - SS_res/SS_tot | Variance explained (0–1, higher better) |
| MAPE | mean(\|y-ŷ\|/y)×100 | Percentage error |

---

## 🌐 Streamlit Dashboard Pages

| Page | Description |
|---|---|
| 🏠 Overview | KPI cards, salary distributions, department/location breakdowns |
| 📊 EDA | Interactive distributions, correlation heatmap, demographic analysis |
| 🤖 ML Models | Model leaderboard, metric comparison, actual vs predicted, residuals |
| 🔮 Predict Salary | Real-time salary prediction with peer comparison chart |
| 🗂️ Data Explorer | Filterable table, summary stats, CSV export |

---

## 🔮 Prediction API

```python
from src.predict import predict_salary, load_artifacts

preprocessor, model, _ = load_artifacts()

employee = {
    "age": 30,
    "gender": "Female",
    "education_level": "Master's",
    "department": "Data Science",
    "job_title": "Senior",
    "location": "San Francisco",
    "years_experience": 7,
    "years_at_company": 3,
    "performance_score": 4.2,
    "num_projects": 12,
    "training_hours": 80,
    "certifications": 3,
    "team_size": 8,
}

result = predict_salary(employee, preprocessor, model)
print(result)
# → {'predicted_salary': 189808.15, 'lower_bound': 170827.34, 'upper_bound': 208788.97}
```

---

## 🧠 Key Findings

1. **Gradient Boosting** achieved the best R² of **0.9487**, explaining ~95% of salary variance.
2. **Experience**, **job title**, **department**, and **location** are the strongest salary predictors.
3. San Francisco and New York carry the highest location premiums (+35–40%).
4. PhD holders earn ~59% more than High School graduates on average.
5. Each additional year of experience adds ~5% salary (diminishing after 10 years).
6. Performance score has a meaningful positive correlation (+8% per point above average).
7. Linear models (Ridge, Lasso) perform surprisingly well due to strong linear relationships
   in the data — outperforming Random Forest, which overfits small clusters.

---

## 📖 Viva Questions & Answers

**Q: Why is Gradient Boosting better than Random Forest here?**
A: Gradient Boosting builds trees sequentially, correcting errors from previous trees.
This reduces bias systematically. Random Forest averages parallel trees — strong for
high-variance data but less effective when the underlying relationships are more additive.

**Q: What is the purpose of StandardScaler?**
A: Normalises numeric features to zero mean and unit variance. Prevents features with
large magnitudes (like salary or age) from dominating gradient-based models.

**Q: How do you prevent data leakage?**
A: The preprocessor is fitted ONLY on training data and then applied to test data.
The train-test split is performed BEFORE any fitting happens.

**Q: What does R² = 0.9487 mean practically?**
A: The model explains ~95% of the variance in employee salaries. The remaining 5%
is unexplained noise — representing factors we don't have data on (negotiation skill,
company-specific bands, etc.).

**Q: Why use GridSearchCV over RandomSearchCV?**
A: GridSearchCV exhaustively searches the defined parameter space, guaranteeing the
best combination within that space. Acceptable here because our parameter grids are
small (≤ 16 combinations per model). RandomSearchCV is preferred for larger spaces.

**Q: What is joblib used for?**
A: joblib serializes Python objects (models, preprocessors) to disk. This enables
model persistence — train once, deploy many times — without retraining.

**Q: Why ordinal encoding for education and job title?**
A: These have a meaningful rank order (Junior < Senior < Director). Ordinal encoding
preserves this order. OneHotEncoding would lose it by treating each level as independent.

---

## 👨‍💻 Author

**Arun B** — B.Tech AI & Data Science  
Dr. Sivanthi Adithanar College of Engineering, Thiruchendur

---

## 📄 License

MIT License — free for academic and personal use.
