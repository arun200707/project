"""
=============================================================
  evaluate.py — Model Evaluation Charts & Feature Importance
=============================================================
Generates:
  • Actual vs Predicted scatter plots
  • Residual plots
  • Feature importance bar charts (top-20)
  • Model comparison bar chart
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib

ROOT    = Path(__file__).parent.parent
FIGURES = ROOT / "reports" / "figures"
REPORTS = ROOT / "reports"
MODELS  = ROOT / "models"
FIGURES.mkdir(parents=True, exist_ok=True)

BG, FG, ACC = "#0E1117", "#FAFAFA", "#4F8BF9"
GRID_CLR     = "#2E3748"


def _style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": "#1A1F2E",
        "axes.edgecolor":   GRID_CLR, "axes.labelcolor": FG,
        "text.color": FG, "xtick.color": FG, "ytick.color": FG,
        "grid.color": GRID_CLR, "grid.linewidth": 0.5,
        "legend.facecolor": "#1A1F2E", "legend.edgecolor": GRID_CLR,
        "legend.labelcolor": FG, "axes.titlesize": 13,
    })


def _save(fig, name: str):
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {path.name}")


def fmt_k(x, _):
    return f"${x/1000:.0f}K"


# ── Actual vs Predicted ──────────────────────────────────
def plot_actual_vs_predicted(y_true, y_pred, model_name: str):
    _style()
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(y_true, y_pred, alpha=0.4, s=12, color=ACC, edgecolors="none")
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], color="#FF6B6B", linewidth=2, linestyle="--", label="Perfect fit")

    # Shaded ±10 % band
    ax.fill_between([mn, mx], [mn*0.9, mx*0.9], [mn*1.1, mx*1.1],
                    color="#FF6B6B", alpha=0.07, label="±10 % band")

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
    ax.set_xlabel("Actual Salary")
    ax.set_ylabel("Predicted Salary")
    ax.set_title(f"Actual vs Predicted — {model_name}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    slug = model_name.lower().replace(" ", "_")
    _save(fig, f"eval_{slug}_actual_vs_pred")


# ── Residual plot ────────────────────────────────────────
def plot_residuals(y_true, y_pred, model_name: str):
    _style()
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Residual Analysis — {model_name}", fontsize=15, color=FG, fontweight="bold")

    # Residuals vs Predicted
    ax = axes[0]
    ax.scatter(y_pred, residuals, alpha=0.4, s=12, color=ACC, edgecolors="none")
    ax.axhline(0, color="#FF6B6B", linewidth=2, linestyle="--")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
    ax.set_xlabel("Predicted Salary")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title("Residuals vs Predicted")
    ax.grid(True)

    # Residual histogram
    ax = axes[1]
    ax.hist(residuals, bins=50, color=ACC, alpha=0.8, edgecolor="none", density=True)
    # KDE overlay
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(residuals)
    xs  = np.linspace(residuals.min(), residuals.max(), 300)
    ax.plot(xs, kde(xs), color="#FF6B6B", linewidth=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution")
    ax.grid(True, axis="y")

    plt.tight_layout()
    slug = model_name.lower().replace(" ", "_")
    _save(fig, f"eval_{slug}_residuals")


# ── Feature importance ───────────────────────────────────
def plot_feature_importance(model_name: str, top_n: int = 20):
    """Read importance JSON and plot horizontal bar chart."""
    slug = model_name.lower().replace(" ", "_")
    path = REPORTS / f"{slug}_importances.json"
    if not path.exists():
        return

    with open(path) as f:
        imp = json.load(f)

    series = pd.Series(imp).sort_values(ascending=True).tail(top_n)
    _style()
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(series)))
    ax.barh(series.index, series.values, color=colors, edgecolor="none")
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
    ax.grid(True, axis="x")
    plt.tight_layout()
    _save(fig, f"feat_imp_{slug}")


# ── Model comparison ─────────────────────────────────────
def plot_model_comparison(results_csv: Path):
    _style()
    df = pd.read_csv(results_csv).sort_values("R²", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Model Comparison", fontsize=16, color=FG, fontweight="bold")

    palette = plt.cm.tab10(np.linspace(0, 1, len(df)))

    for ax, metric, label in zip(
        axes,
        ["R²", "RMSE", "MAE"],
        ["R² Score (higher = better)",
         "RMSE — Root Mean Squared Error ($)",
         "MAE — Mean Absolute Error ($)"]
    ):
        ascending = metric in ("RMSE", "MAE")
        sub = df.sort_values(metric, ascending=not ascending)
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sub)))
        ax.barh(sub["Model"], sub[metric], color=colors, alpha=0.85)
        if metric in ("RMSE", "MAE"):
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
        ax.set_title(label, fontsize=11)
        ax.grid(True, axis="x")
        for i, (val, name) in enumerate(zip(sub[metric], sub["Model"])):
            lbl = f"{val:.4f}" if metric == "R²" else f"${val/1000:.1f}K"
            ax.text(val * 0.01 + sub[metric].min() * 0.01,
                    i, lbl, va="center", fontsize=8.5, color=FG)

    plt.tight_layout()
    _save(fig, "model_comparison")


# ── Orchestrator ─────────────────────────────────────────
def run_all_evaluation(X_test, y_test, feature_names: list[str]):
    """Run evaluation charts for all persisted models."""
    model_map = {
        "Linear Regression":   "linear_regression",
        "Ridge Regression":    "ridge_regression",
        "Random Forest":       "random_forest",
        "Gradient Boosting":   "gradient_boosting",
        "Decision Tree":       "decision_tree",
    }

    print("\n📈  Running model evaluation charts …")
    for display_name, slug in model_map.items():
        model_path = MODELS / f"{slug}.joblib"
        if not model_path.exists():
            continue
        model  = joblib.load(model_path)
        y_pred = model.predict(X_test)
        plot_actual_vs_predicted(y_test, y_pred, display_name)
        plot_residuals(y_test, y_pred, display_name)
        plot_feature_importance(display_name)

    results_csv = REPORTS / "model_comparison.csv"
    if results_csv.exists():
        plot_model_comparison(results_csv)

    print(f"✅  Evaluation figures saved → {FIGURES}")


if __name__ == "__main__":
    from preprocess import load_and_preprocess
    DATA = ROOT / "data" / "employees.csv"
    (X_train, X_test, y_train, y_test,
     feature_names, preprocessor, df) = load_and_preprocess(DATA)
    run_all_evaluation(X_test, y_test, feature_names)
