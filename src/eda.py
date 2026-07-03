"""
=============================================================
  eda.py — Exploratory Data Analysis & Visualisations
=============================================================
Generates and saves all EDA charts to reports/figures/.

Usage:
    python src/eda.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker 
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────
PALETTE  = "viridis"
BG       = "#0E1117"
FG       = "#FAFAFA"
ACC      = "#4F8BF9"
GRID_CLR = "#2E3748"

def _style():
    """Apply a dark, professional style to all matplotlib charts."""
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    "#1A1F2E",
        "axes.edgecolor":    GRID_CLR,
        "axes.labelcolor":   FG,
        "text.color":        FG,
        "xtick.color":       FG,
        "ytick.color":       FG,
        "grid.color":        GRID_CLR,
        "grid.linewidth":    0.5,
        "font.family":       "sans-serif",
        "axes.titlesize":    14,
        "axes.labelsize":    11,
        "legend.facecolor":  "#1A1F2E",
        "legend.edgecolor":  GRID_CLR,
        "legend.labelcolor": FG,
    })

def _save(fig, name: str):
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {path.name}")


# ── Individual plots ─────────────────────────────────────

def plot_salary_distribution(df: pd.DataFrame):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Salary Distribution", fontsize=16, color=FG, fontweight="bold")

    # Histogram + KDE
    ax = axes[0]
    ax.hist(df["salary"], bins=50, color=ACC, edgecolor="none", alpha=0.8, density=True)
    df["salary"].plot.kde(ax=ax, color="#FF6B6B", linewidth=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.set_xlabel("Salary (USD)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution + KDE")
    ax.grid(True, axis="y")

    # Box plot grouped by education
    ax = axes[1]
    order = ["High School", "Bachelor's", "Master's", "PhD"]
    colors = [ACC, "#FF6B6B", "#4ECDC4", "#FFE66D"]
    data_by_edu = [df[df["education_level"] == edu]["salary"].values for edu in order]
    bp = ax.boxplot(data_by_edu, patch_artist=True, medianprops={"color": "white", "linewidth": 2})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)
    ax.set_xticklabels(order, rotation=15)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.set_title("Salary by Education Level")
    ax.grid(True, axis="y")

    plt.tight_layout()
    _save(fig, "01_salary_distribution")


def plot_salary_by_department(df: pd.DataFrame):
    _style()
    fig, ax = plt.subplots(figsize=(12, 6))
    dept_stats = (
        df.groupby("department")["salary"]
        .agg(["mean", "median", "std"])
        .sort_values("mean", ascending=True)
    )
    y_pos = np.arange(len(dept_stats))
    bars  = ax.barh(y_pos, dept_stats["mean"], color=ACC, alpha=0.85, height=0.5)
    ax.errorbar(dept_stats["mean"], y_pos,
                xerr=dept_stats["std"],
                fmt="none", color=FG, alpha=0.5, capsize=4, linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(dept_stats.index)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.set_title("Average Salary by Department (± 1 Std Dev)", fontsize=14)
    ax.set_xlabel("Mean Salary (USD)")
    ax.grid(True, axis="x")
    plt.tight_layout()
    _save(fig, "02_salary_by_department")


def plot_experience_salary(df: pd.DataFrame):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Experience vs Salary", fontsize=16, color=FG, fontweight="bold")

    # Scatter: experience vs salary, coloured by department
    ax = axes[0]
    depts = df["department"].unique()
    cmap  = plt.get_cmap("tab10", len(depts))
    for i, dept in enumerate(sorted(depts)):
        sub = df[df["department"] == dept]
        ax.scatter(sub["years_experience"], sub["salary"],
                   alpha=0.5, s=15, color=cmap(i), label=dept)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary (USD)")
    ax.set_title("Experience vs Salary (by Department)")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True)

    # Bar: average salary by job title (ordered)
    ax = axes[1]
    order = ["Junior", "Mid-Level", "Senior", "Lead", "Manager", "Director"]
    means = [df[df["job_title"] == t]["salary"].mean() for t in order]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(order)))
    bars = ax.bar(order, means, color=colors, edgecolor="none")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.set_xticklabels(order, rotation=20)
    ax.set_title("Average Salary by Job Title")
    ax.set_ylabel("Salary (USD)")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1500,
                f"${val/1000:.0f}K", ha="center", va="bottom", fontsize=9, color=FG)
    ax.grid(True, axis="y")

    plt.tight_layout()
    _save(fig, "03_experience_salary")


def plot_correlation_heatmap(df: pd.DataFrame):
    _style()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        corr, mask=mask, cmap=cmap, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 8},
        linewidths=0.5, linecolor=GRID_CLR,
        ax=ax, cbar_kws={"shrink": 0.8},
        square=True,
    )
    ax.set_title("Correlation Matrix — Numeric Features", fontsize=14, pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    _save(fig, "04_correlation_heatmap")


def plot_performance_salary(df: pd.DataFrame):
    _style()
    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(
        df["performance_score"], df["salary"],
        c=df["years_experience"], cmap="viridis",
        alpha=0.6, s=20, edgecolors="none"
    )
    cb = plt.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("Years of Experience", color=FG)
    cb.ax.yaxis.set_tick_params(color=FG)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=FG)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.set_xlabel("Performance Score (1–5)")
    ax.set_ylabel("Salary (USD)")
    ax.set_title("Performance Score vs Salary (coloured by Experience)")
    ax.grid(True)
    plt.tight_layout()
    _save(fig, "05_performance_salary")


def plot_location_salary(df: pd.DataFrame):
    _style()
    fig, ax = plt.subplots(figsize=(10, 5))
    loc_median = df.groupby("location")["salary"].median().sort_values(ascending=False)
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(loc_median)))
    bars = ax.bar(loc_median.index, loc_median.values, color=colors, edgecolor="none")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.set_xlabel("Location")
    ax.set_ylabel("Median Salary (USD)")
    ax.set_title("Median Salary by Location")
    for bar, val in zip(bars, loc_median.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1500,
                f"${val/1000:.0f}K", ha="center", va="bottom", fontsize=9, color=FG)
    ax.grid(True, axis="y")
    plt.tight_layout()
    _save(fig, "06_location_salary")


def plot_gender_salary(df: pd.DataFrame):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Gender & Salary Analysis", fontsize=16, color=FG, fontweight="bold")

    # Violin plot
    ax = axes[0]
    genders = df["gender"].unique()
    data    = [df[df["gender"] == g]["salary"].values for g in genders]
    parts   = ax.violinplot(data, positions=range(len(genders)),
                            showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(ACC)
        pc.set_alpha(0.7)
    ax.set_xticks(range(len(genders)))
    ax.set_xticklabels(genders)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.set_title("Salary Distribution by Gender")
    ax.grid(True, axis="y")

    # Mean salary + count
    ax = axes[1]
    stats = df.groupby("gender")["salary"].agg(["mean", "count"])
    ax.bar(stats.index, stats["mean"], color=[ACC, "#FF6B6B", "#4ECDC4"], alpha=0.85)
    for i, (g, row) in enumerate(stats.iterrows()):
        ax.text(i, row["mean"] + 1500, f"n={row['count']}", ha="center", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.set_title("Mean Salary by Gender")
    ax.grid(True, axis="y")

    plt.tight_layout()
    _save(fig, "07_gender_salary")


def plot_top_correlations(df: pd.DataFrame):
    """Bar chart of top numeric correlations with salary."""
    _style()
    numeric_df = df.select_dtypes(include=[np.number])
    corr_salary = numeric_df.corr()["salary"].drop("salary").sort_values(key=abs, ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [ACC if v > 0 else "#FF6B6B" for v in corr_salary]
    ax.barh(corr_salary.index, corr_salary.values, color=colors, alpha=0.85)
    ax.axvline(0, color=FG, linewidth=0.8)
    ax.set_xlabel("Pearson Correlation with Salary")
    ax.set_title("Feature Correlations with Salary")
    ax.grid(True, axis="x")
    plt.tight_layout()
    _save(fig, "08_feature_correlations")


# ── Orchestrator ─────────────────────────────────────────
def run_all_eda(df: pd.DataFrame):
    print("\n📊  Running EDA plots …")
    plot_salary_distribution(df)
    plot_salary_by_department(df)
    plot_experience_salary(df)
    plot_correlation_heatmap(df)
    plot_performance_salary(df)
    plot_location_salary(df)
    plot_gender_salary(df)
    plot_top_correlations(df)
    print(f"✅  All EDA figures saved → {FIGURES}")


if __name__ == "__main__":
    from preprocess import load_raw, engineer_features
    DATA = ROOT / "data" / "employees.csv"
    df   = engineer_features(load_raw(DATA))
    run_all_eda(df)
