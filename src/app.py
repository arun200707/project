"""
=============================================================
  app.py — Streamlit Interactive Analytics Dashboard
=============================================================
Launch with:
    streamlit run src/app.py
"""

import sys
from pathlib import Path
import warnings

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Employee Salary Prediction & Analytics by arun",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS theme ─────────────────────────────────────────────
st.markdown("""
<style>
  /* Global background */
  .stApp { background: #0E1117; }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1A1F2E 0%, #252B3B 100%);
    border: 1px solid #2E3748;
    border-radius: 12px;
    padding: 16px 20px;
  }
  div[data-testid="metric-container"] label {
    color: #8892A4 !important; font-size: 13px !important;
  }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #4F8BF9 !important; font-size: 28px !important; font-weight: 700;
  }

  /* Section headers */
  .section-header {
    background: linear-gradient(135deg, #1A1F2E, #252B3B);
    border-left: 4px solid #4F8BF9;
    border-radius: 8px;
    padding: 12px 20px;
    margin: 20px 0 12px 0;
    font-size: 20px;
    font-weight: 700;
    color: #FAFAFA;
  }

  /* Prediction card */
  .pred-card {
    background: linear-gradient(135deg, #1A1F2E, #0E4C96);
    border: 2px solid #4F8BF9;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin: 16px 0;
  }
  .pred-amount {
    font-size: 52px;
    font-weight: 800;
    color: #4F8BF9;
    letter-spacing: -1px;
  }
  .pred-range {
    color: #8892A4;
    font-size: 15px;
    margin-top: 8px;
  }

  /* Badge */
  .badge {
    display: inline-block;
    background: #4F8BF9;
    color: white;
    border-radius: 12px;
    padding: 3px 12px;
    font-size: 12px;
    font-weight: 600;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] { background: #131720 !important; }
  section[data-testid="stSidebar"] * { color: #FAFAFA !important; }
</style>
""", unsafe_allow_html=True)

# ── Colour palette for Plotly ─────────────────────────────
PLOTLY_COLORS = px.colors.qualitative.Plotly
DARK_TEMPLATE  = dict(
    layout=dict(
        paper_bgcolor="#0E1117",
        plot_bgcolor ="#1A1F2E",
        font=dict(color="#FAFAFA", family="Inter, sans-serif"),
        xaxis=dict(gridcolor="#2E3748", zerolinecolor="#2E3748"),
        yaxis=dict(gridcolor="#2E3748", zerolinecolor="#2E3748"),
    )
)


# ── Data & artifact loading (cached) ─────────────────────
@st.cache_data
def load_data():
    path = ROOT / "data" / "employees.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_ml_artifacts():
    try:
        preprocessor  = joblib.load(ROOT / "models" / "preprocessor.joblib")
        model         = joblib.load(ROOT / "models" / "best_model.joblib")
        feature_names = joblib.load(ROOT / "models" / "feature_names.joblib")
        return preprocessor, model, feature_names
    except FileNotFoundError:
        st.error("❌ Model files not found. Please ensure models are trained.")
        return None, None, None
    except (AttributeError, ImportError, ModuleNotFoundError) as e:
        st.error(f"❌ Model compatibility error (sklearn version mismatch): {e}")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None


@st.cache_data
def load_results():
    path = ROOT / "reports" / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


# ── Helper ────────────────────────────────────────────────
def fmt(n: float) -> str:
    return f"${n:,.0f}"


def predict_salary_st(employee: dict, preprocessor, model) -> dict:
    from preprocess import engineer_features
    df = pd.DataFrame([employee])
    df = engineer_features(df)
    X  = preprocessor.transform(df)
    p  = model.predict(X)[0]
    return {"predicted": p, "lower": p * 0.90, "upper": p * 1.10}


# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 💰 Salary Analytics")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview",
         "📊 EDA",
         "🤖 ML Models",
         "🔮 Predict Salary",
         "🗂️ Data Explorer"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Dataset Info**")
    df_raw = load_data()
    if df_raw is not None:
        st.metric("Employees", f"{len(df_raw):,}")
        st.metric("Features",  f"{df_raw.shape[1]-1}")
        st.metric("Avg Salary", fmt(df_raw["salary"].mean()))
    else:
        st.warning("Run `python src/run_pipeline.py` first!")

    st.markdown("---")
    st.markdown("<small style='color:#8892A4'>Final Year Mini Project<br>AI & Data Science Dept</small>",
                unsafe_allow_html=True)


# ── Load artefacts ────────────────────────────────────────
df          = load_data()
results_df  = load_results()
preprocessor, model, feature_names = load_ml_artifacts()
artifacts_ready = preprocessor is not None
data_ready      = df is not None


# ══════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("Employee Salary Prediction & Analytics System")
    st.markdown("##### AI-powered insights into compensation across roles, departments & locations")
    st.markdown("---")

    if not data_ready:
        st.error("Dataset not found. Please run `python src/run_pipeline.py` first.")
        st.stop()

    # ── KPI row ──────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Employees", f"{len(df):,}")
    c2.metric("Avg Salary",       fmt(df["salary"].mean()))
    c3.metric("Median Salary",    fmt(df["salary"].median()))
    c4.metric("Max Salary",       fmt(df["salary"].max()))
    c5.metric("Departments",      str(df["department"].nunique()))

    st.markdown("")

    # ── Row 1: Salary distribution + dept breakdown ───────
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="section-header">Salary Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x="salary", nbins=60,
                           color_discrete_sequence=["#4F8BF9"],
                           labels={"salary": "Salary (USD)"})
        fig.update_traces(opacity=0.85)
        fig.update_layout(**DARK_TEMPLATE["layout"],
                          xaxis_tickprefix="$", xaxis_tickformat=",.0f",
                          bargap=0.03, height=350, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">By Department</div>', unsafe_allow_html=True)
        dept_avg = df.groupby("department")["salary"].mean().sort_values(ascending=True)
        fig = go.Figure(go.Bar(
            x=dept_avg.values, y=dept_avg.index,
            orientation="h",
            marker=dict(color=dept_avg.values, colorscale="Viridis"),
            text=[fmt(v) for v in dept_avg.values],
            textposition="outside",
        ))
        fig.update_layout(**DARK_TEMPLATE["layout"],
                          xaxis_tickprefix="$", xaxis_tickformat=",.0f",
                          height=350, margin=dict(l=0,r=60,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Job title + Location ───────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Salary by Job Title</div>', unsafe_allow_html=True)
        order = ["Junior", "Mid-Level", "Senior", "Lead", "Manager", "Director"]
        box_df = df[df["job_title"].isin(order)].copy()
        box_df["job_title"] = pd.Categorical(box_df["job_title"], categories=order, ordered=True)
        fig = px.box(box_df.sort_values("job_title"), x="job_title", y="salary",
                     color="job_title",
                     color_discrete_sequence=px.colors.sequential.Viridis)
        fig.update_layout(**DARK_TEMPLATE["layout"],
                          yaxis_tickprefix="$", yaxis_tickformat=",.0f",
                          showlegend=False, height=370,
                          margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Median Salary by Location</div>', unsafe_allow_html=True)
        loc = df.groupby("location")["salary"].median().sort_values(ascending=False)
        fig = px.bar(x=loc.index, y=loc.values,
                     color=loc.values, color_continuous_scale="Plasma",
                     labels={"x": "Location", "y": "Median Salary"})
        fig.update_layout(**DARK_TEMPLATE["layout"],
                          yaxis_tickprefix="$", yaxis_tickformat=",.0f",
                          coloraxis_showscale=False,
                          height=370, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # ── Best model banner ─────────────────────────────────
    if results_df is not None:
        best = results_df.loc[results_df["R²"].idxmax()]
        st.success(
            f"🏆 **Best Model:** {best['Model']}  |  "
            f"R² = **{best['R²']:.4f}**  |  "
            f"RMSE = **${best['RMSE']:,.0f}**  |  "
            f"MAE = **${best['MAE']:,.0f}**"
        )


# ══════════════════════════════════════════════════════════
#  PAGE: EDA
# ══════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    if not data_ready:
        st.error("Dataset not found. Run the pipeline first.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Distributions", "🔗 Correlations",
         "🏢 Department & Location", "👤 Demographics"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Salary Histogram + KDE")
            fig = px.histogram(df, x="salary", nbins=60,
                               marginal="violin",
                               color_discrete_sequence=["#4F8BF9"])
            fig.update_layout(**DARK_TEMPLATE["layout"],
                              xaxis_tickprefix="$", height=380)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Salary vs Experience")
            fig = px.scatter(df, x="years_experience", y="salary",
                             color="department", opacity=0.55, size_max=6,
                             trendline="ols",
                             color_discrete_sequence=PLOTLY_COLORS)
            fig.update_layout(**DARK_TEMPLATE["layout"],
                              yaxis_tickprefix="$", height=380)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Performance Score vs Salary (coloured by Experience)")
        fig = px.scatter(df, x="performance_score", y="salary",
                         color="years_experience",
                         color_continuous_scale="Viridis",
                         opacity=0.6,
                         labels={"performance_score": "Performance Score",
                                 "years_experience": "Yrs Exp"})
        fig.update_layout(**DARK_TEMPLATE["layout"], yaxis_tickprefix="$", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr().round(2)
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig.update_layout(**DARK_TEMPLATE["layout"], height=560)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top Correlations with Salary")
        corr_salary = (numeric_df.corr()["salary"]
                       .drop("salary").abs()
                       .sort_values(ascending=True))
        fig = px.bar(x=corr_salary.values, y=corr_salary.index,
                     orientation="h", color=corr_salary.values,
                     color_continuous_scale="Blues",
                     labels={"x": "Absolute Correlation", "y": "Feature"})
        fig.update_layout(**DARK_TEMPLATE["layout"],
                          coloraxis_showscale=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Salary Distribution by Department")
            fig = px.box(df, x="department", y="salary",
                         color="department",
                         color_discrete_sequence=PLOTLY_COLORS)
            fig.update_layout(**DARK_TEMPLATE["layout"],
                              yaxis_tickprefix="$",
                              showlegend=False,
                              xaxis_tickangle=-30, height=420)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Avg Salary: Department × Job Title")
            pivot = df.pivot_table("salary", "department", "job_title",
                                   aggfunc="mean").fillna(0)
            order = ["Junior", "Mid-Level", "Senior", "Lead", "Manager", "Director"]
            pivot = pivot[[c for c in order if c in pivot.columns]]
            fig = px.imshow(pivot / 1000, text_auto=".0f",
                            color_continuous_scale="Viridis",
                            labels={"color": "Salary (K$)"})
            fig.update_layout(**DARK_TEMPLATE["layout"], height=420)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Salary Bubble Map by Location")
        loc_stats = df.groupby("location").agg(
            avg_salary=("salary", "mean"),
            count=("salary", "count"),
        ).reset_index()
        fig = px.scatter(loc_stats, x="location", y="avg_salary",
                         size="count", color="avg_salary",
                         color_continuous_scale="Plasma",
                         size_max=50)
        fig.update_layout(**DARK_TEMPLATE["layout"],
                          yaxis_tickprefix="$", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Salary by Gender")
            fig = px.violin(df, x="gender", y="salary",
                            color="gender", box=True, points="outliers",
                            color_discrete_sequence=["#4F8BF9", "#FF6B6B", "#4ECDC4"])
            fig.update_layout(**DARK_TEMPLATE["layout"],
                              yaxis_tickprefix="$",
                              showlegend=False, height=420)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Education Level vs Salary")
            edu_order = ["High School", "Bachelor's", "Master's", "PhD"]
            edu_df = df[df["education_level"].isin(edu_order)].copy()
            edu_df["education_level"] = pd.Categorical(
                edu_df["education_level"], categories=edu_order, ordered=True)
            fig = px.box(edu_df.sort_values("education_level"),
                         x="education_level", y="salary",
                         color="education_level",
                         color_discrete_sequence=px.colors.sequential.Plasma)
            fig.update_layout(**DARK_TEMPLATE["layout"],
                              yaxis_tickprefix="$",
                              showlegend=False, height=420)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Age Distribution by Department")
        fig = px.violin(df, x="department", y="age",
                        color="department",
                        color_discrete_sequence=PLOTLY_COLORS,
                        box=True)
        fig.update_layout(**DARK_TEMPLATE["layout"],
                          xaxis_tickangle=-30, showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════
#  PAGE: ML MODELS
# ══════════════════════════════════════════════════════════
elif page == "🤖 ML Models":
    st.title("🤖 Machine Learning Model Analysis")
    if results_df is None:
        st.error("No model results found. Run `python src/run_pipeline.py` first.")
        st.stop()

    # ── Leaderboard ──────────────────────────────────────
    st.markdown('<div class="section-header">Model Leaderboard</div>', unsafe_allow_html=True)

    best_idx = results_df["R²"].idxmax()
    for i, row in results_df.sort_values("R²", ascending=False).iterrows():
        badge = ' <span class="badge">🏆 Best</span>' if i == best_idx else ""
        cols  = st.columns([2, 1, 1, 1, 1, 1])
        cols[0].markdown(f"**{row['Model']}**{badge}", unsafe_allow_html=True)
        cols[1].metric("R²",   f"{row['R²']:.4f}")
        cols[2].metric("RMSE", fmt(row["RMSE"]))
        cols[3].metric("MAE",  fmt(row["MAE"]))
        cols[4].metric("MAPE", f"{row['MAPE (%)']:.1f}%")
        cols[5].metric("Time", f"{row['Train Time']:.2f}s")
        st.divider()

    # ── Comparison charts ─────────────────────────────────
    tab1, tab2 = st.tabs(["📊 Metric Comparison", "📈 Actual vs Predicted"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(results_df.sort_values("R²"),
                         x="R²", y="Model", orientation="h",
                         color="R²", color_continuous_scale="Viridis",
                         title="R² Score (higher = better)")
            fig.update_layout(**DARK_TEMPLATE["layout"],
                              coloraxis_showscale=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(results_df.sort_values("RMSE", ascending=True),
                         x="RMSE", y="Model", orientation="h",
                         color="RMSE", color_continuous_scale="RdYlGn_r",
                         title="RMSE (lower = better)")
            fig.update_layout(**DARK_TEMPLATE["layout"],
                              xaxis_tickprefix="$",
                              coloraxis_showscale=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

        # Radar chart
        st.subheader("Multi-Metric Radar Chart")
        metrics = ["R²", "CV R² Mean"]
        avail_m = [m for m in metrics if m in results_df.columns]
        fig = go.Figure()
        for _, row in results_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[m] for m in avail_m],
                theta=avail_m,
                fill="toself",
                name=row["Model"],
            ))
        fig.update_layout(**DARK_TEMPLATE["layout"],
                          polar=dict(
                              bgcolor="#1A1F2E",
                              radialaxis=dict(color="#FAFAFA"),
                              angularaxis=dict(color="#FAFAFA"),
                          ),
                          height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if not artifacts_ready or not data_ready:
            st.warning("Models not loaded. Run the pipeline first.")
        else:
            from preprocess import load_and_preprocess
            (X_tr, X_te, y_tr, y_te,
             fn, pp, _) = load_and_preprocess(ROOT / "data" / "employees.csv")

            available_models = {
                "Random Forest":     "random_forest",
                "Gradient Boosting": "gradient_boosting",
                "Linear Regression": "linear_regression",
                "Ridge Regression":  "ridge_regression",
                "Decision Tree":     "decision_tree",
            }
            sel = st.selectbox("Select model", list(available_models.keys()))
            slug = available_models[sel]
            mp   = ROOT / "models" / f"{slug}.joblib"
            if mp.exists():
                m      = joblib.load(mp)
                y_pred = m.predict(X_te)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_te, y=y_pred,
                    mode="markers",
                    marker=dict(color="#4F8BF9", opacity=0.5, size=5),
                    name="Predictions"
                ))
                mn, mx = min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())
                fig.add_trace(go.Scatter(
                    x=[mn, mx], y=[mn, mx],
                    mode="lines",
                    line=dict(color="#FF6B6B", dash="dash", width=2),
                    name="Perfect fit"
                ))
                fig.update_layout(
                    **DARK_TEMPLATE["layout"],
                    xaxis=dict(tickprefix="$", **DARK_TEMPLATE["layout"]["xaxis"]),
                    yaxis=dict(tickprefix="$", **DARK_TEMPLATE["layout"]["yaxis"]),
                    title=f"Actual vs Predicted — {sel}",
                    height=460,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Residuals
                resid = y_te - y_pred
                fig2 = px.scatter(x=y_pred, y=resid,
                                  labels={"x": "Predicted Salary", "y": "Residual"},
                                  color=np.abs(resid),
                                  color_continuous_scale="RdYlGn_r",
                                  title="Residuals vs Predicted",
                                  opacity=0.6)
                fig2.add_hline(y=0, line_dash="dash", line_color="#FF6B6B")
                fig2.update_layout(**DARK_TEMPLATE["layout"],
                                   xaxis_tickprefix="$", yaxis_tickprefix="$",
                                   height=350)
                st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════
#  PAGE: PREDICT SALARY
# ══════════════════════════════════════════════════════════
elif page == "🔮 Predict Salary":
    st.title("🔮 Predict Employee Salary")
    if not artifacts_ready:
        st.error("Models not trained. Run `python src/run_pipeline.py` first.")
        st.stop()

    st.markdown("Fill in the employee details below to get an instant salary prediction.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Personal Info")
        age    = st.slider("Age", 22, 65, 30)
        gender = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
        edu    = st.selectbox("Education Level",
                              ["High School", "Bachelor's", "Master's", "PhD"],
                              index=1)

    with col2:
        st.subheader("🏢 Work Info")
        dept     = st.selectbox("Department",
                                ["Engineering", "Sales", "Marketing", "Finance",
                                 "HR", "Operations", "Data Science", "Product"])
        title    = st.selectbox("Job Title",
                                ["Junior", "Mid-Level", "Senior",
                                 "Lead", "Manager", "Director"])
        location = st.selectbox("Location",
                                ["New York", "San Francisco", "Chicago",
                                 "Austin", "Seattle", "Boston", "Remote"])

    with col3:
        st.subheader("📈 Performance")
        yrs_exp     = st.slider("Years of Experience", 0, 35, 5)
        yrs_company = st.slider("Years at Company",    0, 30, 2)
        perf        = st.slider("Performance Score",   1.0, 5.0, 3.5, 0.1)
        projects    = st.slider("Num Projects",        1, 25, 8)
        training    = st.slider("Training Hours",      0, 120, 40)
        certs       = st.slider("Certifications",      0, 10, 1)
        team_sz     = st.slider("Team Size",           1, 60, 8)

    st.markdown("---")

    if st.button("🚀 Predict Salary", use_container_width=True, type="primary"):
        employee = {
            "age": age, "gender": gender,
            "education_level": edu,
            "department": dept, "job_title": title,
            "location": location,
            "years_experience": yrs_exp,
            "years_at_company": yrs_company,
            "performance_score": perf,
            "num_projects": projects,
            "training_hours": training,
            "certifications": certs,
            "team_size": team_sz,
        }

        result = predict_salary_st(employee, preprocessor, model)
        p, lo, hi = result["predicted"], result["lower"], result["upper"]

        st.markdown(f"""
        <div class="pred-card">
          <div style="color:#8892A4;font-size:16px;margin-bottom:8px">
            Predicted Annual Salary
          </div>
          <div class="pred-amount">{fmt(p)}</div>
          <div class="pred-range">
            90% Confidence Range: {fmt(lo)} – {fmt(hi)}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Comparable peers
        if data_ready:
            peers = df[
                (df["department"] == dept) &
                (df["job_title"]  == title) &
                (df["years_experience"].between(max(0, yrs_exp-3), yrs_exp+3))
            ]["salary"]

            if len(peers) > 3:
                col1, col2, col3 = st.columns(3)
                col1.metric("Peer Median",  fmt(peers.median()),
                            delta=fmt(p - peers.median()))
                col2.metric("Peer 25th %",  fmt(peers.quantile(0.25)))
                col3.metric("Peer 75th %",  fmt(peers.quantile(0.75)))

                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=peers, nbinsx=25,
                    name="Comparable Peers",
                    marker_color="#4F8BF9", opacity=0.7
                ))
                fig.add_vline(x=p,     line_color="#FFE66D", line_width=3,
                              line_dash="solid",  annotation_text="Your Prediction")
                fig.add_vline(x=peers.median(), line_color="#FF6B6B",
                              line_dash="dash", annotation_text="Peer Median")
                fig.update_layout(
                    **DARK_TEMPLATE["layout"],
                    title=f"Salary Distribution — {dept} / {title} Peers",
                    xaxis_tickprefix="$", height=370,
                )
                st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════
#  PAGE: DATA EXPLORER
# ══════════════════════════════════════════════════════════
elif page == "🗂️ Data Explorer":
    st.title("🗂️ Data Explorer")
    if not data_ready:
        st.error("Dataset not found. Run the pipeline first.")
        st.stop()

    # ── Filters ──────────────────────────────────────────
    with st.expander("⚙️ Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        sel_dept  = col1.multiselect("Department", sorted(df["department"].unique()),
                                     default=sorted(df["department"].unique()))
        sel_edu   = col2.multiselect("Education",  sorted(df["education_level"].unique()),
                                     default=sorted(df["education_level"].unique()))
        sal_range = col3.slider("Salary Range",
                                int(df["salary"].min()), int(df["salary"].max()),
                                (int(df["salary"].min()), int(df["salary"].max())))

    filtered = df[
        df["department"].isin(sel_dept) &
        df["education_level"].isin(sel_edu) &
        df["salary"].between(*sal_range)
    ]

    st.info(f"Showing **{len(filtered):,}** of {len(df):,} records")

    # Summary stats
    st.subheader("Summary Statistics")
    st.dataframe(
        filtered.describe().round(2).T,
        use_container_width=True,
    )

    # Raw table
    st.subheader("Raw Data")
    st.dataframe(
        filtered.style.format({"salary": "${:,.0f}", "performance_score": "{:.1f}"}),
        use_container_width=True,
        height=400,
    )

    # Download
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_employees.csv",
        mime="text/csv",
        use_container_width=True,
    )
