import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ============================================================
# Loan Approval Analytics - Streamlit Dashboard
# ============================================================

st.set_page_config(
    page_title="Loan Approval Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

sns.set_theme(style="whitegrid")

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    raw = pd.read_csv("loan_sanction_train.csv")
    clean = raw.copy()

    for col in ["Gender", "Married", "Dependents", "Self_Employed"]:
        clean[col] = clean[col].fillna(clean[col].mode()[0])

    for col in ["LoanAmount", "Loan_Amount_Term", "Credit_History"]:
        clean[col] = clean[col].fillna(clean[col].median())

    clean["Dependents"] = clean["Dependents"].replace("3+", "3").astype(int)
    return raw, clean

try:
    df, clean_df = load_data()
except FileNotFoundError:
    st.error(
        "Dataset not found. Put 'loan_sanction_train.csv' in the same folder as app.py."
    )
    st.stop()

# -----------------------------
# Feature engineering
# -----------------------------
@st.cache_data
def create_features(clean):
    fe = clean.copy()

    fe["Total_Household_Income"] = (
        fe["ApplicantIncome"] + fe["CoapplicantIncome"]
    )

    q1, q3 = fe["Total_Household_Income"].quantile([0.25, 0.75])
    fe["Income_Category"] = pd.cut(
        fe["Total_Household_Income"],
        [-np.inf, q1, q3, np.inf],
        labels=["Low", "Medium", "High"]
    )

    lq1, lq3 = fe["LoanAmount"].quantile([0.25, 0.75])
    fe["Loan_Amount_Category"] = pd.cut(
        fe["LoanAmount"],
        [-np.inf, lq1, lq3, np.inf],
        labels=["Low", "Medium", "High"]
    )

    fe["Income_to_Loan_Ratio"] = (
        fe["Total_Household_Income"] /
        fe["LoanAmount"].replace(0, np.nan)
    )

    # Analytical approximation only; no interest rate is present in the dataset.
    fe["Estimated_Monthly_EMI"] = (
        fe["LoanAmount"] * 1000 /
        fe["Loan_Amount_Term"].replace(0, np.nan)
    )

    fe["EMI_Burden_Ratio"] = (
        fe["Estimated_Monthly_EMI"] /
        fe["Total_Household_Income"].replace(0, np.nan)
    )

    fe["EMI_Burden_Category"] = pd.cut(
        fe["EMI_Burden_Ratio"],
        [-np.inf, 0.05, 0.10, np.inf],
        labels=["Low", "Medium", "High"]
    )

    median_ratio = fe["Income_to_Loan_Ratio"].median()

    fe["Applicant_Risk_Category"] = np.select(
        [
            fe["Credit_History"].eq(0),
            (fe["Credit_History"].eq(1)) &
            (fe["Income_to_Loan_Ratio"] < median_ratio)
        ],
        ["Higher Risk", "Moderate Risk"],
        default="Lower Risk"
    )

    return fe

fe = create_features(clean_df)

# -----------------------------
# Helper functions
# -----------------------------
def approval_pivot(col):
    p = pd.crosstab(clean_df[col], clean_df["Loan_Status"])
    p["Total"] = p.sum(axis=1)
    p["Approval_Rate_%"] = p.get("Y", 0) / p["Total"] * 100
    return p.round(2)

def show_figure(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def approval_rate_table(col):
    rate = (
        clean_df.groupby(col)["Loan_Status"]
        .apply(lambda s: (s == "Y").mean() * 100)
        .round(2)
        .to_frame("Approval_Rate_%")
    )
    return rate

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🏦 Loan Approval Analytics")
st.sidebar.caption("Interactive EDA Dashboard")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Dashboard",
        "📊 Dataset Overview",
        "🧹 Data Cleaning",
        "📈 Univariate Analysis",
        "🔄 Bivariate Analysis",
        "🌐 Multivariate Analysis",
        "📋 Pivot & Crosstab",
        "🔥 Correlation Analysis",
        "🧪 Statistical Analysis",
        "⚙️ Feature Engineering",
        "💼 Business Insights",
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Project: Loan Approval Analytics using Python\n\n"
    "Tools: Pandas, NumPy, Matplotlib, Seaborn, SciPy and Streamlit"
)

# ============================================================
# 1. DASHBOARD
# ============================================================
if page == "🏠 Dashboard":
    st.title("🏦 Loan Approval Analytics")
    st.subheader("Exploratory Data Analysis using Python")

    st.write(
        "An interactive analysis of loan applications to identify "
        "patterns associated with loan approval and rejection."
    )

    approved = int((clean_df["Loan_Status"] == "Y").sum())
    rejected = int((clean_df["Loan_Status"] == "N").sum())
    approval_rate = approved / len(clean_df) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Applications", f"{len(clean_df):,}")
    c2.metric("Approved", f"{approved:,}")
    c3.metric("Rejected", f"{rejected:,}")
    c4.metric("Approval Rate", f"{approval_rate:.2f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Loan Status Distribution")
        counts = clean_df["Loan_Status"].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(["Approved (Y)", "Rejected (N)"],
               [counts.get("Y", 0), counts.get("N", 0)])
        ax.set_ylabel("Applications")
        ax.set_title("Loan Approval Status")
        show_figure(fig)

    with col2:
        st.markdown("### Credit History vs Approval")
        rate = approval_rate_table("Credit_History")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(rate.index.astype(str), rate["Approval_Rate_%"])
        ax.set_xlabel("Credit History")
        ax.set_ylabel("Approval Rate (%)")
        ax.set_title("Observed Approval Rate by Credit History")
        show_figure(fig)

    st.markdown("### Key Project Findings")
    st.info(
        "• Credit history shows a very large difference in observed approval rates.\n\n"
        "• Semiurban applicants have the highest observed approval rate among property-area groups.\n\n"
        "• Applicant income and loan amount have a moderate positive relationship.\n\n"
        "• Mean applicant income does not show a statistically significant difference between approval groups in the Welch t-test.\n\n"
        "• These are observed associations, not causal conclusions."
    )

# ============================================================
# 2. DATASET OVERVIEW
# ============================================================
elif page == "📊 Dataset Overview":
    st.title("📊 Dataset Understanding")

    c1, c2 = st.columns(2)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### Columns")
    st.write(df.columns.tolist())

    st.markdown("### Data Types")
    st.dataframe(
        df.dtypes.astype(str).to_frame("dtype"),
        use_container_width=True
    )

    st.markdown("### Summary Statistics")
    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True
    )

# ============================================================
# 3. DATA CLEANING
# ============================================================
elif page == "🧹 Data Cleaning":
    st.title("🧹 Data Quality Assessment & Cleaning")

    missing = pd.DataFrame({
        "Missing_Count": df.isna().sum(),
        "Missing_Percent": (df.isna().mean() * 100).round(2)
    }).query("Missing_Count > 0").sort_values(
        "Missing_Count", ascending=False
    )

    st.markdown("### Missing Values Before Cleaning")
    if missing.empty:
        st.success("No missing values found.")
    else:
        st.dataframe(missing, use_container_width=True)

    duplicate_count = int(df.duplicated().sum())
    st.metric("Duplicate Rows", duplicate_count)

    st.markdown("### Cleaning Decisions")
    st.write(
        """
        - No duplicate rows are removed because the duplicate check found none.
        - Missing categorical values are filled with the mode.
        - Missing numerical values are filled with the median because income/loan
          variables are skewed and the median is less sensitive to extreme values.
        - `Dependents` is standardized from `3+` to numeric `3`.
        - No raw records are manually edited.
        - Outliers are retained because extreme income/loan values can be legitimate observations.
        """
    )

    st.markdown("### Missing Values After Cleaning")
    after = clean_df.isna().sum().to_frame("Missing")
    st.dataframe(after, use_container_width=True)

    st.metric("Duplicates After Cleaning", int(clean_df.duplicated().sum()))

    st.markdown("### Outlier Analysis")
    num_cols = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term"
    ]

    out = []
    for col in num_cols:
        q1, q3 = clean_df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((clean_df[col] < lo) | (clean_df[col] > hi)).sum())
        out.append([col, q1, q3, lo, hi, n])

    out_df = pd.DataFrame(
        out,
        columns=[
            "Feature", "Q1", "Q3",
            "Lower_Bound", "Upper_Bound", "Outlier_Count"
        ]
    )
    st.dataframe(out_df.round(2), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(
        [clean_df[c].dropna() for c in num_cols],
        tick_labels=num_cols
    )
    ax.set_title("Numerical Outlier Analysis")
    ax.tick_params(axis="x", rotation=20)
    show_figure(fig)

# ============================================================
# 4. UNIVARIATE
# ============================================================
elif page == "📈 Univariate Analysis":
    st.title("📈 Univariate Analysis")

    st.markdown("### Numerical Distributions")
    num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, num_cols):
        ax.hist(clean_df[col], bins=30)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
    plt.tight_layout()
    show_figure(fig)

    st.markdown("### Categorical Distributions")
    cat_cols = [
        "Gender", "Married", "Education",
        "Self_Employed", "Property_Area", "Loan_Status"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.ravel(), cat_cols):
        counts = clean_df[col].value_counts()
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_title(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    show_figure(fig)

    st.markdown("### Loan Status Summary")
    status_table = clean_df["Loan_Status"].value_counts().to_frame("Count")
    status_table["Percent"] = (
        clean_df["Loan_Status"].value_counts(normalize=True) * 100
    ).round(2)
    st.dataframe(status_table, use_container_width=True)

    st.info(
        "Observation: 422 of 614 applications are approved (68.73%) "
        "and 192 are rejected (31.27%). Approval is the majority class."
    )

# ============================================================
# 5. BIVARIATE
# ============================================================
elif page == "🔄 Bivariate Analysis":
    st.title("🔄 Bivariate Analysis")

    st.markdown("### Applicant Income vs Loan Amount")
    fig, ax = plt.subplots(figsize=(8, 5))
    for status, grp in clean_df.groupby("Loan_Status"):
        ax.scatter(
            grp["ApplicantIncome"],
            grp["LoanAmount"],
            label=status,
            alpha=0.6
        )
    ax.set_xlabel("Applicant Income")
    ax.set_ylabel("Loan Amount")
    ax.set_title("Applicant Income vs Loan Amount")
    ax.legend(title="Loan Status")
    show_figure(fig)

    st.markdown("### Approval Rates by Applicant Features")
    features = [
        "Gender", "Married", "Education",
        "Self_Employed", "Property_Area",
        "Dependents", "Credit_History"
    ]

    selected_feature = st.selectbox(
        "Select a feature",
        features
    )

    rate = approval_rate_table(selected_feature)
    st.dataframe(rate, use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(
        rate.index.astype(str),
        rate["Approval_Rate_%"]
    )
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Approval Rate (%)")
    ax.set_title(f"Observed Approval Rate by {selected_feature}")
    ax.tick_params(axis="x", rotation=20)
    show_figure(fig)

    st.markdown("### Bivariate Observations")
    st.write(
        """
        - Applicant income and loan amount have a moderate positive relationship, but with substantial variation.
        - Credit history shows a very large difference in observed approval rates.
        - Semiurban applicants have the highest observed approval rate among property-area groups.
        - Married applicants and graduates show higher observed approval rates than their comparison groups.
        - Male and female approval rates are relatively close.
        - Self-employment shows little difference in approval rate in this dataset.
        """
    )
    st.caption("These are observed associations, not causal conclusions.")

# ============================================================
# 6. MULTIVARIATE
# ============================================================
elif page == "🌐 Multivariate Analysis":
    st.title("🌐 Multivariate Analysis")

    st.markdown("### Numerical Relationships")
    cols = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Credit_History"
    ]

    fig = plt.figure(figsize=(10, 10))
    axes = pd.plotting.scatter_matrix(
        clean_df[cols],
        figsize=(10, 10),
        diagonal="hist",
        alpha=0.6
    )
    plt.suptitle("Multivariate Numerical Relationships", y=1.02)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("### Loan Status Across Multiple Factors")
    multi_col = st.selectbox(
        "Select grouping variable",
        ["Property_Area", "Education", "Married", "Credit_History"]
    )

    cross = pd.crosstab(
        clean_df[multi_col],
        clean_df["Loan_Status"],
        normalize="index"
    ).mul(100).round(2)

    st.dataframe(cross, use_container_width=True)

# ============================================================
# 7. PIVOT & CROSSTAB
# ============================================================
elif page == "📋 Pivot & Crosstab":
    st.title("📋 Group-Based Analysis, Pivot Tables & Crosstabs")

    selected = st.selectbox(
        "Select feature for approval pivot",
        ["Property_Area", "Education", "Credit_History", "Married"]
    )

    st.markdown(f"### Approval Pivot — {selected}")
    st.dataframe(
        approval_pivot(selected),
        use_container_width=True
    )

    st.markdown("### Crosstab: Property Area × Loan Status")
    st.dataframe(
        pd.crosstab(
            clean_df["Property_Area"],
            clean_df["Loan_Status"],
            margins=True
        ),
        use_container_width=True
    )

# ============================================================
# 8. CORRELATION
# ============================================================
elif page == "🔥 Correlation Analysis":
    st.title("🔥 Correlation Analysis")

    numeric_cols = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Dependents"
    ]

    corr = clean_df[numeric_cols].corr()

    st.markdown("### Correlation Matrix")
    st.dataframe(corr.round(3), use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(corr, aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(
                j, i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center"
            )

    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    show_figure(fig)

    st.info(
        "Observation: Applicant income and loan amount have a moderate "
        "positive correlation (about 0.565). The remaining numerical "
        "relationships are mostly weak. Correlation indicates association, not causation."
    )

# ============================================================
# 9. STATISTICAL ANALYSIS
# ============================================================
elif page == "🧪 Statistical Analysis":
    st.title("🧪 Statistical Validation")

    st.markdown("## Chi-Square Tests")

    features = [
        "Gender", "Married", "Education",
        "Self_Employed", "Property_Area",
        "Dependents", "Credit_History"
    ]

    chi = []
    for col in features:
        tab = pd.crosstab(clean_df[col], clean_df["Loan_Status"])
        stat, p, dof, expected = stats.chi2_contingency(tab)
        chi.append([
            col,
            stat,
            p,
            dof,
            "Reject H0" if p < 0.05 else "Fail to Reject H0"
        ])

    chi_results = pd.DataFrame(
        chi,
        columns=["Feature", "Chi2", "p_value", "df", "Decision"]
    )

    st.dataframe(
        chi_results.style.format({
            "Chi2": "{:.4f}",
            "p_value": "{:.6f}"
        }),
        use_container_width=True
    )

    st.caption(
        "At α=0.05, statistical evidence of association is observed for "
        "Married, Education, Property_Area and Credit_History in this sample."
    )

    st.markdown("---")
    st.markdown("## Welch's t-test — Applicant Income")

    approved = clean_df.loc[
        clean_df["Loan_Status"] == "Y", "ApplicantIncome"
    ]
    rejected = clean_df.loc[
        clean_df["Loan_Status"] == "N", "ApplicantIncome"
    ]

    t_stat, p_value = stats.ttest_ind(
        approved,
        rejected,
        equal_var=False
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Approved Mean", f"{approved.mean():.2f}")
    c2.metric("Rejected Mean", f"{rejected.mean():.2f}")
    c3.metric("p-value", f"{p_value:.6f}")

    st.write(f"**t-statistic:** {t_stat:.4f}")
    st.write(
        "**Decision:** " +
        ("Reject H0" if p_value < 0.05 else "Fail to Reject H0")
    )

    st.info(
        "Interpretation: the p-value is approximately 0.913, so there is "
        "not sufficient evidence at α=0.05 that mean applicant income "
        "differs between approved and rejected groups."
    )

    st.markdown("---")
    st.markdown("## Covariance")

    numeric = clean_df[
        [
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History"
        ]
    ]

    st.dataframe(numeric.cov().round(3), use_container_width=True)

    st.markdown("## 95% Confidence Interval for Mean Applicant Income")

    x = clean_df["ApplicantIncome"]
    ci = stats.t.interval(
        0.95,
        len(x) - 1,
        loc=x.mean(),
        scale=stats.sem(x)
    )

    st.write(
        f"**95% CI:** ({ci[0]:.2f}, {ci[1]:.2f})"
    )

# ============================================================
# 10. FEATURE ENGINEERING
# ============================================================
elif page == "⚙️ Feature Engineering":
    st.title("⚙️ Feature Engineering")

    st.write(
        "The following features are created from the cleaned dataset "
        "to support applicant segmentation and analytical profiling."
    )

    feature_cols = [
        "Total_Household_Income",
        "Income_Category",
        "Loan_Amount_Category",
        "Income_to_Loan_Ratio",
        "Estimated_Monthly_EMI",
        "EMI_Burden_Ratio",
        "EMI_Burden_Category",
        "Applicant_Risk_Category"
    ]

    st.dataframe(
        fe[feature_cols].head(10),
        use_container_width=True
    )

    st.markdown("### Feature Purpose")
    st.write(
        """
        - **Total_Household_Income:** combines applicant and co-applicant income.
        - **Income_Category:** supports applicant segmentation.
        - **Loan_Amount_Category:** supports loan-size segmentation.
        - **Income_to_Loan_Ratio:** provides a relative capacity-to-request measure.
        - **Estimated_Monthly_EMI / EMI_Burden_Ratio:** analytical burden proxies.
        - **Applicant_Risk_Category:** simple profiling based on credit history and relative income-to-loan capacity.
        """
    )

    st.warning(
        "The estimated EMI is an analytical approximation because the dataset "
        "does not contain an interest rate. Applicant Risk Category is not a formal credit score."
    )

    selected_fe = st.selectbox(
        "Select engineered feature",
        [
            "Income_Category",
            "Loan_Amount_Category",
            "EMI_Burden_Category",
            "Applicant_Risk_Category"
        ]
    )

    st.markdown(f"### {selected_fe} × Loan Status")
    engineered_table = pd.crosstab(
        fe[selected_fe],
        fe["Loan_Status"],
        normalize="index"
    ).mul(100).round(2)

    st.dataframe(engineered_table, use_container_width=True)

# ============================================================
# 11. BUSINESS INSIGHTS
# ============================================================
elif page == "💼 Business Insights":
    st.title("💼 Final Business Insights & Recommendations")

    st.markdown("### Key Insights")

    insights = [
        "Credit history: applicants with credit history = 1 have a substantially higher observed approval rate than those with credit history = 0.",
        "Property area: Semiurban applicants show the highest observed approval rate among the three property-area groups.",
        "Income alone: the t-test does not show a statistically significant difference in mean applicant income between approval groups.",
        "Income and loan amount: there is a moderate positive relationship, meaning larger requested loans tend to occur with higher applicant income.",
        "Class imbalance: approvals are the majority outcome (422/614).",
        "Statistical associations: Married, Education, Property_Area and Credit_History are statistically associated with Loan_Status at α=0.05 in this sample."
    ]

    for i, insight in enumerate(insights, 1):
        st.write(f"**{i}.** {insight}")

    st.markdown("### Recommendations")
    recommendations = [
        "Use credit history together with financial and applicant information.",
        "Monitor approval patterns by property area.",
        "Use household income and income-to-loan ratio for applicant profiling.",
        "Investigate high-income rejected applications for additional decision factors.",
        "Account for class imbalance in any future predictive model."
    ]

    for item in recommendations:
        st.write(f"• {item}")

    st.caption(
        "Recommendations are based on the observed patterns in this dataset. "
        "They should not be interpreted as causal or as a formal lending policy."
    )

    st.markdown("---")
    st.markdown("### Project Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Applications", "614")
    c2.metric("Approved", "422")
    c3.metric("Rejected", "192")

    st.success(
        "The dashboard combines data cleaning, exploratory analysis, "
        "visualization, statistical validation and feature engineering "
        "into one interactive analytical application."
    )
