import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(
    page_title="Loan Approval Analytics",
    page_icon="🏦",
    layout="wide"
)

# -----------------------------
# Load & Clean Data
# -----------------------------
@st.cache_data
def load_data():
    file = Path(__file__).parent / "loan_sanction_train.csv"

    if not file.exists():
        raise FileNotFoundError("loan_sanction_train.csv not found")

    df = pd.read_csv(file)
    clean = df.copy()

    cat_cols = ["Gender", "Married", "Dependents", "Self_Employed"]
    num_cols = ["LoanAmount", "Loan_Amount_Term", "Credit_History"]

    for col in cat_cols:
        clean[col] = clean[col].fillna(clean[col].mode()[0])

    for col in num_cols:
        clean[col] = clean[col].fillna(clean[col].median())

    clean["Dependents"] = (
        clean["Dependents"].replace("3+", "3").astype(int)
    )

    return df, clean


try:
    df, clean = load_data()
except Exception as e:
    st.error(f"Dataset error: {e}")
    st.stop()


# -----------------------------
# Feature Engineering
# -----------------------------
@st.cache_data
def feature_engineering(data):
    fe = data.copy()

    fe["Total_Household_Income"] = (
        fe["ApplicantIncome"] + fe["CoapplicantIncome"]
    )

    q1, q3 = fe["Total_Household_Income"].quantile([.25, .75])
    fe["Income_Category"] = pd.cut(
        fe["Total_Household_Income"],
        [-np.inf, q1, q3, np.inf],
        labels=["Low", "Medium", "High"]
    )

    q1, q3 = fe["LoanAmount"].quantile([.25, .75])
    fe["Loan_Amount_Category"] = pd.cut(
        fe["LoanAmount"],
        [-np.inf, q1, q3, np.inf],
        labels=["Low", "Medium", "High"]
    )

    fe["Income_to_Loan_Ratio"] = (
        fe["Total_Household_Income"] /
        fe["LoanAmount"].replace(0, np.nan)
    )

    fe["Estimated_Monthly_EMI"] = (
        fe["LoanAmount"] * 1000 /
        fe["Loan_Amount_Term"].replace(0, np.nan)
    )

    fe["EMI_Burden_Ratio"] = (
        fe["Estimated_Monthly_EMI"] /
        fe["Total_Household_Income"].replace(0, np.nan)
    )

    median = fe["Income_to_Loan_Ratio"].median()

    fe["Applicant_Risk_Category"] = np.select(
        [
            fe["Credit_History"].eq(0),
            (fe["Credit_History"].eq(1)) &
            (fe["Income_to_Loan_Ratio"] < median)
        ],
        ["Higher Risk", "Moderate Risk"],
        default="Lower Risk"
    )

    return fe


fe = feature_engineering(clean)


# -----------------------------
# Helper
# -----------------------------
def approval_rate(col):
    return (
        clean.groupby(col)["Loan_Status"]
        .apply(lambda x: (x == "Y").mean() * 100)
        .round(2)
    )


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🏦 Loan Approval Analytics")

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
        "💼 Business Insights"
    ]
)


# ============================================================
# 1. DASHBOARD
# ============================================================
if page == "🏠 Dashboard":

    st.title("🏦 Loan Approval Analytics")
    st.subheader("Exploratory Data Analysis using Python")

    approved = (clean["Loan_Status"] == "Y").sum()
    rejected = (clean["Loan_Status"] == "N").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applications", len(clean))
    c2.metric("Approved", approved)
    c3.metric("Rejected", rejected)
    c4.metric("Approval Rate", f"{approved/len(clean)*100:.2f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Loan Status Distribution")
        st.bar_chart(clean["Loan_Status"].value_counts())

    with col2:
        st.write("### Approval Rate by Credit History")
        st.bar_chart(approval_rate("Credit_History"))

    st.info(
        "Credit history shows a large difference in observed approval rates. "
        "Applicant income and loan amount have a moderate positive relationship."
    )


# ============================================================
# 2. DATASET OVERVIEW
# ============================================================
elif page == "📊 Dataset Overview":

    st.title("📊 Dataset Understanding")

    c1, c2 = st.columns(2)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])

    st.write("### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.write("### Data Types")
    st.dataframe(
        df.dtypes.astype(str).to_frame("Data Type"),
        use_container_width=True
    )

    st.write("### Summary Statistics")
    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True
    )


# ============================================================
# 3. DATA CLEANING
# ============================================================
elif page == "🧹 Data Cleaning":

    st.title("🧹 Data Quality & Cleaning")

    st.write("### Missing Values Before Cleaning")
    st.dataframe(
        df.isna().sum().to_frame("Missing Values"),
        use_container_width=True
    )

    st.metric("Duplicate Rows", df.duplicated().sum())

    st.write("### Cleaning Methods")
    st.markdown("""
    - Categorical missing values → Mode
    - Numerical missing values → Median
    - `3+` Dependents → `3`
    - Duplicate rows → Checked
    - Outliers → Retained with justification
    """)

    st.write("### Missing Values After Cleaning")
    st.dataframe(
        clean.isna().sum().to_frame("Missing Values"),
        use_container_width=True
    )

    st.success("Data cleaning completed successfully.")


# ============================================================
# 4. UNIVARIATE ANALYSIS
# ============================================================
elif page == "📈 Univariate Analysis":

    st.title("📈 Univariate Analysis")

    st.write("### Numerical Distributions")

    for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]:
        fig, ax = plt.subplots()
        ax.hist(clean[col], bins=30)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        plt.close(fig)

    st.write("### Categorical Distributions")

    col = st.selectbox(
        "Select Feature",
        ["Gender", "Married", "Education",
         "Self_Employed", "Property_Area", "Loan_Status"]
    )

    st.bar_chart(clean[col].value_counts())


# ============================================================
# 5. BIVARIATE ANALYSIS
# ============================================================
elif page == "🔄 Bivariate Analysis":

    st.title("🔄 Bivariate Analysis")

    fig, ax = plt.subplots()

    for status, group in clean.groupby("Loan_Status"):
        ax.scatter(
            group["ApplicantIncome"],
            group["LoanAmount"],
            label=status,
            alpha=.6
        )

    ax.set_xlabel("Applicant Income")
    ax.set_ylabel("Loan Amount")
    ax.set_title("Applicant Income vs Loan Amount")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    st.write("### Approval Rate by Feature")

    col = st.selectbox(
        "Select Feature",
        [
            "Gender", "Married", "Education",
            "Self_Employed", "Property_Area",
            "Dependents", "Credit_History"
        ]
    )

    st.dataframe(
        approval_rate(col).to_frame("Approval Rate (%)"),
        use_container_width=True
    )


# ============================================================
# 6. MULTIVARIATE ANALYSIS
# ============================================================
elif page == "🌐 Multivariate Analysis":

    st.title("🌐 Multivariate Analysis")

    cols = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Credit_History"
    ]

    st.write("### Numerical Relationships")

    fig = plt.figure(figsize=(10, 8))
    pd.plotting.scatter_matrix(
        clean[cols],
        figsize=(10, 8),
        diagonal="hist"
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    col = st.selectbox(
        "Select Grouping Variable",
        ["Property_Area", "Education",
         "Married", "Credit_History"]
    )

    st.write("### Loan Status Distribution")
    st.dataframe(
        pd.crosstab(
            clean[col],
            clean["Loan_Status"],
            normalize="index"
        ).mul(100).round(2),
        use_container_width=True
    )


# ============================================================
# 7. PIVOT & CROSSTAB
# ============================================================
elif page == "📋 Pivot & Crosstab":

    st.title("📋 Pivot Tables & Crosstabs")

    col = st.selectbox(
        "Select Feature",
        ["Property_Area", "Education",
         "Credit_History", "Married"]
    )

    pivot = pd.crosstab(
        clean[col],
        clean["Loan_Status"]
    )

    pivot["Total"] = pivot.sum(axis=1)
    pivot["Approval_Rate_%"] = (
        pivot.get("Y", 0) / pivot["Total"] * 100
    ).round(2)

    st.dataframe(
        pivot,
        use_container_width=True
    )

    st.write("### Property Area × Loan Status")

    st.dataframe(
        pd.crosstab(
            clean["Property_Area"],
            clean["Loan_Status"],
            margins=True
        ),
        use_container_width=True
    )


# ============================================================
# 8. CORRELATION
# ============================================================
elif page == "🔥 Correlation Analysis":

    st.title("🔥 Correlation Analysis")

    cols = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Dependents"
    ]

    corr = clean[cols].corr()

    st.write("### Correlation Matrix")
    st.dataframe(
        corr.round(3),
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(corr)
    fig.colorbar(im)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)

    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(
                j, i,
                f"{corr.iloc[i,j]:.2f}",
                ha="center",
                va="center"
            )

    ax.set_title("Correlation Heatmap")
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "Applicant Income and Loan Amount show a moderate positive "
        "correlation of approximately 0.565."
    )


# ============================================================
# 9. STATISTICAL ANALYSIS
# ============================================================
elif page == "🧪 Statistical Analysis":

    st.title("🧪 Statistical Analysis")

    st.write("### Chi-Square Tests")

    features = [
        "Gender", "Married", "Education",
        "Self_Employed", "Property_Area",
        "Dependents", "Credit_History"
    ]

    results = []

    for col in features:

        table = pd.crosstab(
            clean[col],
            clean["Loan_Status"]
        )

        chi2, p, dof, _ = stats.chi2_contingency(table)

        results.append([
            col,
            round(chi2, 4),
            round(p, 6),
            "Reject H0" if p < .05 else "Fail to Reject H0"
        ])

    st.dataframe(
        pd.DataFrame(
            results,
            columns=[
                "Feature",
                "Chi-Square",
                "p-value",
                "Decision"
            ]
        ),
        use_container_width=True
    )

    st.write("### Welch's t-test")

    approved = clean.loc[
        clean["Loan_Status"] == "Y",
        "ApplicantIncome"
    ]

    rejected = clean.loc[
        clean["Loan_Status"] == "N",
        "ApplicantIncome"
    ]

    t_stat, p = stats.ttest_ind(
        approved,
        rejected,
        equal_var=False
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Approved Mean", f"{approved.mean():.2f}")
    c2.metric("Rejected Mean", f"{rejected.mean():.2f}")
    c3.metric("p-value", f"{p:.6f}")

    st.write("### Covariance")
    st.dataframe(
        clean[
            [
                "ApplicantIncome",
                "CoapplicantIncome",
                "LoanAmount",
                "Loan_Amount_Term",
                "Credit_History"
            ]
        ].cov().round(3),
        use_container_width=True
    )


# ============================================================
# 10. FEATURE ENGINEERING
# ============================================================
elif page == "⚙️ Feature Engineering":

    st.title("⚙️ Feature Engineering")

    features = [
        "Total_Household_Income",
        "Income_Category",
        "Loan_Amount_Category",
        "Income_to_Loan_Ratio",
        "Estimated_Monthly_EMI",
        "EMI_Burden_Ratio",
        "Applicant_Risk_Category"
    ]

    st.dataframe(
        fe[features].head(10),
        use_container_width=True
    )

    st.markdown("""
    **Created Features:**

    - Total Household Income
    - Income Category
    - Loan Amount Category
    - Income-to-Loan Ratio
    - Estimated Monthly EMI
    - EMI Burden Ratio
    - Applicant Risk Category
    """)

    st.warning(
        "Estimated EMI is only an analytical approximation because "
        "the dataset does not contain an interest rate."
    )


# ============================================================
# 11. BUSINESS INSIGHTS
# ============================================================
elif page == "💼 Business Insights":

    st.title("💼 Business Insights & Recommendations")

    st.markdown("""
    ### Key Insights

    **1. Credit History**  
    Credit history shows a substantial difference in observed approval rates.

    **2. Property Area**  
    Semiurban applicants show the highest observed approval rate.

    **3. Applicant Income**  
    The Welch t-test does not show a statistically significant difference
    in mean applicant income between approval groups.

    **4. Income & Loan Amount**  
    Applicant income and loan amount have a moderate positive relationship.

    **5. Class Distribution**  
    Approved applications are the majority class.

    ### Recommendations

    - Consider credit history along with other applicant information.
    - Monitor approval patterns across property areas.
    - Use household income and income-to-loan ratio for analysis.
    - Investigate rejected applications for additional decision factors.
    - Consider class imbalance in future predictive models.
    """)

    st.caption(
        "These findings are based on observed patterns and should not "
        "be interpreted as causal conclusions or formal lending policy."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Applications", 614)
    c2.metric("Approved", 422)
    c3.metric("Rejected", 192)