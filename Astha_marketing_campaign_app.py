"""
=====================================================================================
MARKETING CAMPAIGN RESPONSE PREDICTION USING MACHINE LEARNING
Predicting Customer Response to Marketing Campaigns for Targeted Customer Engagement
=====================================================================================

A complete, single-file Streamlit analytics dashboard.

Run:
    streamlit run marketing_campaign_app.py

Dataset (NOT bundled - no synthetic data is ever generated):
    Kaggle "Customer Personality Analysis" -> marketing_campaign.csv
    https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis
    Place it at ./data/marketing_campaign.csv or upload it from the sidebar.

Design notes
------------
* Everything (cleaning, feature engineering, ML, UI) lives in this one file.
* No metric, importance, revenue or prediction is hard-coded - all values are computed.
* All preprocessing that "learns" from data (imputation, scaling, encoding) sits inside
  an sklearn Pipeline that is fitted on the training split only, so there is no leakage.
* Optional packages (Plotly, XGBoost, SHAP, Seaborn) degrade gracefully when missing.
"""

from __future__ import annotations

import importlib.util
import io
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---- Optional dependencies (the app must keep working without them) -----------------
try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_OK = True
except Exception:  # pragma: no cover
    px = go = None
    PLOTLY_OK = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MPL_OK = True
except Exception:  # pragma: no cover
    plt = None
    MPL_OK = False

try:
    import seaborn as sns

    SEABORN_OK = True
except Exception:  # pragma: no cover
    sns = None
    SEABORN_OK = False

try:
    from xgboost import XGBClassifier

    XGB_OK = True
except Exception:  # ImportError, or a missing native library such as libomp
    XGBClassifier = None
    XGB_OK = False

SHAP_OK = importlib.util.find_spec("shap") is not None  # imported lazily (slow import)

# =====================================================================================
# CONSTANTS
# =====================================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TARGET = "Response"
ID_COL = "ID"
DEFAULT_DATA_PATH = os.path.join(".", "data", "marketing_campaign.csv")
CURRENT_YEAR = datetime.now().year
MIN_AGE, MAX_AGE = 18, 100  # defensible bounds for a customer database

SPEND_COLS = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
CHANNEL_COLS = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
CAMPAIGN_COLS = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]
HISTORY_FEATURES = CAMPAIGN_COLS + ["Previous_Campaign_Acceptance"]
NON_FEATURE_COLS = {ID_COL, TARGET, "Dt_Customer", "Year_Birth", "Z_CostContact", "Z_Revenue"}
KNOWN_NUMERIC = (
    ["Year_Birth", "Income", "Kidhome", "Teenhome", "Recency", "NumDealsPurchases", "NumWebVisitsMonth", "Complain",
     "Z_CostContact", "Z_Revenue"]
    + SPEND_COLS + CHANNEL_COLS + CAMPAIGN_COLS
)
PRODUCT_LABELS = {
    "MntWines": "Wine", "MntFruits": "Fruits", "MntMeatProducts": "Meat",
    "MntFishProducts": "Fish", "MntSweetProducts": "Sweets", "MntGoldProds": "Gold products",
}
CHANNEL_LABELS = {
    "NumWebPurchases": "Web", "NumCatalogPurchases": "Catalog",
    "NumStorePurchases": "Store", "NumDealsPurchases": "Deals",
}
RESPONSE_LABELS = {0: "Did not respond", 1: "Responded"}
COLOR_MAP = {"Did not respond": "#9AA5B1", "Responded": "#1F6FEB"}
ORDERS = {
    "Age_Group": ["Under 40", "40-49", "50-59", "60-69", "70+"],
    "Income_Segment": ["Low", "Lower-Mid", "Upper-Mid", "High"],
    "Spending_Segment": ["Low", "Medium-Low", "Medium-High", "High"],
    "Campaign_History": ["None", "1 campaign", "2+ campaigns"],
}
PAGES = [
    "Dashboard", "Data Overview", "EDA & Insights", "Campaign Analysis", "Model Training",
    "Prediction", "Customer Targeting", "Model Explainability", "Business Recommendations", "About Project",
]

DATA_DICTIONARY = {
    "ID": ("Unique customer identifier", "Identifier"),
    "Year_Birth": ("Customer's year of birth", "Numeric"),
    "Education": ("Customer's education level", "Categorical"),
    "Marital_Status": ("Customer's marital status", "Categorical"),
    "Income": ("Customer's yearly household income", "Numeric"),
    "Kidhome": ("Number of small children in the household", "Numeric"),
    "Teenhome": ("Number of teenagers in the household", "Numeric"),
    "Dt_Customer": ("Date the customer enrolled with the company", "Date"),
    "Recency": ("Days since the customer's last purchase", "Numeric"),
    "MntWines": ("Amount spent on wine in the last 2 years", "Numeric"),
    "MntFruits": ("Amount spent on fruits in the last 2 years", "Numeric"),
    "MntMeatProducts": ("Amount spent on meat in the last 2 years", "Numeric"),
    "MntFishProducts": ("Amount spent on fish in the last 2 years", "Numeric"),
    "MntSweetProducts": ("Amount spent on sweets in the last 2 years", "Numeric"),
    "MntGoldProds": ("Amount spent on gold products in the last 2 years", "Numeric"),
    "NumDealsPurchases": ("Purchases made with a discount", "Numeric"),
    "NumWebPurchases": ("Purchases made through the company website", "Numeric"),
    "NumCatalogPurchases": ("Purchases made using a catalogue", "Numeric"),
    "NumStorePurchases": ("Purchases made directly in stores", "Numeric"),
    "NumWebVisitsMonth": ("Website visits in the last month", "Numeric"),
    "AcceptedCmp1": ("1 if the customer accepted the offer in campaign 1", "Binary"),
    "AcceptedCmp2": ("1 if the customer accepted the offer in campaign 2", "Binary"),
    "AcceptedCmp3": ("1 if the customer accepted the offer in campaign 3", "Binary"),
    "AcceptedCmp4": ("1 if the customer accepted the offer in campaign 4", "Binary"),
    "AcceptedCmp5": ("1 if the customer accepted the offer in campaign 5", "Binary"),
    "Complain": ("1 if the customer complained in the last 2 years", "Binary"),
    "Z_CostContact": ("Constant contact-cost field (no predictive information)", "Numeric (constant)"),
    "Z_Revenue": ("Constant revenue field (no predictive information)", "Numeric (constant)"),
    "Response": ("1 if the customer accepted the offer in the LAST campaign (TARGET)", "Binary (target)"),
    "Country": ("Customer's country", "Categorical"),
}

ENGINEERED_DICTIONARY = {
    "Age": ("Current year minus Year_Birth", "Numeric",
            "Age is easier to interpret than birth year and lets us test life-stage effects on response."),
    "Customer_Tenure_Days": ("Days between enrolment (Dt_Customer) and the latest enrolment date in the data", "Numeric",
                             "Measures loyalty/relationship length; a raw date cannot be used by most models."),
    "Total_Spending": ("Sum of the six Mnt* product-category spends", "Numeric",
                       "Single measure of customer value; usually more stable than any one category."),
    "Total_Purchases": ("Web + Catalog + Store purchases", "Numeric",
                        "Overall purchasing activity across the three channels."),
    "Total_Children": ("Kidhome + Teenhome", "Numeric", "Household size proxy that affects purchasing priorities."),
    "Is_Parent": ("1 if Total_Children > 0 else 0", "Binary", "Simple family-status flag."),
    "Previous_Campaign_Acceptance": ("Sum of AcceptedCmp1..5", "Numeric",
                                     "How many earlier campaigns the customer accepted (historical responsiveness)."),
    "Avg_Spend_Per_Purchase": ("Total_Spending / max(Total_Purchases, 1)", "Numeric",
                               "Basket-value indicator; separates big-ticket from frequent small buyers."),
    "Web_Visits_Per_Web_Purchase": ("NumWebVisitsMonth / max(NumWebPurchases, 1)", "Numeric",
                                    "Browsing-to-buying ratio; high values suggest interest without conversion."),
    "Deal_Purchase_Ratio": ("NumDealsPurchases / max(Total_Purchases, 1)", "Numeric",
                            "Discount sensitivity: how much of the buying is deal-driven."),
    "Web_Purchase_Share": ("NumWebPurchases / max(Total_Purchases, 1)", "Numeric", "Share of purchases made online."),
    "Catalog_Purchase_Share": ("NumCatalogPurchases / max(Total_Purchases, 1)", "Numeric",
                               "Share of purchases made via catalogue."),
    "Store_Purchase_Share": ("NumStorePurchases / max(Total_Purchases, 1)", "Numeric",
                            "Share of purchases made in physical stores."),
}


# =====================================================================================
# UI HELPERS (CSS, tables, downloads, KPI cards)
# =====================================================================================
def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1400px;}
        .hero {background: linear-gradient(120deg, #0B3C5D 0%, #1F6FEB 100%); color: #fff;
               padding: 1.4rem 1.8rem; border-radius: 14px; margin-bottom: 1.2rem;}
        .hero h1 {margin: 0; font-size: 1.9rem; letter-spacing: .04em; color: #fff;}
        .hero p {margin: .3rem 0 0 0; font-size: 1.02rem; opacity: .92; color: #fff;}
        .section-title {font-size: 1.35rem; font-weight: 700; margin: .2rem 0 .1rem 0;}
        .section-sub {opacity: .75; margin-bottom: .8rem;}
        div[data-testid="stMetric"] {background: rgba(31,111,235,.07); border: 1px solid rgba(31,111,235,.25);
               border-radius: 12px; padding: .7rem .9rem;}
        div[data-testid="stMetricValue"] {font-size: 1.55rem;}
        .note-box {border-left: 4px solid #1F6FEB; background: rgba(31,111,235,.07);
                   padding: .7rem 1rem; border-radius: 6px; margin: .5rem 0 1rem 0;}
        .warn-box {border-left: 4px solid #E0A800; background: rgba(224,168,0,.10);
                   padding: .7rem 1rem; border-radius: 6px; margin: .5rem 0 1rem 0;}
        .pred-yes {background: rgba(25,135,84,.12); border: 1px solid rgba(25,135,84,.5); color: inherit;
                   padding: 1rem 1.2rem; border-radius: 12px; font-size: 1.5rem; font-weight: 700;}
        .pred-no {background: rgba(220,53,69,.10); border: 1px solid rgba(220,53,69,.45); color: inherit;
                  padding: 1rem 1.2rem; border-radius: 12px; font-size: 1.5rem; font-weight: 700;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def note(text: str, warn: bool = False) -> None:
    css = "warn-box" if warn else "note-box"
    st.markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)


def _stretch_kwargs() -> dict:
    """Streamlit >= 1.50 uses width='stretch'; older versions use use_container_width=True."""
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
        return {"width": "stretch"} if (major, minor) >= (1, 50) else {"use_container_width": True}
    except Exception:
        return {"use_container_width": True}


def st_table(df: pd.DataFrame, hide_index: bool = True) -> None:
    """dataframe rendering that works across Streamlit versions."""
    st.dataframe(df, hide_index=hide_index, **_stretch_kwargs())


def st_plot(fig) -> None:
    """Render a Plotly or Matplotlib figure (whichever backend created it)."""
    if fig is None:
        st.info("Charts need Plotly or Matplotlib. Install with: pip install plotly matplotlib")
        return
    if PLOTLY_OK and isinstance(fig, go.Figure):
        st.plotly_chart(fig, **_stretch_kwargs())
    else:
        st.pyplot(fig)
        if MPL_OK:
            plt.close(fig)


def download_df_button(label: str, df: pd.DataFrame, filename: str, key: str) -> None:
    st.download_button(label, data=df.to_csv(index=False).encode("utf-8"), file_name=filename,
                       mime="text/csv", key=key)


def kpi_row(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


def fmt_money(x) -> str:
    return "N/A" if x is None or pd.isna(x) else f"${x:,.0f}"


def fmt_pct(x, digits: int = 1) -> str:
    return "N/A" if x is None or pd.isna(x) else f"{x:.{digits}f}%"


# =====================================================================================
# CHART HELPERS (Plotly first, Matplotlib/Seaborn fallback)
# =====================================================================================
def fig_bar(data, x, y, title, xlabel=None, ylabel=None, color=None, horizontal=False,
            text_fmt=".1f", height=420, color_map=None):
    data = data.copy()
    xlabel, ylabel = xlabel or x, ylabel or y
    if not horizontal:
        data[x] = data[x].astype(str)
    if PLOTLY_OK:
        fig = px.bar(data, x=y if horizontal else x, y=x if horizontal else y, color=color, barmode="group",
                     orientation="h" if horizontal else "v", color_discrete_map=color_map, text_auto=text_fmt,
                     template="plotly_white", title=title, height=height)
        fig.update_layout(xaxis_title=ylabel if horizontal else xlabel, yaxis_title=xlabel if horizontal else ylabel,
                          legend_title_text=color or "", margin=dict(l=10, r=10, t=55, b=10))
        if not horizontal:
            fig.update_xaxes(type="category")
        return fig
    if MPL_OK:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if color:
            pv = data.pivot_table(index=x, columns=color, values=y, aggfunc="mean", sort=False)
            pv.plot(kind="barh" if horizontal else "bar", ax=ax, rot=0)
        else:
            (ax.barh if horizontal else ax.bar)(data[x].astype(str), data[y], color="#1F6FEB")
        ax.set_title(title)
        ax.set_xlabel(ylabel if horizontal else xlabel)
        ax.set_ylabel(xlabel if horizontal else ylabel)
        fig.tight_layout()
        return fig
    return None


def fig_hist(data, x, title, xlabel=None, ylabel="Number of customers", color=None, nbins=30, color_map=None):
    xlabel = xlabel or x
    if PLOTLY_OK:
        fig = px.histogram(data, x=x, color=color, nbins=nbins, barmode="overlay", opacity=0.75,
                           color_discrete_map=color_map, template="plotly_white", title=title, height=400)
        fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel, legend_title_text=color or "",
                          margin=dict(l=10, r=10, t=55, b=10))
        return fig
    if MPL_OK:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        if color:
            for name, grp in data.groupby(color):
                ax.hist(grp[x].dropna(), bins=nbins, alpha=0.6, label=str(name))
            ax.legend()
        else:
            ax.hist(data[x].dropna(), bins=nbins, color="#1F6FEB", alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        return fig
    return None


def fig_pie(labels, values, title):
    if PLOTLY_OK:
        fig = px.pie(names=labels, values=values, hole=0.45, title=title, template="plotly_white",
                     color=labels, color_discrete_map=COLOR_MAP, height=380)
        fig.update_traces(textinfo="label+percent", hovertemplate="%{label}: %{value:,} customers (%{percent})")
        return fig
    if MPL_OK:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(values, labels=labels, autopct="%1.1f%%", colors=[COLOR_MAP.get(l, "#888") for l in labels])
        ax.set_title(title)
        return fig
    return None


def fig_curve(curves, title, xlabel, ylabel, diagonal=False, hline=None, hline_label="", height=450):
    """curves: list of (name, x_values, y_values)."""
    if PLOTLY_OK:
        fig = go.Figure()
        for name, xs, ys in curves:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name))
        if diagonal:
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                                     line=dict(dash="dash", color="grey")))
        if hline is not None:
            fig.add_hline(y=hline, line_dash="dash", line_color="grey", annotation_text=hline_label)
        fig.update_layout(template="plotly_white", title=title, xaxis_title=xlabel, yaxis_title=ylabel,
                          height=height, margin=dict(l=10, r=10, t=55, b=10),
                          legend=dict(orientation="h", y=-0.2))
        return fig
    if MPL_OK:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name, xs, ys in curves:
            ax.plot(xs, ys, label=name)
        if diagonal:
            ax.plot([0, 1], [0, 1], "--", color="grey", label="Random guess")
        if hline is not None:
            ax.axhline(hline, ls="--", color="grey", label=hline_label or "baseline")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        return fig
    return None


def fig_heatmap(matrix: pd.DataFrame, title, fmt=".2f", height=520, colorscale="Blues",
                midpoint=None, show_text=True):
    if PLOTLY_OK:
        fig = px.imshow(matrix, text_auto=fmt if show_text else False, aspect="auto",
                        color_continuous_scale=colorscale, color_continuous_midpoint=midpoint,
                        title=title, height=height, template="plotly_white")
        fig.update_layout(margin=dict(l=10, r=10, t=55, b=10))
        return fig
    if MPL_OK:
        fig, ax = plt.subplots(figsize=(8, 6))
        if SEABORN_OK:
            sns.heatmap(matrix, annot=show_text, fmt=fmt, cmap=colorscale if colorscale != "RdBu_r" else "coolwarm",
                        center=midpoint, ax=ax)
        else:
            im = ax.imshow(matrix.values, cmap="Blues")
            ax.set_xticks(range(matrix.shape[1]))
            ax.set_xticklabels(matrix.columns, rotation=90)
            ax.set_yticks(range(matrix.shape[0]))
            ax.set_yticklabels(matrix.index)
            fig.colorbar(im)
        ax.set_title(title)
        fig.tight_layout()
        return fig
    return None


def fig_gauge(prob_pct: float, title: str):
    if not PLOTLY_OK:
        return None
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob_pct, number={"suffix": "%", "valueformat": ".1f"},
        title={"text": title},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1F6FEB"},
               "steps": [{"range": [0, 40], "color": "#E9EDF2"}, {"range": [40, 70], "color": "#CFE0FA"},
                         {"range": [70, 100], "color": "#A9C8F7"}]}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


# =====================================================================================
# DATA LOADING
# =====================================================================================
def _read_csv_flexible(raw_bytes: bytes) -> pd.DataFrame:
    """The Kaggle file is TAB separated; other copies use commas/semicolons. Try all."""
    if not raw_bytes or not raw_bytes.strip():
        raise ValueError("The file is empty.")
    last_error = None
    for sep in ("\t", ",", ";", "|"):
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), sep=sep, encoding="utf-8", encoding_errors="replace")
            if df.shape[1] >= 3 and df.shape[0] >= 1:
                return df
        except Exception as exc:  # try the next separator
            last_error = exc
    raise ValueError("Could not parse the file as a delimited table" + (f" ({last_error})." if last_error else "."))


@st.cache_data(show_spinner=False)
def load_data_from_bytes(raw_bytes: bytes) -> pd.DataFrame:
    return _read_csv_flexible(raw_bytes)


@st.cache_data(show_spinner=False)
def load_data_from_path(path: str, mtime: float) -> pd.DataFrame:  # mtime busts the cache when the file changes
    with open(path, "rb") as fh:
        return _read_csv_flexible(fh.read())


def load_data(uploaded_file):
    """Returns (DataFrame | None, source description | None)."""
    if uploaded_file is not None:
        return load_data_from_bytes(uploaded_file.getvalue()), f"Uploaded file: {uploaded_file.name}"
    if os.path.exists(DEFAULT_DATA_PATH):
        return (load_data_from_path(DEFAULT_DATA_PATH, os.path.getmtime(DEFAULT_DATA_PATH)),
                f"Local file: {DEFAULT_DATA_PATH}")
    return None, None


def parse_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%d-%m-%Y", errors="coerce")
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return parsed


# =====================================================================================
# DATA CLEANING
# =====================================================================================
def standardise_columns(df: pd.DataFrame):
    """Light, row-preserving standardisation shared by training data and uploaded customers."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    info = {"invalid_income": 0, "negative_values": 0, "unparsed_dates": 0}
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]) and not pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].map(lambda v: (v.strip() or np.nan) if isinstance(v, str) else v)
    for c in KNOWN_NUMERIC:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in SPEND_COLS + CHANNEL_COLS + ["NumDealsPurchases", "NumWebVisitsMonth", "Recency"]:
        if c in out.columns:
            neg = out[c] < 0
            info["negative_values"] += int(neg.sum())
            out.loc[neg, c] = np.nan
    if "Income" in out.columns:
        bad = out["Income"] <= 0
        info["invalid_income"] = int(bad.sum())
        out.loc[bad, "Income"] = np.nan  # invalid -> missing; imputed later INSIDE the pipeline
    if "Dt_Customer" in out.columns:
        before = out["Dt_Customer"].notna().sum()
        out["Dt_Customer"] = parse_dates(out["Dt_Customer"])
        info["unparsed_dates"] = int(before - out["Dt_Customer"].notna().sum())
    return out, info


def clean_data(raw: pd.DataFrame):
    """Cleans the training data. Returns (clean_df, log_rows)."""
    log: list[dict] = []
    df, info = standardise_columns(raw)
    n_start = len(df)

    if TARGET not in df.columns:
        raise ValueError(f"The target column '{TARGET}' was not found. Columns detected: {', '.join(df.columns)}")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    n_bad_target = int(df[TARGET].isna().sum())
    if n_bad_target:
        df = df[df[TARGET].notna()]
        log.append({"Step": "Missing target", "Records affected": n_bad_target, "Action": "Rows removed",
                    "Reason": "A supervised model cannot learn from rows without a known outcome."})
    df[TARGET] = df[TARGET].astype(int)
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError(f"'{TARGET}' must be binary (0/1). Found values: {sorted(df[TARGET].unique().tolist())}")

    dup_cols = [c for c in df.columns if c != ID_COL]
    n_dup = int(df.duplicated(subset=dup_cols).sum())
    log.append({"Step": "Duplicate records", "Records affected": n_dup,
                "Action": "Removed (kept first)" if n_dup else "None found",
                "Reason": "Rows identical in every field except ID are treated as duplicate customers; keeping them "
                          "would overweight them and can leak between train and test."})
    if n_dup:
        df = df.drop_duplicates(subset=dup_cols, keep="first")

    log.append({"Step": "Invalid income (<= 0)", "Records affected": info["invalid_income"],
                "Action": "Set to missing, then median-imputed inside the ML pipeline",
                "Reason": "A non-positive yearly income is not a valid value. Rows are kept because the other "
                          "fields are valid; the value is imputed on training data only (no leakage)."})
    log.append({"Step": "Negative counts/spend", "Records affected": info["negative_values"],
                "Action": "Set to missing" if info["negative_values"] else "None found",
                "Reason": "Purchases, visits, recency and spend cannot be negative."})

    n_missing_income = int(df["Income"].isna().sum()) if "Income" in df.columns else 0
    log.append({"Step": "Missing values (all columns)", "Records affected": int(df.isna().any(axis=1).sum()),
                "Action": "Numeric: median imputation | Categorical: mode imputation (inside the pipeline)",
                "Reason": f"Imputation statistics are learned from the training split only. "
                          f"Income missing after cleaning: {n_missing_income}."})

    if "Year_Birth" in df.columns:
        age = CURRENT_YEAR - df["Year_Birth"]
        bad_age = age.notna() & ~age.between(MIN_AGE, MAX_AGE)
        n_age = int(bad_age.sum())
        log.append({"Step": f"Unrealistic age (outside {MIN_AGE}-{MAX_AGE})", "Records affected": n_age,
                    "Action": "Rows removed" if n_age else "None found",
                    "Reason": "Ages above 100 (e.g. birth years 1893-1900) are almost certainly data-entry errors; "
                              "they would distort age-based statistics."})
        df = df[~bad_age]

    if "Dt_Customer" in df.columns:
        log.append({"Step": "Enrolment date parsing", "Records affected": info["unparsed_dates"],
                    "Action": "Converted Dt_Customer to datetime", "Reason": "Needed to derive Customer_Tenure_Days."})

    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if const_cols:
        log.append({"Step": "Constant columns", "Records affected": len(const_cols),
                    "Action": f"Excluded from modelling: {', '.join(const_cols)}",
                    "Reason": "A column with a single value carries no information."})

    log.insert(0, {"Step": "Starting records", "Records affected": n_start, "Action": "-", "Reason": "Raw dataset size."})
    log.append({"Step": "Final records", "Records affected": len(df), "Action": "-", "Reason": "Records used downstream."})
    return df.reset_index(drop=True), log


# =====================================================================================
# FEATURE ENGINEERING
# =====================================================================================
def _row_total(df: pd.DataFrame, cols: list[str]):
    present = [c for c in cols if c in df.columns]
    if not present:
        return None
    return df[present].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=False)


def feature_engineering(df: pd.DataFrame, ref_date=None):
    """Adds business features. Works for training data AND for new raw customer rows."""
    out = df.copy()
    if "Age" not in out.columns and "Year_Birth" in out.columns:
        out["Age"] = CURRENT_YEAR - pd.to_numeric(out["Year_Birth"], errors="coerce")
    if "Customer_Tenure_Days" not in out.columns and "Dt_Customer" in out.columns:
        dts = out["Dt_Customer"]
        if not pd.api.types.is_datetime64_any_dtype(dts):
            dts = parse_dates(dts)
        if ref_date is None or pd.isna(ref_date):
            ref_date = dts.max()
        if pd.notna(ref_date):
            out["Customer_Tenure_Days"] = (ref_date - dts).dt.days

    total = _row_total(out, SPEND_COLS)
    if total is not None:
        out["Total_Spending"] = total
    total = _row_total(out, CHANNEL_COLS)
    if total is not None:
        out["Total_Purchases"] = total
    total = _row_total(out, ["Kidhome", "Teenhome"])
    if total is not None:
        out["Total_Children"] = total
        out["Is_Parent"] = (total > 0).astype(float).where(total.notna())
    total = _row_total(out, CAMPAIGN_COLS)
    if total is not None:
        out["Previous_Campaign_Acceptance"] = total

    if "Total_Purchases" in out.columns:
        den = out["Total_Purchases"].clip(lower=1)
        if "Total_Spending" in out.columns:
            out["Avg_Spend_Per_Purchase"] = out["Total_Spending"] / den
        if "NumDealsPurchases" in out.columns:
            out["Deal_Purchase_Ratio"] = pd.to_numeric(out["NumDealsPurchases"], errors="coerce") / den
        for col, name in [("NumWebPurchases", "Web_Purchase_Share"), ("NumCatalogPurchases", "Catalog_Purchase_Share"),
                          ("NumStorePurchases", "Store_Purchase_Share")]:
            if col in out.columns:
                out[name] = pd.to_numeric(out[col], errors="coerce") / den
    if "NumWebVisitsMonth" in out.columns and "NumWebPurchases" in out.columns:
        out["Web_Visits_Per_Web_Purchase"] = (
            pd.to_numeric(out["NumWebVisitsMonth"], errors="coerce")
            / pd.to_numeric(out["NumWebPurchases"], errors="coerce").clip(lower=1)
        )
    return out, ref_date


def impute_for_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive-only imputation (median / mode) so EDA charts include every customer.
    The ML pipeline does its own leakage-free imputation on the training split."""
    out = df.copy()
    for c in out.columns:
        if out[c].isna().any():
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].fillna(out[c].median())
            elif not pd.api.types.is_datetime64_any_dtype(out[c]):
                mode = out[c].mode(dropna=True)
                if len(mode):
                    out[c] = out[c].fillna(mode.iloc[0])
    return out


def _quartile_segment(series: pd.Series, labels: list[str]) -> pd.Series:
    ranks = series.rank(method="first")
    return pd.qcut(ranks, q=len(labels), labels=labels).astype(object)


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Reporting-only segments used by EDA / campaign analysis (never fed to the model)."""
    out = df.copy()
    out["Response_Label"] = out[TARGET].map(RESPONSE_LABELS)
    if "Age" in out.columns:
        out["Age_Group"] = pd.cut(out["Age"], bins=[0, 39, 49, 59, 69, 200],
                                  labels=ORDERS["Age_Group"]).astype(object)
    if "Income" in out.columns:
        out["Income_Segment"] = _quartile_segment(out["Income"], ORDERS["Income_Segment"])
    if "Total_Spending" in out.columns:
        out["Spending_Segment"] = _quartile_segment(out["Total_Spending"], ORDERS["Spending_Segment"])
    if "Previous_Campaign_Acceptance" in out.columns:
        out["Campaign_History"] = pd.cut(out["Previous_Campaign_Acceptance"], bins=[-1, 0, 1, 100],
                                         labels=ORDERS["Campaign_History"]).astype(object)
    ch = {"Web": "NumWebPurchases", "Catalog": "NumCatalogPurchases", "Store": "NumStorePurchases"}
    ch = {k: v for k, v in ch.items() if v in out.columns}
    if ch:
        sub = out[list(ch.values())]
        top = sub.idxmax(axis=1).map({v: k for k, v in ch.items()})
        tie = sub.eq(sub.max(axis=1), axis=0).sum(axis=1) > 1
        top = top.where(~tie, "Mixed (tie)")
        top = top.where(sub.sum(axis=1) > 0, "No purchases")
        out["Preferred_Channel"] = top.astype(object)
    return out


@st.cache_data(show_spinner=False)
def prepare_data(raw: pd.DataFrame):
    """Clean -> engineer -> descriptive frame. Cached so filters never trigger re-cleaning."""
    cleaned, log = clean_data(raw)
    if len(cleaned) < 50:
        raise ValueError(f"Insufficient data: only {len(cleaned)} usable records remain after cleaning (need >= 50).")
    df_model, ref_date = feature_engineering(cleaned)
    df_eda = add_segments(impute_for_eda(df_model))
    dup_all = int(raw.duplicated().sum())
    dup_excl_id = int(raw.duplicated(subset=[c for c in raw.columns if str(c).strip() != ID_COL]).sum())
    return {"df_model": df_model, "df_eda": df_eda, "log": log, "ref_date": ref_date,
            "dup_all": dup_all, "dup_excl_id": dup_excl_id}


# =====================================================================================
# MACHINE LEARNING PIPELINE
# =====================================================================================
def select_features(df: pd.DataFrame, include_history: bool = True):
    """Decide which columns may be used as predictors. Returns (numeric, categorical, excluded{col: reason}).

    Leakage logic
    -------------
    * `Response` is the outcome of the LATEST campaign -> never a predictor.
    * AcceptedCmp1..5 refer to EARLIER campaigns (known before the latest campaign is launched) ->
      allowed by default (documented assumption), but can be switched off to test dependence on them.
    * ID, raw dates, birth year (replaced by Age) and constant columns carry no legitimate signal.
    """
    numeric, categorical, excluded = [], [], {}
    for c in df.columns:
        if c == TARGET:
            excluded[c] = "Target variable - must never appear in X"
        elif c == ID_COL:
            excluded[c] = "Identifier - no predictive meaning, would only let a model memorise rows"
        elif c == "Dt_Customer" or pd.api.types.is_datetime64_any_dtype(df[c]):
            excluded[c] = "Raw date - replaced by Customer_Tenure_Days"
        elif c == "Year_Birth" and "Age" in df.columns:
            excluded[c] = "Replaced by Age (identical information)"
        elif c in ("Z_CostContact", "Z_Revenue") or df[c].nunique(dropna=True) <= 1:
            excluded[c] = "Constant column - carries no information"
        elif (not include_history) and c in HISTORY_FEATURES:
            excluded[c] = "Historical campaign acceptance switched off by the user (leakage stress-test)"
        elif pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
        elif df[c].nunique(dropna=True) <= 30:
            categorical.append(c)
        else:
            excluded[c] = "High-cardinality free text - not suitable for one-hot encoding"
    return numeric, categorical, excluded


def coerce_features(X: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    X = X.copy()
    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in categorical:
        X[c] = X[c].map(lambda v: np.nan if pd.isna(v) else str(v).strip()).astype(object)
    return X


def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessing_pipeline(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                              ("scaler", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                              ("onehot", _make_ohe())]), categorical))
    return ColumnTransformer(transformers, remainder="drop")


def get_model_zoo(pos_weight: float, use_class_weight: bool) -> dict:
    cw = "balanced" if use_class_weight else None
    zoo = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight=cw, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE,
            class_weight="balanced_subsample" if use_class_weight else None),
    }
    if XGB_OK:
        zoo["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", tree_method="hist", random_state=RANDOM_STATE, n_jobs=1,
            scale_pos_weight=pos_weight if use_class_weight else 1.0)
    else:
        try:
            gb = HistGradientBoostingClassifier(random_state=RANDOM_STATE, class_weight=cw)
        except TypeError:  # older scikit-learn has no class_weight here
            gb = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
        zoo["Hist Gradient Boosting (XGBoost fallback)"] = gb
    return zoo


def best_f1_threshold(y_true, proba) -> float:
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.nanargmax(f1))]) if len(thr) else 0.5


def evaluate_model(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, proba)
    prec_c, rec_c, _ = precision_recall_curve(y_true, proba)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "threshold": float(threshold),
        "cm": confusion_matrix(y_true, pred, labels=[0, 1]),
        "fpr": fpr, "tpr": tpr, "prec_curve": prec_c, "rec_curve": rec_c,
    }


def train_models(df: pd.DataFrame, include_history: bool, use_class_weight: bool, ref_date, progress_cb=None) -> dict:
    """Train + evaluate all models. Everything that learns from data lives inside the Pipeline and is
    fitted on the TRAINING split only. The decision threshold is tuned on cross-validated training
    predictions (never on the test set)."""
    numeric, categorical, excluded = select_features(df, include_history)
    feats = numeric + categorical
    if not feats:
        raise ValueError("No usable predictor columns were found in the dataset.")
    y = df[TARGET].astype(int)
    counts = y.value_counts()
    if len(counts) < 2 or counts.min() < 10:
        raise ValueError("Insufficient data: each class needs at least 10 examples to train and evaluate models.")

    X = coerce_features(df[feats], numeric, categorical)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    zoo = get_model_zoo(pos_weight, use_class_weight)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    models, metrics, proba_test, rows = {}, {}, {}, []
    for i, (name, estimator) in enumerate(zoo.items()):
        if progress_cb:
            progress_cb(i / len(zoo), f"Training {name} ({i + 1}/{len(zoo)}) - cross-validating on the training split...")
        pipe = Pipeline([("prep", build_preprocessing_pipeline(numeric, categorical)), ("model", estimator)])
        oof = cross_val_predict(clone(pipe), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        thr = best_f1_threshold(y_train, oof)
        cv_pr, cv_roc = average_precision_score(y_train, oof), roc_auc_score(y_train, oof)
        cv_f1 = f1_score(y_train, (oof >= thr).astype(int), zero_division=0)
        if progress_cb:
            progress_cb((i + 0.6) / len(zoo), f"Training {name} ({i + 1}/{len(zoo)}) - fitting final model...")
        pipe.fit(X_train, y_train)
        p = pipe.predict_proba(X_test)[:, 1]
        m = evaluate_model(y_test, p, thr)
        m.update({"cv_pr_auc": cv_pr, "cv_roc_auc": cv_roc, "cv_f1": cv_f1})
        models[name], metrics[name], proba_test[name] = pipe, m, p
        rows.append({"Model": name, "Accuracy": m["accuracy"], "Precision": m["precision"], "Recall": m["recall"],
                     "F1": m["f1"], "ROC-AUC": m["roc_auc"], "PR-AUC": m["pr_auc"],
                     "Decision threshold": m["threshold"], "CV PR-AUC (train)": cv_pr, "CV ROC-AUC (train)": cv_roc})

    comparison = pd.DataFrame(rows)
    selected = comparison.sort_values(["CV PR-AUC (train)", "CV ROC-AUC (train)"], ascending=False).iloc[0]["Model"]
    if progress_cb:
        progress_cb(1.0, "Training complete.")
    return {
        "models": models, "metrics": metrics, "comparison": comparison, "selected": selected,
        "numeric": numeric, "categorical": categorical, "excluded": excluded, "feature_cols": feats,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "proba_test": proba_test,
        "include_history": include_history, "use_class_weight": use_class_weight, "ref_date": ref_date,
        "n_rows": len(df), "boost_name": [k for k in zoo if k not in ("Logistic Regression", "Random Forest")][0],
    }


# ---- feature importance / explainability helpers -------------------------------------
def get_feature_names(pipe: Pipeline) -> list[str]:
    names = pipe.named_steps["prep"].get_feature_names_out()
    return [n.split("__", 1)[1] if "__" in n else n for n in names]


def permutation_feature_importance(pipe, X_test, y_test, n_repeats: int = 5) -> pd.DataFrame:
    r = permutation_importance(pipe, X_test, y_test, scoring="average_precision", n_repeats=n_repeats,
                               random_state=RANDOM_STATE, n_jobs=1)
    return pd.DataFrame({"Feature": list(X_test.columns), "Importance": r.importances_mean,
                         "Std": r.importances_std}).sort_values("Importance", ascending=False)


def compute_feature_importance(training: dict, name: str):
    """Returns (DataFrame sorted by Importance desc, method description)."""
    pipe = training["models"][name]
    model = pipe.named_steps["model"]
    names = get_feature_names(pipe)
    if hasattr(model, "coef_") and len(names) == len(model.coef_.ravel()):
        coef = model.coef_.ravel()
        df = pd.DataFrame({"Feature": names, "Importance": np.abs(coef), "Coefficient": coef})
        df["Direction"] = np.where(df["Coefficient"] >= 0, "Raises response probability", "Lowers response probability")
        return df.sort_values("Importance", ascending=False), \
            "Absolute logistic-regression coefficient on standardised features (sign = direction of effect). " \
            "Correlated features (e.g. Total_Spending and its product components) share credit, so read individual " \
            "coefficients with caution."
    if hasattr(model, "feature_importances_") and len(names) == len(model.feature_importances_):
        df = pd.DataFrame({"Feature": names, "Importance": model.feature_importances_})
        return df.sort_values("Importance", ascending=False), \
            "Impurity-based (split-gain) importance from the fitted tree ensemble; shows magnitude, not direction."
    df = permutation_feature_importance(pipe, training["X_test"], training["y_test"])
    return df, "Permutation importance on the test set (drop in PR-AUC when a feature is shuffled)."


def feature_direction_text(feature: str, training: dict, name: str) -> str:
    """Plain-language direction of association between a feature and the model's predicted probability."""
    X_test, proba = training["X_test"], pd.Series(training["proba_test"][name], index=training["X_test"].index)
    if feature in X_test.columns and feature in training["numeric"]:
        rho = X_test[feature].corr(proba, method="spearman")
        if pd.isna(rho) or abs(rho) < 0.05:
            return "no clear monotonic relationship with predicted probability"
        return ("higher values are associated with a HIGHER predicted response probability" if rho > 0
                else "higher values are associated with a LOWER predicted response probability")
    for c in training["categorical"]:
        if feature.startswith(c + "_"):
            level = feature[len(c) + 1:]
            mask = X_test[c] == level
            if mask.sum() >= 5 and (~mask).sum() >= 5:
                diff = proba[mask].mean() - proba[~mask].mean()
                return (f"customers with {c} = '{level}' have an average predicted probability "
                        f"{abs(diff) * 100:.1f} percentage points {'higher' if diff > 0 else 'lower'} than others")
    return "direction could not be determined reliably"


def compute_shap_figure(training: dict, name: str, max_rows: int = 300):
    """Optional SHAP summary plot. Raises on any problem so the caller can fall back gracefully."""
    import shap  # lazy import: optional dependency

    pipe = training["models"][name]
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    Xt = prep.transform(training["X_test"].iloc[:max_rows])
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    names = get_feature_names(pipe)
    if hasattr(model, "coef_"):
        bg = prep.transform(training["X_train"].iloc[:200])
        bg = bg.toarray() if hasattr(bg, "toarray") else np.asarray(bg)
        values = shap.LinearExplainer(model, bg).shap_values(Xt)
    elif hasattr(model, "feature_importances_"):
        values = shap.TreeExplainer(model).shap_values(Xt)
    else:
        raise NotImplementedError("SHAP is not configured for this model type.")
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    plt.figure(figsize=(8, 6))
    shap.summary_plot(values, Xt, feature_names=names, show=False, max_display=15)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


# ---- scoring -------------------------------------------------------------------------
def align_features(fe_df: pd.DataFrame, training: dict):
    """Align (already feature-engineered) customer rows to the columns the model was trained on."""
    feats = training["feature_cols"]
    missing = [c for c in feats if c not in fe_df.columns]
    X = fe_df.reindex(columns=feats)
    X = coerce_features(X, training["numeric"], training["categorical"])
    return X, missing


def assign_segment(p, high: float, medium: float):
    p = np.asarray(p)
    return np.where(p >= high, "High", np.where(p >= medium, "Medium", "Low"))


def gains_table(y_true, proba) -> pd.DataFrame:
    y = np.asarray(y_true)
    order = np.argsort(-np.asarray(proba), kind="mergesort")
    y_sorted = y[order]
    n, total_pos = len(y), max(int(y.sum()), 1)
    rows = []
    for pct in (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
        k = int(np.ceil(n * pct / 100))
        captured = y_sorted[:k].sum() / total_pos * 100
        rows.append({"Customers contacted (%)": pct, "Customers contacted": k,
                     "Responders captured (%)": round(captured, 1), "Lift (x)": round(captured / pct, 2)})
    return pd.DataFrame(rows)


def simulate_campaign(proba, budget: float, cost: float, revenue: float, y_actual=None) -> dict:
    proba = np.asarray(proba, dtype=float)
    n = len(proba)
    max_contacts = int(budget // cost) if cost > 0 else 0
    k = min(max_contacts, n)
    order = np.argsort(-proba, kind="mergesort")
    top = proba[order][:k]
    exp_resp = float(top.sum())
    est_cost = k * cost
    exp_rev = exp_resp * revenue
    roi = (exp_rev - est_cost) / est_cost * 100 if est_cost > 0 else np.nan
    rand_resp = k * float(proba.mean()) if n else 0.0
    rand_rev = rand_resp * revenue
    rand_roi = (rand_rev - est_cost) / est_cost * 100 if est_cost > 0 else np.nan
    out = {"list_size": n, "max_contacts": max_contacts, "contacts": k, "expected_responses": exp_resp,
           "expected_revenue": exp_rev, "cost": est_cost, "roi": roi, "random_responses": rand_resp,
           "random_revenue": rand_rev, "random_roi": rand_roi}
    if y_actual is not None:
        out["actual_responses_top"] = int(np.asarray(y_actual)[order][:k].sum())
        out["actual_responses_random"] = k * float(np.mean(y_actual))
    return out


# =====================================================================================
# ANALYTICS HELPERS & BUSINESS INSIGHT GENERATION (all numbers computed from data)
# =====================================================================================
def rate_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    order = ORDERS.get(col)
    g = df.groupby(col, observed=True)[TARGET].agg(["mean", "count", "sum"]).reset_index()
    g.columns = [col, "Response rate (%)", "Customers", "Responders"]
    g["Response rate (%)"] = g["Response rate (%)"] * 100
    if order:
        g[col] = pd.Categorical(g[col], categories=order, ordered=True)
        g = g.sort_values(col)
        g[col] = g[col].astype(str)
    else:
        g = g.sort_values("Response rate (%)", ascending=False)
    return g.reset_index(drop=True)


def describe_rate_table(tbl: pd.DataFrame, col: str, overall: float, label: str, min_n: int = 15) -> str:
    valid = tbl[tbl["Customers"] >= min_n]
    if valid.empty:
        return f"Not enough customers per {label.lower()} group for a reliable comparison."
    best, worst = valid.loc[valid["Response rate (%)"].idxmax()], valid.loc[valid["Response rate (%)"].idxmin()]
    lift = best["Response rate (%)"] / overall if overall > 0 else np.nan
    return (f"By {label.lower()}, **{best[col]}** responds most ({best['Response rate (%)']:.1f}%, "
            f"{lift:.1f}x the overall {overall:.1f}%) and **{worst[col]}** responds least "
            f"({worst['Response rate (%)']:.1f}%). Group sizes: {int(best['Customers'])} vs {int(worst['Customers'])}.")


def generate_business_insights(df: pd.DataFrame, training: dict | None, model_name: str | None) -> dict:
    """Returns {section: [markdown bullets]}. Every number is derived from the current data/model."""
    ins = {"Targeting": [], "Campaign efficiency": [], "Customer engagement": [], "Channel strategy": [],
           "Campaign history": [], "Caveats": []}
    if df.empty:
        return ins
    overall = df[TARGET].mean() * 100
    min_n = max(15, int(0.02 * len(df)))

    for col, label in [("Income_Segment", "Income segment"), ("Age_Group", "Age group"),
                       ("Education", "Education"), ("Marital_Status", "Marital status"), ("Country", "Country")]:
        if col in df.columns:
            ins["Targeting"].append(describe_rate_table(rate_table(df, col), col, overall, label, min_n))
    if training and model_name:
        imp, _ = compute_feature_importance(training, model_name)
        top = ", ".join(imp["Feature"].head(5).tolist())
        ins["Targeting"].append(f"The most influential model features ({model_name}) are: **{top}**. "
                                "Customers who score highly on these are the natural first targets.")
    if training and model_name:
        gt = gains_table(training["y_test"], training["proba_test"][model_name])
        for pct in (20, 30):
            r = gt[gt["Customers contacted (%)"] == pct].iloc[0]
            ins["Campaign efficiency"].append(
                f"On the held-out test set, contacting only the top **{pct}%** of customers by predicted probability "
                f"would have reached **{r['Responders captured (%)']:.1f}%** of all responders "
                f"(lift {r['Lift (x)']:.2f}x versus untargeted contact), i.e. {100 - pct}% fewer contacts.")
        ins["Campaign efficiency"].append(
            "These are historical test-set results; actual savings depend on real contact costs and on the campaign "
            "behaving like the past one.")
    else:
        ins["Campaign efficiency"].append("Train the models (Model Training page) to quantify how much outreach "
                                          "probability-based targeting could save.")

    resp, non = df[df[TARGET] == 1], df[df[TARGET] == 0]
    diffs = []
    for c, label in [("Recency", "Recency (days since last purchase)"), ("Total_Spending", "Total spending"),
                     ("Total_Purchases", "Total purchases"), ("NumWebVisitsMonth", "Web visits per month"),
                     ("NumDealsPurchases", "Deal purchases"), ("Income", "Income"),
                     ("Customer_Tenure_Days", "Customer tenure (days)")]:
        if c in df.columns and len(resp) and len(non) and non[c].mean() != 0:
            a, b = resp[c].mean(), non[c].mean()
            diffs.append((abs(a / b - 1), f"**{label}**: responders average {a:,.1f} versus {b:,.1f} for "
                                          f"non-responders ({(a / b - 1) * 100:+.0f}%)."))
    ins["Customer engagement"] = [t for _, t in sorted(diffs, reverse=True)[:5]] or \
        ["Engagement columns were not found in this dataset."]

    if "Preferred_Channel" in df.columns:
        tbl = rate_table(df, "Preferred_Channel")
        ins["Channel strategy"].append(describe_rate_table(tbl, "Preferred_Channel", overall, "preferred purchase channel", min_n))
    for c in CHANNEL_COLS + ["NumDealsPurchases"]:
        if c in df.columns and len(resp) and len(non) and non[c].mean() > 0:
            change = (resp[c].mean() / non[c].mean() - 1) * 100
            verdict = f"({change:+.0f}%)" if abs(change) >= 5 else f"({change:+.0f}%, no material difference)"
            ins["Channel strategy"].append(
                f"**{CHANNEL_LABELS[c]} purchases**: responders average {resp[c].mean():.2f} versus "
                f"{non[c].mean():.2f} {verdict}.")

    if "Campaign_History" in df.columns:
        tbl = rate_table(df, "Campaign_History")
        none = tbl[tbl["Campaign_History"] == "None"]["Response rate (%)"]
        rest = df[df["Campaign_History"] != "None"]
        if len(none) and len(rest) >= 5:
            r_rest = rest[TARGET].mean() * 100
            ins["Campaign history"].append(
                f"Customers who accepted at least one earlier campaign respond at **{r_rest:.1f}%** versus "
                f"**{none.iloc[0]:.1f}%** for customers with no prior acceptance "
                f"({r_rest / none.iloc[0]:.1f}x)." if none.iloc[0] > 0 else
                f"Customers with prior acceptances respond at {r_rest:.1f}%.")
        ins["Campaign history"].append(describe_rate_table(tbl, "Campaign_History", overall, "campaign history", 5))
    else:
        ins["Campaign history"].append("Previous-campaign columns (AcceptedCmp1-5) were not found.")

    ins["Caveats"] = [
        "All findings are **associations in historical data**, not proven causes.",
        "Predicted probabilities are estimates, not guarantees of individual behaviour.",
        "AcceptedCmp1-5 are assumed to describe campaigns that ran BEFORE the latest one (see About Project).",
    ]
    return ins


def build_results_summary(prep: dict, training: dict | None, source: str) -> str:
    """Plain-text summary of computed results - handy for copying into the project report."""
    lines = ["MARKETING CAMPAIGN RESPONSE PREDICTION - RESULTS SUMMARY", "=" * 60, f"Data source: {source}",
             f"Records after cleaning: {len(prep['df_model'])}", f"Response rate: {prep['df_model'][TARGET].mean() * 100:.2f}%",
             "", "CLEANING LOG"]
    for r in prep["log"]:
        lines.append(f"- {r['Step']}: {r['Records affected']} | {r['Action']}")
    if training:
        lines += ["", f"Train/test split: {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}, stratified, "
                      f"random_state={RANDOM_STATE}", f"Features used ({len(training['feature_cols'])}): "
                                                     f"{', '.join(training['feature_cols'])}",
                  f"Historical campaign features included: {training['include_history']}",
                  f"Class weights used: {training['use_class_weight']}", "", "MODEL COMPARISON (test set)",
                  training["comparison"].round(4).to_string(index=False), "",
                  f"Model selected by criterion (highest cross-validated PR-AUC on training data): {training['selected']}"]
        for n, m in training["metrics"].items():
            tn, fp, fn, tp = m["cm"].ravel()
            lines.append(f"Confusion matrix {n}: TN={tn} FP={fp} FN={fn} TP={tp}")
    return "\n".join(lines)


# =====================================================================================
# SESSION-STATE ACCESSORS
# =====================================================================================
def short_name(name: str) -> str:
    return name.split(" (")[0]


def deployed_model_name(training: dict) -> str:
    name = st.session_state.get("deploy_choice")  # persistent copy (widget keys vanish when the page changes)
    return name if name in training["models"] else training["selected"]


def decision_threshold(training: dict) -> float:
    name = deployed_model_name(training)
    if st.session_state.get("use_tuned_thr", True):
        return float(training["metrics"][name]["threshold"])
    return float(st.session_state.get("custom_thr", 0.5))


def segment_thresholds() -> tuple[float, float]:
    high = float(st.session_state.get("thr_high", 0.70))
    med = float(st.session_state.get("thr_med", 0.40))
    return high, (med if med < high else max(high - 0.05, 0.0))


def require_training(training) -> bool:
    if training is None:
        st.warning("No trained model yet. Open **Model Training** in the sidebar and click **Train Models** first.")
        return False
    return True


def compute_kpis(df: pd.DataFrame) -> dict:
    def mean_of(col):
        return df[col].mean() if col in df.columns and len(df) else np.nan

    return {"n": len(df), "rate": df[TARGET].mean() * 100 if len(df) else np.nan, "income": mean_of("Income"),
            "spend": mean_of("Total_Spending"), "recency": mean_of("Recency")}


# =====================================================================================
# REUSABLE PANELS
# =====================================================================================
def render_class_distribution(df: pd.DataFrame, key: str = "cd") -> None:
    counts = df[TARGET].value_counts().reindex([0, 1]).fillna(0).astype(int)
    total = int(counts.sum())
    tbl = pd.DataFrame({"Class": ["0 -> Did not respond", "1 -> Responded"], "Customers": counts.values,
                        "Percentage (%)": (counts.values / max(total, 1) * 100).round(2)})
    c1, c2 = st.columns([1, 1])
    with c1:
        st_plot(fig_pie([RESPONSE_LABELS[0], RESPONSE_LABELS[1]], counts.values, "Response distribution"))
    with c2:
        st_table(tbl)
        minority = counts.min() / max(total, 1) * 100
        ratio = counts.max() / max(counts.min(), 1)
        if minority < 30:
            note(f"<b>Class imbalance detected.</b> Only {minority:.1f}% of customers responded "
                 f"(about 1 responder for every {ratio:.1f} non-responders). Implications: accuracy is misleading "
                 f"(always predicting 'no response' would already score {100 - minority:.1f}%), so precision, recall, "
                 f"F1, ROC-AUC and PR-AUC are used; splits are stratified; and the decision threshold is tuned "
                 f"instead of using 0.5 blindly.")
        else:
            note(f"Classes are reasonably balanced (minority class = {minority:.1f}%). Accuracy is more "
                 f"informative here, but precision/recall are still reported.")


def response_by_chart(df: pd.DataFrame, col: str, label: str) -> None:
    if col not in df.columns:
        st.info(f"Column for '{label}' is not available in this dataset.")
        return
    tbl = rate_table(df, col)
    overall = df[TARGET].mean() * 100
    st_plot(fig_bar(tbl, col, "Response rate (%)", f"Response rate by {label}", xlabel=label,
                    ylabel="Response rate (%)"))
    st.caption(describe_rate_table(tbl, col, overall, label, max(15, int(0.02 * len(df)))))
    with st.expander(f"Table: response by {label}"):
        st_table(tbl.round(2))


def render_model_evaluation_charts(training: dict, name: str) -> None:
    m = training["metrics"][name]
    tn, fp, fn, tp = m["cm"].ravel()
    t1, t2, t3 = st.tabs(["Confusion matrix", "ROC curves", "Precision-Recall curves"])
    with t1:
        cm_df = pd.DataFrame(m["cm"], index=["Actual: Did not respond", "Actual: Responded"],
                             columns=["Predicted: Did not respond", "Predicted: Responded"])
        c1, c2 = st.columns([1, 1])
        with c1:
            st_plot(fig_heatmap(cm_df, f"Confusion matrix - {short_name(name)} (threshold {m['threshold']:.2f})",
                                fmt=".0f", height=380))
        with c2:
            st.markdown(
                f"- **True Positive (TP) = {tp}** - predicted to respond and did respond.\n"
                f"- **True Negative (TN) = {tn}** - predicted not to respond and did not.\n"
                f"- **False Positive (FP) = {fp}** - predicted to respond but did **not**. Marketing consequence: "
                f"*unnecessary campaign cost* (wasted contacts).\n"
                f"- **False Negative (FN) = {fn}** - predicted not to respond but actually **did**. Marketing "
                f"consequence: *missed opportunity* (lost revenue).\n\n"
                f"At this threshold the model contacts {tp + fp} customers on the test set and reaches {tp} of the "
                f"{tp + fn} real responders. Lowering the threshold catches more responders (fewer FN) but wastes more "
                f"contacts (more FP); raising it does the opposite.")
    with t2:
        curves = [(f"{n} (AUC = {mm['roc_auc']:.3f})", mm["fpr"], mm["tpr"]) for n, mm in training["metrics"].items()]
        st_plot(fig_curve(curves, "ROC curves (test set)", "False Positive Rate", "True Positive Rate (Recall)",
                          diagonal=True))
        st.caption("ROC-AUC is the probability that the model ranks a random responder above a random "
                   "non-responder (0.5 = random guessing, 1.0 = perfect ranking).")
    with t3:
        prevalence = float(training["y_test"].mean())
        curves = [(f"{n} (AP = {mm['pr_auc']:.3f})", mm["rec_curve"], mm["prec_curve"])
                  for n, mm in training["metrics"].items()]
        st_plot(fig_curve(curves, "Precision-Recall curves (test set)", "Recall", "Precision", hline=prevalence,
                          hline_label=f"No-skill baseline = {prevalence:.2f}"))
        st.caption("When positive responses are rare, ROC curves can look optimistic because the huge number of "
                   "true negatives keeps the false-positive rate low. The Precision-Recall curve focuses only on the "
                   "responders: how many of the customers we contact are real responders (precision) versus how "
                   "many real responders we reach (recall). The dashed line is the precision of random targeting.")


# =====================================================================================
# PAGE: DASHBOARD
# =====================================================================================
def render_dashboard(df: pd.DataFrame, training) -> None:
    st.markdown('<div class="hero"><h1>MARKETING CAMPAIGN ANALYTICS</h1>'
                '<p>Customer Response Prediction</p></div>', unsafe_allow_html=True)
    k = compute_kpis(df)
    best, auc = "Not trained", "N/A"
    if training:
        name = deployed_model_name(training)
        best, auc = short_name(name), f"{training['metrics'][name]['roc_auc']:.3f}"
    kpi_row([("Customers", f"{k['n']:,}"), ("Response Rate", fmt_pct(k["rate"])), ("Avg Income", fmt_money(k["income"])),
             ("Avg Spending", fmt_money(k["spend"])), ("Best Model", best), ("ROC-AUC", auc)])
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        counts = df[TARGET].value_counts().reindex([0, 1]).fillna(0).astype(int)
        st_plot(fig_pie([RESPONSE_LABELS[0], RESPONSE_LABELS[1]], counts.values, "Response distribution"))
    with c2:
        if training:
            imp, _ = compute_feature_importance(training, deployed_model_name(training))
            top = imp.head(10).sort_values("Importance")
            st_plot(fig_bar(top, "Feature", "Importance", "Top influential factors", xlabel="Feature",
                            ylabel="Importance", horizontal=True, text_fmt=".3f", height=380))
        else:
            st.info("Top influential factors appear here after the models are trained.")

    st.markdown("#### Campaign insights")
    ins = generate_business_insights(df, training, deployed_model_name(training) if training else None)
    bullets = ins["Targeting"][:2] + ins["Campaign history"][:1] + ins["Channel strategy"][:1]
    for b in bullets:
        st.markdown(f"- {b}")

    st.markdown("#### Model performance")
    if training:
        st_table(training["comparison"].round(3))
        st.caption(f"Deployment model: **{best}** - selected by highest cross-validated PR-AUC on the training split.")
    else:
        st.info("Model performance appears here after training.")


# =====================================================================================
# PAGE: DATA OVERVIEW
# =====================================================================================
def build_data_dictionary(raw: pd.DataFrame, df_model: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in raw.columns:
        c = str(c).strip()
        if c in DATA_DICTIONARY:
            desc, typ = DATA_DICTIONARY[c]
        else:
            desc = "Not in the standard dictionary (auto-detected column)"
            typ = "Numeric" if pd.api.types.is_numeric_dtype(raw[c]) else "Categorical"
        rows.append({"Feature": c, "Description": desc, "Type": typ, "Origin": "Original"})
    for c, (desc, typ, why) in ENGINEERED_DICTIONARY.items():
        if c in df_model.columns:
            rows.append({"Feature": c, "Description": f"{desc}. {why}", "Type": typ, "Origin": "Engineered"})
    return pd.DataFrame(rows)


def render_data_overview(raw: pd.DataFrame, prep: dict, df: pd.DataFrame, source: str) -> None:
    section_header("Data Overview", f"Source: {source}. KPIs reflect the sidebar filters.")
    k = compute_kpis(df)
    kpi_row([("Total Customers", f"{k['n']:,}"), ("Total Features (raw columns)", f"{raw.shape[1]}"),
             ("Response Rate", fmt_pct(k["rate"])), ("Average Income", fmt_money(k["income"])),
             ("Average Spending", fmt_money(k["spend"])), ("Average Recency", f"{k['recency']:.1f} days"
                                                           if pd.notna(k["recency"]) else "N/A")])
    tabs = st.tabs(["Preview", "Data types & missing", "Duplicates", "Numerical summary", "Categorical summary",
                    "Cleaning log", "Data dictionary", "Download"])
    with tabs[0]:
        st.caption("First 50 rows of the raw file exactly as loaded.")
        st_table(raw.head(50), hide_index=False)
    with tabs[1]:
        info = pd.DataFrame({"Column": raw.columns, "Data type": raw.dtypes.astype(str).values,
                             "Non-null": raw.notna().sum().values, "Missing": raw.isna().sum().values,
                             "Missing (%)": (raw.isna().mean() * 100).round(2).values,
                             "Unique values": raw.nunique().values})
        st_table(info)
        total_missing = int(raw.isna().sum().sum())
        if total_missing == 0:
            st.success("No missing values in the raw file.")
        else:
            st.warning(f"{total_missing} missing cells across {int((raw.isna().sum() > 0).sum())} column(s); "
                       f"handled by median (numeric) / mode (categorical) imputation inside the ML pipeline.")
    with tabs[2]:
        c1, c2 = st.columns(2)
        c1.metric("Fully identical rows (raw)", prep["dup_all"])
        c2.metric("Identical except ID", prep["dup_excl_id"])
        st.caption("Rows identical in every field except ID are treated as duplicate customers and removed.")
    with tabs[3]:
        num = raw.select_dtypes(include="number")
        if num.shape[1]:
            st_table(num.describe().T.round(2).reset_index().rename(columns={"index": "Feature"}))
        else:
            st.info("No numeric columns found.")
    with tabs[4]:
        cat = raw.select_dtypes(exclude="number")
        if cat.shape[1]:
            st_table(cat.describe().T.reset_index().rename(columns={"index": "Feature"}).astype(str))
        else:
            st.info("No categorical columns found.")
    with tabs[5]:
        st_table(pd.DataFrame(prep["log"])[["Step", "Records affected", "Action", "Reason"]])
    with tabs[6]:
        st_table(build_data_dictionary(raw, prep["df_model"]))
        st.markdown("**Engineered feature logic**")
        eng = pd.DataFrame([{"Feature": c, "Formula": d, "Business rationale": w}
                            for c, (d, _, w) in ENGINEERED_DICTIONARY.items() if c in prep["df_model"].columns])
        st_table(eng)
    with tabs[7]:
        export = prep["df_eda"].drop(columns=["Response_Label"], errors="ignore")
        st.caption("Cleaned + feature-engineered dataset (descriptive median/mode imputation applied; the model "
                   "imputes separately inside its pipeline).")
        download_df_button("Download cleaned dataset (CSV)", export, "cleaned_marketing_campaign.csv", "dl_clean")


# =====================================================================================
# PAGE: EDA
# =====================================================================================
def render_eda(df: pd.DataFrame) -> None:
    section_header("EDA & Insights", "Every chart updates with the sidebar filters.")
    tabs = st.tabs(["Target", "Demographics", "Customer behaviour", "Response analysis", "Correlations"])
    with tabs[0]:
        st.markdown("**Target variable:** `Response`  |  `0` -> Did not respond  |  `1` -> Responded")
        render_class_distribution(df)
    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            if "Age" in df.columns:
                st_plot(fig_hist(df, "Age", "Age distribution", xlabel="Age (years)", color="Response_Label",
                                 color_map=COLOR_MAP, nbins=25))
        with c2:
            if "Income" in df.columns:
                st_plot(fig_hist(df, "Income", "Income distribution", xlabel="Annual income", color="Response_Label",
                                 color_map=COLOR_MAP, nbins=40))
        c3, c4 = st.columns(2)
        for col, label, target in [("Education", "Education level", c3), ("Marital_Status", "Marital status", c4)]:
            with target:
                if col in df.columns:
                    vc = df[col].value_counts().reset_index()
                    vc.columns = [col, "Customers"]
                    st_plot(fig_bar(vc, col, "Customers", f"{label} distribution", xlabel=label,
                                    ylabel="Number of customers", text_fmt=".0f"))
    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            if "Total_Spending" in df.columns:
                st_plot(fig_hist(df, "Total_Spending", "Total spending distribution", xlabel="Total spending",
                                 color="Response_Label", color_map=COLOR_MAP, nbins=40))
        with c2:
            if "Recency" in df.columns:
                st_plot(fig_hist(df, "Recency", "Recency distribution", xlabel="Days since last purchase",
                                 color="Response_Label", color_map=COLOR_MAP, nbins=30))
        rows = [{"Category": PRODUCT_LABELS[c], "Response group": lab, "Average spend": g[c].mean()}
                for c in SPEND_COLS if c in df.columns for lab, g in df.groupby("Response_Label")]
        if rows:
            st_plot(fig_bar(pd.DataFrame(rows), "Category", "Average spend", "Average spending by product category",
                            xlabel="Product category", ylabel="Average spend", color="Response group",
                            color_map=COLOR_MAP, text_fmt=".0f"))
        rows = [{"Channel": CHANNEL_LABELS[c], "Response group": lab, "Average purchases": g[c].mean()}
                for c in CHANNEL_COLS + ["NumDealsPurchases"] if c in df.columns for lab, g in df.groupby("Response_Label")]
        if rows:
            st_plot(fig_bar(pd.DataFrame(rows), "Channel", "Average purchases", "Average purchases by channel",
                            xlabel="Purchase channel", ylabel="Average purchases", color="Response group",
                            color_map=COLOR_MAP, text_fmt=".2f"))
        c3, c4 = st.columns(2)
        with c3:
            if "NumWebVisitsMonth" in df.columns:
                st_plot(fig_hist(df, "NumWebVisitsMonth", "Web visits per month", xlabel="Web visits (last month)",
                                 color="Response_Label", color_map=COLOR_MAP, nbins=15))
        with c4:
            if "NumDealsPurchases" in df.columns:
                st_plot(fig_hist(df, "NumDealsPurchases", "Deal purchases", xlabel="Purchases made with a discount",
                                 color="Response_Label", color_map=COLOR_MAP, nbins=15))
    with tabs[3]:
        st.metric("Overall response rate (filtered)", fmt_pct(df[TARGET].mean() * 100))
        pairs = [("Education", "education"), ("Marital_Status", "marital status"), ("Income_Segment", "income segment"),
                 ("Age_Group", "age group"), ("Campaign_History", "previous campaign acceptance"),
                 ("Spending_Segment", "spending level"), ("Preferred_Channel", "preferred purchase channel")]
        for i in range(0, len(pairs), 2):
            cols = st.columns(2)
            for j, (col, label) in enumerate(pairs[i:i + 2]):
                with cols[j]:
                    response_by_chart(df, col, label)
    with tabs[4]:
        cols = [c for c in df.select_dtypes(include="number").columns
                if c not in (ID_COL, "Z_CostContact", "Z_Revenue") and df[c].nunique() > 1]
        corr = df[cols].corr()
        st_plot(fig_heatmap(corr, "Correlation heatmap (numeric features)", fmt=".2f", height=720,
                            colorscale="RdBu_r", midpoint=0, show_text=False))
        tgt = corr[TARGET].drop(TARGET).sort_values(key=lambda s: s.abs(), ascending=False).head(10)
        st.markdown("**Features most correlated with `Response`** (Pearson; correlation is not causation)")
        st_table(tgt.round(3).rename("Correlation with Response").reset_index().rename(columns={"index": "Feature"}))


# =====================================================================================
# PAGE: CAMPAIGN ANALYSIS
# =====================================================================================
def render_campaign_analysis(df: pd.DataFrame) -> None:
    section_header("Campaign Analysis", "Performance of the latest campaign and its relationship to history, "
                                        "segments and channels (filtered data).")
    overall = df[TARGET].mean() * 100
    responders = int(df[TARGET].sum())
    accept_cols = [c for c in CAMPAIGN_COLS if c in df.columns]
    avg_prev = np.mean([df[c].mean() * 100 for c in accept_cols]) if accept_cols else np.nan
    kpi_row([("Overall response rate", fmt_pct(overall)), ("Responders", f"{responders:,}"),
             ("Customers contacted", f"{len(df):,}"), ("Avg acceptance of earlier campaigns", fmt_pct(avg_prev))])
    if accept_cols:
        rows = sorted([{"Campaign": f"Campaign {c[-1]}", "Acceptance rate (%)": df[c].mean() * 100}
                       for c in accept_cols], key=lambda r: r["Campaign"])
        rows.append({"Campaign": "Latest (Response)", "Acceptance rate (%)": overall})
        st_plot(fig_bar(pd.DataFrame(rows), "Campaign", "Acceptance rate (%)", "Acceptance rate by campaign",
                        xlabel="Campaign", ylabel="Acceptance rate (%)"))
        st.caption("Campaigns 1-5 are earlier campaigns; the latest campaign (the target) is compared against them.")
    c1, c2 = st.columns(2)
    with c1:
        response_by_chart(df, "Campaign_History", "campaign history (earlier acceptances)")
    with c2:
        response_by_chart(df, "Spending_Segment", "customer value segment (spending quartile)")
    c3, c4 = st.columns(2)
    with c3:
        response_by_chart(df, "Preferred_Channel", "preferred purchase channel")
    with c4:
        response_by_chart(df, "Income_Segment", "income segment")
    if {"Income_Segment", "Spending_Segment"} <= set(df.columns):
        pv = (df.pivot_table(index="Income_Segment", columns="Spending_Segment", values=TARGET, aggfunc="mean") * 100)
        pv = pv.reindex(index=ORDERS["Income_Segment"], columns=ORDERS["Spending_Segment"])
        st_plot(fig_heatmap(pv, "Response rate (%) by income segment x spending segment", fmt=".1f", height=420,
                            colorscale="Blues"))
        st.caption("Cells built from few customers are noisy - check the counts before acting on a single cell.")
    prof_cols = [c for c in ["Age", "Income", "Recency", "Total_Spending", "Total_Purchases", "NumWebVisitsMonth",
                             "NumDealsPurchases", "Previous_Campaign_Acceptance", "Customer_Tenure_Days"]
                 if c in df.columns]
    if prof_cols:
        st.markdown("#### Responder vs non-responder profile (averages)")
        prof = df.groupby("Response_Label")[prof_cols].mean().T.round(2)
        prof.index.name = "Feature"
        st_table(prof.reset_index())
    st.markdown("#### Interpretation")
    ins = generate_business_insights(df, None, None)
    for b in ins["Campaign history"] + ins["Channel strategy"][:1]:
        st.markdown(f"- {b}")


# =====================================================================================
# PAGE: MODEL TRAINING
# =====================================================================================
def render_model_training(prep: dict, source: str) -> None:
    section_header("Model Training", "Train three classifiers with a leakage-safe sklearn Pipeline, "
                                     f"stratified {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} split, "
                                     f"random_state={RANDOM_STATE}.")
    df = prep["df_model"]
    st.markdown("#### Target and class balance")
    render_class_distribution(df)

    with st.expander("Feature selection & data-leakage logic", expanded=False):
        num, cat, excl = select_features(df, st.session_state.get("include_history", True))
        st.markdown(
            "**Assumption:** `Response` is the outcome of the **latest** campaign. `AcceptedCmp1-5` describe "
            "**earlier** campaigns, so they are known *before* the latest campaign is launched and are treated as "
            "legitimate historical predictors. If in your business they were recorded *after* the campaign, tick "
            "the option below to exclude them.")
        st.markdown(f"**Numeric predictors ({len(num)}):** {', '.join(num)}")
        st.markdown(f"**Categorical predictors ({len(cat)}):** {', '.join(cat) if cat else 'none'}")
        st.markdown("**Excluded columns:**")
        st_table(pd.DataFrame([{"Column": k, "Reason": v} for k, v in excl.items()]))
        eng = pd.DataFrame([{"Engineered feature": c, "Formula": d, "Why it matters": w}
                            for c, (d, _, w) in ENGINEERED_DICTIONARY.items() if c in df.columns])
        st_table(eng)

    st.markdown("#### Training options")
    c1, c2 = st.columns(2)
    with c1:
        include_history = st.checkbox("Include historical campaign acceptance features (AcceptedCmp1-5)", value=True,
                                      key="include_history")
    with c2:
        use_cw = st.checkbox("Use class weights (raises recall, but inflates predicted probabilities)", value=False,
                             key="use_class_weight")
    if not XGB_OK:
        st.info("XGBoost is not available here - HistGradientBoostingClassifier will be used as the boosted model. "
                "Install with `pip install xgboost` for XGBoost.")

    if st.button("Train Models", type="primary", key="train_btn"):
        bar = st.progress(0.0, text="Starting training...")
        try:
            training = train_models(df, include_history, use_cw, prep["ref_date"],
                                    progress_cb=lambda f, t: bar.progress(float(min(max(f, 0.0), 1.0)), text=t))
            training["id"] = datetime.now().isoformat()
            st.session_state["training"] = training
            st.session_state.pop("deploy_model", None)
            st.session_state.pop("deploy_choice", None)
            st.session_state["last_pred"] = None
            st.success("Training finished. Results below; other pages now use the trained models.")
        except ValueError as exc:
            st.error(f"Training could not start: {exc}")
        except Exception as exc:  # keep the UI friendly; no raw stack trace
            st.error(f"Model training failed ({type(exc).__name__}): {exc}. Check the dataset and package versions.")
        finally:
            bar.empty()

    training = st.session_state.get("training")
    if training is None:
        st.info("Click **Train Models** to fit Logistic Regression, Random Forest and a boosted-tree model.")
        return

    st.markdown("---")
    st.markdown("#### Model comparison (held-out test set)")
    comp = training["comparison"].round(3)
    st_table(comp)
    st.caption(f"Train rows: {len(training['y_train'])}, test rows: {len(training['y_test'])}. Precision, recall and F1 "
               "use each model's decision threshold, tuned on cross-validated *training* predictions (never on the "
               "test set). ROC-AUC and PR-AUC are threshold-independent.")

    st.markdown("#### Model selection criterion")
    note("<b>Criterion (documented):</b> the model with the highest <b>cross-validated PR-AUC on the training split</b> "
         "is proposed for deployment (ties broken by cross-validated ROC-AUC). PR-AUC is preferred because responders "
         "are the minority class and the business cares about finding them without wasting contacts. The test set is "
         "<b>not</b> used to choose the model, so the test metrics above remain an honest estimate. This is a "
         "transparent rule, not a claim that the model is universally best - you can override it below, e.g. if your "
         "business values recall over precision.")
    names = list(training["models"])
    st.selectbox("Model used for prediction / targeting / explainability", names,
                 index=names.index(deployed_model_name(training)), key="deploy_model")
    st.session_state["deploy_choice"] = st.session_state["deploy_model"]
    name = deployed_model_name(training)
    m = training["metrics"][name]
    kpi_row([("Selected model", short_name(name)), ("Precision", f"{m['precision']:.3f}"), ("Recall", f"{m['recall']:.3f}"),
             ("F1", f"{m['f1']:.3f}"), ("ROC-AUC", f"{m['roc_auc']:.3f}"), ("PR-AUC", f"{m['pr_auc']:.3f}")])
    st.markdown(
        f"**Trade-off:** at threshold **{m['threshold']:.2f}**, about **{m['precision'] * 100:.1f}%** of customers the "
        f"model flags actually respond (*precision* - controls wasted contact cost) and it finds **{m['recall'] * 100:.1f}%** "
        f"of all real responders (*recall* - controls missed opportunity). *F1* balances the two; *ROC-AUC/PR-AUC* judge "
        f"the ranking quality independent of any threshold.")
    render_model_evaluation_charts(training, name)

    d1, d2 = st.columns(2)
    with d1:
        download_df_button("Download model comparison (CSV)", training["comparison"].round(4), "model_comparison.csv",
                           "dl_comp")
    with d2:
        st.download_button("Download results summary (TXT)", data=build_results_summary(prep, training, source).encode("utf-8"),
                           file_name="results_summary.txt", mime="text/plain", key="dl_summary")


# =====================================================================================
# PAGE: PREDICTION
# =====================================================================================
def _int_input(label: str, series: pd.Series, key: str, floor_max: int = 10):
    med = int(round(series.median())) if series.notna().any() else 0
    mx = max(int(series.max() * 1.5) if series.notna().any() else floor_max, med + 1, floor_max)
    return st.number_input(label, min_value=0, max_value=mx, value=med, step=1, key=key)


def render_prediction(prep: dict, training) -> None:
    section_header("Customer Prediction", "Enter a customer profile to estimate the probability of responding.")
    if not require_training(training):
        return
    df = prep["df_model"]
    name = deployed_model_name(training)
    row: dict = {}
    with st.form("customer_form"):
        st.markdown("##### Customer information")
        c1, c2, c3 = st.columns(3)
        with c1:
            if "Age" in df.columns:
                row["Age"] = st.number_input("Age", MIN_AGE, MAX_AGE, int(np.clip(round(df["Age"].median()), MIN_AGE, MAX_AGE)),
                                             1, key="in_age")
            if "Income" in df.columns:
                row["Income"] = st.number_input("Annual income", 0.0, float(max(df["Income"].max() * 2, 200000.0)),
                                                float(df["Income"].median()), 1000.0, key="in_income")
            if "Recency" in df.columns:
                row["Recency"] = _int_input("Recency (days since last purchase)", df["Recency"], "in_recency", 365)
            if "Customer_Tenure_Days" in df.columns:
                row["Customer_Tenure_Days"] = _int_input("Customer tenure (days)", df["Customer_Tenure_Days"], "in_tenure", 365)
        with c2:
            for cat_col in training["categorical"]:
                opts = sorted(df[cat_col].dropna().astype(str).unique())
                if not opts:
                    continue
                mode = df[cat_col].mode(dropna=True)
                default = opts.index(str(mode.iloc[0])) if len(mode) and str(mode.iloc[0]) in opts else 0
                row[cat_col] = st.selectbox(cat_col.replace("_", " "), opts, index=default, key=f"in_{cat_col}")
            for c in ("Kidhome", "Teenhome"):
                if c in df.columns:
                    row[c] = st.number_input(f"{c} (children at home)", 0, 6, int(round(df[c].median())), 1, key=f"in_{c}")
            if "Complain" in df.columns:
                row["Complain"] = 1 if st.checkbox("Complained in the last 2 years", value=False, key="in_complain") else 0
        with c3:
            for c in SPEND_COLS:
                if c in df.columns:
                    row[c] = _int_input(f"{PRODUCT_LABELS[c]} spending", df[c], f"in_{c}", 100)
        st.markdown("##### Purchase behaviour and history")
        d = st.columns(5)
        beh = [("NumWebPurchases", "Web purchases"), ("NumCatalogPurchases", "Catalog purchases"),
               ("NumStorePurchases", "Store purchases"), ("NumWebVisitsMonth", "Web visits / month"),
               ("NumDealsPurchases", "Deal purchases")]
        for col_ui, (c, label) in zip(d, beh):
            if c in df.columns:
                with col_ui:
                    row[c] = _int_input(label, df[c], f"in_{c}", 10)
        accepted = []
        avail_cmp = [c for c in CAMPAIGN_COLS if c in df.columns]
        if avail_cmp:
            accepted = st.multiselect("Earlier campaigns this customer accepted", avail_cmp, default=[], key="in_accepted")
        submitted = st.form_submit_button("Predict Campaign Response")

    if submitted:
        for c in CAMPAIGN_COLS:
            if c in df.columns:
                row[c] = 1 if c in accepted else 0
        try:
            fe, _ = feature_engineering(pd.DataFrame([row]), ref_date=training["ref_date"])
            X, missing = align_features(fe, training)
            st.session_state["last_pred"] = {"X": X, "raw": pd.DataFrame([row]), "missing": missing,
                                             "train_id": training.get("id")}
        except Exception as exc:
            st.error(f"Could not build the prediction input ({type(exc).__name__}): {exc}")
            return

    last = st.session_state.get("last_pred")
    if not last or last.get("train_id") != training.get("id"):
        st.info("Fill the form and click **Predict Campaign Response**.")
        return
    try:
        p = float(training["models"][name].predict_proba(last["X"])[0, 1])
    except Exception as exc:
        st.error(f"Prediction failed - the input does not match the trained model ({type(exc).__name__}). "
                 "Re-submit the form.")
        return
    thr, (high, med) = decision_threshold(training), segment_thresholds()
    likely = p >= thr
    segment = str(assign_segment([p], high, med)[0])
    if last["missing"]:
        st.warning("These model features were not provided and were filled with the training median/mode: "
                   + ", ".join(last["missing"]))
    st.markdown("---")
    r1, r2 = st.columns([1, 1])
    with r1:
        st.markdown(f'<div class="{"pred-yes" if likely else "pred-no"}">'
                    f'{"LIKELY TO RESPOND" if likely else "UNLIKELY TO RESPOND"}</div>', unsafe_allow_html=True)
        st.markdown(f"### Response Probability: {p * 100:.1f}%")
        st.progress(float(min(max(p, 0.0), 1.0)))
        st.markdown(f"> The model estimates a **{p * 100:.1f}%** probability of response based on the provided "
                    f"features. This is a statistical estimate, not a certainty about this customer's behaviour.")
        st.caption(f"Model: {name}. Class rule: predicted 'responds' if probability >= {thr:.2f} "
                   f"(threshold {'tuned on training data' if st.session_state.get('use_tuned_thr', True) else 'set manually'}).")
    with r2:
        g = fig_gauge(p * 100, "Predicted response probability")
        if g is not None:
            st_plot(g)
        st.markdown(f"**Marketing action tier: {segment} predicted response**")
        st.markdown(f"- Probability >= {high:.2f} -> High predicted response\n"
                    f"- {med:.2f} to {high:.2f} -> Medium predicted response\n"
                    f"- Probability < {med:.2f} -> Low predicted response")
        base = float(training["y_train"].mean())
        st.caption(f"For context, the average response rate in the training data is {base * 100:.1f}%. Tiers are "
                   f"model-based groupings (adjustable in the sidebar), not guaranteed customer segments.")
    out = last["raw"].copy()
    out["Response_Probability"] = round(p, 4)
    out["Prediction"] = int(likely)
    out["Predicted_Response_Tier"] = segment
    download_df_button("Download this prediction (CSV)", out, "single_customer_prediction.csv", "dl_single_pred")


# =====================================================================================
# PAGE: CUSTOMER TARGETING (+ business simulation)
# =====================================================================================
def render_customer_targeting(prep: dict, training) -> None:
    section_header("Customer Targeting", "Upload a customer file to score it and download a campaign targeting list.")
    if not require_training(training):
        return
    name = deployed_model_name(training)
    pipe = training["models"][name]
    thr, (high, med) = decision_threshold(training), segment_thresholds()
    up = st.file_uploader("Upload customer CSV (same layout as marketing_campaign.csv; Response column optional)",
                          type=["csv", "tsv", "txt"], key="target_upload")
    proba_upload = None
    if up is not None:
        try:
            raw_up = load_data_from_bytes(up.getvalue())
            cust, _ = standardise_columns(raw_up)
            fe, _ = feature_engineering(cust, ref_date=training["ref_date"])
            X, missing = align_features(fe, training)
            if len(missing) > 0.5 * len(training["feature_cols"]):
                st.error("The uploaded file does not look like the training data - more than half of the required "
                         f"columns are missing: {', '.join(missing)}")
                return
            if missing:
                st.warning("Missing columns (filled with training median/mode): " + ", ".join(missing))
            if "Age" in fe.columns:
                bad_age = int((~fe["Age"].between(MIN_AGE, MAX_AGE) & fe["Age"].notna()).sum())
                if bad_age:
                    st.warning(f"{bad_age} customers have an age outside {MIN_AGE}-{MAX_AGE}; they were scored as given.")
            proba_upload = pipe.predict_proba(X)[:, 1]
            ids = cust[ID_COL] if ID_COL in cust.columns else pd.Series(np.arange(1, len(cust) + 1))
            table = pd.DataFrame({"Customer": ids.values, "Probability": np.round(proba_upload, 4),
                                  "Prediction": (proba_upload >= thr).astype(int),
                                  "Segment": assign_segment(proba_upload, high, med)})
            table = table.sort_values("Probability", ascending=False).reset_index(drop=True)
            table.insert(0, "Rank", np.arange(1, len(table) + 1))
            kpi_row([("Customers scored", f"{len(table):,}"), ("Predicted responders", f"{int(table['Prediction'].sum()):,}"),
                     ("High", f"{int((table['Segment'] == 'High').sum()):,}"),
                     ("Medium", f"{int((table['Segment'] == 'Medium').sum()):,}"),
                     ("Low", f"{int((table['Segment'] == 'Low').sum()):,}")])
            c1, c2 = st.columns(2)
            with c1:
                seg = table["Segment"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
                seg.columns = ["Segment", "Customers"]
                st_plot(fig_bar(seg, "Segment", "Customers", "Customers by predicted response tier",
                                xlabel="Predicted response tier", ylabel="Customers", text_fmt=".0f"))
            with c2:
                st_plot(fig_hist(table, "Probability", "Distribution of predicted probabilities",
                                 xlabel="Predicted response probability", nbins=30))
            st.caption(f"Model: {name}. Prediction = 1 if probability >= {thr:.2f}. Tiers: High >= {high:.2f}, "
                       f"Medium >= {med:.2f}, otherwise Low. The average training response rate is "
                       f"{training['y_train'].mean() * 100:.1f}%, so with high tier cut-offs few customers may reach "
                       "'High' - adjust the thresholds in the sidebar if needed.")
            st_table(table)
            download_df_button("Download campaign_targeting_predictions.csv", table,
                               "campaign_targeting_predictions.csv", "dl_targeting")
        except ValueError as exc:
            st.error(f"Could not read the uploaded file: {exc}")
            return
        except Exception as exc:
            st.error(f"Scoring failed ({type(exc).__name__}): {exc}. Check that the file has the expected columns.")
            return
    else:
        st.info("Upload a CSV to score customers. Without an upload, the simulation below uses the held-out test set.")

    render_simulation(training, name, proba_upload)


def render_simulation(training: dict, name: str, proba_upload) -> None:
    st.markdown("---")
    section_header("Campaign Budget Simulation",
                   "Scenario simulation based on model probabilities and YOUR assumptions - not guaranteed financial results.")
    options = ["Held-out test set (backtest with known outcomes)"]
    if proba_upload is not None:
        options.insert(0, "Uploaded customer list")
    source = st.radio("Customers to simulate on", options, horizontal=True, key="sim_source")
    if source.startswith("Uploaded"):
        proba, y_actual = proba_upload, None
    else:
        proba, y_actual = training["proba_test"][name], training["y_test"].values
    df_model = st.session_state["prep"]["df_model"]
    cost_default = float(df_model["Z_CostContact"].median()) if "Z_CostContact" in df_model.columns else 3.0
    rev_default = float(df_model["Z_Revenue"].median()) if "Z_Revenue" in df_model.columns else 11.0
    c1, c2, c3 = st.columns(3)
    budget = c1.number_input("Campaign budget", min_value=0.0, value=float(max(round(len(proba) * cost_default * 0.3, -1), 100.0)),
                             step=100.0, key="sim_budget")
    cost = c2.number_input("Cost per contact", min_value=0.01, value=max(cost_default, 0.01), step=0.5, key="sim_cost")
    revenue = c3.number_input("Revenue per successful response", min_value=0.0, value=rev_default, step=1.0, key="sim_rev")
    st.caption("Defaults are pre-filled from the dataset's Z_CostContact / Z_Revenue fields when present; edit freely.")
    s = simulate_campaign(proba, budget, cost, revenue, y_actual)
    st.markdown(f"**Maximum contacts = Budget / Cost per contact = {s['max_contacts']:,}** "
                f"(the list has {s['list_size']:,} customers, so **{s['contacts']:,}** are contacted, highest probability first).")
    kpi_row([("Expected responses", f"{s['expected_responses']:.1f}"), ("Expected revenue", f"{s['expected_revenue']:,.0f}"),
             ("Estimated campaign cost", f"{s['cost']:,.0f}"), ("Estimated ROI", fmt_pct(s["roi"], 0))])
    cmp_rows = [{"Strategy": "Model-ranked targeting", "Contacts": s["contacts"],
                 "Expected responses": round(s["expected_responses"], 1), "Expected revenue": round(s["expected_revenue"], 0),
                 "Cost": round(s["cost"], 0), "ROI (%)": round(s["roi"], 0) if pd.notna(s["roi"]) else np.nan},
                {"Strategy": "Untargeted (random) contact", "Contacts": s["contacts"],
                 "Expected responses": round(s["random_responses"], 1), "Expected revenue": round(s["random_revenue"], 0),
                 "Cost": round(s["cost"], 0), "ROI (%)": round(s["random_roi"], 0) if pd.notna(s["random_roi"]) else np.nan}]
    st_table(pd.DataFrame(cmp_rows))
    if y_actual is not None:
        st.markdown(f"**Backtest with real outcomes:** among the top {s['contacts']:,} customers ranked by the model, "
                    f"**{s['actual_responses_top']}** actually responded, versus about "
                    f"**{s['actual_responses_random']:.1f}** expected from random contact of the same size.")
    note("Expected responses = sum of the model's probabilities for the contacted customers. Probabilities are "
         "estimates (calibration is not guaranteed, especially when class weights are used), revenue per response "
         "and costs are your assumptions, and past behaviour may not repeat. Use this to compare scenarios, not to "
         "promise results.", warn=True)


# =====================================================================================
# PAGE: MODEL EXPLAINABILITY
# =====================================================================================
def render_explainability(training) -> None:
    section_header("Model Explainability", "Which factors drive the model's predictions - in business language.")
    if not require_training(training):
        return
    names = list(training["models"])
    name = st.selectbox("Model to explain", names, index=names.index(deployed_model_name(training)), key="explain_model")
    imp, method = compute_feature_importance(training, name)
    t1, t2, t3 = st.tabs(["Feature importance", "Business interpretation", "SHAP (optional)"])
    with t1:
        top_n = (st.slider("Number of features to show", 5, min(30, len(imp)), min(15, len(imp)), key="imp_topn")
                 if len(imp) > 5 else len(imp))
        top = imp.head(top_n).sort_values("Importance")
        if "Direction" in top.columns:
            st_plot(fig_bar(top, "Feature", "Coefficient", f"Logistic Regression coefficients (top {top_n})",
                            xlabel="Feature", ylabel="Standardised coefficient", horizontal=True, text_fmt=".2f",
                            color="Direction", height=max(380, 26 * top_n),
                            color_map={"Raises response probability": "#1F6FEB", "Lowers response probability": "#D9534F"}))
        else:
            st_plot(fig_bar(top, "Feature", "Importance", f"Feature importance - {short_name(name)} (top {top_n})",
                            xlabel="Feature", ylabel="Importance", horizontal=True, text_fmt=".3f",
                            height=max(380, 26 * top_n)))
        st.caption(method)
        st_table(imp.head(top_n).round(4))
        download_df_button("Download feature importance (CSV)", imp.round(5), "feature_importance.csv", "dl_imp")
        if st.button("Compute permutation importance (model-agnostic check)", key="perm_btn"):
            with st.spinner("Shuffling features on the test set..."):
                perm = permutation_feature_importance(training["models"][name], training["X_test"], training["y_test"])
            st.markdown("**Permutation importance** - drop in PR-AUC when a feature is randomly shuffled:")
            st_table(perm.head(15).round(4))
    with t2:
        st.markdown("Reading the model in plain language (associations from the fitted model, **not causal proof**):")
        for feat in imp["Feature"].head(8):
            st.markdown(f"- **{feat}** - {feature_direction_text(feat, training, name)}.")
        st.markdown(
            "\n**How to use this:** features at the top explain most of the variation in predicted response. "
            "Marketers can prioritise customers who look like past responders on these dimensions, and treat the rest "
            "as lower-priority or test them with cheaper channels. Direction statements come from the relationship "
            "between each feature and the model's predicted probabilities on the test set.")
    with t3:
        if not SHAP_OK:
            st.info("SHAP is not installed, so the coefficient / feature-importance explanation above is shown instead. "
                    "Optional install: `pip install shap`")
        else:
            st.markdown("SHAP shows how much each feature pushes an individual prediction up (positive) or down "
                        "(negative) relative to the average prediction.")
            if st.button("Generate SHAP summary plot", key="shap_btn"):
                with st.spinner("Computing SHAP values..."):
                    try:
                        st_plot(compute_shap_figure(training, name))
                        st.caption("Each dot is a customer. Horizontal position = impact on the prediction (right = "
                                   "raises response probability, left = lowers it). Colour = feature value (red high, blue low).")
                    except Exception as exc:
                        st.info(f"SHAP could not be computed for this model ({type(exc).__name__}). "
                                "The feature-importance explanation above remains valid.")


# =====================================================================================
# PAGE: BUSINESS RECOMMENDATIONS
# =====================================================================================
def render_business_recommendations(df: pd.DataFrame, training) -> None:
    section_header("Business Recommendations", "Generated from the current (filtered) data and trained model - "
                                               "they change when the data or filters change.")
    name = deployed_model_name(training) if training else None
    ins = generate_business_insights(df, training, name)
    icons = {"Targeting": "🎯", "Campaign efficiency": "💰", "Customer engagement": "🤝", "Channel strategy": "🛒",
             "Campaign history": "📈", "Caveats": "⚠️"}
    for section in ["Targeting", "Campaign efficiency", "Customer engagement", "Channel strategy", "Campaign history"]:
        st.markdown(f"#### {icons[section]} {section}")
        for b in ins[section]:
            st.markdown(f"- {b}")
    if training:
        st.markdown("#### 📊 Cumulative gains (held-out test set)")
        gt = gains_table(training["y_test"], training["proba_test"][name])
        st_plot(fig_bar(gt, "Customers contacted (%)", "Responders captured (%)",
                        "Share of responders reached by contacting the top X% of customers",
                        xlabel="Customers contacted (%, highest probability first)", ylabel="Responders captured (%)"))
        st_table(gt)
    st.markdown(f"#### {icons['Caveats']} Caveats")
    for b in ins["Caveats"]:
        st.markdown(f"- {b}")
    st.markdown("#### Prediction vs recommendation")
    note("A <b>prediction</b> is the model's estimated probability that a customer responds. A <b>recommendation</b> is a "
         "business decision (whom to contact, with what offer, at what cost) that also uses costs, margins and "
         "judgement. The model informs the decision; it does not guarantee customer behaviour.")


# =====================================================================================
# PAGE: ABOUT PROJECT
# =====================================================================================
def render_about() -> None:
    section_header("About Project", "Marketing Campaign Response Prediction Using Machine Learning")
    st.markdown("#### Project objective")
    st.markdown(
        "Build a machine-learning system that estimates the **probability that a customer will respond to a marketing "
        "campaign**, so marketing teams can prioritise likely responders, reduce unnecessary contact costs, improve "
        "conversion and understand which customer characteristics are associated with response. The project keeps a "
        "clear line between a **prediction** (a probability) and a **business recommendation** (a decision that also "
        "uses costs and judgement); it never claims to guarantee customer behaviour.")
    st.markdown("#### Dataset")
    st.markdown(
        "Kaggle - *Customer Personality Analysis* (`marketing_campaign.csv`): "
        "https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis  \n"
        "Customer demographics, product spending, purchase channels, earlier-campaign acceptance and the outcome of the "
        "latest campaign (`Response`, 1 = responded). The file is **tab-separated**; the app detects the delimiter "
        "automatically. Columns are detected dynamically, and no synthetic data is ever generated.")
    st.markdown("#### Methodology")
    st.code("Data\n ↓\nCleaning\n ↓\nFeature Engineering\n ↓\nEDA\n ↓\nClassification\n ↓\nEvaluation\n ↓\n"
            "Prediction\n ↓\nMarketing Insights", language="text")
    st.markdown("#### ML architecture")
    st.code("Raw Data -> Cleaning -> Feature Engineering -> Stratified Train/Test Split (80/20)\n"
            "   -> ColumnTransformer [ numeric: median imputation + scaling | categorical: mode imputation + one-hot ]\n"
            "   -> Model (Logistic Regression | Random Forest | XGBoost or HistGradientBoosting)\n"
            "   -> Probability -> Threshold -> Prediction / Tier", language="text")
    st.markdown("#### Key assumptions and design decisions")
    st.markdown(
        "- **Target leakage:** `Response` is never used as a predictor. `AcceptedCmp1-5` are treated as *earlier* "
        "campaigns (known before the latest campaign) and may be excluded with one checkbox to test dependence on them.\n"
        "- **No preprocessing leakage:** imputation, scaling and encoding are learned inside the Pipeline on the training "
        "split only. Cleaning steps that need no learned statistics (removing invalid ages, duplicates) run first.\n"
        "- **Invalid income (<= 0)** is converted to missing and median-imputed, not deleted.\n"
        "- **Age** = current year - Year_Birth; ages outside 18-100 are removed as data-entry errors.\n"
        "- **Class imbalance** is handled by stratified splitting, threshold tuning (on cross-validated training "
        "predictions) and imbalance-aware metrics; class weights are optional.\n"
        "- **Model selection** uses cross-validated PR-AUC on the training split, keeping the test set untouched.\n"
        "- **Business simulation** is a scenario tool driven by user assumptions, not a financial forecast.")
    st.markdown("#### Technologies")
    st.markdown("Python, Pandas, NumPy, Scikit-learn, XGBoost, Streamlit, Plotly, Matplotlib, Seaborn, SHAP (optional).")
    st.markdown("#### Optional-package status in this environment")
    st_table(pd.DataFrame([
        {"Package": "Plotly", "Available": "Yes" if PLOTLY_OK else "No (Matplotlib fallback)"},
        {"Package": "Matplotlib", "Available": "Yes" if MPL_OK else "No"},
        {"Package": "Seaborn", "Available": "Yes" if SEABORN_OK else "No"},
        {"Package": "XGBoost", "Available": "Yes" if XGB_OK else "No (HistGradientBoosting fallback)"},
        {"Package": "SHAP", "Available": "Yes" if SHAP_OK else "No (feature-importance explanation)"},
    ]))
    st.markdown("#### Limitations")
    st.markdown(
        "- One retailer's data from a single campaign; patterns may not transfer to other companies or periods.\n"
        "- Roughly 2,200 customers with about 15% responders gives noisy estimates - metrics vary with the split.\n"
        "- Findings are associations, not causal effects; no experiment (A/B test) was run.\n"
        "- Predicted probabilities are estimates and may be poorly calibrated for individual customers.\n"
        "- Currency and the exact time reference of `Recency` are not documented in the source data.")


# =====================================================================================
# SIDEBAR
# =====================================================================================
def render_sidebar_nav():
    st.sidebar.markdown("## 📊 Marketing Campaign Analytics")
    page = st.sidebar.radio("Navigation", PAGES, key="nav_page", label_visibility="collapsed")
    with st.sidebar.expander("📁 Data source", expanded=True):
        uploaded = st.file_uploader("Upload marketing_campaign.csv", type=["csv", "tsv", "txt"], key="main_upload")
        st.caption(f"No upload? The app looks for `{DEFAULT_DATA_PATH}`.")
    return page, uploaded


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    with st.sidebar.expander("🔎 Filters", expanded=False):
        st.caption("Filters apply to KPIs and charts (not to model training). Leave a list empty to include everything.")
        for col, label in [("Education", "Education"), ("Marital_Status", "Marital Status"), ("Country", "Country")]:
            if col in df.columns:
                opts = sorted(df[col].dropna().astype(str).unique())
                chosen = st.multiselect(label, opts, default=opts, key=f"flt_{col}")
                if chosen:
                    mask &= df[col].astype(str).isin(chosen)
        if "Age" in df.columns and df["Age"].min() < df["Age"].max():
            lo, hi = int(np.floor(df["Age"].min())), int(np.ceil(df["Age"].max()))
            a = st.slider("Age range", lo, hi, (lo, hi), key="flt_age")
            mask &= df["Age"].between(a[0], a[1])
        if "Income" in df.columns and df["Income"].min() < df["Income"].max():
            lo = float(np.floor(df["Income"].min() / 1000) * 1000)
            hi = float(np.ceil(df["Income"].max() / 1000) * 1000)
            a = st.slider("Income range", lo, hi, (lo, hi), step=1000.0, key="flt_income")
            mask &= df["Income"].between(a[0], a[1])
        if "Recency" in df.columns and df["Recency"].min() < df["Recency"].max():
            lo, hi = int(np.floor(df["Recency"].min())), int(np.ceil(df["Recency"].max()))
            a = st.slider("Recency range (days)", lo, hi, (lo, hi), key="flt_recency")
            mask &= df["Recency"].between(a[0], a[1])
    return df[mask]


def sidebar_thresholds() -> None:
    with st.sidebar.expander("🎯 Targeting thresholds", expanded=False):
        st.slider("High predicted response if probability >=", 0.05, 0.95, 0.70, 0.05, key="thr_high")
        st.slider("Medium predicted response if probability >=", 0.0, 0.90, 0.40, 0.05, key="thr_med")
        if st.session_state.get("thr_med", 0.4) >= st.session_state.get("thr_high", 0.7):
            st.warning("Medium threshold must be below the High threshold; it is adjusted automatically.")
        use_tuned = st.checkbox("Use tuned decision threshold for the 0/1 prediction", value=True, key="use_tuned_thr")
        st.slider("Custom decision threshold", 0.05, 0.95, 0.50, 0.01, key="custom_thr", disabled=use_tuned)


# =====================================================================================
# MAIN
# =====================================================================================
def show_missing_dataset_help() -> None:
    st.error("No dataset found. Provide `marketing_campaign.csv` in one of two ways.")
    st.markdown(
        "1. **Upload it** with the *Data source* box in the sidebar, or\n"
        f"2. **Place it** at `{DEFAULT_DATA_PATH}` and refresh.\n\n"
        "Download it from Kaggle: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis")
    st.code("MarketingCampaignProject/\n│\n├── marketing_campaign_app.py\n│\n└── data/\n    └── marketing_campaign.csv",
            language="text")
    st.info("This app never creates synthetic data. The *About Project* page works without a dataset.")


def main() -> None:
    st.set_page_config(page_title="Marketing Campaign Analytics", page_icon="📊", layout="wide")
    inject_css()
    st.session_state.setdefault("training", None)
    page, uploaded = render_sidebar_nav()

    raw, source, load_error = None, None, None
    try:
        raw, source = load_data(uploaded)
    except Exception as exc:
        load_error = str(exc)
    if load_error:
        st.error(f"The dataset could not be read: {load_error} Please upload a valid CSV/TSV file.")
    if raw is None:
        if page == "About Project":
            render_about()
        elif not load_error:
            show_missing_dataset_help()
        return

    try:
        prep = prepare_data(raw)
    except ValueError as exc:
        st.error(f"The dataset cannot be used: {exc}")
        if page == "About Project":
            render_about()
        return
    except Exception as exc:
        st.error(f"Unexpected problem while preparing the data ({type(exc).__name__}): {exc}")
        return

    sig = f"{raw.shape}-{int(pd.util.hash_pandas_object(raw, index=True).sum())}"
    if st.session_state.get("data_sig") != sig:  # new dataset -> old models are invalid
        for k in ("training", "last_pred", "deploy_choice", "deploy_model"):
            st.session_state.pop(k, None)
        st.session_state["training"] = None
        st.session_state["data_sig"] = sig
    st.session_state.update({"raw_df": raw, "prep": prep, "source": source})

    st.sidebar.caption(f"✅ {source}")
    df_f = sidebar_filters(prep["df_eda"])
    sidebar_thresholds()
    training = st.session_state.get("training")

    if page == "About Project":
        render_about()
        return
    if page in ("Dashboard", "Data Overview", "EDA & Insights", "Campaign Analysis", "Business Recommendations") \
            and df_f.empty:
        st.warning("The current filters exclude every customer. Widen the filters in the sidebar.")
        return

    if page == "Dashboard":
        render_dashboard(df_f, training)
    elif page == "Data Overview":
        render_data_overview(raw, prep, df_f, source)
    elif page == "EDA & Insights":
        render_eda(df_f)
    elif page == "Campaign Analysis":
        render_campaign_analysis(df_f)
    elif page == "Model Training":
        render_model_training(prep, source)
    elif page == "Prediction":
        render_prediction(prep, training)
    elif page == "Customer Targeting":
        render_customer_targeting(prep, training)
    elif page == "Model Explainability":
        render_explainability(training)
    elif page == "Business Recommendations":
        render_business_recommendations(df_f, training)


if __name__ == "__main__":
    main()
