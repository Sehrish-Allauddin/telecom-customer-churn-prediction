"""Polished Streamlit dashboard for telecom churn analytics and prediction."""

# ruff: noqa: E501

import json
import logging
from contextlib import contextmanager
from collections.abc import Sequence
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    CHURN_THRESHOLD,
    DATA_PATH,
    FIGURES_DIR,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    METADATA_PATH,
    MODEL_PATH,
    PREDICTION_LOG_PATH,
    RANDOM_STATE,
)
from src.explainability import explain_customer
from src.logging_config import configure_logging
from src.monitoring import (
    feature_drift_report,
    log_prediction,
    prediction_summary,
    read_prediction_events,
)
from src.predict import load_model, predict_batch, validate_customer_input
from src.preprocessing import engineer_features

configure_logging()
logger = logging.getLogger(__name__)

PAGE_NAMES = [
    "Overview",
    "Customer Analysis",
    "Churn Analysis",
    "Revenue Analysis",
    "High-Risk Customers",
    "Prediction",
    "Model Performance",
    "Explainability",
    "Retention Insights",
    "Monitoring & Batch Scoring",
]
PALETTE = {
    "navy": "#F3F5F8",
    "silver": "#4C8DCA",
    "silver_soft": "#20242A",
    "green": "#45A878",
    "green_soft": "#1D3027",
    "amber": "#D99A42",
    "amber_soft": "#3A2D1D",
    "red": "#E16B72",
    "red_soft": "#3A2024",
    "ink": "#F3F5F8",
    "muted": "#A0A8B4",
    "line": "#2B3038",
    "surface": "#111318",
    "canvas": "#070809",
}
CHART_COLORS = [PALETTE["silver"], PALETTE["green"], PALETTE["amber"], PALETTE["red"]]
METRIC_ICONS = {
    "customers": (
        "<circle cx='9' cy='8' r='3.5'/><path d='M2.5 20v-1.5a6.5 6.5 0 0 1 13 0V20'/><path "
        "d='M16 5.2a3.5 3.5 0 0 1 0 6.7M18 14a5.8 5.8 0 0 1 3.5 5.3'/>"
    ),
    "churn": "<path d='M3 19V5M3 19h18'/><path d='m6 9 4 4 4-3 5 5'/>",
    "risk": "<path d='M12 3 21 20H3L12 3Z'/><path d='M12 9v4m0 3h.01'/>",
    "revenue": "<path d='M12 3v18M17 7.5c0-1.4-1.7-2.5-4.5-2.5S8 6.1 8 7.5 9.4 10 12.5 10 17 11.1 17 12.5 15.3 15 12.5 15 8 13.9 8 12.5'/>",
    "tenure": "<circle cx='12' cy='12' r='9'/><path d='M12 7v5l3 2'/>",
    "model": "<path d='M12 3 21 12 12 21 3 12 12 3Z'/><circle cx='12' cy='12' r='3'/>",
    "monitor": "<path d='M3 3v18h18'/><path d='m7 14 4-4 3 3 6-7'/>",
}


@st.cache_resource(show_spinner=False)
def load_dashboard_model(
    model_path: str, metadata_path: str, model_mtime: int, metadata_mtime: int
) -> Any:
    """Cache the fitted pipeline until either model artifact changes."""
    del model_mtime, metadata_mtime
    return load_model(Path(model_path), Path(metadata_path))


def get_model() -> Any:
    return load_dashboard_model(
        str(MODEL_PATH),
        str(METADATA_PATH),
        MODEL_PATH.stat().st_mtime_ns,
        METADATA_PATH.stat().st_mtime_ns,
    )


@st.cache_data(show_spinner=False)
def load_dashboard_data(path: str) -> pd.DataFrame:
    """Load actual source records and normalize target and numeric fields."""
    frame = pd.read_csv(path)
    target = next((name for name in frame.columns if name.strip().lower() == "churn"), None)
    if target is None:
        raise ValueError("The dataset does not contain a Churn column.")
    frame = frame.rename(columns={target: "Churn"})
    frame["Churn"] = frame["Churn"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    for column in ("tenure", "MonthlyCharges", "TotalCharges"):
        if column in frame:
            frame[column] = pd.to_numeric(
                frame[column].replace(r"^\s*$", pd.NA, regex=True), errors="coerce"
            )
    return frame.dropna(subset=["Churn"])


@st.cache_data(show_spinner="Scoring customer records…")
def score_dashboard_data(
    data_path: str,
    model_path: str,
    metadata_path: str,
    data_mtime: int,
    model_mtime: int,
    metadata_mtime: int,
    medium_threshold: float,
    high_threshold: float,
) -> pd.DataFrame:
    """Cache actual model scores until the dataset/model/risk bands change."""
    del data_mtime
    data = load_dashboard_data(data_path)
    model = load_dashboard_model(model_path, metadata_path, model_mtime, metadata_mtime)
    features = engineer_features(data.drop(columns=["Churn"]))
    data["PredictedChurnProbability"] = model.predict_proba(features)[:, 1]
    data["PredictedRiskLevel"] = pd.cut(
        data["PredictedChurnProbability"],
        [0.0, medium_threshold, high_threshold, 1.000001],
        labels=["Low Risk", "Medium Risk", "High Risk"],
        right=False,
    ).astype("string")
    return data


def chart_layout(fig: go.Figure, height: int = 220) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 34, "r": 14, "t": 26, "b": 30},
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["surface"],
        font={"color": PALETTE["ink"], "family": "Inter, Arial, sans-serif", "size": 12},
        hoverlabel={
            "bgcolor": "#20242A",
            "bordercolor": "#414750",
            "font": {"color": "#F5F5F5", "size": 12},
        },
        legend={"orientation": "h", "y": 1.12, "x": 1, "xanchor": "right"},
    )
    fig.update_xaxes(showline=True, linecolor=PALETTE["line"], gridcolor=PALETTE["line"])
    fig.update_yaxes(showline=False, gridcolor=PALETTE["line"], zeroline=False)
    return fig


def chart_card(title: str, description: str, fig: go.Figure, key: str) -> None:
    # Scoped chart container: spacing changes here must not affect KPI cards or other UI.
    with st.container(border=True, key=f"chart-card-{key}"):
        st.markdown(
            f"<div class='chart-heading'><div class='card-title'>{escape(title)}</div>"
            f"<div class='chart-description'>{escape(description)}</div></div>",
            unsafe_allow_html=True,
        )
        fig.update_layout(title=None)
        st.plotly_chart(
            fig,
            width="stretch",
            key=key,
            config={"responsive": True, "displayModeBar": False},
        )


def metric_card(
    container: Any,
    label: str,
    value: str,
    icon: str,
    accent: str = "#3C78A8",
    note: str = "",
) -> None:
    """Render a reusable monochrome analytics KPI card."""
    icon_svg = METRIC_ICONS[icon]
    note_html = f'<div class="kpi-note">{escape(note)}</div>' if note else ""
    container.markdown(
        f"""<div class="kpi-card" style="--kpi-accent:{accent}">
          <div class="kpi-top"><div class="kpi-label">{escape(label)}</div>
            <div class="kpi-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
              stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">{icon_svg}</svg></div>
          </div>
          <div class="kpi-value">{escape(value)}</div>
          {note_html}</div>""",
        unsafe_allow_html=True,
    )


@contextmanager
def metric_row(key: str, count: int) -> Any:
    """Keep KPI columns in a scoped, compact row without affecting chart layouts."""
    with st.container(key=f"kpi-row-{key}"):
        yield st.columns(count, gap="small")


def section_heading(title: str, description: str) -> None:
    st.markdown(f"<div class='section-title'>{escape(title)}</div>", unsafe_allow_html=True)
    st.caption(description)


def show_churn_rate(
    data: pd.DataFrame,
    title: str,
    column: str,
    selected_value: Any = None,
    order: Sequence[str] | None = None,
) -> go.Figure | None:
    """Build observed churn rates with cohort counts and interactive details."""
    if column not in data or data.empty:
        return None
    rates = data.groupby(column, dropna=False, observed=False).agg(
        churn_rate=("Churn", "mean"),
        churned_customers=("Churn", "sum"),
        customers=("Churn", "size"),
    )
    rates["churn_rate"] *= 100
    rates["stayed_customers"] = rates["customers"] - rates["churned_customers"]
    rates = rates.loc[rates["customers"] > 0].reset_index()
    rates[column] = rates[column].astype(str)
    if order is not None:
        order = [str(value) for value in order]
        rates[column] = pd.Categorical(rates[column], categories=order, ordered=True)
        rates = rates.sort_values(column).dropna(subset=[column])
    else:
        rates = rates.sort_values("churn_rate", ascending=False)
    colors = [
        (
            PALETTE["amber"]
            if selected_value is not None and str(value) == str(selected_value)
            else PALETTE["silver"]
        )
        for value in rates[column]
    ]
    fig = go.Figure(
        go.Bar(
            x=rates[column].astype(str),
            y=rates["churn_rate"],
            marker_color=colors,
            marker_line_color="#27384A",
            marker_line_width=1,
            text=[f"{rate:.1f}%" for rate in rates["churn_rate"]],
            textposition="outside",
            customdata=rates[["customers", "churned_customers", "stayed_customers"]],
            hovertemplate=(
                "<b>%{x}</b><br>Historical churn rate: %{y:.1f}%"
                "<br>Customers: %{customdata[0]:,}<br>Churned: %{customdata[1]:,}"
                "<br>Stayed: %{customdata[2]:,}<extra></extra>"
            ),
        )
    )
    fig.update_layout(title={"text": title, "x": 0, "font": {"size": 16}}, showlegend=False)
    fig.update_yaxes(title="Historical churn rate", ticksuffix="%", rangemode="tozero")
    return chart_layout(fig, 220)


def show_empty_state(message: str) -> None:
    st.info(message, icon="ℹ️")


st.set_page_config(
    page_title="Telecom Analytics · Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,500,0,0');
    :root { color-scheme:dark; --canvas:#08090B; --surface:#111214; --surface-raised:#151618;
      --ink:#F5F5F5; --muted:#9A9DA3; --line:#292C31; --blue:#4C8DCA; }
    html, body, [class*="css"] { font-family:'DM Sans',Inter,Arial,sans-serif; color:var(--ink); }
    .stApp, [data-testid="stHeader"] { background:var(--canvas); }
    [data-testid="stSidebar"] { background:#0C0D10; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap:10px; }
    [data-testid="stSidebar"] [data-testid="stSidebarHeader"] button {
      position:absolute; right:10px; top:8px; width:32px; height:32px;
      border:1px solid #383A3E; border-radius:50%; background:#17181A; color:#E9E9EB; }
    [data-testid="stSidebar"] [data-testid="stSidebarHeader"] button:hover {
      background:#242629; border-color:#626873; }
    [data-testid="stRadio"] [role="radiogroup"] { gap:4px; }
    [data-testid="stRadio"] label[data-baseweb="radio"] {
      display:flex; align-items:center; gap:10px; min-height:38px; padding:6px 10px;
      border:1px solid transparent; border-radius:10px; transition:background .16s ease,border-color .16s ease; }
    [data-testid="stRadio"] label[data-baseweb="radio"]:hover { background:#171A20; border-color:#252A32; }
    [data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child { display:none; }
    [data-testid="stRadio"] label[data-baseweb="radio"]::before {
      content:""; font-family:'Material Symbols Rounded'; font-size:19px; font-weight:500;
      line-height:1; color:#858E9A; width:21px; flex:0 0 21px; font-feature-settings:'liga'; }
    [data-testid="stRadio"] label[data-baseweb="radio"]:nth-of-type(1)::before { content:"space_dashboard"; }
    [data-testid="stRadio"] label[data-baseweb="radio"]:nth-of-type(2)::before { content:"group"; }
    [data-testid="stRadio"] label[data-baseweb="radio"]:nth-of-type(3)::before { content:"trending_down"; }
    [data-testid="stRadio"] label[data-baseweb="radio"]:nth-of-type(4)::before { content:"payments"; }
    [data-testid="stRadio"] label[data-baseweb="radio"]:nth-of-type(5)::before { content:"warning"; }
    [data-testid="stRadio"] label[data-baseweb="radio"]:nth-of-type(6)::before { content:"my_location"; }
    [data-testid="stRadio"] label[data-baseweb="radio"]:nth-of-type(7)::before { content:"monitoring"; }
    [data-testid="stRadio"] label[data-baseweb="radio"]:nth-of-type(8)::before { content:"hub"; }
    [data-testid="stRadio"] label[data-baseweb="radio"]:nth-of-type(9)::before { content:"lightbulb"; }
    [data-testid="stRadio"] label[data-baseweb="radio"]:nth-of-type(10)::before { content:"database"; }
    [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
      background:#20242A; border-color:#363C45; }
    [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p,
    [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked)::before { color:#FFFFFF!important; font-weight:700; }
    .block-container { max-width:1500px; padding:20px 24px 40px; }
    [data-testid="stMain"] [data-testid="stVerticalBlock"] { gap:12px; }

    /* Professional chart-card layout: add separation without changing the existing theme. */
    [class*="st-key-chart-card-"] {
      width:100%!important;
      margin:0 0 10px 0!important;
      padding:0!important;
      overflow:visible!important;
    }
    [class*="st-key-chart-card-"] .chart-heading {
      padding:10px 16px 0 16px!important;
    }
    [class*="st-key-chart-card-"] .chart-heading .card-title {
      margin-left:0!important;
    }
    [class*="st-key-chart-card-"] .chart-heading .chart-description {
      margin:3px 0 0 0!important;
      color:#8D929A!important;
      font-size:0.92rem!important;
      line-height:1.35!important;
    }
    [class*="st-key-chart-card-"] [data-testid="stPlotlyChart"] {
      width:100%!important;
      max-width:100%!important;
      overflow:visible!important;
    }

    /* Customer Analysis only: controlled vertical breathing room. */
    .customer-analysis-chart-top-gap { height:10px; width:100%; }
    .customer-analysis-table-gap { height:16px; width:100%; }

    /* Revenue Analysis only: small breathing room below KPI cards. */
    .revenue-analysis-chart-top-gap { height:10px; width:100%; }

    /* One content-sized layout for every KPI row. */
    /* KPI layout: compact, separated, and content-safe. Keep all other dashboard styling unchanged. */
    [class*="st-key-kpi-row-"] {
      width:100%!important;
      overflow:visible!important;
      margin:0 0 4px!important;
      padding:0 0 4px!important;
    }
    [class*="st-key-kpi-row-"] [data-testid="stHorizontalBlock"] {
      display:flex!important;
      flex-wrap:wrap!important;
      justify-content:flex-start!important;
      align-items:stretch!important;
      gap:16px!important;
      width:100%!important;
      overflow:visible!important;
      margin:0!important;
      padding:0!important;
    }
    [class*="st-key-kpi-row-"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
      flex:0 0 195px!important;
      width:195px!important;
      max-width:195px!important;
      min-width:195px!important;
      padding:0!important;
      margin:0!important;
      overflow:visible!important;
    }
    [class*="st-key-kpi-row-"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div {
      width:100%!important;
      min-width:0!important;
      overflow:visible!important;
    }
    .kpi-card {
      box-sizing:border-box;
      display:block;
      width:100%;
      max-width:195px;
      min-height:0;
      height:auto;
      margin:0!important;
      padding:8px 11px;
      border:1px solid var(--line);
      border-radius:14px;
      background:var(--surface);
      box-shadow:0 2px 8px rgba(0,0,0,.12);
      transition:border-color .16s ease,transform .16s ease;
    }
    .kpi-card:hover { transform:translateY(-1px); border-color:#414750; }
    .kpi-top {
      display:flex;
      justify-content:space-between;
      align-items:center;
      gap:8px;
      min-height:28px;
    }
    .kpi-label {
      color:var(--muted);
      font-size:11px;
      line-height:1.2;
      font-weight:600;
      overflow-wrap:anywhere;
    }
    .kpi-icon {
      box-sizing:border-box;
      width:28px;
      height:28px;
      flex:0 0 28px;
      display:grid;
      place-items:center;
      border:1px solid #363C45;
      border-radius:9px;
      background:#20242A;
      color:var(--kpi-accent);
    }
    .kpi-icon svg { width:14px; height:14px; }
    .kpi-value {
      margin-top:4px;
      color:var(--ink);
      font-size:22px;
      line-height:1.08;
      font-weight:700;
      letter-spacing:-.025em;
      overflow-wrap:anywhere;
    }
    .kpi-note {
      margin-top:2px;
      color:#969EAA;
      font-size:9px;
      line-height:1.2;
      overflow-wrap:anywhere;
    }

    h1 { color:var(--ink)!important; font-size:26px!important; letter-spacing:-.03em; }
    h2 { color:var(--ink)!important; font-size:21px!important; }
    h3 { color:var(--ink)!important; font-size:17px!important; }
    [data-testid="stVerticalBlockBorderWrapper"] { background:var(--surface);
      border-color:var(--line); border-radius:14px; }
    [data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stPlotlyChart"]):hover {
      border-color:#414750; }
    [data-testid="stPlotlyChart"] { background:var(--surface); border-radius:9px; }
    [data-testid="stDataFrame"] { background:var(--surface); border:1px solid var(--line); border-radius:12px; }
    [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"],
    [data-testid="stWidgetLabel"] { color:var(--ink); }
    [data-baseweb="select"] > div, [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div { background:#191C21; border-color:#343A43; }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] [data-baseweb="input"] > div,
    [data-testid="stMultiSelect"] [role="group"] {
      background:#191C21!important; border:1px solid #343A43!important; color:#F5F5F5!important; }
    [data-testid="stMain"] [data-baseweb="select"] > div,
    [data-testid="stMain"] [data-baseweb="input"] > div,
    [data-testid="stMain"] [data-baseweb="textarea"] > div {
      background-color:#191C21!important; border-color:#343A43!important; }
    [data-testid="stMain"] [data-baseweb="select"] input,
    [data-testid="stMain"] [data-baseweb="input"] input,
    [data-testid="stMain"] [data-baseweb="textarea"] textarea {
      background-color:#191C21!important; color:#F5F5F5!important; }
    [data-testid="stMain"] [role="combobox"],
    [data-testid="stMain"] [role="combobox"] > div {
      background-color:#191C21!important; color:#F5F5F5!important; }
    [data-testid="stMain"] [role="group"] {
      background-color:#191C21!important; border-color:#343A43!important; color:#F5F5F5!important; }
    [data-testid="stMain"] [role="group"] input {
      background-color:transparent!important; color:#F5F5F5!important; }
    [data-testid="stMain"] [role="group"] button {
      background-color:#20242A!important; border-color:#343A43!important; color:#F5F5F5!important; }
    [data-testid="stMain"] [data-testid="stNumberInput"] > div,
    [data-testid="stMain"] [data-testid="stNumberInput"] input {
      background-color:#191C21!important; border-color:#343A43!important; color:#F5F5F5!important; }
    [data-testid="stMain"] [data-testid="stNumberInput"] button {
      background-color:#20242A!important; border-color:#343A43!important; color:#F5F5F5!important; }
    [data-baseweb="select"] input, [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea { color:var(--ink); }
    [data-baseweb="tag"] { background:#292E36; color:var(--ink); }
    [data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
      min-height:38px; border-radius:9px; font-weight:600; }
    [data-testid="stButton"] button[kind="primary"] { background:#E5E7EB; border-color:#E5E7EB;
      color:#14171C; transition:background .16s ease,transform .16s ease; }
    [data-testid="stButton"] button[kind="primary"]:hover { background:#FFFFFF; transform:translateY(-1px); }
    [data-testid="stButton"] button[kind="secondary"]:hover,
    [data-testid="stDownloadButton"] button:hover { background:#20242A; border-color:#4A515C; }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"] {
      background:#20242A!important; border:1px solid #363C45!important; color:#F5F5F5!important; }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"]:hover {
      background:#292E36!important; border-color:#4A515C!important; }
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="tag"] {
      background:#292E36!important; border:1px solid #3B424D!important; color:#F5F5F5!important; }
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="tag"] * {
      color:#F5F5F5!important; }
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] [role="group"] span {
      background:#292E36!important; border-color:#3B424D!important; color:#F5F5F5!important; }
    [data-testid="stMultiSelect"] [role="group"] [role="option"] {
      background:#292E36!important; color:#F5F5F5!important; }
    .sidebar-brand { display:flex; align-items:center; gap:11px; padding:8px 3px 6px; }
    .sidebar-logo { box-sizing:border-box; width:36px; height:36px; flex:0 0 36px;
      display:grid; place-items:center; color:#F5F5F5; background:#252A31;
      border:1px solid #3B414A; border-radius:10px; }
    .sidebar-logo svg { width:20px; height:20px; }
    .sidebar-brand-name { color:var(--ink); font-size:13px; font-weight:700; letter-spacing:.035em; }
    .sidebar-brand-sub { color:var(--muted); font-size:11px; margin-top:3px; }
    [data-testid="stSidebar"] [data-testid="stDivider"] { border-color:var(--line); margin:4px 0 8px; }
    .sidebar-insight { margin-top:12px; padding:14px; border:1px solid var(--line);
      border-radius:13px; background:var(--surface-raised); }
    .sidebar-insight-head { display:flex; align-items:center; gap:8px; color:#BEC5CE;
      font-size:10px; font-weight:700; letter-spacing:.08em; margin-bottom:8px; }
    .sidebar-insight-head svg { width:18px; height:18px; }
    .sidebar-insight-title { color:var(--ink); font-size:14px; font-weight:700; line-height:1.3; }
    .sidebar-insight-copy { color:var(--muted); font-size:12px; line-height:1.4; margin-top:6px; }
    .section-title { color:var(--ink); font-weight:700; font-size:19px; margin:8px 0 2px; }
    .card-title { color:var(--ink); font-weight:700; font-size:16px; margin:2px 0 4px; }
    .status-pill { display:inline-flex; align-items:center; gap:7px; padding:7px 11px;
      border-radius:100px; background:#20242A; color:#D2D8E0; font-size:12px; font-weight:700;
      border:1px solid #363C45; white-space:nowrap; }
    .status-dot { width:8px; height:8px; border-radius:50%; background:#AEB8C4; }
    .header-meta { color:var(--muted); font-size:10px; text-align:right; margin-top:5px; }
    .soft-note { color:var(--muted); font-size:13px; }
    .risk-box { padding:12px 16px; border-radius:12px; font-size:16px; font-weight:700; margin:4px 0 12px; }
    @media (max-height:720px) { [data-testid="stSidebar"] .sidebar-insight { display:none; }
      [data-testid="stRadio"] label[data-baseweb="radio"] { min-height:34px; } }
    @media (max-width:900px) { .block-container { padding:16px 16px 32px; }
      [class*="st-key-kpi-row-"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                flex:0 0 185px!important; width:185px!important; max-width:185px!important;
                min-width:185px!important; }
             .kpi-card { max-width:185px; padding:8px 10px; } .kpi-value { font-size:21px; } }
    @media (max-width:600px) { .block-container { padding:12px 10px 24px; }
      [class*="st-key-kpi-row-"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex:1 1 100%!important; width:100%!important; min-width:0!important;
        max-width:420px!important; margin-bottom:12px!important; }
     .kpi-card { max-width:420px; padding:8px 10px; } .kpi-value { font-size:20px; }
      .kpi-icon { width:27px; height:27px; flex-basis:27px; } }
    </style>""",
    unsafe_allow_html=True,
)

model_ready = MODEL_PATH.exists() and METADATA_PATH.exists()
metadata: dict[str, Any] = {}
if METADATA_PATH.exists():
    try:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Model metadata could not be read; model information will be unavailable.")

try:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset is missing: {DATA_PATH}")
    full_data = score_dashboard_data(
        str(DATA_PATH),
        str(MODEL_PATH),
        str(METADATA_PATH),
        DATA_PATH.stat().st_mtime_ns,
        MODEL_PATH.stat().st_mtime_ns,
        METADATA_PATH.stat().st_mtime_ns,
        MEDIUM_RISK_THRESHOLD,
        HIGH_RISK_THRESHOLD,
    )
except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
    logger.exception("Could not load dashboard data.")
    full_data = pd.DataFrame()
    data_error = str(error)
else:
    data_error = ""

with st.sidebar:
    st.markdown(
        """<div class="sidebar-brand">
          <div class="sidebar-logo" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
              stroke-linecap="round" stroke-linejoin="round">
              <path d="M3 19V5M3 19h18"/><path d="m6 15 4-4 3 2 6-7"/>
              <path d="M15 6h4v4"/>
            </svg>
          </div>
          <div><div class="sidebar-brand-name">TELECOM ANALYTICS</div>
            <div class="sidebar-brand-sub">Customer Churn Intelligence</div></div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.divider()
    selected_page = st.radio(
        "Workspace",
        PAGE_NAMES,
        label_visibility="collapsed",
        key="workspace_navigation",
    )
    st.divider()
    st.markdown("#### Customer filters")
    st.caption("Filter analytics, charts, and customer tables.")

    filter_columns = {
        "contract": "Contract",
        "internet": "InternetService",
        "payment": "PaymentMethod",
        "support": "TechSupport",
    }
    filter_options = {
        key: (
            sorted(full_data[column].dropna().astype(str).unique().tolist())
            if column in full_data
            else []
        )
        for key, column in filter_columns.items()
    }
    tenure_values = pd.to_numeric(full_data.get("tenure", pd.Series(dtype=float)), errors="coerce")
    monthly_values = pd.to_numeric(
        full_data.get("MonthlyCharges", pd.Series(dtype=float)), errors="coerce"
    )
    total_values = pd.to_numeric(
        full_data.get("TotalCharges", pd.Series(dtype=float)), errors="coerce"
    )

    def numeric_range(values: pd.Series, default: tuple[float, float]) -> tuple[float, float]:
        clean = values.dropna()
        if clean.empty:
            return default
        low, high = float(clean.min()), float(clean.max())
        return (low, high if high > low else low + 1.0)

    tenure_range = numeric_range(tenure_values, (0, 72))
    monthly_range = numeric_range(monthly_values, (0, 200))
    total_range = numeric_range(total_values, (0, 10000))
    filter_defaults = {
        "filter_tenure": (int(tenure_range[0]), int(tenure_range[1])),
        "filter_contract": filter_options["contract"],
        "filter_internet": filter_options["internet"],
        "filter_payment": filter_options["payment"],
        "filter_monthly": monthly_range,
        "filter_total": total_range,
        "filter_support": filter_options["support"],
    }
    if st.button("Reset filters", width="stretch", key="reset_customer_filters"):
        for state_key, default_value in filter_defaults.items():
            st.session_state[state_key] = default_value

    tenure_widget_default = (
        {"value": filter_defaults["filter_tenure"]}
        if "filter_tenure" not in st.session_state
        else {}
    )
    monthly_widget_default = (
        {"value": monthly_range} if "filter_monthly" not in st.session_state else {}
    )
    total_widget_default = {"value": total_range} if "filter_total" not in st.session_state else {}
    tenure_filter = st.slider(
        "Tenure (months)",
        min_value=int(tenure_range[0]),
        max_value=int(tenure_range[1]),
        key="filter_tenure",
        **tenure_widget_default,
    )
    contract_filter = st.multiselect(
        "Contract",
        filter_options["contract"],
        key="filter_contract",
        **(
            {"default": filter_options["contract"]}
            if "filter_contract" not in st.session_state
            else {}
        ),
    )
    internet_filter = st.multiselect(
        "Internet service",
        filter_options["internet"],
        key="filter_internet",
        **(
            {"default": filter_options["internet"]}
            if "filter_internet" not in st.session_state
            else {}
        ),
    )
    payment_filter = st.multiselect(
        "Payment method",
        filter_options["payment"],
        key="filter_payment",
        **(
            {"default": filter_options["payment"]}
            if "filter_payment" not in st.session_state
            else {}
        ),
    )
    monthly_filter = st.slider(
        "Monthly charges ($)",
        min_value=float(monthly_range[0]),
        max_value=float(monthly_range[1]),
        step=1.0,
        key="filter_monthly",
        **monthly_widget_default,
    )
    total_filter = st.slider(
        "Total charges ($)",
        min_value=float(total_range[0]),
        max_value=float(total_range[1]),
        step=50.0,
        key="filter_total",
        **total_widget_default,
    )
    support_filter = st.multiselect(
        "Tech support",
        filter_options["support"],
        key="filter_support",
        **(
            {"default": filter_options["support"]}
            if "filter_support" not in st.session_state
            else {}
        ),
    )
    st.markdown(
        """<div class="sidebar-insight">
          <div class="sidebar-insight-head">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
              stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M3 19V5M3 19h18"/><path d="M7 15v-4m5 4V7m5 8V9"/>
            </svg>
            DATA-DRIVEN RETENTION
          </div>
          <div class="sidebar-insight-title">Turn Customer Data<br>into Smarter Decisions</div>
          <div class="sidebar-insight-copy">Predict churn, identify risks<br>
            and improve customer retention.
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

filtered_data = full_data.copy()
if not filtered_data.empty:
    if "tenure" in filtered_data:
        filtered_data = filtered_data[filtered_data["tenure"].between(*tenure_filter)]
    if contract_filter and "Contract" in filtered_data:
        filtered_data = filtered_data[filtered_data["Contract"].astype(str).isin(contract_filter)]
    if internet_filter and "InternetService" in filtered_data:
        filtered_data = filtered_data[
            filtered_data["InternetService"].astype(str).isin(internet_filter)
        ]
    if payment_filter and "PaymentMethod" in filtered_data:
        filtered_data = filtered_data[
            filtered_data["PaymentMethod"].astype(str).isin(payment_filter)
        ]
    if "MonthlyCharges" in filtered_data:
        filtered_data = filtered_data[
            filtered_data["MonthlyCharges"].between(*monthly_filter)
            | filtered_data["MonthlyCharges"].isna()
        ]
    if "TotalCharges" in filtered_data:
        filtered_data = filtered_data[
            filtered_data["TotalCharges"].between(*total_filter)
            | filtered_data["TotalCharges"].isna()
        ]
    if support_filter and "TechSupport" in filtered_data:
        filtered_data = filtered_data[filtered_data["TechSupport"].astype(str).isin(support_filter)]

with st.container():
    header_left, header_right = st.columns([5, 1])
    with header_left:
        st.markdown("# Customer Churn Prediction")
        st.caption("Predict, understand and reduce telecom customer churn using machine learning.")
    with header_right:
        status_label = "Ready" if model_ready else "Unavailable"
        status_color = PALETTE["silver"] if model_ready else PALETTE["red"]
        status_bg = PALETTE["silver_soft"] if model_ready else PALETTE["red_soft"]
        st.markdown(
            f"<div style='text-align:right;padding-top:30px'><span class='status-pill' "
            f"style='background:{status_bg};color:{status_color};border-color:{status_color}33'>"
            f"<span class='status-dot' style='background:{status_color}'></span>"
            f"Model status · {status_label}</span></div>",
            unsafe_allow_html=True,
        )
        model_version = escape(str(metadata.get("model_version", "Unavailable")))
        trained_date = escape(str(metadata.get("training_date", "Unavailable")))
        st.markdown(
            f"<div class='header-meta'>v{model_version} &nbsp;·&nbsp; Updated {trained_date}</div>",
            unsafe_allow_html=True,
        )
st.divider()

if data_error:
    st.error(f"Dashboard data is unavailable: {data_error}")
elif filtered_data.empty:
    show_empty_state("The dataset contains no customer records.")
else:
    if selected_page == "Overview":
        section_heading(
            "Business overview", "A live summary of customer records and model-scored churn risk."
        )
        total_customers = len(filtered_data)
        churn_rate = float(filtered_data["Churn"].mean())
        high_risk = int((filtered_data["PredictedChurnProbability"] >= HIGH_RISK_THRESHOLD).sum())
        revenue_exposure = float(
            (
                filtered_data["MonthlyCharges"].fillna(0)
                * filtered_data["PredictedChurnProbability"]
            ).sum()
        )
        with metric_row("overview", 4) as kpis:
            metric_card(
                kpis[0],
                "Total customers",
                f"{total_customers:,}",
                "customers",
                note="Source dataset records",
            )
            metric_card(
                kpis[1],
                "Historical churn rate",
                f"{churn_rate:.1%}",
                "churn",
                note="Observed customer labels",
            )
            metric_card(
                kpis[2],
                "High-risk customers",
                f"{high_risk:,}",
                "risk",
                PALETTE["red"],
                f"Model score ≥ {HIGH_RISK_THRESHOLD:.0%}",
            )
            metric_card(
                kpis[3],
                "Estimated monthly exposure",
                f"${revenue_exposure:,.0f}",
                "revenue",
                PALETTE["amber"],
                "Score-weighted estimate · not causal",
            )
        st.caption(
            "Revenue exposure is a model-based estimate, not predicted causal loss or savings. Historical churn comes from observed dataset labels."  # noqa: E501
        )
        chart_columns = st.columns(2, gap="medium")
        with chart_columns[0]:
            fig = show_churn_rate(
                filtered_data,
                "Churn rate by contract",
                "Contract",
                order=["Month-to-month", "One year", "Two year"],
            )
            if fig:
                chart_card(
                    "Churn rate by contract",
                    "Observed churn share in each contract group.",
                    fig,
                    "overview_contract",
                )
        with chart_columns[1]:
            chart_data = filtered_data.copy()
            if "tenure" in chart_data:
                chart_data["Tenure group"] = pd.cut(
                    chart_data["tenure"],
                    [-1, 12, 24, 48, 72],
                    labels=["0–12 months", "13–24 months", "25–48 months", "49–72 months"],
                )
                fig = show_churn_rate(
                    chart_data,
                    "Churn rate by tenure",
                    "Tenure group",
                    order=["0–12 months", "13–24 months", "25–48 months", "49–72 months"],
                )
                if fig:
                    chart_card(
                        "Churn rate by tenure",
                        "Observed churn by customer tenure range.",
                        fig,
                        "overview_tenure",
                    )
        chart_columns = st.columns(2, gap="medium")
        with chart_columns[0]:
            dist = (
                filtered_data["Churn"]
                .map({0: "Stayed", 1: "Churned"})
                .value_counts()
                .reindex(["Stayed", "Churned"], fill_value=0)
            )
            fig = go.Figure(
                go.Pie(
                    labels=dist.index,
                    values=dist.values,
                    hole=0.64,
                    marker_colors=[PALETTE["silver"], PALETTE["red"]],
                    textinfo="label+percent",
                    hovertemplate="<b>%{label}</b><br>Customers: %{value:,}<br>Share: %{percent}<extra></extra>",  # noqa: E501
                )
            )
            fig.update_layout(
                showlegend=False,
                annotations=[
                    dict(
                        text=f"{len(filtered_data):,}<br>customers",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font={"size": 14, "color": PALETTE["navy"]},
                    )
                ],
            )
            chart_card(
                "Customer churn distribution",
                "Observed customer outcomes in the source dataset.",
                chart_layout(fig, 220),
                "overview_distribution",
            )
        with chart_columns[1]:
            if "Contract" in filtered_data:
                revenue = (
                    filtered_data.groupby("Contract", observed=False)
                    .agg(
                        revenue_exposure=(
                            "PredictedChurnProbability",
                            lambda s: float(
                                (s * filtered_data.loc[s.index, "MonthlyCharges"].fillna(0)).sum()
                            ),
                        ),
                        customers=("Churn", "size"),
                    )
                    .reset_index()
                )
                fig = go.Figure(
                    go.Bar(
                        x=revenue["Contract"],
                        y=revenue["revenue_exposure"],
                        marker_color=PALETTE["silver"],
                        customdata=revenue[["customers"]],
                        hovertemplate="<b>%{x}</b><br>Estimated monthly exposure: $%{y:,.0f}<br>Customers: %{customdata[0]:,}<extra></extra>",  # noqa: E501
                    )
                )
                fig.update_layout(
                    showlegend=False,
                    title="Exposure by contract",
                    yaxis_title="Model-based monthly exposure ($)",
                )
                chart_card(
                    "Revenue exposure by contract",
                    "Estimated sum of monthly charges × model churn probability.",
                    chart_layout(fig, 220),
                    "overview_exposure",
                )

    elif selected_page == "Customer Analysis":
        section_heading(
            "Customer analysis",
            "Explore customer profiles and billing patterns in the source dataset.",
        )
        avg_tenure = float(filtered_data["tenure"].mean()) if "tenure" in filtered_data else 0
        with metric_row("customer-analysis", 4) as cols:
            metric_card(cols[0], "Customers", f"{len(filtered_data):,}", "customers")
            metric_card(cols[1], "Average tenure", f"{avg_tenure:.1f} months", "tenure")
            metric_card(
                cols[2],
                "Average monthly charge",
                f"${filtered_data['MonthlyCharges'].mean():,.2f}",
                "revenue",
            )
            metric_card(
                cols[3],
                "Median total charges",
                f"${filtered_data['TotalCharges'].median():,.2f}",
                "revenue",
            )
        # Small, page-specific breathing room: keep charts clearly below KPI cards
        # without changing the global dashboard layout.
        st.markdown("<div class=\"customer-analysis-chart-top-gap\"></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="small")
        with c1:
            counts = (
                filtered_data.groupby("Contract", observed=False)
                .size()
                .rename("customers")
                .reset_index()
                .sort_values("customers", ascending=False)
            )
            fig = go.Figure(
                go.Bar(
                    x=counts["Contract"],
                    y=counts["customers"],
                    marker_color=PALETTE["silver"],
                    text=counts["customers"],
                    texttemplate="%{text:,}",
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Customers: %{y:,}<extra></extra>",
                )
            )
            fig.update_layout(showlegend=False, yaxis_title="Customers")
            chart_card(
                "Customer count by contract",
                "Number of observed dataset records in each group.",
                chart_layout(fig, 250),
                "customer_contract_count",
            )
        with c2:
            if {"MonthlyCharges", "TotalCharges"}.issubset(filtered_data.columns):
                fig = go.Figure(
                    go.Scatter(
                        x=filtered_data["MonthlyCharges"],
                        y=filtered_data["TotalCharges"],
                        mode="markers",
                        marker={
                            "color": filtered_data["PredictedChurnProbability"],
                            "colorscale": [
                                [0, PALETTE["green"]],
                                [0.5, PALETTE["amber"]],
                                [1, PALETTE["red"]],
                            ],
                            "cmin": 0,
                            "cmax": 1,
                            "showscale": True,
                            "colorbar": {"title": "Risk score"},
                            "opacity": 0.65,
                        },
                        customdata=filtered_data[
                            [
                                "Contract",
                                "tenure",
                                "PredictedRiskLevel",
                                "PredictedChurnProbability",
                            ]
                        ],
                        hovertemplate="Monthly charges: $%{x:.2f}<br>Total charges: $%{y:.2f}<br>Contract: %{customdata[0]}<br>Tenure: %{customdata[1]} months<br>Risk: %{customdata[2]}<br>Churn score: %{customdata[3]:.1%}<extra></extra>",  # noqa: E501
                    )
                )
                fig.update_layout(
                    xaxis_title="Monthly charges ($)", yaxis_title="Total charges ($)"
                )
                chart_card(
                    "Customer billing profile",
                    "Each point represents a real filtered customer; color shows its model score.",
                    chart_layout(fig, 250),
                    "customer_billing_scatter",
                )
        table_columns = [
            c
            for c in [
                "customerID",
                "tenure",
                "Contract",
                "InternetService",
                "TechSupport",
                "PaymentMethod",
                "MonthlyCharges",
                "TotalCharges",
                "PredictedChurnProbability",
                "PredictedRiskLevel",
                "Churn",
            ]
            if c in filtered_data
        ]
        display = (
            filtered_data.sort_values("PredictedChurnProbability", ascending=False)[table_columns]
            .head(500)
            .copy()
        )
        display["PredictedChurnProbability"] = display["PredictedChurnProbability"].map(
            lambda value: f"{value:.1%}"
        )
        display["Churn"] = display["Churn"].map({0: "Stayed", 1: "Churned"})
        # Preserve a clean visual separation between the chart row and the records table.
        st.markdown("<div class=\"customer-analysis-table-gap\"></div>", unsafe_allow_html=True)
        st.markdown("#### Customer records")
        st.caption(
            f"Showing up to 500 of {len(filtered_data):,} matching customers, ordered by model risk score."  # noqa: E501
        )
        st.dataframe(display, hide_index=True, width="stretch", height=430)

    elif selected_page == "Churn Analysis":
        section_heading(
            "Churn analysis", "Historical churn patterns from observed customer outcomes."
        )
        columns = st.columns(2, gap="large")
        for slot, (title, column, order) in zip(
            columns * 2,
            [
                ("Churn by contract", "Contract", ["Month-to-month", "One year", "Two year"]),
                ("Churn by internet service", "InternetService", ["DSL", "Fiber optic", "No"]),
                ("Churn by tech support", "TechSupport", ["No", "Yes", "No internet service"]),
                ("Churn by payment method", "PaymentMethod", None),
            ],
            strict=True,
        ):
            with slot:
                fig = show_churn_rate(filtered_data, title, column, order=order)
                if fig:
                    chart_card(
                        title,
                        "Churn rates and customer counts are computed from source records.",
                        fig,
                        f"churn_{column}",
                    )
                else:
                    show_empty_state(f"{column} is not present in the source data.")
        st.caption(
            "These are associations in the historical dataset and do not establish that a service or contract caused churn."  # noqa: E501
        )

    elif selected_page == "Revenue Analysis":
        section_heading(
            "Revenue analysis", "Explore customer charges and model-based revenue exposure."
        )
        exposure = (
            filtered_data["MonthlyCharges"].fillna(0) * filtered_data["PredictedChurnProbability"]
        )
        with metric_row("revenue-analysis", 3) as cols:
            metric_card(
                cols[0],
                "Monthly charges",
                f"${filtered_data['MonthlyCharges'].fillna(0).sum():,.0f}",
                "revenue",
            )
            metric_card(
                cols[1],
                "Model-based exposure",
                f"${exposure.sum():,.0f}",
                "risk",
                PALETTE["amber"],
            )
            metric_card(
                cols[2],
                "Average monthly charge",
                f"${filtered_data['MonthlyCharges'].mean():,.2f}",
                "revenue",
            )

        # Revenue Analysis only: keep charts slightly separated from KPI cards
        # and reduce the excessive horizontal gap between the two chart cards.
        st.markdown(
            "<div class=\"revenue-analysis-chart-top-gap\"></div>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2, gap="small")
        with c1:
            if "Contract" in filtered_data:
                revenue = (
                    filtered_data.assign(Exposure=exposure)
                    .groupby("Contract", observed=False)
                    .agg(
                        exposure=("Exposure", "sum"),
                        monthly_revenue=("MonthlyCharges", "sum"),
                        customers=("Churn", "size"),
                    )
                    .reset_index()
                )
                fig = go.Figure()
                fig.add_bar(
                    x=revenue["Contract"],
                    y=revenue["monthly_revenue"],
                    name="Monthly charges",
                    marker_color=PALETTE["silver"],
                    customdata=revenue[["customers"]],
                    hovertemplate="%{x}<br>Monthly charges: $%{y:,.0f}<br>Customers: %{customdata[0]:,}<extra></extra>",  # noqa: E501
                )
                fig.add_bar(
                    x=revenue["Contract"],
                    y=revenue["exposure"],
                    name="Estimated exposure",
                    marker_color=PALETTE["amber"],
                    customdata=revenue[["customers"]],
                    hovertemplate="%{x}<br>Estimated exposure: $%{y:,.0f}<br>Customers: %{customdata[0]:,}<extra></extra>",  # noqa: E501
                )
                fig.update_layout(barmode="group", yaxis_title="Amount ($)")
                chart_card(
                    "Revenue and estimated exposure",
                    "Grouped totals by contract; exposure is score-weighted, not actual loss.",
                    chart_layout(fig),
                    "revenue_grouped",
                )
        with c2:
            if {"MonthlyCharges", "TotalCharges"}.issubset(filtered_data.columns):
                plot = filtered_data.copy()
                plot["Outcome"] = plot["Churn"].map({0: "Stayed", 1: "Churned"})
                fig = go.Figure()
                for outcome, color in [("Stayed", PALETTE["silver"]), ("Churned", PALETTE["red"])]:
                    group = plot[plot["Outcome"] == outcome]
                    fig.add_trace(
                        go.Scatter(
                            x=group["MonthlyCharges"],
                            y=group["TotalCharges"],
                            mode="markers",
                            name=outcome,
                            marker={"color": color, "opacity": 0.48, "size": 6},
                            customdata=group[
                                ["Contract", "tenure", "PaymentMethod", "PredictedChurnProbability"]
                            ],
                            hovertemplate="%{fullData.name}<br>Monthly: $%{x:.2f}<br>Total: $%{y:.2f}<br>Contract: %{customdata[0]}<br>Tenure: %{customdata[1]}<br>Payment: %{customdata[2]}<br>Model score: %{customdata[3]:.1%}<extra></extra>",  # noqa: E501
                        )
                    )
                fig.update_layout(
                    xaxis_title="Monthly charges ($)",
                    yaxis_title="Total charges ($)",
                    hovermode="closest",
                )
                chart_card(
                    "Customer charges by observed outcome",
                    "Point-level billing details appear on hover.",
                    chart_layout(fig),
                    "revenue_scatter",
                )
        st.caption(
            "Exposure = monthly charge × model churn probability. It is a prioritization aid, not causal loss or guaranteed savings."  # noqa: E501
        )

    elif selected_page == "High-Risk Customers":
        section_heading(
            "High-risk customers",
            f"Customers with model churn scores at or above the configured {HIGH_RISK_THRESHOLD:.0%} high-risk threshold.",
        )
        high_risk_data = filtered_data[
            filtered_data["PredictedChurnProbability"] >= HIGH_RISK_THRESHOLD
        ].copy()
        risk_exposure = float(
            (
                high_risk_data["MonthlyCharges"].fillna(0)
                * high_risk_data["PredictedChurnProbability"]
            ).sum()
        )
        with metric_row("high-risk", 3) as risk_metrics:
            metric_card(
                risk_metrics[0],
                "High-risk customers",
                f"{len(high_risk_data):,}",
                "risk",
                PALETTE["red"],
            )
            metric_card(
                risk_metrics[1],
                "Share of filtered customers",
                f"{len(high_risk_data) / len(filtered_data):.1%}" if len(filtered_data) else "0.0%",
                "churn",
                PALETTE["amber"],
            )
            metric_card(
                risk_metrics[2],
                "Estimated monthly exposure",
                f"${risk_exposure:,.0f}",
                "revenue",
            )
        if high_risk_data.empty:
            show_empty_state(
                "No customers in the current filter selection meet the high-risk threshold."
            )
        else:
            risk_columns = [
                column
                for column in [
                    "customerID",
                    "tenure",
                    "Contract",
                    "InternetService",
                    "TechSupport",
                    "MonthlyCharges",
                    "PredictedChurnProbability",
                    "PredictedRiskLevel",
                    "Churn",
                ]
                if column in high_risk_data
            ]
            risk_table = (
                high_risk_data.sort_values("PredictedChurnProbability", ascending=False)[
                    risk_columns
                ]
                .head(500)
                .copy()
            )
            risk_table["PredictedChurnProbability"] *= 100
            risk_table["Churn"] = risk_table["Churn"].map({0: "Stayed", 1: "Churned"})
            risk_table = risk_table.style.map(
                lambda value: {
                    "High Risk": "background-color: #3A2024; color: #F2A0A5; font-weight: 700",
                    "Medium Risk": "background-color: #3A2D1D; color: #F0C47E; font-weight: 700",
                    "Low Risk": "background-color: #1D3027; color: #94D0AC; font-weight: 700",
                }.get(value, ""),
                subset=["PredictedRiskLevel"],
            )
            st.caption(
                f"Showing up to 500 of {len(high_risk_data):,} filtered high-risk customers, ranked by model score."
            )
            st.dataframe(
                risk_table,
                hide_index=True,
                width="stretch",
                height=420,
                column_config={
                    "PredictedChurnProbability": st.column_config.ProgressColumn(
                        "Churn probability", format="%.1f%%", min_value=0, max_value=100
                    ),
                    "PredictedRiskLevel": st.column_config.TextColumn("Risk level"),
                },
            )

    elif selected_page == "Prediction":
        section_heading(
            "Customer churn prediction",
            "Enter a customer profile to score using the saved production pipeline.",
        )
        with st.container(border=True):
            with st.form("customer_prediction_form"):
                st.markdown("#### Customer profile")
                p1, p2 = st.columns(2, gap="large")
                with p1:
                    tenure = st.number_input("Tenure (months)", 0, 72, 12)
                with p2:
                    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
                st.markdown("#### Contract & billing")
                b1, b2 = st.columns(2, gap="large")
                with b1:
                    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                    monthly = st.number_input("Monthly charges ($)", 0.0, 1000.0, 70.0, step=1.0)
                with b2:
                    payment = st.selectbox(
                        "Payment method",
                        [
                            "Electronic check",
                            "Mailed check",
                            "Bank transfer (automatic)",
                            "Credit card (automatic)",
                        ],
                    )
                    total = st.number_input("Total charges ($)", 0.0, 100000.0, 840.0, step=10.0)
                st.markdown("#### Support & security")
                s1, s2 = st.columns(2, gap="large")
                with s1:
                    tech_support = st.selectbox(
                        "Tech support",
                        ["No internet service"] if internet == "No" else ["Yes", "No"],
                    )
                with s2:
                    scenario_contract = st.selectbox(
                        "What-if contract",
                        ["No change", "Month-to-month", "One year", "Two year"],
                        help="Scenario model scores are not causal estimates.",
                    )
                submitted = st.form_submit_button(
                    "Predict Churn Risk",
                    type="primary",
                    width="stretch",
                    disabled=not model_ready,
                )
        if not model_ready:
            st.warning("Model artifacts are missing. Train the model before making predictions.")
        st.caption(
            "The existing interface uses default values for profile fields outside these controls; prediction inputs and model pipeline are preserved."  # noqa: E501
        )
        if submitted:
            try:
                with st.spinner("Scoring customer and preparing local explanation…"):
                    model = get_model()
                    internet_default = "No internet service" if internet == "No" else "No"
                    customer = {
                        "gender": "Female",
                        "SeniorCitizen": 0,
                        "Partner": "No",
                        "Dependents": "No",
                        "tenure": tenure,
                        "PhoneService": "Yes",
                        "MultipleLines": "No",
                        "InternetService": internet,
                        "OnlineSecurity": internet_default,
                        "OnlineBackup": internet_default,
                        "DeviceProtection": internet_default,
                        "TechSupport": tech_support,
                        "StreamingTV": internet_default,
                        "StreamingMovies": internet_default,
                        "Contract": contract,
                        "PaperlessBilling": "Yes",
                        "PaymentMethod": payment,
                        "MonthlyCharges": monthly,
                        "TotalCharges": total,
                    }
                    validate_customer_input(customer)
                    raw_features = engineer_features(pd.DataFrame([customer]))
                    probability = float(model.predict_proba(raw_features)[:, 1][0])
                    risk = (
                        "High Risk"
                        if probability >= HIGH_RISK_THRESHOLD
                        else "Medium Risk" if probability >= MEDIUM_RISK_THRESHOLD else "Low Risk"
                    )
                    log_prediction(customer, int(probability >= CHURN_THRESHOLD), probability, risk)
                    recommendations = []
                    if contract == "Month-to-month":
                        recommendations.append(
                            "Discuss whether a longer-term plan fits this customer's needs."
                        )
                    if tenure <= 12:
                        recommendations.append(
                            "Consider a proactive early-tenure onboarding check-in."
                        )
                    if tech_support == "No":
                        recommendations.append(
                            "Offer a support-service review or technical check-in."
                        )
                    if monthly > float(full_data["MonthlyCharges"].median()):
                        recommendations.append("Review plan fit and available lower-cost options.")
                    if not recommendations:
                        recommendations.append(
                            "Offer a service review based on the customer's stated needs."
                        )
                st.markdown("#### Prediction result")
                risk_style = {
                    "Low Risk": (PALETTE["green_soft"], PALETTE["green"]),
                    "Medium Risk": (PALETTE["amber_soft"], PALETTE["amber"]),
                    "High Risk": (PALETTE["red_soft"], PALETTE["red"]),
                }[risk]
                outcome = "CHURN" if probability >= CHURN_THRESHOLD else "NOT CHURN"
                st.markdown(
                    f"<div class='risk-box' style='background:{risk_style[0]};color:{risk_style[1]};border:1px solid {risk_style[1]}33'>● &nbsp; {outcome} &nbsp; · &nbsp; {risk}</div>",  # noqa: E501
                    unsafe_allow_html=True,
                )
                with metric_row("prediction-result", 3) as result_cols:
                    metric_card(result_cols[0], "Churn probability", f"{probability:.1%}", "churn")
                    metric_card(result_cols[1], "Risk band", risk, "risk", risk_style[1])
                    metric_card(
                        result_cols[2],
                        "Expected monthly exposure",
                        f"${monthly*probability:,.2f}",
                        "revenue",
                    )
                st.progress(
                    probability,
                    text=f"Estimated churn probability · decision threshold {CHURN_THRESHOLD:.0%}",
                )
                st.caption(
                    "Expected exposure multiplies the customer's monthly charges by the model probability. It is not causal loss."  # noqa: E501
                )
                if scenario_contract != "No change":
                    scenario_customer = {**customer, "Contract": scenario_contract}
                    scenario_features = engineer_features(pd.DataFrame([scenario_customer]))
                    scenario_probability = float(model.predict_proba(scenario_features)[:, 1][0])
                    st.markdown("#### What-if scenario")
                    score_delta = scenario_probability - probability
                    with metric_row("prediction-scenario", 2) as sc:
                        metric_card(sc[0], "Current model score", f"{probability:.1%}", "churn")
                        metric_card(
                            sc[1],
                            f"Score with {scenario_contract} contract",
                            f"{scenario_probability:.1%}",
                            "model",
                            PALETTE["red"] if score_delta > 0 else PALETTE["green"],
                            f"Change from current: {score_delta:+.1%}",
                        )
                    st.caption(
                        "This is a comparison of model outputs, not an estimate of causal impact."
                    )
                with st.expander("Suggested retention actions", expanded=True):
                    st.caption(
                        "Rule-based suggestions; the project has not established causal intervention effects."  # noqa: E501
                    )
                    for item in recommendations:
                        st.markdown(f"- {item}")
                st.markdown("#### Why is this customer at risk?")
                st.caption(
                    "SHAP values describe contributions to this model score. They are not causal reasons."  # noqa: E501
                )
                background_source = full_data.drop(
                    columns=[
                        "Churn",
                        "customerID",
                        "PredictedChurnProbability",
                        "PredictedRiskLevel",
                    ],
                    errors="ignore",
                )
                background = engineer_features(
                    background_source.sample(
                        n=min(100, len(background_source)), random_state=RANDOM_STATE
                    )
                )
                explanation = explain_customer(model, raw_features, background, top_n=10)
                st.session_state["last_customer_explanation"] = explanation.to_dict(
                    orient="records"
                )
                st.dataframe(explanation, hide_index=True, width="stretch")
            except (
                FileNotFoundError,
                ValueError,
                KeyError,
                TypeError,
                RuntimeError,
                ImportError,
                AttributeError,
            ) as error:
                logger.exception("Prediction or customer explanation failed.")
                st.error(f"Could not complete this prediction: {error}")

    elif selected_page == "Model Performance":
        section_heading(
            "Model performance",
            "Evaluation metrics below come from the saved, untouched test-set report.",
        )
        metrics = metadata.get("metrics", {})
        metric_map = [
            ("Accuracy", "Accuracy"),
            ("Precision", "Precision"),
            ("Recall", "Recall"),
            ("F1 score", "F1"),
            ("ROC-AUC", "ROC_AUC"),
            ("PR-AUC", "PR_AUC"),
            ("Brier score", "Brier_Score"),
        ]
        available = [(label, metrics[key]) for label, key in metric_map if key in metrics]
        for start in range(0, len(available), 4):
            row = available[start : start + 4]
            with metric_row(f"model-{start // 4 + 1}", len(row)) as cols:
                for col, (label, value) in zip(cols, row):
                    fmt = f"{float(value):.3f}" if label == "Brier score" else f"{float(value):.1%}"
                    metric_card(col, label, fmt, "model", note="Held-out test-set metric")
        if not available:
            show_empty_state("No evaluation metrics are available in the model metadata.")
        cv = metadata.get("cross_validation", {})
        if cv:
            st.markdown("#### Training cross-validation")
            with metric_row("cross-validation", 3) as cv_cols:
                metric_card(cv_cols[0], "Selection criterion", "Mean PR-AUC", "model")
                metric_card(
                    cv_cols[1],
                    "5-fold CV PR-AUC",
                    f"{float(cv.get('selected_mean_pr_auc',0)):.3f} ± {float(cv.get('selected_std_pr_auc',0)):.3f}",  # noqa: E501
                    "model",
                )
                metric_card(
                    cv_cols[2],
                    "CV ROC-AUC",
                    f"{float(cv.get('selected_mean_roc_auc',0)):.3f}",
                    "model",
                )
        threshold = metadata.get("threshold", {})
        st.info(
            f"Configured decision threshold: {float(threshold.get('value',CHURN_THRESHOLD)):.0%}. It is a configured operating point; no intervention-cost data was available to optimize it."  # noqa: E501
        )
        for filename, title, description in [
            (
                "precision_recall_curve.png",
                "Precision–recall curve",
                "Held-out test-set precision and recall across thresholds.",
            ),
            (
                "calibration_curve.png",
                "Probability calibration",
                "Held-out calibration summary, if produced during evaluation.",
            ),
        ]:
            path = FIGURES_DIR / filename
            if path.exists():
                with st.container(border=True):
                    st.markdown(f"<div class='card-title'>{title}</div>", unsafe_allow_html=True)
                    st.caption(description)
                    st.image(str(path), width="stretch")

    elif selected_page == "Explainability":
        section_heading(
            "Model explainability",
            "Global drivers describe model behavior across held-out examples; local explanations are available after scoring a customer.",  # noqa: E501
        )
        shap_path = FIGURES_DIR / "shap_summary.png"
        if shap_path.exists():
            with st.container(border=True):
                st.markdown("#### Global feature impact")
                st.caption(
                    "SHAP summary from held-out evaluation rows. Positive and negative contributions reflect model output, not cause."  # noqa: E501
                )
                st.image(str(shap_path), width="stretch")
        else:
            show_empty_state("Global SHAP summary has not been generated yet.")
        top_features = metadata.get("top_shap_features", [])
        if top_features:
            feature_frame = pd.DataFrame(top_features).head(15)
            fig = go.Figure(
                go.Bar(
                    x=feature_frame["Mean absolute SHAP"],
                    y=feature_frame["Feature"],
                    orientation="h",
                    marker_color=PALETTE["silver"],
                    hovertemplate="%{y}<br>Mean |SHAP|: %{x:.3f}<extra></extra>",
                )
            )
            fig.update_layout(
                yaxis={"autorange": "reversed"},
                xaxis_title="Mean absolute SHAP value",
                showlegend=False,
            )
            chart_card(
                "Top global contributors",
                "Average absolute SHAP magnitude, ranked from actual model explanations.",
                chart_layout(fig, max(250, min(320, 18 * len(feature_frame) + 60))),
                "global_shap_bar",
            )
        st.markdown("#### Local customer explanation")
        local_records = st.session_state.get("last_customer_explanation", [])
        if local_records:
            local = pd.DataFrame(local_records)
            local["Color"] = local["SHAP value"].map(
                lambda value: PALETTE["red"] if value >= 0 else PALETTE["green"]
            )
            fig = go.Figure(
                go.Bar(
                    x=local["SHAP value"],
                    y=local["Feature"],
                    orientation="h",
                    marker_color=local["Color"],
                    customdata=local[["Direction"]],
                    hovertemplate=(
                        "<b>%{y}</b><br>SHAP value: %{x:.4f}" "<br>%{customdata[0]}<extra></extra>"
                    ),
                )
            )
            fig.update_layout(
                yaxis={"autorange": "reversed"},
                xaxis_title="SHAP contribution (model score units)",
                showlegend=False,
            )
            chart_card(
                "Feature contributions for the last scored customer",
                "Red features pushed the model score toward churn; green features pushed it away.",
                chart_layout(fig, max(250, min(320, 20 * len(local) + 55))),
                "local_shap_contributions",
            )
            st.dataframe(local.drop(columns="Color"), hide_index=True, width="stretch")
        else:
            st.caption(
                "Run a prediction on the Prediction page to generate a local SHAP breakdown for that customer."  # noqa: E501
            )

    elif selected_page == "Retention Insights":
        section_heading(
            "Customer retention insights",
            "Use historical segment patterns and held-out ranking results to guide outreach planning.",  # noqa: E501
        )
        risk_data = filtered_data.assign(
            ModelExposure=filtered_data["MonthlyCharges"].fillna(0)
            * filtered_data["PredictedChurnProbability"]
        )
        risk_summary = (
            risk_data.groupby("PredictedRiskLevel", observed=False)
            .agg(
                customers=("Churn", "size"),
                observed_churn_rate=("Churn", "mean"),
                mean_score=("PredictedChurnProbability", "mean"),
                monthly_charges=("MonthlyCharges", "sum"),
                revenue_exposure=("ModelExposure", "sum"),
            )
            .reset_index()
            .dropna(subset=["PredictedRiskLevel"])
        )
        risk_summary["observed_churn_rate"] *= 100
        if not risk_summary.empty:
            risk_summary["color"] = risk_summary["PredictedRiskLevel"].map(
                {
                    "Low Risk": PALETTE["green"],
                    "Medium Risk": PALETTE["amber"],
                    "High Risk": PALETTE["red"],
                }
            )
            fig = go.Figure(
                go.Bar(
                    x=risk_summary["PredictedRiskLevel"],
                    y=risk_summary["customers"],
                    marker_color=risk_summary["color"],
                    customdata=risk_summary[
                        [
                            "mean_score",
                            "observed_churn_rate",
                            "monthly_charges",
                            "revenue_exposure",
                        ]
                    ],
                    hovertemplate=(
                        "<b>%{x}</b><br>Customers: %{y:,}"
                        "<br>Mean model score: %{customdata[0]:.1%}"
                        "<br>Observed churn: %{customdata[1]:.1f}%"
                        "<br>Monthly charges: $%{customdata[2]:,.0f}"
                        "<br>Estimated exposure: $%{customdata[3]:,.0f}<extra></extra>"
                    ),
                )
            )
            fig.update_layout(yaxis_title="Customers", showlegend=False)
            chart_card(
                "Risk segments and outcomes",
                "Model risk bands alongside historical outcomes and monthly charges.",
                chart_layout(fig),
                "retention_risk_segments",
            )
            st.dataframe(
                risk_summary[
                    [
                        "PredictedRiskLevel",
                        "customers",
                        "observed_churn_rate",
                        "mean_score",
                        "monthly_charges",
                        "revenue_exposure",
                    ]
                ].rename(
                    columns={
                        "PredictedRiskLevel": "Risk level",
                        "customers": "Customers",
                        "observed_churn_rate": "Observed churn rate (%)",
                        "mean_score": "Mean model score",
                        "monthly_charges": "Monthly charges ($)",
                        "revenue_exposure": "Estimated exposure ($)",
                    }
                ),
                hide_index=True,
                width="stretch",
            )
        top_k = metadata.get("business_test", {}).get("top_k", [])
        if top_k:
            st.markdown("#### Held-out outreach ranking")
            st.caption(
                "These test-set estimates describe ranking performance at different outreach capacities; they are not customer-level promises."  # noqa: E501
            )
            frame = pd.DataFrame(top_k)
            frame["Precision"] = (frame["precision_at_k"] * 100).map(lambda x: f"{x:.1f}%")
            frame["Recall"] = (frame["recall_at_k"] * 100).map(lambda x: f"{x:.1f}%")
            frame["Lift"] = frame["lift"].map(lambda x: f"{x:.2f}×")
            st.dataframe(
                frame[
                    [
                        "top_fraction",
                        "targeted_customers",
                        "actual_churners_captured",
                        "Precision",
                        "Recall",
                        "Lift",
                    ]
                ].rename(
                    columns={
                        "top_fraction": "Top-ranked share",
                        "targeted_customers": "Targeted customers",
                        "actual_churners_captured": "Churners captured",
                    }
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            show_empty_state("Held-out Top-K analysis is unavailable in model metadata.")
        st.caption(
            "Historical segments reveal associations. They do not prove that any listed action will prevent churn."  # noqa: E501
        )

    elif selected_page == "Monitoring & Batch Scoring":
        section_heading(
            "Prediction monitoring & batch scoring",
            "Inspect logged model outputs and score uploaded customer records.",
        )
        events = read_prediction_events(PREDICTION_LOG_PATH)
        summary = prediction_summary(events)
        with metric_row("monitoring", 3) as monitor_cols:
            metric_card(
                monitor_cols[0], "Logged predictions", f"{summary['prediction_count']:,}", "monitor"
            )
            metric_card(
                monitor_cols[1],
                "Predicted churn rate",
                f"{summary['predicted_churn_rate']:.1%}",
                "churn",
            )
            metric_card(
                monitor_cols[2],
                "Mean churn score",
                f"{summary['mean_churn_probability']:.1%}",
                "model",
            )
        st.caption(
            "These summaries describe model predictions, not verified customer outcomes or measured performance."  # noqa: E501
        )
        drift = feature_drift_report(full_data, events)
        if "feature" in drift:
            st.markdown("#### Feature distribution drift")
            st.dataframe(drift, hide_index=True, width="stretch")
            st.caption(
                "PSI flags are heuristic signals and do not prove model degradation. Ground-truth labels are not collected here."  # noqa: E501
            )
        else:
            show_empty_state(
                f"Need at least {drift.iloc[0]['minimum_events']} logged predictions before distribution drift checks are shown."  # noqa: E501
            )
        st.markdown("#### Batch customer scoring")
        st.caption(
            "Upload a CSV with the raw customer feature columns. customerID is optional; any Churn column is ignored. Output includes score and risk band."  # noqa: E501
        )
        uploaded = st.file_uploader("Upload customer CSV", type=["csv"], key="monitor_batch_upload")
        if uploaded is not None:
            try:
                batch_frame = pd.read_csv(uploaded)
                if len(batch_frame) > 10_000:
                    raise ValueError("Upload is limited to 10,000 customer rows at a time.")
                with st.spinner("Scoring uploaded customer rows…"):
                    results = predict_batch(batch_frame, get_model())
                st.success(f"Scored {len(results):,} customers.")
                st.dataframe(results.head(100), hide_index=True, width="stretch")
                st.download_button(
                    "Download prediction results",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                )
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                logger.exception("Batch customer scoring failed.")
                st.error(f"Could not process this CSV: {error}")
