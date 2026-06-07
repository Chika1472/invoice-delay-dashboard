from __future__ import annotations

from datetime import date
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.invoice_delay_service import (
    get_payment_term,
    infer_amount_bin,
    load_or_train_artifact,
    make_prediction_frame,
    predict_invoice_delay,
)


st.set_page_config(
    page_title="Invoice Delay Risk",
    layout="wide",
    initial_sidebar_state="collapsed",
)


CUSTOM_CSS = """
<style>
    :root {
        --ink: #161a22;
        --muted: #5e6572;
        --paper: #ffffff;
        --page: #e7e9ee;
        --line: #c9ced8;
        --line-dark: #9ea7b8;
        --navy: #263765;
        --navy-soft: #5f7198;
        --navy-pale: #eef1f6;
        --red-muted: #8d5963;
        --tan-muted: #e5dcc6;
    }

    .stApp {
        background: #ffffff;
        color: var(--ink);
    }

    .block-container {
        max-width: min(1720px, calc(100vw - 28px));
        min-height: calc(100vh - 28px);
        margin: 0 auto;
        padding: 22px 28px 24px;
        background: #ffffff;
        border: 0;
        box-shadow: none;
    }

    [data-testid="stSidebar"] {
        background: #f7f8fb;
        border-right: 1px solid var(--line);
    }

    .stApp,
    .stApp p,
    .stApp span,
    .stApp label,
    .stApp div,
    [data-testid="stMarkdownContainer"],
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {
        color: var(--ink) !important;
    }

    h1, h2, h3, p {
        letter-spacing: 0;
    }

    .report-top {
        display: grid;
        grid-template-columns: 1fr auto;
        align-items: start;
        gap: 20px;
        border-bottom: 1px solid var(--line);
        padding-bottom: 14px;
        margin-bottom: 14px;
    }

    .report-title {
        color: var(--navy) !important;
        font-size: clamp(2rem, 2.45vw, 3rem);
        font-weight: 900;
        line-height: 1.05;
        margin: 0;
    }

    .report-kicker {
        color: var(--muted) !important;
        font-size: 0.92rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 7px;
    }

    .brand-block {
        color: var(--navy) !important;
        font-size: 1.05rem;
        font-weight: 900;
        text-align: right;
        line-height: 1.3;
        white-space: nowrap;
    }

    .doc-meta {
        color: var(--muted) !important;
        font-size: 0.84rem;
        margin-top: 7px;
        text-align: right;
    }

    div[data-testid="stMetric"] {
        background: #fbfbfd;
        border: 1px solid var(--line);
        border-radius: 0;
        padding: 0.55rem 0.7rem;
        box-shadow: none;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] p {
        color: var(--muted) !important;
    }

    div[data-testid="stMetricValue"] div {
        color: var(--navy) !important;
        font-size: 1.45rem !important;
        font-weight: 850 !important;
    }

    button[data-baseweb="tab"] p {
        color: var(--navy) !important;
        font-weight: 800;
    }

    button[data-baseweb="tab"][aria-selected="true"] p {
        color: var(--red-muted) !important;
    }

    .panel-head,
    .input-head {
        background: var(--navy-soft);
        border: 1px solid var(--navy-soft);
        color: #ffffff !important;
        text-align: center;
        padding: 0.62rem 0.75rem;
        font-size: 1rem;
        font-weight: 900;
    }

    .stApp .panel-head,
    .stApp .input-head,
    .panel-head *,
    .input-head * {
        color: #ffffff !important;
    }

    div[data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 0;
        padding: 0.72rem 0.88rem 0.76rem;
        box-shadow: none;
    }

    div[data-testid="stForm"] [data-testid="stVerticalBlock"] {
        gap: 0.48rem;
    }

    [data-testid="stWidgetLabel"] p {
        font-size: 0.86rem;
        font-weight: 850;
        color: #262b35 !important;
    }

    input,
    textarea,
    [data-baseweb="input"] input,
    [data-baseweb="select"] > div,
    [data-baseweb="select"] span,
    [data-baseweb="textarea"] textarea {
        background: #ffffff !important;
        color: var(--ink) !important;
        border-color: var(--line-dark) !important;
    }

    [data-baseweb="input"] > div,
    [data-baseweb="select"] > div {
        min-height: 2.34rem;
        border-radius: 0 !important;
    }

    [data-testid="stNumberInput"] button {
        background: #ffffff !important;
        color: var(--navy) !important;
        border-color: var(--line-dark) !important;
    }

    [data-baseweb="radio"] div,
    [data-baseweb="checkbox"] div {
        color: var(--ink) !important;
    }

    .stButton button,
    div[data-testid="stFormSubmitButton"] button {
        border-radius: 0;
        border: 1px solid var(--navy-soft);
        background: var(--navy-soft);
        color: #ffffff !important;
        font-weight: 850;
        min-height: 2.5rem;
    }

    .stButton button *,
    div[data-testid="stFormSubmitButton"] button *,
    .stApp div[data-testid="stFormSubmitButton"] button p {
        color: #ffffff !important;
    }

    .stButton button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        border-color: var(--navy);
        background: var(--navy);
        color: #ffffff !important;
    }

    div[data-testid="stFormSubmitButton"] {
        margin-top: 0.2rem;
    }

    .probability-panel,
    .model-judge-panel,
    .note-panel,
    .model-result-box {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 0;
        overflow: hidden;
    }

    .model-judge-panel {
        margin-top: 0;
    }

    .panel-body-note {
        border-top: 1px solid var(--line);
        padding: 0.72rem 0.88rem;
        color: #303643 !important;
        font-size: 0.95rem;
        line-height: 1.45;
    }

    .panel-body-note strong {
        color: var(--navy) !important;
    }

    .notice {
        color: #38404d !important;
        font-size: 0.82rem;
        line-height: 1.55;
        padding: 0.72rem 0.86rem;
    }

    .small-note {
        color: var(--muted) !important;
        font-size: 0.82rem;
    }

    .stCaptionContainer,
    .stCaptionContainer p {
        color: var(--muted) !important;
        font-size: 0.78rem !important;
        line-height: 1.35;
    }

    @media (max-width: 1000px) {
        .block-container {
            max-width: calc(100vw - 12px);
            margin: 6px;
            padding: 16px;
        }
        .report-top {
            display: block;
        }
        .brand-block,
        .doc-meta {
            text-align: left;
            margin-top: 8px;
        }
    }
</style>
"""


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


@st.cache_resource(show_spinner=False)
def get_artifact(force_retrain: bool = False):
    return load_or_train_artifact(force_retrain=force_retrain)


def payment_options(artifact):
    terms = artifact.payment_terms

    def nearest_term(target_days: int):
        candidates = [term for term in terms if term["allowed_days"] > 0]
        if not candidates:
            return None
        return min(candidates, key=lambda term: (abs(term["allowed_days"] - target_days), -term["count"]))

    def most_common_term(predicate):
        candidates = [term for term in terms if predicate(term["allowed_days"])]
        if not candidates:
            return None
        return max(candidates, key=lambda term: term["count"])

    grouped_terms = [
        ("5일", nearest_term(5)),
        ("10일", nearest_term(10)),
        ("15일", nearest_term(15)),
        ("20일", nearest_term(20)),
        ("25일", nearest_term(25)),
        ("30일 이상", most_common_term(lambda days: days >= 30)),
    ]

    options = {}
    for label, term in grouped_terms:
        if term is not None:
            options[label] = term["code"]

    if not options and terms:
        options[terms[0]["label"]] = terms[0]["code"]
    return options


def outstanding_options(artifact):
    return [
        key
        for key in ["없음", "낮음", "보통", "높음", "매우 높음"]
        if key in artifact.outstanding_levels
    ]


def default_payment_index(labels: list[str]) -> int:
    for idx, label in enumerate(labels):
        if "15일" in label:
            return idx
    return 0


MAX_UPLOAD_ROWS = 15
RAW_UPLOAD_REQUIRED_COLUMNS = ["total_open_amount", "cust_payment_terms", "baseline_create_date"]


def clean_cell(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def format_identifier(value, fallback: str) -> str:
    if pd.isna(value):
        return fallback
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    return text if text else fallback


def parse_upload_date(value) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT

    text = str(value).strip()
    if not text:
        return pd.NaT
    if text.endswith(".0"):
        text = text[:-2]

    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) == 8:
        parsed = pd.to_datetime(digits, format="%Y%m%d", errors="coerce")
        if not pd.isna(parsed):
            return parsed

    return pd.to_datetime(text, errors="coerce")


def format_upload_date(value) -> str:
    parsed = parse_upload_date(value)
    if pd.isna(parsed):
        return "-"
    return parsed.strftime("%Y-%m-%d")


def upload_amount_usd(row: pd.Series) -> float:
    amount = pd.to_numeric(pd.Series([row.get("total_open_amount", row.get("amount_in_usd", 0.0))]), errors="coerce").iloc[0]
    if pd.isna(amount):
        amount = 0.0

    if "amount_in_usd" in row.index and "total_open_amount" not in row.index:
        return max(float(np.expm1(amount)), 0.0)

    currency = clean_cell(row.get("invoice_currency", "USD")).upper()
    if currency == "CAD":
        amount = float(amount) * 0.75
    return max(float(amount), 0.0)


def choose_payment_term_code(artifact, row: pd.Series, baseline_timestamp: pd.Timestamp) -> str:
    known_terms = {str(term["code"]).upper(): term["code"] for term in artifact.payment_terms}
    uploaded_code = clean_cell(row.get("cust_payment_terms", "")).upper()
    if uploaded_code in known_terms:
        return known_terms[uploaded_code]

    due_timestamp = parse_upload_date(row.get("due_in_date", pd.NaT))
    if not pd.isna(baseline_timestamp) and not pd.isna(due_timestamp):
        allowed_days = max(int((due_timestamp - baseline_timestamp).days), 0)
        return min(
            artifact.payment_terms,
            key=lambda term: (abs(int(term["allowed_days"]) - allowed_days), -int(term["count"])),
        )["code"]

    return artifact.payment_terms[0]["code"]


def upload_is_open(value) -> bool:
    text = clean_cell(value).lower()
    return text in {"1", "1.0", "true", "t", "yes", "y", "open"}


def business_late_flag(clear_timestamp: pd.Timestamp, due_timestamp: pd.Timestamp) -> bool | None:
    if pd.isna(clear_timestamp) or pd.isna(due_timestamp):
        return None
    if clear_timestamp <= due_timestamp:
        return False
    business_days = int(np.busday_count(due_timestamp.date(), clear_timestamp.date()))
    return business_days > 5


def upload_contexts(frame: pd.DataFrame) -> dict[int, dict]:
    working = frame.copy()
    working["_row_order"] = np.arange(len(working))
    working["_baseline_ts"] = working.get("baseline_create_date", pd.Series(pd.NaT, index=working.index)).apply(parse_upload_date)
    working["_clear_ts"] = working.get("clear_date", pd.Series(pd.NaT, index=working.index)).apply(parse_upload_date)
    working["_due_ts"] = working.get("due_in_date", pd.Series(pd.NaT, index=working.index)).apply(parse_upload_date)
    working["_amount_log"] = working.apply(lambda row: float(np.log1p(upload_amount_usd(row))), axis=1)
    working["_customer_key"] = working.apply(
        lambda row: clean_cell(row.get("cust_number", "")) or f"__row_{int(row['_row_order'])}",
        axis=1,
    )

    contexts: dict[int, dict] = {}
    for _, current in working.iterrows():
        order = int(current["_row_order"])
        baseline_ts = current["_baseline_ts"]
        same_customer = working["_customer_key"] == current["_customer_key"]
        if pd.isna(baseline_ts):
            prior_rows = working.iloc[0:0]
        else:
            prior_rows = working[same_customer & (working["_baseline_ts"] < baseline_ts)]

        cleared_rows = prior_rows[
            prior_rows["_clear_ts"].notna()
            & prior_rows["_due_ts"].notna()
            & (prior_rows["_clear_ts"] <= baseline_ts if not pd.isna(baseline_ts) else False)
        ].sort_values("_clear_ts")

        late_flags = [
            flag
            for flag in (
                business_late_flag(row["_clear_ts"], row["_due_ts"])
                for _, row in cleared_rows.iterrows()
            )
            if flag is not None
        ]
        late_days = [
            max(int((row["_clear_ts"] - row["_due_ts"]).days), 0)
            for _, row in cleared_rows.iterrows()
            if business_late_flag(row["_clear_ts"], row["_due_ts"])
        ]

        open_rows = prior_rows[
            prior_rows.apply(
                lambda row: upload_is_open(row.get("isOpen", 0))
                or pd.isna(row["_clear_ts"])
                or (not pd.isna(baseline_ts) and row["_clear_ts"] > baseline_ts),
                axis=1,
            )
        ]

        previous_rate = float(np.mean(late_flags)) if late_flags else 0.0
        recent_flags = late_flags[-3:]
        recent_rate = float(np.mean(recent_flags)) if recent_flags else previous_rate
        contexts[order] = {
            "previous_late_rate": previous_rate,
            "recent_late_rate": recent_rate,
            "average_late_days": float(np.mean(late_days)) if late_days else 0.0,
            "sum_outstanding_amount_past": float(open_rows["_amount_log"].sum()) if not open_rows.empty else 0.0,
            "cleared_count": int(len(cleared_rows)),
            "current_transaction_count": int(len(open_rows) + 1),
            "is_new_customer": int(len(cleared_rows) < 3),
            "is_last_late": bool(late_flags[-1]) if late_flags else False,
        }
    return contexts


def upload_row_prediction_frame(artifact, row: pd.Series, context: dict) -> tuple[pd.DataFrame, dict]:
    if all(column in row.index for column in artifact.feature_columns):
        frame = pd.DataFrame([{column: row[column] for column in artifact.feature_columns}], columns=artifact.feature_columns)
        return frame, {"payment_term_code": clean_cell(row.get("cust_payment_terms", "")), "baseline": parse_upload_date(row.get("baseline_create_date", pd.NaT))}

    feature_row = dict(artifact.default_row)
    baseline_timestamp = parse_upload_date(row.get("baseline_create_date", pd.NaT))
    if pd.isna(baseline_timestamp):
        baseline_timestamp = pd.Timestamp(date.today())

    amount_usd = upload_amount_usd(row)
    amount_log = float(np.log1p(amount_usd))
    payment_term_code = choose_payment_term_code(artifact, row, baseline_timestamp)
    payment_term = get_payment_term(artifact, payment_term_code)
    due_timestamp = baseline_timestamp + pd.Timedelta(days=int(payment_term["allowed_days"]))

    feature_row["business_code"] = clean_cell(row.get("business_code", feature_row.get("business_code", "")))
    feature_row["amount_in_usd"] = amount_log
    feature_row["amount_bin"] = infer_amount_bin(amount_log, artifact.amount_bin_thresholds)
    feature_row["cust_payment_terms"] = payment_term["code"]
    feature_row["cust_payment_terms_grp"] = payment_term["group"]
    feature_row["Allowed_Pay_Days"] = int(payment_term["allowed_days"])
    feature_row["baseline_month"] = int(baseline_timestamp.month)
    feature_row["baseline_day"] = int(baseline_timestamp.day)
    feature_row["baseline_dayofweek"] = int(baseline_timestamp.dayofweek)
    feature_row["due_weekend_flag"] = int(due_timestamp.dayofweek >= 5)

    previous_late_rate = float(np.clip(context["previous_late_rate"], 0.0, 1.0))
    recent_late_rate = float(np.clip(context["recent_late_rate"], 0.0, 1.0))

    feature_row["cust_allowed_pay_days_late_rate_past"] = previous_late_rate
    feature_row["ratio_paid_invoices_late_past"] = previous_late_rate
    feature_row["recent_3_late_rate"] = recent_late_rate
    feature_row["recent_5_late_rate"] = recent_late_rate
    feature_row["recent_10_late_rate"] = (recent_late_rate + previous_late_rate) / 2
    feature_row["recent_20_late_rate"] = previous_late_rate
    feature_row["late_rate_time_decay_lambda_0_8"] = recent_late_rate * 0.7 + previous_late_rate * 0.3
    feature_row["avg_days_late_paid_late_past"] = max(float(context["average_late_days"]), 0.0)
    feature_row["sum_outstanding_amount_past"] = max(float(context["sum_outstanding_amount_past"]), 0.0)
    feature_row["cleared_count"] = max(int(context["cleared_count"]), 0)
    feature_row["current_transaction_count"] = max(int(context["current_transaction_count"]), 0)
    feature_row["is_new_customer"] = int(context["is_new_customer"])
    feature_row["is_last_late"] = 1.0 if context["is_last_late"] else 0.0

    metadata = {
        "payment_term_code": payment_term["code"],
        "payment_term_label": f"{int(payment_term['allowed_days'])}일",
        "baseline": baseline_timestamp,
        "amount_usd": amount_usd,
    }
    return pd.DataFrame([feature_row], columns=artifact.feature_columns), metadata


def make_upload_predictions(artifact, upload_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    limited_frame = upload_frame.head(MAX_UPLOAD_ROWS).copy()
    contexts = upload_contexts(limited_frame)
    records = []
    base_probability_rows = []

    for position, (_, row) in enumerate(limited_frame.iterrows(), start=1):
        prediction_frame, metadata = upload_row_prediction_frame(artifact, row, contexts[position - 1])
        prediction = predict_invoice_delay(artifact, prediction_frame)
        probability_pct = prediction["probability"] * 100
        base_probability_rows.append(prediction["base_probabilities"])

        records.append(
            {
                "No": position,
                "인보이스 ID": format_identifier(row.get("invoice_id", row.get("doc_id", np.nan)), f"ROW-{position:02d}"),
                "고객명": clean_cell(row.get("name_customer", "")) or "-",
                "고객번호": format_identifier(row.get("cust_number", np.nan), "-"),
                "청구 금액(USD)": round(float(metadata.get("amount_usd", upload_amount_usd(row))), 2),
                "결제 조건": metadata.get("payment_term_label", clean_cell(row.get("cust_payment_terms", "")) or "-"),
                "청구 기준일": metadata.get("baseline", pd.NaT).strftime("%Y-%m-%d") if not pd.isna(metadata.get("baseline", pd.NaT)) else format_upload_date(row.get("baseline_create_date", pd.NaT)),
                "연체 확률(%)": round(probability_pct, 1),
                "위험 등급": prediction["risk_grade"]["label"],
                "업무 판단": prediction["risk_grade"]["message"],
                "Logistic Regression": round(prediction["base_probabilities"].get("Logistic Regression", 0.0) * 100, 1),
                "Random Forest": round(prediction["base_probabilities"].get("Random Forest", 0.0) * 100, 1),
                "XGBoost": round(prediction["base_probabilities"].get("XGBoost", 0.0) * 100, 1),
                "LightGBM": round(prediction["base_probabilities"].get("LightGBM", 0.0) * 100, 1),
            }
        )

    mean_base_probabilities = {
        model_name: float(np.mean([row[model_name] for row in base_probability_rows]))
        for model_name in artifact.base_models.keys()
    } if base_probability_rows else {}
    return pd.DataFrame(records), mean_base_probabilities


def batch_probability_chart(result_frame: pd.DataFrame):
    if result_frame.empty:
        return go.Figure()

    frame = result_frame.sort_values("연체 확률(%)", ascending=True).copy()
    labels = frame["No"].astype(str) + " · " + frame["인보이스 ID"].astype(str)
    color_map = {"높음": "#8d5963", "주의": "#b98b5f", "관찰": "#60769b", "낮음": "#344879"}
    colors = [color_map.get(label, "#60769b") for label in frame["위험 등급"]]

    fig = go.Figure(
        go.Bar(
            x=frame["연체 확률(%)"],
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{value:.1f}%" for value in frame["연체 확률(%)"]],
            textposition="auto",
            textfont={"color": "#ffffff", "size": 12},
            hovertemplate="%{y}<br>연체 확률 %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(300, 70 + len(frame) * 30),
        xaxis=dict(
            range=[0, 100],
            title={"text": "연체 확률", "font": {"color": "#303746", "size": 13}},
            tickfont={"color": "#303746", "size": 12},
            gridcolor="#e1e4eb",
            zerolinecolor="#c9ced8",
        ),
        yaxis=dict(title="", tickfont={"color": "#202735", "size": 12}),
        margin=dict(l=150, r=24, t=12, b=40),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#202735"),
    )
    return fig


def gauge_chart(probability: float, threshold: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 42, "color": "#263765"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#6a7280",
                    "tickfont": {"color": "#303746"},
                },
                "bar": {"color": "#394d7d", "thickness": 0.24},
                "bgcolor": "#ffffff",
                "borderwidth": 1,
                "bordercolor": "#c9ced8",
                "steps": [
                    {"range": [0, 20], "color": "#eef1f6"},
                    {"range": [20, threshold * 100], "color": "#e5e9f1"},
                    {"range": [threshold * 100, 50], "color": "#e8dec7"},
                    {"range": [50, 100], "color": "#e3c9c7"},
                ],
                "threshold": {
                    "line": {"color": "#8d5963", "width": 4},
                    "thickness": 0.75,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(
        height=315,
        margin=dict(l=20, r=20, t=12, b=0),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#161a22"),
    )
    return fig


def base_probability_chart(base_probabilities: dict[str, float]):
    names = list(base_probabilities.keys())
    values = [base_probabilities[name] * 100 for name in names]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=["#344879", "#60769b", "#955b66", "#6f7686"][: len(names)],
            text=[f"{value:.1f}%" for value in values],
            textposition="auto",
            textfont={"color": "#ffffff", "size": 13},
            hovertemplate="%{y}<br>%{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=250,
        xaxis=dict(
            range=[0, 100],
            title={"text": "연체 확률", "font": {"color": "#303746", "size": 13}},
            tickfont={"color": "#303746", "size": 12},
            gridcolor="#e1e4eb",
            zerolinecolor="#c9ced8",
        ),
        yaxis=dict(
            autorange="reversed",
            title="",
            tickfont={"color": "#202735", "size": 13},
        ),
        margin=dict(l=134, r=26, t=10, b=40),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#202735"),
    )
    return fig


def confusion_matrix_chart(matrix: list[list[int]]):
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["정상 예측", "연체 예측"],
            y=["실제 정상", "실제 연체"],
            colorscale=[[0, "#f1f3f7"], [0.55, "#8b99bb"], [1, "#263765"]],
            text=matrix,
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}: %{z:,}<extra></extra>",
            showscale=False,
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=8, r=8, t=12, b=8),
        paper_bgcolor="#ffffff",
        font=dict(color="#161a22"),
    )
    return fig


def model_comparison_chart(rows: list[dict]):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return go.Figure()
    frame = frame.sort_values("macro_f1", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=frame["macro_f1"] * 100,
            y=frame["model"],
            orientation="h",
            marker_color="#394d7d",
            text=[f"{value * 100:.1f}%" for value in frame["macro_f1"]],
            textposition="auto",
            textfont={"color": "#ffffff", "size": 13},
            hovertemplate="%{y}<br>Macro F1 %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=260,
        xaxis=dict(range=[0, 100], title="Macro F1"),
        yaxis=dict(title=""),
        margin=dict(l=12, r=24, t=12, b=36),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#161a22"),
    )
    return fig


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with st.spinner("모델을 불러오는 중입니다."):
    artifact = get_artifact()

metrics = artifact.metrics
tuned_metrics = metrics["stacking_tuned"]

st.markdown(
    """
    <div class="report-top">
        <div>
            <div class="report-kicker">Accounts Receivable Risk Desk</div>
            <div class="report-title">기업 연체위험 평가등급 확인 대시보드</div>
        </div>
        <div>
            <div class="brand-block">Invoice Delay Risk Analytics</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
metric_cols[0].metric("Accuracy", percent(tuned_metrics["accuracy"]))
metric_cols[1].metric("Macro F1", percent(tuned_metrics["macro_f1"]))
metric_cols[2].metric("Recall", percent(tuned_metrics["recall"]))
metric_cols[3].metric("운영 기준선", percent(artifact.threshold))

tab_single, tab_upload = st.tabs(["실시간 예측", "CSV 업로드 평가"])

with tab_single:
    input_col, result_col, note_col = st.columns([0.92, 1.1, 0.78], gap="large")

    labels_to_codes = payment_options(artifact)
    payment_labels = list(labels_to_codes.keys())
    outstanding_labels = outstanding_options(artifact)

    with input_col:
        st.markdown('<div class="input-head">평가 입력</div>', unsafe_allow_html=True)
        with st.form("invoice-risk-form"):
            invoice_amount = st.number_input(
                "청구 금액 (USD)",
                min_value=0.0,
                value=16800.0,
                step=500.0,
                format="%.2f",
            )
            selected_payment_label = st.selectbox(
                "결제 조건",
                payment_labels,
                index=default_payment_index(payment_labels),
            )
            baseline_date = st.date_input("청구 기준일", value=date.today())
            customer_type = st.radio(
                "고객 관계",
                ["기존 고객", "신규 고객"],
                horizontal=True,
            )
            previous_late_rate_pct = st.slider("과거 연체 비율", 0, 100, 5, step=1)
            outstanding_level = st.select_slider(
                "미수금 규모",
                outstanding_labels,
                value="보통" if "보통" in outstanding_labels else outstanding_labels[0],
            )
            st.caption("없음/낮음/보통/높음/매우 높음은 학습 데이터 기준 미수 잔액 누적 수준입니다.")
            last_transaction_late = st.toggle("직전 거래가 연체됨", value=False)
            st.form_submit_button("평가 실행", type="primary", width="stretch")

    recent_late_rate_pct = previous_late_rate_pct
    average_late_days = 3.0
    previous_transaction_count = 20
    active_invoice_count = 2

    prediction_frame = make_prediction_frame(
        artifact,
        invoice_amount_usd=invoice_amount,
        payment_term_code=labels_to_codes[selected_payment_label],
        baseline_date=baseline_date,
        existing_customer=customer_type == "기존 고객",
        previous_late_rate=previous_late_rate_pct / 100,
        recent_late_rate=recent_late_rate_pct / 100,
        average_late_days=average_late_days,
        outstanding_level=outstanding_level,
        previous_transaction_count=previous_transaction_count,
        active_invoice_count=active_invoice_count,
        last_transaction_late=last_transaction_late,
    )
    prediction = predict_invoice_delay(artifact, prediction_frame)
    probability = prediction["probability"]
    service_grade = prediction["risk_grade"]

    with result_col:
        st.markdown(
            '<div class="probability-panel"><div class="panel-head">모델 확률 그래프</div></div>',
            unsafe_allow_html=True,
        )

with tab_upload:
    st.markdown('<div class="input-head">CSV 업로드 평가</div>', unsafe_allow_html=True)
    uploaded_csv = st.file_uploader(
        "dataset.csv 형식의 인보이스 CSV 파일을 업로드하세요.",
        type=["csv"],
        accept_multiple_files=False,
    )

    with st.expander("업로드 형식 확인", expanded=uploaded_csv is None):
        st.markdown(
            """
            기본 형식은 `data/dataset.csv`와 동일합니다. 필수 컬럼은
            `total_open_amount`, `cust_payment_terms`, `baseline_create_date`입니다.
            `invoice_currency`, `cust_number`, `name_customer`, `invoice_id`,
            `due_in_date`, `clear_date`, `isOpen`이 있으면 고객 이력과 미수 정보를 함께 반영합니다.
            최대 15행까지 평가하며, 초과분은 앞 15행만 사용합니다.
            """
        )

    if uploaded_csv is not None:
        try:
            upload_frame = pd.read_csv(uploaded_csv)
        except Exception as exc:
            st.error(f"CSV 파일을 읽는 중 오류가 발생했습니다: {exc}")
            upload_frame = pd.DataFrame()

        if not upload_frame.empty:
            missing_columns = [
                column for column in RAW_UPLOAD_REQUIRED_COLUMNS
                if column not in upload_frame.columns and not all(feature in upload_frame.columns for feature in artifact.feature_columns)
            ]

            if missing_columns:
                st.error(f"필수 컬럼이 없습니다: {', '.join(missing_columns)}")
            else:
                if len(upload_frame) > MAX_UPLOAD_ROWS:
                    st.warning(f"업로드 파일은 {len(upload_frame):,}행입니다. 앞 {MAX_UPLOAD_ROWS}행만 평가합니다.")

                result_frame, mean_base_probabilities = make_upload_predictions(artifact, upload_frame)

                summary_cols = st.columns(4)
                summary_cols[0].metric("평가 건수", f"{len(result_frame):,}")
                summary_cols[1].metric("평균 연체 확률", f"{result_frame['연체 확률(%)'].mean():.1f}%")
                summary_cols[2].metric("주의 이상", f"{int((result_frame['연체 확률(%)'] >= artifact.threshold * 100).sum()):,}")
                summary_cols[3].metric("최고 위험 확률", f"{result_frame['연체 확률(%)'].max():.1f}%")

                chart_col, judge_col = st.columns([1.12, 0.88], gap="large")
                with chart_col:
                    st.markdown('<div class="probability-panel"><div class="panel-head">업로드 인보이스별 연체 확률</div></div>', unsafe_allow_html=True)
                    st.plotly_chart(
                        batch_probability_chart(result_frame),
                        width="stretch",
                        config={"displayModeBar": False},
                    )

                with judge_col:
                    st.markdown('<div class="model-judge-panel"><div class="panel-head">모델별 평균 판단</div></div>', unsafe_allow_html=True)
                    st.plotly_chart(
                        base_probability_chart(mean_base_probabilities),
                        width="stretch",
                        config={"displayModeBar": False},
                    )
                    st.markdown(
                        f"""
                        <div class="note-panel">
                            <div class="notice">
                                1. 업로드 결과는 파일 내 인보이스 조건과 동일 고객의 선행 이력을 기준으로 산출합니다.<br>
                                2. 파일 내 이력이 부족한 고객은 신규 또는 이력 부족 고객으로 보수적으로 처리합니다.<br>
                                3. 운영 기준선은 {artifact.threshold * 100:.1f}%입니다.<br>
                                4. 평가 기준은 지급일 이후 5영업일 초과 연체입니다.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown('<div class="model-result-box"><div class="panel-head">CSV 평가 결과</div></div>', unsafe_allow_html=True)
                display_columns = [
                    "No",
                    "인보이스 ID",
                    "고객명",
                    "청구 금액(USD)",
                    "결제 조건",
                    "청구 기준일",
                    "연체 확률(%)",
                    "위험 등급",
                    "업무 판단",
                    "Logistic Regression",
                    "Random Forest",
                    "XGBoost",
                    "LightGBM",
                ]
                st.dataframe(
                    result_frame[display_columns],
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "연체 확률(%)": st.column_config.ProgressColumn(
                            "연체 확률(%)",
                            min_value=0,
                            max_value=100,
                            format="%.1f%%",
                        ),
                        "청구 금액(USD)": st.column_config.NumberColumn("청구 금액(USD)", format="$%.2f"),
                    },
                )
                st.download_button(
                    "평가 결과 CSV 다운로드",
                    data=result_frame.to_csv(index=False).encode("utf-8-sig"),
                    file_name="invoice_delay_batch_predictions.csv",
                    mime="text/csv",
                    width="stretch",
                )
        st.plotly_chart(
            gauge_chart(probability, artifact.threshold),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.markdown(
            f"""
            <div class="panel-body-note">
                모델 산출 연체확률은 <strong>{probability * 100:.1f}%</strong>입니다.<br>
                현재 업무 판단: <strong>{escape(service_grade["label"])}</strong><br>
                {escape(service_grade["message"])}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with note_col:
        st.markdown(
            '<div class="model-judge-panel"><div class="panel-head">모델별 판단</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            base_probability_chart(prediction["base_probabilities"]),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.markdown(
            f"""
            <div class="note-panel">
                <div class="notice">
                    1. 본 확인서는 입력 조건과 과거 이력 정보를 기준으로 한 연체위험 평가 결과입니다.<br>
                    2. 실제 회수 가능성은 거래처 상황, 분쟁 여부, 지급 승인 절차 등에 따라 변동될 수 있습니다.<br>
                    3. 운영 기준선은 {artifact.threshold * 100:.1f}%입니다.<br>
                    4. 평가 기준은 지급일 이후 5영업일 초과 연체입니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
