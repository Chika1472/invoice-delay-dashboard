from __future__ import annotations

from datetime import date
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.invoice_delay_service import (
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

with st.container():
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
