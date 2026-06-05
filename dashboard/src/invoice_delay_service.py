from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model._logistic")

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


DATA_PATH = PROJECT_ROOT / "data" / "train_eda_revised.csv"
if not DATA_PATH.exists():
    DATA_PATH = PROJECT_ROOT.parent / "data" / "train_eda_revised.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "invoice_delay_stacking.joblib"

SEED = 42
DATE_COL = "baseline_create_date"
TARGET_COL = "target"
VALID_SIZE = 0.2
BEST_THRESHOLD = 0.32
ARTIFACT_VERSION = "2026-06-05-tuned-stacking-threshold-032"

DROP_COLS = [
    "cust_number",
    "name_customer",
    "clear_date",
    "buisness_year",
    "due_in_date",
    "posting_id",
    "baseline_create_date",
    "target_old",
    "year_month",
    "year_quarter",
    "business_days_late",
]

OUTSTANDING_LEVELS = ["없음", "낮음", "보통", "높음", "매우 높음"]


@dataclass
class InvoiceDelayArtifact:
    version: str
    meta_model: LogisticRegression
    base_models: dict[str, Pipeline]
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    default_row: dict[str, Any]
    payment_terms: list[dict[str, Any]]
    outstanding_levels: dict[str, float]
    amount_bin_thresholds: list[tuple[int, float]]
    threshold: float
    metrics: dict[str, Any]


def load_training_frame(data_path: Path = DATA_PATH) -> pd.DataFrame:
    frame = pd.read_csv(data_path)
    frame[DATE_COL] = pd.to_datetime(frame[DATE_COL])
    return frame.sort_values(DATE_COL).reset_index(drop=True)


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    drop_cols = [col for col in DROP_COLS if col in frame.columns]
    model_frame = frame.drop(columns=drop_cols).copy()
    y = model_frame[TARGET_COL].astype(int)
    X = model_frame.drop(columns=TARGET_COL)
    return X, y


def make_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_columns),
            ("cat", Pipeline(categorical_steps), categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_model_pipeline(
    model: Any,
    numeric_columns: list[str],
    categorical_columns: list[str],
    scale_numeric: bool = False,
) -> Pipeline:
    return Pipeline(
        [
            (
                "preprocess",
                make_preprocessor(
                    numeric_columns=numeric_columns,
                    categorical_columns=categorical_columns,
                    scale_numeric=scale_numeric,
                ),
            ),
            ("model", model),
        ]
    )


def make_baseline_models(
    numeric_columns: list[str],
    categorical_columns: list[str],
    scale_pos_weight: float,
) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": make_model_pipeline(
            LogisticRegression(
                C=0.1509600200360336,
                penalty="l2",
                class_weight=None,
                solver="saga",
                max_iter=5000,
                random_state=SEED,
                n_jobs=-1,
            ),
            numeric_columns,
            categorical_columns,
            scale_numeric=True,
        ),
        "Random Forest": make_model_pipeline(
            RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features="log2",
                class_weight=None,
                random_state=SEED,
                n_jobs=-1,
            ),
            numeric_columns,
            categorical_columns,
        ),
        "XGBoost": make_model_pipeline(
            XGBClassifier(
                n_estimators=700,
                max_depth=7,
                learning_rate=0.07475542028557966,
                subsample=0.7089245864240629,
                colsample_bytree=0.997906659156371,
                min_child_weight=1,
                gamma=0.7441507094807528,
                reg_alpha=6.296005477504001e-07,
                reg_lambda=0.003957390330656658,
                random_state=SEED,
                seed=SEED,
                eval_metric="logloss",
                tree_method="hist",
                scale_pos_weight=scale_pos_weight,
                n_jobs=-1,
            ),
            numeric_columns,
            categorical_columns,
        ),
        "LightGBM": make_model_pipeline(
            LGBMClassifier(
                n_estimators=400,
                learning_rate=0.04580308219230547,
                num_leaves=80,
                max_depth=10,
                min_child_samples=11,
                subsample=0.8745184318634152,
                colsample_bytree=0.7288069659920371,
                reg_alpha=0.2902177620992576,
                reg_lambda=0.4616548566638855,
                class_weight=None,
                random_state=SEED,
                verbose=-1,
                n_jobs=1,
            ),
            numeric_columns,
            categorical_columns,
        ),
    }


def evaluate_binary_model(
    y_true: pd.Series,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    pred = (probability >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }


def make_default_row(X_train: pd.DataFrame) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for column in X_train.columns:
        series = X_train[column]
        if pd.api.types.is_numeric_dtype(series):
            defaults[column] = float(series.median())
        else:
            mode = series.mode(dropna=True)
            defaults[column] = mode.iloc[0] if not mode.empty else ""
    return defaults


def make_payment_terms(raw: pd.DataFrame) -> list[dict[str, Any]]:
    def mode_value(series: pd.Series) -> Any:
        mode = series.mode(dropna=True)
        return mode.iloc[0] if not mode.empty else series.dropna().iloc[0]

    term_frame = (
        raw.groupby("cust_payment_terms")
        .agg(
            payment_group=("cust_payment_terms_grp", mode_value),
            allowed_days=("Allowed_Pay_Days", mode_value),
            count=("cust_payment_terms", "size"),
        )
        .reset_index()
        .sort_values(["count", "allowed_days"], ascending=[False, True])
    )

    terms: list[dict[str, Any]] = []
    for row in term_frame.to_dict("records"):
        allowed_days = int(row["allowed_days"])
        day_label = "즉시/특수 조건" if allowed_days <= 0 else f"{allowed_days}일 조건"
        terms.append(
            {
                "code": row["cust_payment_terms"],
                "group": row["payment_group"],
                "allowed_days": allowed_days,
                "count": int(row["count"]),
                "label": f"{day_label} · ERP {row['cust_payment_terms']}",
            }
        )
    return terms


def make_outstanding_levels(X_train: pd.DataFrame) -> dict[str, float]:
    if "sum_outstanding_amount_past" not in X_train.columns:
        return {level: 0.0 for level in OUTSTANDING_LEVELS}

    series = X_train["sum_outstanding_amount_past"].astype(float)
    return {
        "없음": 0.0,
        "낮음": float(series.quantile(0.25)),
        "보통": float(series.quantile(0.50)),
        "높음": float(series.quantile(0.75)),
        "매우 높음": float(series.quantile(0.90)),
    }


def make_amount_bin_thresholds(X_train: pd.DataFrame) -> list[tuple[int, float]]:
    if "amount_bin" not in X_train.columns:
        return [(0, float("inf"))]

    grouped = (
        X_train.groupby("amount_bin")["amount_in_usd"]
        .max()
        .reset_index()
        .sort_values("amount_bin")
    )
    return [(int(row["amount_bin"]), float(row["amount_in_usd"])) for row in grouped.to_dict("records")]


def infer_amount_bin(amount_log: float, thresholds: list[tuple[int, float]]) -> int:
    for bin_id, upper_bound in thresholds:
        if amount_log <= upper_bound:
            return bin_id
    return thresholds[-1][0]


def stacking_predict(
    base_models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
) -> tuple[LogisticRegression, dict[str, Pipeline], np.ndarray, dict[str, np.ndarray]]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_preds = np.zeros((len(X_train), len(base_models)))
    valid_preds = np.zeros((len(X_valid), len(base_models)))
    fitted_base_models: dict[str, Pipeline] = {}
    base_valid_probabilities: dict[str, np.ndarray] = {}

    X = X_train.reset_index(drop=True)
    y = y_train.reset_index(drop=True)

    for model_idx, (model_name, model) in enumerate(base_models.items()):
        valid_fold_preds = np.zeros((len(X_valid), skf.n_splits))

        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            fold_model = clone(model)
            fold_model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            oof_preds[val_idx, model_idx] = fold_model.predict_proba(X.iloc[val_idx])[:, 1]
            valid_fold_preds[:, fold_idx] = fold_model.predict_proba(X_valid)[:, 1]

        valid_preds[:, model_idx] = valid_fold_preds.mean(axis=1)

        final_base = clone(model)
        final_base.fit(X_train, y_train)
        fitted_base_models[model_name] = final_base
        base_valid_probabilities[model_name] = final_base.predict_proba(X_valid)[:, 1]

    meta_model = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=1.0,
        max_iter=3000,
        random_state=SEED,
    )
    meta_model.fit(oof_preds, y_train)
    stacking_valid_proba = meta_model.predict_proba(valid_preds)[:, 1]
    return meta_model, fitted_base_models, stacking_valid_proba, base_valid_probabilities


def fit_final_stacking(
    base_models: dict[str, Pipeline],
    X_full: pd.DataFrame,
    y_full: pd.Series,
) -> tuple[LogisticRegression, dict[str, Pipeline]]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    X = X_full.reset_index(drop=True)
    y = y_full.reset_index(drop=True)
    oof_preds = np.zeros((len(X), len(base_models)))
    fitted_base_models: dict[str, Pipeline] = {}

    for model_idx, (model_name, model) in enumerate(base_models.items()):
        for tr_idx, val_idx in skf.split(X, y):
            fold_model = clone(model)
            fold_model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            oof_preds[val_idx, model_idx] = fold_model.predict_proba(X.iloc[val_idx])[:, 1]

        final_base = clone(model)
        final_base.fit(X_full, y_full)
        fitted_base_models[model_name] = final_base

    meta_model = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=1.0,
        max_iter=3000,
        random_state=SEED,
    )
    meta_model.fit(oof_preds, y_full)
    return meta_model, fitted_base_models


def train_invoice_delay_artifact(data_path: Path = DATA_PATH) -> InvoiceDelayArtifact:
    raw = load_training_frame(data_path)
    split_idx = int(len(raw) * (1 - VALID_SIZE))
    train_inner_raw = raw.iloc[:split_idx].copy()
    valid_raw = raw.iloc[split_idx:].copy()

    X_full, y_full = split_xy(raw)
    X_train, y_train = split_xy(train_inner_raw)
    X_valid, y_valid = split_xy(valid_raw)
    categorical_columns = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()
    positive = y_train.sum()
    negative = len(y_train) - positive
    scale_pos_weight = negative / positive
    base_models = make_baseline_models(numeric_columns, categorical_columns, scale_pos_weight)

    _, _, stacking_valid_proba, base_valid_probabilities = stacking_predict(
        base_models=base_models,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
    )
    final_meta_model, final_base_models = fit_final_stacking(base_models, X_full, y_full)

    base_model_metrics = []
    for model_name, probability in base_valid_probabilities.items():
        base_metrics = evaluate_binary_model(y_valid, probability, threshold=0.5)
        base_model_metrics.append(
            {
                "model": model_name,
                "accuracy": base_metrics["accuracy"],
                "macro_f1": base_metrics["macro_f1"],
                "roc_auc": base_metrics["roc_auc"],
            }
        )

    stacking_default_metrics = evaluate_binary_model(y_valid, stacking_valid_proba, threshold=0.5)
    stacking_tuned_metrics = evaluate_binary_model(y_valid, stacking_valid_proba, threshold=BEST_THRESHOLD)

    metrics = {
        "train_rows": int(len(train_inner_raw)),
        "validation_rows": int(len(valid_raw)),
        "feature_count": int(X_train.shape[1]),
        "target_definition": "business_days_late > 5",
        "train_late_rate": float(y_train.mean()),
        "validation_late_rate": float(y_valid.mean()),
        "stacking_default": stacking_default_metrics,
        "stacking_tuned": stacking_tuned_metrics,
        "base_models": sorted(base_model_metrics, key=lambda item: item["macro_f1"], reverse=True),
        "meta_model_coefficients": {
            model_name: float(coef)
            for model_name, coef in zip(final_base_models.keys(), final_meta_model.coef_[0])
        },
    }

    return InvoiceDelayArtifact(
        version=ARTIFACT_VERSION,
        meta_model=final_meta_model,
        base_models=final_base_models,
        feature_columns=X_full.columns.tolist(),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        default_row=make_default_row(X_full),
        payment_terms=make_payment_terms(raw),
        outstanding_levels=make_outstanding_levels(X_train),
        amount_bin_thresholds=make_amount_bin_thresholds(X_full),
        threshold=BEST_THRESHOLD,
        metrics=metrics,
    )


def save_artifact(artifact: InvoiceDelayArtifact, model_path: Path = MODEL_PATH) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)


def load_or_train_artifact(
    model_path: Path = MODEL_PATH,
    data_path: Path = DATA_PATH,
    force_retrain: bool = False,
) -> InvoiceDelayArtifact:
    if model_path.exists() and not force_retrain:
        artifact = joblib.load(model_path)
        if getattr(artifact, "version", None) == ARTIFACT_VERSION:
            return artifact

    artifact = train_invoice_delay_artifact(data_path)
    save_artifact(artifact, model_path)
    return artifact


def get_payment_term(artifact: InvoiceDelayArtifact, payment_term_code: str) -> dict[str, Any]:
    for term in artifact.payment_terms:
        if term["code"] == payment_term_code:
            return term
    return artifact.payment_terms[0]


def make_prediction_frame(
    artifact: InvoiceDelayArtifact,
    *,
    invoice_amount_usd: float,
    payment_term_code: str,
    baseline_date: Any,
    existing_customer: bool,
    previous_late_rate: float,
    recent_late_rate: float,
    average_late_days: float,
    outstanding_level: str,
    previous_transaction_count: int,
    active_invoice_count: int,
    last_transaction_late: bool,
) -> pd.DataFrame:
    row = dict(artifact.default_row)
    payment_term = get_payment_term(artifact, payment_term_code)
    baseline_timestamp = pd.Timestamp(baseline_date)
    amount_log = float(np.log1p(max(invoice_amount_usd, 0.0)))

    row["amount_in_usd"] = amount_log
    row["amount_bin"] = infer_amount_bin(amount_log, artifact.amount_bin_thresholds)
    row["cust_payment_terms"] = payment_term["code"]
    row["cust_payment_terms_grp"] = payment_term["group"]
    row["Allowed_Pay_Days"] = int(payment_term["allowed_days"])
    row["baseline_month"] = int(baseline_timestamp.month)
    row["baseline_day"] = int(baseline_timestamp.day)
    row["baseline_dayofweek"] = int(baseline_timestamp.dayofweek)

    due_date = baseline_timestamp + pd.Timedelta(days=int(payment_term["allowed_days"]))
    row["due_weekend_flag"] = int(due_date.dayofweek >= 5)

    previous_late_rate = float(np.clip(previous_late_rate, 0.0, 1.0))
    recent_late_rate = float(np.clip(recent_late_rate, 0.0, 1.0))

    row["is_new_customer"] = 0 if existing_customer else 1
    row["cust_allowed_pay_days_late_rate_past"] = previous_late_rate
    row["ratio_paid_invoices_late_past"] = previous_late_rate
    row["recent_3_late_rate"] = recent_late_rate
    row["recent_5_late_rate"] = recent_late_rate
    row["recent_10_late_rate"] = (recent_late_rate + previous_late_rate) / 2
    row["recent_20_late_rate"] = previous_late_rate
    row["late_rate_time_decay_lambda_0_8"] = recent_late_rate * 0.7 + previous_late_rate * 0.3
    row["avg_days_late_paid_late_past"] = max(float(average_late_days), 0.0)
    row["sum_outstanding_amount_past"] = artifact.outstanding_levels.get(outstanding_level, 0.0)
    row["cleared_count"] = max(int(previous_transaction_count), 0)
    row["current_transaction_count"] = max(int(active_invoice_count), 0)
    row["is_last_late"] = 1.0 if last_transaction_late else 0.0

    return pd.DataFrame([row], columns=artifact.feature_columns)


def risk_grade(probability: float, threshold: float = BEST_THRESHOLD) -> dict[str, str]:
    if probability >= 0.5:
        return {
            "label": "높음",
            "tone": "danger",
            "message": "선제 연락, 결제 조건 재확인, 승인 보류 검토가 필요합니다.",
        }
    if probability >= threshold:
        return {
            "label": "주의",
            "tone": "warning",
            "message": "담당자 확인과 결제 일정 리마인드가 권장됩니다.",
        }
    if probability >= 0.2:
        return {
            "label": "관찰",
            "tone": "watch",
            "message": "현재 조건은 수용 가능하지만 최근 이력 변화를 확인하세요.",
        }
    return {
        "label": "낮음",
        "tone": "safe",
        "message": "통상적인 처리 흐름으로 관리 가능한 수준입니다.",
    }


def predict_invoice_delay(
    artifact: InvoiceDelayArtifact,
    prediction_frame: pd.DataFrame,
) -> dict[str, Any]:
    base_probabilities = {
        model_name: float(model.predict_proba(prediction_frame)[:, 1][0])
        for model_name, model in artifact.base_models.items()
    }
    meta_input = np.array([[base_probabilities[name] for name in artifact.base_models.keys()]])
    probability = float(artifact.meta_model.predict_proba(meta_input)[:, 1][0])
    grade = risk_grade(probability, artifact.threshold)
    return {
        "probability": probability,
        "prediction": int(probability >= artifact.threshold),
        "risk_grade": grade,
        "base_probabilities": base_probabilities,
    }
