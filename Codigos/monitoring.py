"""
monitoring.py
Monitoreo del pipeline: PSI de score, AUC y Recall acumulado por decil.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score

import inference as inf

TARGET_COL = "target"
EPS = 1e-6


def psi_flag(psi: float) -> str:
    """Etiqueta de alerta según PSI."""
    if psi < 0.10:
        return "OK"
    if psi < 0.25:
        return "WARN"
    return "ALERT"


def _read_scores(score_path: str) -> np.ndarray:
    scores = pd.read_csv(score_path)
    if "predictions" in scores.columns:
        return scores["predictions"].astype(float).to_numpy()
    return scores.iloc[:, 0].astype(float).to_numpy()


def calculate_psi(expected_scores: np.ndarray, actual_scores: np.ndarray, buckets: int = 10) -> float:
    """Calcula PSI usando cortes por cuantiles de la distribución esperada."""
    expected_scores = np.asarray(expected_scores, dtype=float)
    actual_scores = np.asarray(actual_scores, dtype=float)

    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = np.quantile(expected_scores, quantiles)
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) <= 2:
        breakpoints = np.linspace(min(expected_scores.min(), actual_scores.min()), max(expected_scores.max(), actual_scores.max()), buckets + 1)

    expected_counts, _ = np.histogram(expected_scores, bins=breakpoints)
    actual_counts, _ = np.histogram(actual_scores, bins=breakpoints)

    expected_pct = np.maximum(expected_counts / max(expected_counts.sum(), 1), EPS)
    actual_pct = np.maximum(actual_counts / max(actual_counts.sum(), 1), EPS)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def compute_recall_by_decile(y_true, scores, n_deciles: int = 10) -> pd.DataFrame:
    """Calcula recall acumulado por decil de score. Decil 1 = mayor score."""
    df = pd.DataFrame({"score": scores, "target": y_true}).dropna()
    df["target"] = df["target"].astype(int)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["decil"] = pd.qcut(df.index + 1, q=n_deciles, labels=range(1, n_deciles + 1))

    total_positives = df["target"].sum()
    rows = []
    accumulated = 0
    for decil, group in df.groupby("decil", observed=False):
        accumulated += group["target"].sum()
        recall_acc = accumulated / total_positives if total_positives > 0 else 0
        rows.append({
            "decil": int(decil),
            "clientes": int(len(group)),
            "positivos_decil": int(group["target"].sum()),
            "recall_acumulado": float(recall_acc),
        })
    return pd.DataFrame(rows)


def _find_raw_file(raw_dir: str, partition: int | str) -> str | None:
    matches = sorted(glob.glob(os.path.join(raw_dir, f"p{partition}_*.csv")))
    return matches[0] if matches else None


def run_monitoring(
    train_path: str,
    score_path: str,
    models_dir: str,
    output_dir: str,
    raw_eval_path: str | None = None,
    psi_threshold: float = 0.25,
) -> dict:
    """Ejecuta monitoreo de score y métricas de validación/OOT si existe target."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model, metadata = inf.load_model_and_metadata(models_dir)
    train_scores = inf.perform_inference(train_path, model, metadata)["predictions"].to_numpy()
    eval_scores = _read_scores(score_path)

    psi_score = calculate_psi(train_scores, eval_scores)
    flag = psi_flag(psi_score)

    result = {
        "psi_score": psi_score,
        "psi_flag": flag,
        "requires_retraining": bool(psi_score >= psi_threshold),
        "auc": None,
        "recall_at_50": None,
    }

    recall_table = pd.DataFrame()
    if raw_eval_path and os.path.exists(raw_eval_path):
        raw_eval = pd.read_csv(raw_eval_path, usecols=lambda c: c == TARGET_COL)
        if TARGET_COL in raw_eval.columns and len(raw_eval) == len(eval_scores):
            y_true = raw_eval[TARGET_COL].astype(int)
            result["auc"] = float(roc_auc_score(y_true, eval_scores))
            result["recall_at_50"] = float(recall_score(y_true, eval_scores >= 0.5, zero_division=0))
            recall_table = compute_recall_by_decile(y_true, eval_scores)

    metrics_path = os.path.join(output_dir, "monitoring_metrics.json")
    recall_path = os.path.join(output_dir, "recall_by_decile.csv")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    if not recall_table.empty:
        recall_table.to_csv(recall_path, index=False)

    print(f"PSI Score: {psi_score:.4f} | Flag: {flag} | Reentrenar: {result['requires_retraining']}")
    if result["auc"] is not None:
        print(f"AUC OOT: {result['auc']:.4f}")
    print(f"Métricas guardadas en: {metrics_path}")
    return result


def main(payload: dict, score_path: str) -> dict:
    """Wrapper compatible con el notebook Prefect."""
    partition = payload["params"]["partition"]
    model_name = payload["params"]["model_name"]
    train_path = os.path.join(payload["TRAINING_DATA"], f"train_vars_{model_name}.csv")
    raw_eval_path = _find_raw_file(payload["DIR_RAWDATA"], partition)
    return run_monitoring(
        train_path=train_path,
        score_path=score_path,
        models_dir=payload["MODEL_DIR"],
        output_dir=payload["MONITORING_DIR"],
        raw_eval_path=raw_eval_path,
        psi_threshold=payload.get("params", {}).get("psi_threshold", 0.25),
    )
