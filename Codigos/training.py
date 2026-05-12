"""
training.py
Entrenamiento con hiperparametrización usando Optuna para XGBoost.
Si MLflow está instalado, registra parámetros, métricas y modelo.
"""

from __future__ import annotations

import json
import os
import pickle
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

TARGET_COL = "target"
RANDOM_STATE = 42

try:
    import optuna
except ImportError:  # permite importar el módulo aunque falte optuna
    optuna = None

try:
    import mlflow
    import mlflow.xgboost
except ImportError:
    mlflow = None


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza tipos numéricos para evitar errores de entrenamiento."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "bool":
            df[col] = df[col].astype(int)
        elif str(df[col].dtype).startswith("int"):
            df[col] = df[col].astype(int)
        elif str(df[col].dtype).startswith("float"):
            df[col] = df[col].astype(float).round(4)
    return df


def _xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COL not in df.columns:
        raise ValueError(f"No se encontró la columna target: {TARGET_COL}")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    return X, y


def _align_columns(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for col in set(X_train.columns) - set(X_test.columns):
        X_test[col] = 0
    for col in set(X_test.columns) - set(X_train.columns):
        X_train[col] = 0
    return X_train.sort_index(axis=1), X_test[X_train.sort_index(axis=1).columns]


def _suggest_xgb_params(trial: Any) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 80, 350),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "subsample": trial.suggest_float("subsample", 0.70, 1.00),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.00),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 2.0, log=True),
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def _fallback_params() -> dict[str, Any]:
    """Parámetros base si Optuna no está instalado."""
    return {
        "n_estimators": 180,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "gamma": 0.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def save_model(model: xgb.XGBClassifier, performance: dict[str, float], params: dict[str, Any], save_dir: str, feature_columns: list[str]) -> str:
    """Guarda modelo y metadata en una carpeta timestamp."""
    folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_dir = Path(save_dir) / folder_name
    model_save_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_save_dir / "xgb_model.pkl"
    metadata_path = model_save_dir / "xgb_metadata.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    metadata = {
        "ml_name": "xgb",
        "performance": performance,
        "hyperparameters": params,
        "feature_columns": feature_columns,
        "library_versions": {
            "xgboost": xgb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "python_platform": platform.platform(),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, default=str)

    print(f"Modelo guardado en: {model_path}")
    print(f"Metadata guardada en: {metadata_path}")
    return str(model_save_dir)


def train_and_log(train_path: str, test_path: str, model_save_dir: str, n_trials: int = 20, experiment_name: str = "cu_venta_e2e") -> tuple[str | None, xgb.XGBClassifier]:
    """Entrena XGBoost con HPO y registra resultados si MLflow está disponible."""
    train_df = preprocess_dataframe(pd.read_csv(train_path))
    test_df = preprocess_dataframe(pd.read_csv(test_path))

    X_train, y_train = _xy(train_df)
    X_test, y_test = _xy(test_df)
    X_train, X_test = _align_columns(X_train, X_test)

    if optuna is not None:
        def objective(trial: Any) -> float:
            params = _suggest_xgb_params(trial)
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            pred = model.predict_proba(X_test)[:, 1]
            return roc_auc_score(y_test, pred)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best_params = _suggest_xgb_params(study.best_trial)
        best_auc_cv = float(study.best_value)
    else:
        print("Advertencia: Optuna no está instalado. Se entrenará con parámetros base. Instalar con: pip install optuna")
        best_params = _fallback_params()
        best_auc_cv = np.nan

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    train_pred = model.predict_proba(X_train)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]
    auc_train = roc_auc_score(y_train, train_pred)
    auc_test = roc_auc_score(y_test, test_pred)
    decay = ((auc_train - auc_test) / auc_train) * 100 if auc_train > 0 else np.nan

    performance = {
        "auc_train": float(auc_train),
        "auc_test": float(auc_test),
        "decay_percent": float(decay),
        "best_auc_hpo": float(best_auc_cv) if not np.isnan(best_auc_cv) else None,
    }

    run_id = None
    if mlflow is not None:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.log_params(best_params)
            mlflow.log_metrics({k: v for k, v in performance.items() if v is not None})
            mlflow.xgboost.log_model(model, artifact_path="model", registered_model_name="cu_venta_xgb")
    else:
        print("Advertencia: MLflow no está instalado. Se guardará modelo y metadata localmente.")

    save_model(model, performance, best_params, model_save_dir, list(X_train.columns))
    print(f"AUC Train: {auc_train:.4f} | AUC Test: {auc_test:.4f} | Decay: {decay:.2f}%")
    return run_id, model


def main(train_path: str, test_path: str, model_save_dir: str, n_trials: int = 20) -> tuple[str | None, xgb.XGBClassifier]:
    """Función compatible con el notebook base."""
    return train_and_log(train_path, test_path, model_save_dir, n_trials=n_trials)
