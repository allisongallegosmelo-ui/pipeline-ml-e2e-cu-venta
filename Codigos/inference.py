"""inference.py: carga el último modelo entrenado y genera scores."""

from __future__ import annotations

import glob
import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "bool":
            df[col] = df[col].astype(int)
        elif str(df[col].dtype).startswith("int"):
            df[col] = df[col].astype(int)
        elif str(df[col].dtype).startswith("float"):
            df[col] = df[col].astype(float).round(4)
    return df


def find_latest_model_folder(base_dir: str) -> str:
    """Ubica la carpeta timestamp más reciente dentro del directorio de modelos."""
    latest_folder = None
    latest_timestamp = None
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            try:
                folder_timestamp = datetime.strptime(item, "%Y-%m-%d_%H-%M-%S")
            except ValueError:
                continue
            if latest_timestamp is None or folder_timestamp > latest_timestamp:
                latest_timestamp = folder_timestamp
                latest_folder = item_path
    if latest_folder is None:
        raise FileNotFoundError(f"No se encontró ningún modelo entrenado en {base_dir}")
    print(f"Modelo seleccionado: {latest_folder}")
    return latest_folder


def load_model_and_metadata(models_dir: str):
    latest_folder_path = find_latest_model_folder(models_dir)
    model_files = glob.glob(f"{latest_folder_path}/*.pkl")
    metadata_files = glob.glob(f"{latest_folder_path}/*.json")
    if not model_files or not metadata_files:
        raise FileNotFoundError(f"Falta modelo o metadata en {latest_folder_path}")

    with open(model_files[0], "rb") as f:
        model = pickle.load(f)
    with open(metadata_files[0], "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata


def perform_inference(data_path: str, model, metadata: dict) -> pd.DataFrame:
    df_inference = preprocess_dataframe(pd.read_csv(data_path))
    feature_columns = metadata.get("feature_columns", list(df_inference.columns))

    for col in feature_columns:
        if col not in df_inference.columns:
            df_inference[col] = 0
    df_inference = df_inference[feature_columns]

    predictions = model.predict_proba(df_inference)[:, 1]
    return pd.DataFrame({"predictions": predictions})


def main(models_dir: str, preprocessed_data_path: str, output_dir: str) -> str:
    """Ejecuta inferencia y guarda scores."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = os.path.basename(preprocessed_data_path)
    parts = filename.replace(".csv", "").split("_")  # vars_10_extrac
    if len(parts) < 3:
        raise ValueError(f"Formato inesperado de archivo: {filename}. Esperado: vars_<period>_<model>.csv")

    partition = parts[1]
    model_name = parts[2]
    model, metadata = load_model_and_metadata(models_dir)
    predictions_df = perform_inference(preprocessed_data_path, model, metadata)

    output_path = os.path.join(output_dir, f"inference_{model_name}_{partition}.csv")
    predictions_df.to_csv(output_path, index=False)
    print(f"Predicciones guardadas en: {output_path}")
    return output_path
