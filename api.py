from pathlib import Path
from typing import Optional
import json
import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ============================================================
# Configuración base
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "Datos" / "Best_model"

DEFAULT_INPUT_PATH = (
    BASE_DIR
    / "Datos"
    / "Data Preprocesada"
    / "preprocessed"
    / "vars_10_extrac.csv"
)

DEFAULT_OUTPUT_DIR = BASE_DIR / "Datos" / "Output" / "api"

ID_COLS = ["key_value", "partition", "p_codmes"]
TARGET_COL = "target"


# ============================================================
# Inicialización de FastAPI
# ============================================================

app = FastAPI(
    title="API de Inferencia - Pipeline ML E2E CU Venta",
    description=(
        "API funcional para ejecutar inferencia batch usando el último modelo "
        "entrenado del pipeline ML E2E."
    ),
    version="1.0.0",
)


class PredictRequest(BaseModel):
    input_path: Optional[str] = None
    output_dir: Optional[str] = None


# ============================================================
# Funciones auxiliares
# ============================================================

def get_latest_model_folder(model_dir: Path) -> Path:
    """
    Retorna la carpeta más reciente dentro de Datos/Best_model.
    """
    if not model_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de modelos: {model_dir}")

    folders = [p for p in model_dir.iterdir() if p.is_dir()]

    if not folders:
        raise FileNotFoundError(f"No hay modelos entrenados dentro de: {model_dir}")

    latest_folder = max(folders, key=lambda p: p.stat().st_mtime)
    return latest_folder


def load_model():
    """
    Carga el último modelo entrenado guardado como xgb_model.pkl.
    """
    latest_folder = get_latest_model_folder(MODEL_DIR)

    model_path = latest_folder / "xgb_model.pkl"
    metadata_path = latest_folder / "xgb_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"No existe el archivo del modelo: {model_path}")

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    metadata = {}

    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

    return model, latest_folder, metadata


def prepare_features(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Prepara la matriz de variables para el modelo.
    Usa exactamente las variables esperadas por el modelo entrenado.
    """
    drop_cols = [c for c in ID_COLS + [TARGET_COL] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")

    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)

        missing_cols = [col for col in expected_features if col not in X.columns]
        extra_cols = [col for col in X.columns if col not in expected_features]

        for col in missing_cols:
            X[col] = 0

        X = X[expected_features]

        print(f"Columnas faltantes agregadas: {len(missing_cols)}")
        print(f"Columnas extras eliminadas: {len(extra_cols)}")

    return X


# ============================================================
# Endpoints
# ============================================================

@app.get("/")
def root():
    return {
        "message": "API de inferencia activa",
        "project": "Pipeline ML E2E CU Venta",
        "docs": "Abrir /docs para probar la API",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_dir_exists": MODEL_DIR.exists(),
        "default_input_exists": DEFAULT_INPUT_PATH.exists(),
        "model_dir": str(MODEL_DIR),
        "default_input_path": str(DEFAULT_INPUT_PATH),
    }


@app.get("/model-info")
def model_info():
    try:
        _, latest_folder, metadata = load_model()

        return {
            "status": "ok",
            "latest_model_folder": str(latest_folder),
            "metadata": metadata,
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/predict-batch")
def predict_batch(request: PredictRequest):
    """
    Ejecuta inferencia batch sobre un archivo CSV preprocesado.
    Si no se envía input_path, usa vars_10_extrac.csv por defecto.
    """
    try:
        input_path = Path(request.input_path) if request.input_path else DEFAULT_INPUT_PATH
        output_dir = Path(request.output_dir) if request.output_dir else DEFAULT_OUTPUT_DIR

        if not input_path.exists():
            raise FileNotFoundError(f"No existe el archivo de entrada: {input_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        model, latest_folder, metadata = load_model()

        df = pd.read_csv(input_path)

        X = prepare_features(df, model)

        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X)[:, 1]
        else:
            scores = model.predict(X)

        result = pd.DataFrame()

        if "key_value" in df.columns:
            result["key_value"] = df["key_value"]

        if "partition" in df.columns:
            result["partition"] = df["partition"]

        result["score"] = scores

        output_path = output_dir / "api_predictions.csv"
        result.to_csv(output_path, index=False)

        return {
            "status": "ok",
            "input_path": str(input_path),
            "model_folder": str(latest_folder),
            "rows_scored": int(len(result)),
            "score_mean": float(result["score"].mean()),
            "score_min": float(result["score"].min()),
            "score_max": float(result["score"].max()),
            "output_path": str(output_path),
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))