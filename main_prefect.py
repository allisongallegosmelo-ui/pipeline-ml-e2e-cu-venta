"""
main_prefect.py
Orquestador Prefect del pipeline ML E2E.

Ejecutar en local/Colab:
python main_prefect.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from prefect import flow, task
except ImportError as exc:
    raise ImportError("Prefect no está instalado. Ejecutar: pip install prefect") from exc

BASE_DIR = Path(__file__).resolve().parent
CODE_DIR = BASE_DIR / "Codigos"
sys.path.append(str(CODE_DIR))

import preprocessing as prep
import training as aml
import inference as inf
import monitoring as mon
import postprocessing as posp


@task
def preprocess_data(payload: dict):
    """Ejecuta preprocesamiento según modo training/inference."""
    print("Running preprocessing")
    outputs = prep.main(
        model_name=payload["params"]["model_name"],
        DIR_RAWDATA=payload["DIR_RAWDATA"],
        DIR_PROCESSED=payload["DIR_PROCESSED"],
        type_work=payload["params"]["mode_type"],
        period=payload["params"].get("partition", ""),
    )

    partition_id = payload["params"].get("partition", "")
    model_name = payload["params"]["model_name"]
    dir_pre_processed = outputs.get("vars") or os.path.join(
        payload["DIR_PROCESSED"], "preprocessed", f"vars_{partition_id}_{model_name}.csv"
    )
    dir_pos_processed = outputs.get("post") or os.path.join(
        payload["DIR_PROCESSED"], "postprocessed", f"post_{partition_id}_{model_name}.csv"
    )
    return dir_pre_processed, dir_pos_processed


@task
def prepare_training_data(payload: dict):
    """Genera train/test desde la carpeta Data Entrenamiento."""
    print("Preparing training data")
    model_name = payload["params"]["model_name"]
    return prep.main(
        model_name=model_name,
        DIR_RAWDATA=payload["DIR_TRAIN_RAWDATA"],
        DIR_PROCESSED=payload["DIR_PROCESSED"],
        type_work="training",
        period="",
    )


@task
def train_model(payload: dict):
    """Entrena modelo con HPO."""
    model_name = payload["params"]["model_name"]
    data_train_path = os.path.join(payload["TRAINING_DATA"], f"train_vars_{model_name}.csv")
    data_test_path = os.path.join(payload["TRAINING_DATA"], f"test_vars_{model_name}.csv")
    print("Running training")
    return aml.main(
        train_path=data_train_path,
        test_path=data_test_path,
        model_save_dir=payload["MODEL_DIR"],
        n_trials=payload["params"].get("n_trials", 20),
    )


@task
def run_inference(payload: dict, dir_pre_processed: str):
    """Ejecuta inferencia usando el último modelo disponible."""
    print("Running inference")
    return inf.main(payload["MODEL_DIR"], dir_pre_processed, payload["SCORE_DIR"])


@task
def run_monitoring(payload: dict, score_file_path: str):
    """Ejecuta monitoreo PSI/AUC/Recall por decil."""
    print("Running monitoring")
    return mon.main(payload, score_file_path)


@task
def postprocess_results(payload: dict, dir_pos_processed: str, score_file_path: str):
    """Ejecuta TLV, grupos y réplica."""
    print("Running postprocessing")
    return posp.main(dir_pos_processed, score_file_path, payload["DIR_OUTPUT"])


@flow(name="mlops_pipeline_cu_venta")
def mlops_pipeline(payload: dict):
    """Pipeline E2E: training/inference + monitoreo + postprocessing."""
    if payload["params"].get("prepare_training_data", False):
        prepare_training_data(payload)

    if payload["params"]["mode_type"] == "training":
        train_model(payload)
        payload["params"]["mode_type"] = "inference"

    dir_pre_processed, dir_pos_processed = preprocess_data(payload)
    score_file_path = run_inference(payload, dir_pre_processed)
    monitoring_result = run_monitoring(payload, score_file_path)

    if monitoring_result["requires_retraining"] and payload["params"].get("auto_retrain", True):
        print("Alerta de monitoreo: se ejecutará reentrenamiento automático.")
        prepare_training_data(payload)
        train_model(payload)
        score_file_path = run_inference(payload, dir_pre_processed)
        monitoring_result = run_monitoring(payload, score_file_path)

    final_output_path = postprocess_results(payload, dir_pos_processed, score_file_path)
    return {
        "score_file_path": score_file_path,
        "monitoring_result": monitoring_result,
        "final_output_path": final_output_path,
    }


if __name__ == "__main__":
    payload = {
        "DIR_TRAIN_RAWDATA": str(BASE_DIR / "Datos" / "Data Entrenamiento"),
        "DIR_RAWDATA": str(BASE_DIR / "Datos" / "Data Cruda"),
        "DIR_PROCESSED": str(BASE_DIR / "Datos" / "Data Preprocesada"),
        "MODEL_DIR": str(BASE_DIR / "Datos" / "Best_model"),
        "TRAINING_DATA": str(BASE_DIR / "Datos" / "Data Preprocesada" / "training_data" / "preprocessed"),
        "SCORE_DIR": str(BASE_DIR / "Datos" / "Output" / "score"),
        "MONITORING_DIR": str(BASE_DIR / "Datos" / "Output" / "monitoring"),
        "DIR_OUTPUT": str(BASE_DIR / "Datos" / "Output" / "final"),
        "params": {
            "model_name": "extrac",
            "mode_type": "training",          # training prepara/entrena y luego hace inferencia
            "partition": 10,                   # usa p10_extrac.csv como OOT
            "prepare_training_data": True,
            "auto_retrain": True,
            "psi_threshold": 0.25,
            "n_trials": 10,
        },
    }
    mlops_pipeline(payload)
