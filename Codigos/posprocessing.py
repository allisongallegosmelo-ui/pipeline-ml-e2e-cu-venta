"""
postprocessing.py
Scoring TLV, segmentación en grupos y generación de réplica pipe-delimitada.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd

DIST_GE = [0, 0.035, 0.087, 0.237, 0.393, 0.529, 0.664, 0.787, 0.862, 0.95, 1.0]


def _read_scores(score_path_or_values) -> np.ndarray:
    if isinstance(score_path_or_values, str):
        scores = pd.read_csv(score_path_or_values)
        if "predictions" in scores.columns:
            return scores["predictions"].astype(float).to_numpy()
        return scores.iloc[:, 0].astype(float).to_numpy()
    return np.asarray(score_path_or_values, dtype=float)


def get_groups(scores, df_post: pd.DataFrame) -> pd.DataFrame:
    """Calcula puntuación TLV y grupo de ejecución. Fórmula conservada según clase."""
    df_post = df_post.copy()
    scores = _read_scores(scores)
    if len(scores) != len(df_post):
        raise ValueError(f"Scores ({len(scores)}) y df_post ({len(df_post)}) no tienen la misma cantidad de filas.")

    df_post["prob"] = scores
    df_post["prob_frescura"] = np.where(
        df_post["grp_campecs06m"] == "G1", 0.066,
        np.where(df_post["grp_campecs06m"] == "G2", 0.028,
        np.where(df_post["grp_campecs06m"] == "G3", 0.022,
        np.where(df_post["grp_campecs06m"] == "G4", 0.008, 0.004)))
    )

    df_post["prob_value_contact"] = pd.to_numeric(df_post["prob_value_contact"], errors="coerce").fillna(0.000001)
    df_post["monto"] = pd.to_numeric(df_post["monto"], errors="coerce").fillna(0)
    df_post["puntuacion_tlv"] = (
        df_post["prob"]
        * df_post["prob_value_contact"]
        * np.log(df_post["monto"] + 1)
        * df_post["prob_frescura"]
    )

    df_post["grupo_ejec_tlv"] = pd.qcut(
        df_post["puntuacion_tlv"].rank(method="first"),
        q=DIST_GE,
        labels=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    )
    return df_post


def run_postprocessing(scores, df_post: pd.DataFrame, output_path: str | None = None) -> pd.DataFrame:
    """Wrapper de get_groups con guardado opcional."""
    result = get_groups(scores, df_post)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"Output TLV guardado en: {output_path}")
    return result


def save_replica(
    df_post: pd.DataFrame,
    table: str,
    partition: str | int,
    dir_s3: str = "data/replica/s3",
    dir_athena: str = "data/replica/athena",
    dir_onpremise: str = "data/replica/onpremise",
) -> list[str]:
    """Genera archivos de réplica pipe-delimitados para tres destinos."""
    for folder in [dir_s3, dir_athena, dir_onpremise]:
        Path(folder).mkdir(parents=True, exist_ok=True)

    df_replica = pd.DataFrame({
        "codmes": df_post["partition"],
        "tipdoc": "1",
        "coddoc": df_post["key_value"],
        "puntuacion": df_post["puntuacion_tlv"],
        "modelo": table,
        "fec_replica": datetime.date.today().strftime("%Y%m%d"),
        "grupo_ejec": df_post["grupo_ejec_tlv"],
        "score": df_post["prob"],
        "orden": "",
        "variable1": df_post["codunicocli"].apply(lambda x: str(x).zfill(10)),
        "variable2": df_post["monto"],
        "variable3": "",
    })

    df_replica = df_replica.sort_values("puntuacion", ascending=False)
    df_replica = df_replica.drop_duplicates("coddoc", keep="first")
    df_replica["orden"] = df_replica["puntuacion"].rank(method="first", ascending=False).astype(int)

    filename = f"scr_{table.lower().replace(' ', '_')}_{partition}.txt"
    paths = []
    for folder in [dir_s3, dir_athena, dir_onpremise]:
        path = os.path.join(folder, filename)
        df_replica.to_csv(path, index=False, sep="|")
        paths.append(path)
        print(f"Réplica guardada en: {path}")
    return paths


def main(DIR_POS_PROCESSED: str, DIR_SCORE: str, DIR_OUTPUT: str, table: str = "EC_OMNICANAL") -> str:
    """Ejecuta postprocesamiento completo y genera réplica."""
    filename = os.path.basename(DIR_SCORE)  # inference_extrac_10.csv
    parts = filename.replace(".csv", "").split("_")
    if len(parts) < 3:
        raise ValueError(f"Formato de score inesperado: {filename}")

    model_name = parts[1]
    partition = parts[2]

    df_post = pd.read_csv(DIR_POS_PROCESSED)
    scores = _read_scores(DIR_SCORE)

    output_tlv_path = os.path.join(DIR_OUTPUT, f"output_tlv_{model_name}_{partition}.csv")
    df_result = run_postprocessing(scores, df_post, output_tlv_path)

    save_replica(
        df_result,
        table=table,
        partition=partition,
        dir_s3=os.path.join(DIR_OUTPUT, "replica", "s3"),
        dir_athena=os.path.join(DIR_OUTPUT, "replica", "athena"),
        dir_onpremise=os.path.join(DIR_OUTPUT, "replica", "onpremise"),
    )
    return output_tlv_path
