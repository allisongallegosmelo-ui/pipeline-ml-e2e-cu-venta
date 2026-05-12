"""
preprocessing.py
Preprocesamiento del pipeline ML E2E para CU Venta.

Funciones principales:
- Lee data cruda para entrenamiento o inferencia.
- Limpia nulos y transforma variables.
- Genera archivos de variables para modelo y archivos de postproceso.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COL = "target"
MODEL_NAME_DEFAULT = "extrac"
RANDOM_STATE = 123
TEST_SIZE = 0.33

CATEGORICAL_COLUMNS = ["ent_1erlntcrallsfm01"]

NUMERIC_COLUMNS = [
    "nro_producto_6m", "prom_uso_tc_rccsf3m", "ctd_sms_received",
    "max_usotcribksf06m", "ctd_camptot06m", "dsv_svppallsf06m",
    "prm_svprmecs06m", "ctd_app_productos_m1", "ctd_campecsm01",
    "lin_tcrrstsf03m", "mnt_ptm", "dif_no_gestionado_4meses",
    "max_campecs06m", "beta_pctusotcr12m", "rat_disefepnm01",
    "flg_saltotppe12m", "prom_sow_lintcribksf3m", "openhtml_1m",
    "nprod_1m", "nro_transfer_6m", "max_usotcrrstsf03m",
    "prm_cnt_fee_amt_u7d", "pas_avg6m_max12m", "beta_saltotppe12m",
    "seg_un", "ant_ultprdallsf", "avg_sald_pas_3m", "pas_1m_avg3m",
    "num_incrsaldispefe06m", "cnl_age_p4m_p12m", "cnl_atm_p4m_p12m",
    "cre_lin_tc_rccibk_m07", "prm_svprmlibdis06m", "ingreso_neto",
    "max_nact_12m", "cre_sldtotfinprm03", "dif_contacto_efectivo_10meses",
    "act_1m_avg3m", "monto_consumos_ecommerce_tc", "ctd_camptotm01",
    "prop_atm_4m", "prom_pct_saldopprcc6m", "apppag_1m",
    "nro_configuracion_6m", "act_avg6m_max12m", "sldvig_tcrsrcf",
    "prom_score_acepta_12meses", "telefonos_6meses", "pas_1m_avg6m",
    "ctd_camptototrcnl06m", "prm_saltotrdpj03m", "bpitrx_1m",
    "prm_lintcribksf03m", "ctd_entrdm01", "avg_openhtml_6m", "tea",
    "pct_usotcrm01", "senthtml_1m",
]

POST_COLUMNS = [
    "partition", "key_value", "codunicocli", "grp_campecs06m",
    "prob_value_contact", "monto",
]

MODEL_COLUMNS = NUMERIC_COLUMNS + [
    "ent_1erlntcrallsfm01_INTERBANK",
    "ent_1erlntcrallsfm01_OTRO",
]

CATEGORY_MAP = {"ent_1erlntcrallsfm01": ["INTERBANK", "OTRO"]}


def _ensure_dirs(*dirs: str | os.PathLike) -> None:
    for folder in dirs:
        Path(folder).mkdir(parents=True, exist_ok=True)


def _find_csv_files(raw_dir: str | os.PathLike, type_work: str, period: int | str | None = None) -> list[str]:
    raw_dir = str(raw_dir)
    if type_work == "training":
        files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    elif type_work == "inference":
        if period in [None, ""]:
            raise ValueError("Para inferencia se debe indicar 'period', por ejemplo 10 para p10_extrac.csv.")
        files = sorted(glob.glob(os.path.join(raw_dir, f"p{period}_*.csv")))
    else:
        raise ValueError("type_work debe ser 'training' o 'inference'.")

    if not files:
        raise FileNotFoundError(f"No se encontraron CSV en {raw_dir} para type_work={type_work}, period={period}.")
    return files


def read_rawdata(period: int | str | None, type_work: str, DIR_RAWDATA: str) -> pd.DataFrame:
    """Lee la data cruda según modo training o inference."""
    files = _find_csv_files(DIR_RAWDATA, type_work, period)
    frames = []
    for file in files:
        print(f"Leyendo: {file}")
        frames.append(pd.read_csv(file))
    df = pd.concat(frames, ignore_index=True)
    print(f"Data leída: {df.shape[0]:,} filas x {df.shape[1]:,} columnas")
    return df


def one_hot_encoding(df: pd.DataFrame, col: str, categories: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica one-hot encoding manteniendo categorías fijas entre training e inference."""
    selected_col = df[col].astype(pd.CategoricalDtype(list(categories)))
    new_cols = pd.get_dummies(selected_col, prefix=col)
    df = df.drop(columns=[col])
    df = pd.concat([df, new_cols], axis=1)
    return df, new_cols


def process_vars(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia variables, imputa nulos y genera variables dummy."""
    df = df.copy()
    df = df.replace(["", "null", "None"], np.nan)

    for column in NUMERIC_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(-9999999).astype("float32")

    for column in POST_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    if "partition" in df.columns:
        df["partition"] = df["partition"].astype("string")

    for col, allowed_values in CATEGORY_MAP.items():
        if col not in df.columns:
            df[col] = "OTRO"
        df[col] = df[col].fillna("SV").astype("string")
        df.loc[~df[col].isin([v for v in allowed_values if v != "OTRO"]), col] = "OTRO"
        df, _ = one_hot_encoding(df, col, allowed_values)

    for col in MODEL_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    return df


def save_outputs(df: pd.DataFrame, period: int | str | None, model: str, DIR_PROCESSED: str, type_work: str) -> dict[str, str]:
    """Guarda archivos preprocesados y postprocesados requeridos por el pipeline."""
    pre_dir = Path(DIR_PROCESSED) / "preprocessed"
    post_dir = Path(DIR_PROCESSED) / "postprocessed"
    train_pre_dir = Path(DIR_PROCESSED) / "training_data" / "preprocessed"
    train_post_dir = Path(DIR_PROCESSED) / "training_data" / "postprocessed"
    _ensure_dirs(pre_dir, post_dir, train_pre_dir, train_post_dir)

    outputs: dict[str, str] = {}

    if type_work == "training":
        if TARGET_COL not in df.columns:
            raise ValueError(f"La data de entrenamiento debe contener la columna target: {TARGET_COL}")

        cols = list(dict.fromkeys(MODEL_COLUMNS + POST_COLUMNS))
        x_train, x_test, y_train, y_test = train_test_split(
            df[cols], df[TARGET_COL].astype(int), test_size=TEST_SIZE,
            random_state=RANDOM_STATE, stratify=df[TARGET_COL].astype(int)
        )

        train_vars = pd.concat([y_train.rename(TARGET_COL), x_train[MODEL_COLUMNS]], axis=1)
        test_vars = pd.concat([y_test.rename(TARGET_COL), x_test[MODEL_COLUMNS]], axis=1)

        outputs["train_vars"] = str(train_pre_dir / f"train_vars_{model}.csv")
        outputs["test_vars"] = str(train_pre_dir / f"test_vars_{model}.csv")
        outputs["train_post"] = str(train_post_dir / f"train_post_{model}.csv")
        outputs["test_post"] = str(train_post_dir / f"test_post_{model}.csv")

        train_vars.to_csv(outputs["train_vars"], index=False)
        test_vars.to_csv(outputs["test_vars"], index=False)
        x_train[POST_COLUMNS].to_csv(outputs["train_post"], index=False)
        x_test[POST_COLUMNS].to_csv(outputs["test_post"], index=False)

    elif type_work == "inference":
        if period in [None, ""]:
            raise ValueError("Para inferencia se debe indicar period.")
        outputs["vars"] = str(pre_dir / f"vars_{period}_{model}.csv")
        outputs["post"] = str(post_dir / f"post_{period}_{model}.csv")
        df[MODEL_COLUMNS].to_csv(outputs["vars"], index=False)
        df[POST_COLUMNS].to_csv(outputs["post"], index=False)

    else:
        raise ValueError("type_work debe ser 'training' o 'inference'.")

    print("Archivos generados:")
    for name, path in outputs.items():
        print(f" - {name}: {path}")
    return outputs


def main(model_name: str, DIR_RAWDATA: str, DIR_PROCESSED: str, type_work: str, period: int | str | None = "") -> dict[str, str]:
    """Ejecuta el preprocesamiento completo."""
    df = read_rawdata(period, type_work, DIR_RAWDATA)
    df = process_vars(df)
    return save_outputs(df, period, model_name, DIR_PROCESSED, type_work)
