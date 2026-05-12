# Pipeline ML E2E - CU Venta

Proyecto base para la tarea **Implementación pipeline ML E2E**.

## Estructura

```text
ml_pipeline_tarea/
├── main_prefect.py
├── 7_Pipeline_prefect_actualizado.ipynb
├── Codigos/
│   ├── preprocessing.py
│   ├── training.py
│   ├── inference.py
│   ├── monitoring.py
│   ├── postprocessing.py
│   └── posprocessing.py      # alias de compatibilidad
├── Datos/
│   ├── Data Cruda/
│   ├── Data Entrenamiento/
│   ├── Data Preprocesada/
│   ├── Best_model/
│   └── Output/
├── requirements.txt
└── README.md
```

## Qué hace el pipeline

1. Lee archivos crudos de entrenamiento desde `Datos/Data Entrenamiento`.
2. Genera `train_vars_extrac.csv` y `test_vars_extrac.csv`.
3. Entrena un modelo XGBoost con hiperparametrización mediante Optuna.
4. Guarda el modelo campeón y su metadata.
5. Preprocesa el periodo OOT desde `Datos/Data Cruda`, por ejemplo `p10_extrac.csv`.
6. Genera scores de inferencia.
7. Calcula monitoreo: PSI, AUC y recall por decil cuando existe `target`.
8. Calcula puntuación TLV y grupos de ejecución.
9. Genera archivos de réplica pipe-delimitados para S3, Athena y On Premise.

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python main_prefect.py
```

Por defecto, el `payload` de `main_prefect.py` trabaja con:

- Entrenamiento: `Datos/Data Entrenamiento`
- Inferencia/OOT: `Datos/Data Cruda/p10_extrac.csv`
- Modelo: `Datos/Best_model`
- Outputs: `Datos/Output`

## Modo de trabajo

En `main_prefect.py`, modificar:

```python
"mode_type": "training"
"partition": 10
"n_trials": 20
```

Si ya existe un modelo entrenado y solo se desea inferencia, usar:

```python
"mode_type": "inference"
"prepare_training_data": False
```

## Monitoreo

El archivo `monitoring.py` calcula:

- PSI del score.
- AUC OOT si el periodo crudo contiene `target`.
- Recall acumulado por decil.
- Flag de estabilidad:
  - `OK`: PSI < 0.10
  - `WARN`: 0.10 <= PSI < 0.25
  - `ALERT`: PSI >= 0.25

Si `requires_retraining = true`, el orquestador puede reentrenar automáticamente.

## Nota

El archivo `posprocessing.py` se mantiene como alias para compatibilidad con el notebook original, pero la versión ordenada para la entrega es `postprocessing.py`.
## Repositorio GitHub

El proyecto se encuentra disponible en el siguiente enlace:

https://github.com/allisongallegosmelo-ui/pipeline-ml-e2e-cu-venta
