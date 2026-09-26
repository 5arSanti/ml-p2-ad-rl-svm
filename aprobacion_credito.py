from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

COLUMNAS = [
    "ingreso_mensual_kcop",
    "score_buro",
    "antiguedad_laboral_anios",
    "num_creditos_activos",
    "dti",
    "aprobado",
]

NOMBRE_CSV = "aprobacion_credito_Arias_Becerra_Johel_Santiago_3144.csv"
RUTA_ESPERADA = f"data/{NOMBRE_CSV}"


def ruta_dataset() -> Path:
    candidatos = [
        Path("data") / NOMBRE_CSV,
        Path("../data") / NOMBRE_CSV,
        Path(__file__).resolve().parent / "data" / NOMBRE_CSV,
    ]
    for path in candidatos:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"No se encontró el dataset. Ruta esperada: {RUTA_ESPERADA}"
    )


def cargar_dataset(path: Path | None = None) -> pd.DataFrame:
    destino = Path(path) if path is not None else ruta_dataset()
    if not destino.is_file():
        raise FileNotFoundError(
            f"No se encontró el dataset. Ruta esperada: {RUTA_ESPERADA}"
        )
    df = pd.read_csv(destino)
    if list(df.columns) != COLUMNAS:
        raise ValueError(
            f"columnas distintas a las esperadas {COLUMNAS}; recibidas {list(df.columns)}"
        )
    if len(df) != 1000:
        raise ValueError(f"se esperaban 1000 filas; hay {len(df)}")
    if df.isna().any().any():
        raise ValueError("el dataset no debe tener nulos")
    if not set(df["aprobado"].unique()) <= {0, 1}:
        raise ValueError("aprobado debe ser 0 o 1")
    return df


def partir_train_test(df: pd.DataFrame):
    X = df.drop(columns="aprobado")
    y = df["aprobado"]
    return train_test_split(X, y, test_size=0.25, random_state=11, stratify=y)


def metricas_prueba(y_true, y_pred, y_score) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if tn + fp == 0:
        raise ValueError("especificidad indefinida: no hay negativos reales")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "especificidad": float(tn / (tn + fp)),
        "auc": float(roc_auc_score(y_true, y_score)),
    }


def construir_modelos() -> dict:
    return {
        "Árbol": DecisionTreeClassifier(max_depth=4, random_state=11),
        "Regresión logística": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=11)),
            ]
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, random_state=11)),
            ]
        ),
    }
