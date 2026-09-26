from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

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
