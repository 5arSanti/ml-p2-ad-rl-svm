from pathlib import Path

import pandas as pd
import pytest

from aprobacion_credito import COLUMNAS, cargar_dataset, ruta_dataset


def test_ruta_dataset_existe():
    path = ruta_dataset()
    assert path.is_file()
    assert path.name == "aprobacion_credito_Arias_Becerra_Johel_Santiago_3144.csv"


def test_cargar_dataset_esquema_y_tamano():
    df = cargar_dataset()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1000, 6)
    assert list(df.columns) == COLUMNAS
    assert df.isna().sum().sum() == 0
    assert set(df["aprobado"].unique()) <= {0, 1}
    assert int(df["aprobado"].sum()) == 267


def test_cargar_dataset_falla_si_falta_archivo(tmp_path):
    desaparecido = tmp_path / "no_existe.csv"
    with pytest.raises(FileNotFoundError, match="data/aprobacion_credito_Arias_Becerra_Johel_Santiago_3144.csv"):
        cargar_dataset(desaparecido)


def test_cargar_dataset_falla_si_columnas_distintas(tmp_path):
    malo = tmp_path / "malo.csv"
    pd.DataFrame({"x": [1], "aprobado": [0]}).to_csv(malo, index=False)
    with pytest.raises(ValueError, match="columnas"):
        cargar_dataset(malo)
