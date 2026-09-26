# Comparación de 3 modelos para aprobación de crédito — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Entregar un notebook autónomo que compara Árbol, Regresión logística y SVM sobre el CSV individual, con tabla de 5 métricas, recomendación de negocio y bitácora de IA, listo para Moodle y sustentación.

**Architecture:** Las funciones de carga, split, métricas y construcción de modelos se implementan primero en `aprobacion_credito.py` con TDD (apoyo del repo, no se sube a Moodle). El notebook `notebooks/comparacion_modelos_aprobacion_credito.ipynb` es el entregable: copia esas funciones en celdas, no importa el módulo, y corre de arriba abajo. El CSV vive en `data/`.

**Tech Stack:** Python 3, pandas, numpy, scikit-learn, matplotlib, pytest, jupyter/nbformat/nbclient.

## Global Constraints

- Split cableado: `train_test_split(X, y, test_size=0.25, random_state=11, stratify=y)`.
- Árbol: `DecisionTreeClassifier(max_depth=4, random_state=11)` sin scaler.
- Logística: `Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=11))])`.
- SVM: `Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, random_state=11))])`.
- Métricas solo en test; clase positiva = 1; 3 decimales; especificidad = `TN/(TN+FP)` con `labels=[0, 1]`.
- Sin GridSearch, sin `class_weight`, sin SMOTE, sin umbral distinto de 0.5.
- Notebook autónomo (Moodle recibe solo el `.ipynb` + el alumno debe poder ejecutarlo si coloca el CSV en `data/` o `../data/`).
- Markdown del notebook en español de negocio.
- CSV: `data/aprobacion_credito_Arias_Becerra_Johel_Santiago_3144.csv` (1000 filas, 6 columnas, 0 nulos).
- Recomendación ya calculada con este CSV y este diseño: **regresión logística** (ver Task 7). No elegir el ganador con `idxmax`.

## File structure

| Path | Responsibility |
|---|---|
| `requirements.txt` | pandas, numpy, scikit-learn, matplotlib, jupyter, nbformat, nbclient, pytest |
| `data/aprobacion_credito_Arias_Becerra_Johel_Santiago_3144.csv` | Dataset individual |
| `aprobacion_credito.py` | `COLUMNAS`, `ruta_dataset`, `cargar_dataset`, `partir_train_test`, `metricas_prueba`, `construir_modelos` |
| `tests/test_carga.py` | Contrato del CSV y del cargador |
| `tests/test_split.py` | Tamaños 750/250 y 200/67 positivos |
| `tests/test_metricas.py` | Fórmulas de las 5 métricas |
| `tests/test_modelos.py` | Quién lleva scaler y qué hiperparámetros |
| `notebooks/comparacion_modelos_aprobacion_credito.ipynb` | Entregable Moodle (funciones inline + narrativa) |
| `docs/superpowers/specs/2026-09-26-comparacion-modelos-aprobacion-credito-design.md` | Spec aprobada (no modificar salvo contradicción) |

---

### Task 1: Dependencias, CSV y `cargar_dataset`

**Files:**
- Create: `requirements.txt`
- Create: `data/aprobacion_credito_Arias_Becerra_Johel_Santiago_3144.csv`
- Create: `tests/test_carga.py`
- Create: `aprobacion_credito.py`

**Interfaces:**
- Consumes: CSV fuente en `/home/ubuntu/.cursor/projects/workspace/uploads/aprobacion_credito_Arias_Becerra_Johel_Santiago_3144.csv` (si no está, buscar el adjunto del taller con el mismo nombre).
- Produces: `COLUMNAS: list[str]`; `ruta_dataset() -> Path`; `cargar_dataset(path: Path \| None = None) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_carga.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pip install pytest pandas numpy scikit-learn matplotlib -q && python3 -m pytest tests/test_carga.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'aprobacion_credito'` (o `cannot import`).

- [ ] **Step 3: Copy CSV, write requirements, implement loader**

```text
# requirements.txt
pandas
numpy
scikit-learn
matplotlib
jupyter
nbformat
nbclient
pytest
```

```bash
mkdir -p data
cp /home/ubuntu/.cursor/projects/workspace/uploads/aprobacion_credito_Arias_Becerra_Johel_Santiago_3144.csv \
  data/aprobacion_credito_Arias_Becerra_Johel_Santiago_3144.csv
```

```python
# aprobacion_credito.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_carga.py -v`

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt data/aprobacion_credito_Arias_Becerra_Johel_Santiago_3144.csv aprobacion_credito.py tests/test_carga.py
git commit -m "feat: add dataset loader and schema checks"
```

---

### Task 2: `partir_train_test`

**Files:**
- Create: `tests/test_split.py`
- Modify: `aprobacion_credito.py`

**Interfaces:**
- Consumes: `cargar_dataset() -> pd.DataFrame`
- Produces: `partir_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]` con nombres `Xtr, Xte, ytr, yte`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_split.py
from aprobacion_credito import cargar_dataset, partir_train_test


def test_split_tamanos_y_estrato():
    df = cargar_dataset()
    Xtr, Xte, ytr, yte = partir_train_test(df)
    assert list(Xtr.columns) == [
        "ingreso_mensual_kcop",
        "score_buro",
        "antiguedad_laboral_anios",
        "num_creditos_activos",
        "dti",
    ]
    assert "aprobado" not in Xtr.columns
    assert len(Xtr) == 750
    assert len(Xte) == 250
    assert int(ytr.sum()) == 200
    assert int(yte.sum()) == 67
    assert ytr.mean() == 200 / 750
    assert yte.mean() == 67 / 250


def test_split_es_reproducible():
    df = cargar_dataset()
    a = partir_train_test(df)
    b = partir_train_test(df)
    assert a[0].index.equals(b[0].index)
    assert a[1].index.equals(b[1].index)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_split.py -v`

Expected: FAIL with `ImportError` / `cannot import name 'partir_train_test'`.

- [ ] **Step 3: Write minimal implementation**

Añadir al final de `aprobacion_credito.py`:

```python
from sklearn.model_selection import train_test_split


def partir_train_test(df: pd.DataFrame):
    X = df.drop(columns="aprobado")
    y = df["aprobado"]
    return train_test_split(X, y, test_size=0.25, random_state=11, stratify=y)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_split.py tests/test_carga.py -v`

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add aprobacion_credito.py tests/test_split.py
git commit -m "feat: add class split with seed 11 and stratify"
```

---

### Task 3: `metricas_prueba`

**Files:**
- Create: `tests/test_metricas.py`
- Modify: `aprobacion_credito.py`

**Interfaces:**
- Consumes: `y_true`, `y_pred`, `y_score` tipo array-like.
- Produces: `metricas_prueba(y_true, y_pred, y_score) -> dict[str, float]` con claves `accuracy`, `precision`, `recall`, `especificidad`, `auc`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metricas.py
import pytest

from aprobacion_credito import metricas_prueba


def test_metricas_prueba_caso_manual():
    # TN=2, FP=1, FN=1, TP=2  → N=6
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.4, 0.9, 0.7]
    m = metricas_prueba(y_true, y_pred, y_score)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["especificidad"] == pytest.approx(2 / 3)
    assert m["auc"] == pytest.approx(7 / 9)  # 0.777... = roc_auc_score de este caso


def test_metricas_prueba_precision_cero_sin_positivos_predichos():
    y_true = [0, 0, 1]
    y_pred = [0, 0, 0]
    y_score = [0.1, 0.2, 0.3]
    m = metricas_prueba(y_true, y_pred, y_score)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_metricas.py -v`

Expected: FAIL with `cannot import name 'metricas_prueba'`.

- [ ] **Step 3: Write minimal implementation**

Añadir a `aprobacion_credito.py`:

```python
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_metricas.py tests/test_carga.py tests/test_split.py -v`

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add aprobacion_credito.py tests/test_metricas.py
git commit -m "feat: add shared test-set classification metrics"
```

---

### Task 4: `construir_modelos`

**Files:**
- Create: `tests/test_modelos.py`
- Modify: `aprobacion_credito.py`

**Interfaces:**
- Consumes: nada del dataset.
- Produces: `construir_modelos() -> dict[str, object]` con claves exactas `"Árbol"`, `"Regresión logística"`, `"SVM"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_modelos.py
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from aprobacion_credito import construir_modelos


def test_construir_modelos_contratos():
    modelos = construir_modelos()
    assert list(modelos) == ["Árbol", "Regresión logística", "SVM"]

    arbol = modelos["Árbol"]
    assert isinstance(arbol, DecisionTreeClassifier)
    assert not isinstance(arbol, Pipeline)
    assert arbol.max_depth == 4
    assert arbol.random_state == 11

    for nombre in ("Regresión logística", "SVM"):
        pipe = modelos[nombre]
        assert isinstance(pipe, Pipeline)
        assert isinstance(pipe.named_steps["scaler"], StandardScaler)
        assert "clf" in pipe.named_steps

    log = modelos["Regresión logística"].named_steps["clf"]
    assert isinstance(log, LogisticRegression)
    assert log.max_iter == 1000
    assert log.random_state == 11

    svm = modelos["SVM"].named_steps["clf"]
    assert isinstance(svm, SVC)
    assert svm.kernel == "rbf"
    assert svm.probability is True
    assert svm.random_state == 11
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_modelos.py -v`

Expected: FAIL with `cannot import name 'construir_modelos'`.

- [ ] **Step 3: Write minimal implementation**

Añadir a `aprobacion_credito.py`:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/ -v`

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add aprobacion_credito.py tests/test_modelos.py
git commit -m "feat: add the three required classifiers with correct scaling"
```

---

### Task 5: Integración de métricas reales (no elige ganador)

**Files:**
- Create: `tests/test_evaluacion_test.py`

**Interfaces:**
- Consumes: `cargar_dataset`, `partir_train_test`, `construir_modelos`, `metricas_prueba`.
- Produces: garantía de que el pipeline de test da la tabla conocida de este CSV (3 decimales).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluacion_test.py
import pytest

from aprobacion_credito import (
    cargar_dataset,
    construir_modelos,
    metricas_prueba,
    partir_train_test,
)

ESPERADO = {
    "Árbol": {
        "accuracy": 0.700,
        "precision": 0.426,
        "recall": 0.343,
        "especificidad": 0.831,
        "auc": 0.683,
    },
    "Regresión logística": {
        "accuracy": 0.764,
        "precision": 0.611,
        "recall": 0.328,
        "especificidad": 0.923,
        "auc": 0.805,
    },
    "SVM": {
        "accuracy": 0.768,
        "precision": 0.645,
        "recall": 0.299,
        "especificidad": 0.940,
        "auc": 0.749,
    },
}


def test_metricas_de_test_coinciden_con_la_spec_de_este_csv():
    df = cargar_dataset()
    Xtr, Xte, ytr, yte = partir_train_test(df)
    modelos = construir_modelos()
    for nombre, modelo in modelos.items():
        modelo.fit(Xtr, ytr)
        pred = modelo.predict(Xte)
        score = modelo.predict_proba(Xte)[:, 1]
        m = metricas_prueba(yte, pred, score)
        for clave, valor in ESPERADO[nombre].items():
            assert round(m[clave], 3) == pytest.approx(valor, abs=1e-3), (nombre, clave, m[clave])
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(yte, pred, labels=[0, 1])
        assert int(cm.sum()) == 250
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_evaluacion_test.py -v`

Expected: PASS ya, si Tasks 1–4 están bien (este test es de regresión de números). Si falla por redondeo, no cambiar hiperparámetros: ajustar solo el assert al `round(..., 3)` que imprima el propio test. Si falla porque aún no existen funciones, FAIL de import — terminar Tasks 1–4 primero.

Si el test **pasa de inmediato**, no hay implementación extra. Sigue al commit de “lock” de números.

- [ ] **Step 3: Write minimal implementation**

No hay código nuevo si 1–4 están hechos. Si un redondeo discrepa (p. ej. 0.6825 → 0.682), imprimir `round(m[clave], 3)` y actualizar `ESPERADO`, no los modelos.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_evaluacion_test.py -v`

Expected: 1 passed. Puede aparecer `FutureWarning` de `SVC(probability=True)` en scikit-learn ≥ 1.9. No silenciarlo cambiando a `CalibratedClassifierCV`; la spec fija `probability=True`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_evaluacion_test.py
git commit -m "test: lock test-set metrics for the three models"
```

---

### Task 6: Notebook autónomo hasta la tabla

**Files:**
- Create: `notebooks/comparacion_modelos_aprobacion_credito.ipynb`

**Interfaces:**
- Consumes: las funciones de Tasks 1–4, **copiadas inline** (el notebook no hace `import aprobacion_credito`).
- Produces: notebook ejecutable con secciones 1–5 de la spec (contexto, carga, EDA, split, 3 modelos, tabla, matrices).

- [ ] **Step 1: Write a smoke test that the notebook file will exist with required headings**

```python
# tests/test_notebook_estructura.py
import json
from pathlib import Path

NOTEBOOK = Path("notebooks/comparacion_modelos_aprobacion_credito.ipynb")

TITULOS = [
    "Contexto",
    "Carga y EDA",
    "Split",
    "Entrenamiento",
    "Métricas",
]


def test_notebook_existe_y_tiene_secciones():
    assert NOTEBOOK.is_file()
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    textos = []
    for cell in nb["cells"]:
        textos.append("".join(cell.get("source", [])))
    blob = "\n".join(textos)
    for titulo in TITULOS:
        assert titulo in blob, titulo
    assert "import aprobacion_credito" not in blob
    assert "max_depth=4" in blob
    assert "probability=True" in blob
    assert "test_size=0.25" in blob
    assert "random_state=11" in blob
    assert "stratify=y" in blob
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_notebook_estructura.py -v`

Expected: FAIL with `assert NOTEBOOK.is_file()` or FileNotFound.

- [ ] **Step 3: Write the notebook with nbformat**

Ejecutar este script desde la raíz del repo (crea el archivo; no lo dejes como “generador permanente” salvo que quieras). El contenido de las celdas debe ser el siguiente, en este orden.

**Celda Markdown — Contexto**

```markdown
# Comparación de modelos para aprobación de crédito

Parcial práctico — Machine Learning, Universidad Libre.
Dataset individual: `aprobacion_credito_Arias_Becerra_Johel_Santiago_3144.csv` (1.000 solicitudes).

Trabajo en riesgo: hay que recomendar, con evidencia, un clasificador para aprobar o rechazar crédito entre un **árbol de decisión**, una **regresión logística** y una **SVM**.

Reglas de este notebook (las de clase y las del diseño):

- Un solo split: `test_size=0.25`, `random_state=11`, `stratify=y`.
- Escalamos SVM y logística con `StandardScaler` **dentro** de un `Pipeline` (el scaler se ajusta solo con train). El árbol no se escala.
- Evaluamos solo en test: accuracy, precisión, sensibilidad (recall), especificidad y AUC.
- Criterio de recomendación, declarado **antes** de ver la tabla: mirar recall, especificidad y AUC. Un modelo domina si gana al menos 2 de esas 3 por ≥ 0.03 frente a cada rival. Si el desempeño está cercano, priorizamos interpretabilidad (árbol > logística > SVM). No elegimos por accuracy: el 73.3% de las solicitudes históricas fueron rechazadas.
```

**Celda Markdown — Carga y EDA**

```markdown
## Carga y EDA
```

**Celda código — funciones inline** (copiar el código de `aprobacion_credito.py` completo: `COLUMNAS`, `ruta_dataset`, `cargar_dataset`, `partir_train_test`, `metricas_prueba`, `construir_modelos`, con los mismos imports). No añadir `from aprobacion_credito import ...`.

**Celda código — carga**

```python
df = cargar_dataset()
print(df.shape)
print(df.dtypes)
print(df.isna().sum())
print(df["aprobado"].value_counts())
print("tasa de aprobación", round(df["aprobado"].mean(), 3))
```

**Celda código — EDA tablas**

```python
print(df.describe().T)
print(df.groupby("aprobado")[["ingreso_mensual_kcop", "score_buro", "antiguedad_laboral_anios", "num_creditos_activos", "dti"]].mean())
print(df.corr(numeric_only=True)["aprobado"].sort_values(ascending=False))
```

**Celda Markdown — lectura EDA**

```markdown
Hay **267 aprobados (26.7%)** y 733 rechazados. Sin nulos.

El score de buró es la señal más clara (media ~701 en aprobados vs ~628 en rechazados). El DTI también separa (más bajo entre aprobados). El ingreso mensual casi no cambia entre clases. Este notebook no fabrica variables nuevas: usamos las cinco columnas tal cual.
```

**Celda código — dos gráficos**

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
conteo = df["aprobado"].value_counts().sort_index()
axes[0].bar(["Rechazado (0)", "Aprobado (1)"], conteo.values)
axes[0].set_title("Desbalance de la decisión histórica")
axes[0].set_ylabel("Solicitudes")

df.boxplot(column="score_buro", by="aprobado", ax=axes[1])
axes[1].set_title("Score de buró por decisión histórica")
axes[1].set_xlabel("aprobado")
axes[1].set_ylabel("score_buro")
plt.suptitle("")
plt.tight_layout()
plt.show()
```

**Celda Markdown — Split**

```markdown
## Split

Mismo corte que en clase. `stratify=y` conserva el 26.7% de aprobados en train y test.
```

**Celda código — Split**

```python
Xtr, Xte, ytr, yte = partir_train_test(df)
print(Xtr.shape, Xte.shape)
print("aprobados train", int(ytr.sum()), "tasa", round(ytr.mean(), 4))
print("aprobados test", int(yte.sum()), "tasa", round(yte.mean(), 4))
```

**Celda Markdown — Entrenamiento**

```markdown
## Entrenamiento de los 3 modelos

- Árbol: `max_depth=4` para no memorizar train y poder explicar las reglas. No lleva scaler: los cortes no dependen de la unidad de medida.
- Logística y SVM: `Pipeline` con `StandardScaler`. La SVM es un modelo de distancia; la logística también se beneficia del escalado. `probability=True` en la SVM permite el AUC con `predict_proba` (en scikit-learn reciente puede salir un aviso de deprecación; lo dejamos porque es el contrato de este ejercicio).
```

**Celda código — fit**

```python
modelos = construir_modelos()
for nombre, modelo in modelos.items():
    modelo.fit(Xtr, ytr)
    print("entrenado:", nombre)
```

**Celda Markdown — Métricas**

```markdown
## Métricas en test

Clase positiva = 1 (aprobado). Especificidad = TN / (TN + FP): de los que el histórico rechazó, cuántos rechazamos.
```

**Celda código — tabla**

```python
import pandas as pd

filas = []
predicciones = {}
for nombre, modelo in modelos.items():
    pred = modelo.predict(Xte)
    score = modelo.predict_proba(Xte)[:, 1]
    predicciones[nombre] = pred
    m = metricas_prueba(yte, pred, score)
    filas.append({"modelo": nombre, **m})

tabla = pd.DataFrame(filas).set_index("modelo")
tabla_3 = tabla.round(3)
tabla_3
```

**Celda código — matrices**

```python
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (nombre, pred) in zip(axes, predicciones.items()):
    cm = confusion_matrix(yte, pred, labels=[0, 1])
    print(nombre, "TN FP FN TP", cm.ravel().tolist(), "suma", int(cm.sum()))
    ConfusionMatrixDisplay(cm, display_labels=["Rechazo (0)", "Aprobado (1)"]).plot(ax=ax, colorbar=False)
    ax.set_title(nombre)
plt.tight_layout()
plt.show()
```

Crear el `.ipynb` con `nbformat` (todas las celdas de arriba, `source` como listas de líneas). Kernel: `python3`.

- [ ] **Step 4: Run structure test**

Run: `python3 -m pytest tests/test_notebook_estructura.py -v`

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add notebooks/comparacion_modelos_aprobacion_credito.ipynb tests/test_notebook_estructura.py
git commit -m "feat: add self-contained comparison notebook through metrics table"
```

---

### Task 7: Recomendación, bitácora de IA y guion oral

**Files:**
- Modify: `notebooks/comparacion_modelos_aprobacion_credito.ipynb` (añadir 3 celdas Markdown al final)
- Modify: `tests/test_notebook_estructura.py` (exigir las frases clave)

**Interfaces:**
- Consumes: tabla real de Task 5 (números ya cerrados).
- Produces: secciones 6–8 de la spec, en prosa, sin `idxmax`.

Números de test (250 solicitudes; 67 aprobadas históricas):

| Modelo | Accuracy | Precisión | Recall | Especificidad | AUC | TN, FP, FN, TP |
|---|---|---|---|---|---|---|
| Árbol | 0.700 | 0.426 | 0.343 | 0.831 | 0.683 | 152, 31, 44, 23 |
| Regresión logística | 0.764 | 0.611 | 0.328 | 0.923 | 0.805 | 169, 14, 45, 22 |
| SVM | 0.768 | 0.645 | 0.299 | 0.940 | 0.749 | 172, 11, 47, 20 |

Aplicación del criterio (no es un `if` en código; va en el Markdown):

- Nadie gana 2 de 3 (recall, especificidad, AUC) por ≥ 0.03 **frente a cada** rival.
- El árbol **no** está cerca de los otros: -0.122 AUC y -0.092 especificidad vs logística. Interpretabilidad no lo salva.
- El par cercano es logística vs SVM: recall +0.029, especificidad −0.017, AUC +0.056. Entre esos dos, interpretabilidad elige logística.

**Modelo recomendado: regresión logística.**

- [ ] **Step 1: Extend the structure test (fails until Markdown exists)**

Añadir a `tests/test_notebook_estructura.py`:

```python
FRASES = [
    "Recomiendo la regresión logística",
    "interpretabilidad",
    "Cursor",
    "Guion de 3",
]


def test_notebook_tiene_recomendacion_ia_y_guion():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    blob = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    for frase in FRASES:
        assert frase in blob, frase
    assert "idxmax" not in blob
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_notebook_estructura.py::test_notebook_tiene_recomendacion_ia_y_guion -v`

Expected: FAIL on `Recomiendo la regresión logística` (aún no está).

- [ ] **Step 3: Append the three Markdown cells**

**Recomendación**

```markdown
## Recomendación para producción

**Recomiendo la regresión logística** para este proceso de aprobación.

El criterio se declaró antes de entrenar: equilibrar recall, especificidad y AUC; si el desempeño está cercano, preferir el modelo que el equipo de riesgo pueda explicar.

En las 250 solicitudes de prueba (67 que el histórico había aprobado):

- El **árbol** acierta menos en el ranking (AUC 0.683) y deja pasar más aprobaciones dudosas (31 falsos positivos; especificidad 0.831). Sí es el más fácil de dibujar, pero aquí no está “cerca”: pierde más de 0.03 en AUC y en especificidad frente a la logística. No lo elijo solo por ser interpretable.
- La **SVM** es la más estricta (especificidad 0.940, solo 11 falsos positivos) y su accuracy es 0.768. Su recall es el más bajo (0.299) y su AUC (0.749) queda por debajo de la logística. Además, con kernel RBF no podemos decirle al comité *por qué* se aprobó una solicitud.
- La **regresión logística** tiene el mejor AUC (0.805), especificidad alta (0.923; 14 falsos positivos) y un recall (0.328) muy parecido al del árbol. Frente a la SVM, recall y especificidad están a menos de 0.03; el AUC favorece a la logística por 0.056. En ese empate práctico gana la interpretabilidad: cada variable tiene un signo (el score empuja a aprobar, el DTI a rechazar).

No uso el accuracy (0.764) como argumento principal. Un modelo que rechace todo acertaría cerca del 73% y sería inútil.

**Límite:** el recall ~0.33 significa que, de cada tres solicitudes que el histórico habría aprobado, el modelo solo recupera una. Es un filtro conservador. No ajustamos `class_weight` ni el umbral 0.5 porque el enunciado pide comparar estos tres modelos en igualdad de condiciones, no redefinir la política de cupos.

En una frase: la logística es el punto medio que el área de riesgo puede defender — ordena mejor que la SVM, se explica mejor que la SVM, y no regala el desempeño que perderíamos si nos quedáramos con el árbol.
```

**Bitácora de IA**

```markdown
## Uso de inteligencia artificial

- **Cursor (agente Grok):** diseño del ejercicio, especificación, plan de implementación y estructura del notebook (secciones, split, pipelines, fórmulas de especificidad).
- **Cursor (agente Grok):** dudas sobre por qué la SVM necesita `StandardScaler` y el árbol no; diferencia de `probability=True` para el AUC.
- **Cursor (agente Grok):** redacción inicial de la justificación de negocio a partir de la tabla de test; yo conservo la decisión (logística) y debo poder explicarla en la oral.
- El código de carga, split, modelos y métricas sigue el enunciado y la spec; si al implementar se depuró un error de celda con el asistente, anotarlo aquí en una viñeta extra (herramienta + celda + para qué).
```

**Guion oral**

```markdown
## Guion de 3 a 5 minutos

1. **Problema (20 s).** 1.000 solicitudes; 26.7% aprobadas. Predigo la decisión histórica, no si el cliente va a pagar.
2. **Método (40 s).** Mismo split de clase (`random_state=11`, `stratify`). Escalo logística y SVM dentro del `Pipeline`; el árbol no, con `max_depth=4` para poder contarlo. Especificidad la calculo a mano con la matriz.
3. **Hallazgo (60 s).** En test, la logística tiene el mejor AUC (0.805). La SVM es un poco más estricta; el árbol se entiende más pero ordena peor.
4. **Decisión (90 s).** Recomiendo logística: desempeño cercano a la SVM y se explica con signos. Límite: recall bajo (~0.33), modelo conservador.
5. **IA (20 s).** Usé Cursor para el diseño y para ordenar la justificación. Las semillas, el split y las fórmulas las puedo explicar sin el chat.

Preguntas que debo poder responder: por qué 11, por qué `max_depth=4`, por qué `probability=True`, qué es un FP en crédito, por qué no me creo el accuracy.
```

- [ ] **Step 4: Run structure tests**

Run: `python3 -m pytest tests/test_notebook_estructura.py -v`

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add notebooks/comparacion_modelos_aprobacion_credito.ipynb tests/test_notebook_estructura.py
git commit -m "feat: add production recommendation, AI log, and oral script"
```

---

### Task 8: Ejecutar el notebook y lista de aceptación

**Files:**
- Modify: `notebooks/comparacion_modelos_aprobacion_credito.ipynb` (celdas ejecutadas / outputs si el entorno lo permite)
- Create: `tests/test_notebook_ejecuta.py`

**Interfaces:**
- Consumes: notebook de Tasks 6–7 y CSV en `data/`.
- Produces: notebook que corre de arriba abajo; checklist de la spec sección 8 en verde.

- [ ] **Step 1: Write the failing execution test**

```python
# tests/test_notebook_ejecuta.py
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def test_notebook_corre_de_arriba_abajo():
    path = Path("notebooks/comparacion_modelos_aprobacion_credito.ipynb")
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=180, kernel_name="python3")
    client.execute()
    for cell in nb.cells:
        if cell.cell_type == "code":
            for out in cell.get("outputs", []):
                if out.get("output_type") == "error":
                    raise AssertionError(f"{out.get('ename')}: {out.get('evalue')}")
```

- [ ] **Step 2: Run test to verify it fails or the kernel is missing**

Run: `python3 -m pip install jupyter nbformat nbclient ipykernel -q && python3 -m ipykernel install --user --name python3 && python3 -m pytest tests/test_notebook_ejecuta.py -v`

Expected: FAIL si el notebook está incompleto o el kernel no existe; después de instalar el kernel, debe poder ejecutarse. Si falla por ruta del CSV, `ruta_dataset()` ya cubre `data/` y `../data/` — ejecutar pytest desde la **raíz del repo**.

- [ ] **Step 3: Fix only execution issues (paths, missing imports in the first code cell)**

La primera celda de código del notebook debe incluir **todos** los imports que usan las funciones inline (`pandas`, `Path`, `train_test_split`, `StandardScaler`, `Pipeline`, `DecisionTreeClassifier`, `LogisticRegression`, `SVC`, métricas de sklearn). `matplotlib.pyplot` puede importarse en la celda de gráficos; `ConfusionMatrixDisplay` en la de matrices. No tragar excepciones con `try/except`.

- [ ] **Step 4: Run the full suite and manual checklist**

Run: `python3 -m pytest tests/ -v`

Expected: all passed.

Checklist manual (spec §8):

1. Notebook corre de arriba abajo (el test de esta task).
2. 750 / 250; 200 y 67 positivos.
3. 3 modelos; árbol sin scaler; logística y SVM con `Pipeline`.
4. Tabla 3×5, 3 decimales, sin NaN.
5. Tres matrices suman 250.
6. Markdown nombra logística, cita AUC y especificidad o recall, e interpretabilidad.
7. Bitácora: Cursor + para qué.

- [ ] **Step 5: Commit**

```bash
git add notebooks/comparacion_modelos_aprobacion_credito.ipynb tests/test_notebook_ejecuta.py
git commit -m "test: execute notebook top-to-bottom and lock acceptance"
```

---

## Self-review vs spec

| Spec | Task |
|---|---|
| CSV propio, 1000×6, 0 nulos | 1 |
| Split exacto de clase | 2, 6 |
| 3 modelos, scaler solo SVM y LR, árbol `max_depth=4` | 4, 6 |
| 5 métricas, una tabla, 3 decimales, especificidad a mano | 3, 5, 6 |
| Matrices + marco FP/FN | 6, 7 |
| Recomendación desempeño + interpretabilidad, español | 7 |
| Bitácora IA | 7 |
| Guion oral 3–5 min | 7 |
| Sin GridSearch / class_weight / módulos de producto en Moodle | 6 (notebook inline) |
| Lista de aceptación | 8 |
| EDA: desbalance + boxplot `score_buro` | 6 |

No quedan TBD de implementación: el ganador (logística) y los números de test están cerrados para este CSV.
