# Ampliación visual de la comparación de aprobación de crédito — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ampliar el notebook de aprobación de crédito con el árbol, las reglas, las importancias, los coeficientes, las barras de métricas, la ROC y los heatmaps, cada uno con un párrafo en español, sin cambiar los tres modelos ni la recomendación.

**Architecture:** El entregable sigue siendo un solo notebook autónomo. Las celdas nuevas se insertan dentro de EDA, Entrenamiento y Métricas. `aprobacion_credito.py` no se toca. Los párrafos citan los números ya calculados de este CSV (raíz `score_buro <= 663.50`, coeficientes redondeados a 3 decimales).

**Tech Stack:** Python 3, pandas, numpy, scikit-learn, matplotlib, nbformat, pytest. Sin seaborn.

## Global Constraints

- Spec vigente de modelos: `docs/superpowers/specs/2026-09-26-comparacion-modelos-aprobacion-credito-design.md`. Esta ampliación solo cambia la capa visual: `docs/superpowers/specs/2026-09-28-ampliacion-visual-comparacion-credito-design.md`.
- Split cableado: `train_test_split(X, y, test_size=0.25, random_state=11, stratify=y)`.
- Árbol: `DecisionTreeClassifier(max_depth=4, random_state=11)` sin scaler.
- Logística: `Pipeline` con `StandardScaler` y `LogisticRegression(max_iter=1000, random_state=11)`.
- SVM: `Pipeline` con `StandardScaler` y `SVC(kernel="rbf", probability=True, random_state=11)`.
- Umbral de `predict` = 0.5. Sin `GridSearchCV`, sin `class_weight`, sin SMOTE, sin cuarto modelo, sin `idxmax`.
- No modificar `aprobacion_credito.py` ni `tests/test_evaluacion_test.py`. No añadir `seaborn` ni otra dependencia.
- El notebook no contiene `import aprobacion_credito`.
- AUC de leyenda y tabla, a 3 decimales: árbol 0.683, regresión logística 0.805, SVM 0.749.
- Matrices `labels=[0, 1]`, cada una suma 250. Conteos TN, FP, FN, TP: árbol 152, 31, 44, 23; logística 169, 14, 45, 22; SVM 172, 11, 47, 20.
- Coeficientes del `clf` de la logística, redondeados a 3 decimales, ordenados por magnitud: `score_buro` +1.127 (odds 3.088), `dti` −0.957 (odds 0.384), `num_creditos_activos` −0.491 (odds 0.612), `antiguedad_laboral_anios` +0.359 (odds 1.432), `ingreso_mensual_kcop` +0.008 (odds 1.008).
- Importancias del árbol a 3 decimales: `score_buro` 0.447, `dti` 0.385, `ingreso_mensual_kcop` 0.086, `antiguedad_laboral_anios` 0.050, `num_creditos_activos` 0.032. Profundidad 4, 16 hojas. Primer corte: `score_buro <= 663.50`.
- Frase exacta que debe seguir existiendo: `Recomiendo la regresión logística`.
- Pytest se corre desde la raíz del repo `ml-p2-ad-rl-svm`.

## File structure

| Path | Responsibility |
|---|---|
| `notebooks/comparacion_modelos_aprobacion_credito.ipynb` | Entregable. Se insertan celdas; no se cambian los modelos. |
| `tests/test_notebook_estructura.py` | Exige el orden de figuras y párrafos, y las cadenas de la spec. |
| `tests/test_notebook_ejecuta.py` | Ya existe. La última tarea lo corre de arriba abajo. |
| `aprobacion_credito.py` | No se modifica. |
| `tests/test_evaluacion_test.py` | No se modifica. |

---

### Task 1: Contexto y lectura de EDA debajo de la figura

**Files:**
- Modify: `tests/test_notebook_estructura.py`
- Modify: `notebooks/comparacion_modelos_aprobacion_credito.ipynb`

**Interfaces:**
- Consumes: celda Markdown de contexto (empieza por `# Comparación de modelos`) y celda de código que contiene `df.boxplot`.
- Produces: la celda de contexto contiene la frase del dibujo del árbol; la celda inmediatamente posterior al boxplot contiene `267 aprobados`.

- [ ] **Step 1: Write the failing test**

Añadir al final de `tests/test_notebook_estructura.py`:

```python
def _sources():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c.get("source", [])) for c in nb["cells"]]


def test_contexto_anuncia_la_lectura_de_modelos():
    srcs = _sources()
    assert "dibujo del árbol" in srcs[0]
    assert "curva ROC" in srcs[0]


def test_lectura_eda_va_debajo_de_la_figura():
    srcs = _sources()
    idx_fig = next(i for i, s in enumerate(srcs) if "df.boxplot" in s)
    assert "267 aprobados" in srcs[idx_fig + 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_notebook_estructura.py::test_contexto_anuncia_la_lectura_de_modelos tests/test_notebook_estructura.py::test_lectura_eda_va_debajo_de_la_figura -v`

Expected: FAIL. `dibujo del árbol` no está en la celda 0, y `267 aprobados` está antes del boxplot.

- [ ] **Step 3: Write minimal implementation**

Ejecutar desde la raíz del repo:

```python
import nbformat

path = "notebooks/comparacion_modelos_aprobacion_credito.ipynb"
nb = nbformat.read(path, as_version=4)

frase = (
    "Además de la tabla, el notebook muestra cómo se lee cada modelo: "
    "el dibujo del árbol, los coeficientes de la logística y la curva ROC.\n\n"
)
contexto = nb.cells[0].source
if isinstance(contexto, list):
    contexto = "".join(contexto)
ancla = "Reglas de este notebook"
if frase.strip() not in contexto:
    nb.cells[0].source = contexto.replace(ancla, frase + ancla, 1)

srcs = []
for cell in nb.cells:
    texto = cell.source
    if isinstance(texto, list):
        texto = "".join(texto)
    srcs.append(texto)

idx_lectura = next(i for i, s in enumerate(srcs) if "267 aprobados" in s)
idx_fig = next(i for i, s in enumerate(srcs) if "df.boxplot" in s)
if idx_lectura != idx_fig + 1:
    lectura = nb.cells.pop(idx_lectura)
    if idx_lectura < idx_fig:
        idx_fig -= 1
    nb.cells.insert(idx_fig + 1, lectura)

nbformat.write(nb, path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_notebook_estructura.py -v`

Expected: los dos tests nuevos pasan y los dos tests viejos siguen pasando.

- [ ] **Step 5: Commit**

```bash
git add tests/test_notebook_estructura.py notebooks/comparacion_modelos_aprobacion_credito.ipynb
git commit -m "$(cat <<'EOF'
docs: place the EDA reading under the credit figure

The opening cell now points at the tree, the coefficients, and the ROC.
EOF
)"
```

---

### Task 2: Árbol, reglas e importancias

**Files:**
- Modify: `tests/test_notebook_estructura.py`
- Modify: `notebooks/comparacion_modelos_aprobacion_credito.ipynb`

**Interfaces:**
- Consumes: `_sources()` definido en Task 1; la celda que contiene `print("entrenado:", nombre)` y deja `modelos` y `Xtr` en memoria. El árbol es `modelos["Árbol"]`.
- Produces: la celda de `plot_tree` asigna `arbol = modelos["Árbol"]`. La celda de `export_text` usa ese nombre `arbol`. La celda de importancias lee `modelos["Árbol"].feature_importances_`.

- [ ] **Step 1: Write the failing test**

Añadir al final de `tests/test_notebook_estructura.py`:

```python
def test_arbol_reglas_e_importancias_tienen_parrafo_debajo():
    srcs = _sources()
    idx_fit = next(i for i, s in enumerate(srcs) if 'print("entrenado:"' in s)
    idx_tree = next(i for i, s in enumerate(srcs) if "plot_tree" in s)
    idx_rules = next(i for i, s in enumerate(srcs) if "export_text" in s)
    idx_imp = next(i for i, s in enumerate(srcs) if "feature_importances_" in s)
    assert idx_fit < idx_tree < idx_rules < idx_imp
    assert "primer corte" in srcs[idx_tree + 1]
    assert "663.50" in srcs[idx_tree + 1]
    assert "663.50" in srcs[idx_rules + 1]
    assert "no están en la misma escala" in srcs[idx_imp + 1]
    assert "class_names=[\"Rechazado\", \"Aprobado\"]" in srcs[idx_tree]
    assert "fontsize=8" in srcs[idx_tree]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_notebook_estructura.py::test_arbol_reglas_e_importancias_tienen_parrafo_debajo -v`

Expected: FAIL con `StopIteration` porque `plot_tree` no está en el notebook.

- [ ] **Step 3: Write minimal implementation**

Ejecutar desde la raíz del repo:

```python
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

path = "notebooks/comparacion_modelos_aprobacion_credito.ipynb"
nb = nbformat.read(path, as_version=4)

def texto(cell):
    src = cell.source
    return "".join(src) if isinstance(src, list) else src

idx = next(i for i, c in enumerate(nb.cells) if 'print("entrenado:"' in texto(c))

celdas = [
    new_code_cell(
        "import matplotlib.pyplot as plt\n"
        "from sklearn.tree import plot_tree\n"
        "\n"
        "arbol = modelos[\"Árbol\"]\n"
        "print(\"profundidad\", arbol.get_depth())\n"
        "print(\"hojas\", arbol.get_n_leaves())\n"
        "plt.figure(figsize=(18, 10))\n"
        "plot_tree(\n"
        "    arbol,\n"
        "    feature_names=list(Xtr.columns),\n"
        "    class_names=[\"Rechazado\", \"Aprobado\"],\n"
        "    filled=True,\n"
        "    rounded=True,\n"
        "    fontsize=8,\n"
        ")\n"
        "plt.title(\"Árbol de decisión (max_depth=4)\")\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    ),
    new_markdown_cell(
        "El primer corte es el score de buró, en 663.50. Por encima de ese score el árbol mira el DTI; por debajo también. "
        "La celda imprime profundidad 4 y 16 hojas: `max_depth=4` es el tope elegido para poder contar estas reglas en la sustentación.\n"
    ),
    new_code_cell(
        "from sklearn.tree import export_text\n"
        "\n"
        "print(export_text(arbol, feature_names=list(Xtr.columns), decimals=2))\n"
    ),
    new_markdown_cell(
        "Dos caminos concretos. Si el score pasa de 663.50 y el DTI no pasa de 0.19, la hoja es aprobado. "
        "Si el score no pasa de 663.50, el DTI pasa de 0.25 y la antigüedad no pasa de 24.10 años, la hoja es rechazado. "
        "El ingreso solo aparece en ramas más profundas.\n"
    ),
    new_code_cell(
        "import matplotlib.pyplot as plt\n"
        "import pandas as pd\n"
        "\n"
        "importancias = pd.DataFrame({\n"
        "    \"Variable\": list(Xtr.columns),\n"
        "    \"Importancia\": modelos[\"Árbol\"].feature_importances_,\n"
        "}).sort_values(\"Importancia\", ascending=False).reset_index(drop=True)\n"
        "print(importancias.round(3))\n"
        "\n"
        "plt.figure(figsize=(8, 4))\n"
        "plt.barh(importancias[\"Variable\"], importancias[\"Importancia\"])\n"
        "plt.xlabel(\"Importancia\")\n"
        "plt.ylabel(\"Variable\")\n"
        "plt.title(\"Importancia de variables en el árbol\")\n"
        "plt.gca().invert_yaxis()\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    ),
    new_markdown_cell(
        "El score de buró concentra 0.447 de la importancia y el DTI 0.385. El ingreso queda en 0.086. "
        "Estas importancias dicen qué variable parte más nodos en este árbol. "
        "No se comparan en número con los coeficientes de la logística: no están en la misma escala.\n"
    ),
]

for desplazamiento, celda in enumerate(celdas, start=1):
    nb.cells.insert(idx + desplazamiento, celda)

nbformat.write(nb, path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_notebook_estructura.py -v`

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_notebook_estructura.py notebooks/comparacion_modelos_aprobacion_credito.ipynb
git commit -m "$(cat <<'EOF'
feat: show the credit tree, its rules, and feature importances

The depth-4 tree splits first on bureau score, and the reading stays in Spanish under each figure.
EOF
)"
```

---

### Task 3: Coeficientes de la logística y nota de la SVM

**Files:**
- Modify: `tests/test_notebook_estructura.py`
- Modify: `notebooks/comparacion_modelos_aprobacion_credito.ipynb`

**Interfaces:**
- Consumes: `modelos["Regresión logística"]`, un `Pipeline` con pasos `scaler` y `clf`, y `Xtr.columns`.
- Produces: un `DataFrame` llamado `coeficientes` con las columnas `Variable`, `Coeficiente (escala estandarizada)`, `Odds Ratio` y `Magnitud del efecto`. No lo usa ninguna celda posterior; la recomendación cita los mismos números en prosa.

- [ ] **Step 1: Write the failing test**

Añadir al final de `tests/test_notebook_estructura.py`:

```python
def test_coeficientes_y_nota_svm_van_despues_de_las_importancias():
    srcs = _sources()
    idx_imp = next(i for i, s in enumerate(srcs) if "feature_importances_" in s)
    idx_coef = next(i for i, s in enumerate(srcs) if "Odds Ratio" in s)
    assert idx_coef > idx_imp
    assert "named_steps[\"clf\"]" in srcs[idx_coef]
    assert "3.088" in srcs[idx_coef + 1]
    assert "−0.957" in srcs[idx_coef + 1]
    assert "kernel RBF" in srcs[idx_coef + 2]
    assert "GridSearch" not in "\n".join(srcs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_notebook_estructura.py::test_coeficientes_y_nota_svm_van_despues_de_las_importancias -v`

Expected: FAIL con `StopIteration` porque `Odds Ratio` no está en el notebook.

- [ ] **Step 3: Write minimal implementation**

Ejecutar desde la raíz del repo. Inserta justo después del párrafo de importancias (la celda que contiene `no están en la misma escala`):

```python
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

path = "notebooks/comparacion_modelos_aprobacion_credito.ipynb"
nb = nbformat.read(path, as_version=4)

def texto(cell):
    src = cell.source
    return "".join(src) if isinstance(src, list) else src

idx = next(i for i, c in enumerate(nb.cells) if "no están en la misma escala" in texto(c))

celdas = [
    new_code_cell(
        "import numpy as np\n"
        "import pandas as pd\n"
        "\n"
        "clf = modelos[\"Regresión logística\"].named_steps[\"clf\"]\n"
        "coeficientes = pd.DataFrame({\n"
        "    \"Variable\": list(Xtr.columns),\n"
        "    \"Coeficiente (escala estandarizada)\": clf.coef_[0],\n"
        "})\n"
        "coeficientes[\"Odds Ratio\"] = np.exp(coeficientes[\"Coeficiente (escala estandarizada)\"])\n"
        "coeficientes[\"Magnitud del efecto\"] = coeficientes[\"Coeficiente (escala estandarizada)\"].abs()\n"
        "coeficientes = coeficientes.sort_values(\"Magnitud del efecto\", ascending=False).reset_index(drop=True)\n"
        "coeficientes.round(3)\n"
    ),
    new_markdown_cell(
        "Las dos variables de mayor magnitud, ya en escala estandarizada, son el score de buró "
        "(+1.127, odds ratio 3.088) y el DTI (−0.957, odds ratio 0.384). "
        "Un score más alto empuja a aprobar; un DTI más alto empuja a rechazar. "
        "Eso coincide con las medias del EDA. El ingreso casi no pesa (+0.008).\n"
    ),
    new_markdown_cell(
        "La SVM de este notebook usa kernel RBF. En ese espacio no hay un coeficiente por variable "
        "que el comité pueda leer, como sí hay en la logística. Por eso la SVM entra a la comparación "
        "por accuracy, precisión, recall, especificidad y AUC. La lectura de por qué se aprobó una "
        "solicitud queda en el árbol y en la logística.\n"
    ),
]

for desplazamiento, celda in enumerate(celdas, start=1):
    nb.cells.insert(idx + desplazamiento, celda)

nbformat.write(nb, path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_notebook_estructura.py -v`

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_notebook_estructura.py notebooks/comparacion_modelos_aprobacion_credito.ipynb
git commit -m "$(cat <<'EOF'
feat: show logistic coefficients and why the RBF SVM stays opaque

Score and DTI keep the signs the risk team can defend, on the standardized scale.
EOF
)"
```

---

### Task 4: Barras, ROC y heatmaps

**Files:**
- Modify: `tests/test_notebook_estructura.py`
- Modify: `notebooks/comparacion_modelos_aprobacion_credito.ipynb`

**Interfaces:**
- Consumes: `tabla_3` (índice `Árbol`, `Regresión logística`, `SVM`; columnas `accuracy`, `precision`, `recall`, `especificidad`, `auc`), `modelos`, `predicciones`, `Xte`, `yte`.
- Produces: la celda de matrices usa `imshow` y ya no usa `ConfusionMatrixDisplay`. Sigue imprimiendo `TN FP FN TP` y la suma.

- [ ] **Step 1: Write the failing test**

Añadir al final de `tests/test_notebook_estructura.py`:

```python
def test_barras_roc_y_heatmaps_tienen_parrafo_debajo():
    srcs = _sources()
    blob = "\n".join(srcs)
    idx_tabla = next(i for i, s in enumerate(srcs) if "tabla_3 = tabla.round(3)" in s)
    idx_barras = next(i for i, s in enumerate(srcs) if "Métricas en el conjunto de prueba" in s)
    idx_roc = next(i for i, s in enumerate(srcs) if "roc_curve" in s)
    idx_heat = next(i for i, s in enumerate(srcs) if "imshow" in s)
    assert idx_tabla < idx_barras < idx_roc < idx_heat
    assert "73.3%" in srcs[idx_barras + 1]
    assert "0.805" in srcs[idx_roc + 1]
    assert "0.749" in srcs[idx_roc + 1]
    assert "0.683" in srcs[idx_roc + 1]
    assert "FP es aprobar" in srcs[idx_heat + 1]
    assert "ConfusionMatrixDisplay" not in blob
    assert "labels=[0, 1]" in srcs[idx_heat]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_notebook_estructura.py::test_barras_roc_y_heatmaps_tienen_parrafo_debajo -v`

Expected: FAIL. `roc_curve` no está, y `ConfusionMatrixDisplay` sí está.

- [ ] **Step 3: Write minimal implementation**

Ejecutar desde la raíz del repo. Inserta barras y ROC después de la celda de `tabla_3`. Reemplaza la celda que contiene `ConfusionMatrixDisplay` por el heatmap y añade el párrafo de FP/FN en la celda siguiente.

```python
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

path = "notebooks/comparacion_modelos_aprobacion_credito.ipynb"
nb = nbformat.read(path, as_version=4)

def texto(cell):
    src = cell.source
    return "".join(src) if isinstance(src, list) else src

idx_tabla = next(i for i, c in enumerate(nb.cells) if "tabla_3 = tabla.round(3)" in texto(c))
barras_y_roc = [
    new_code_cell(
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "\n"
        "metricas = [\"accuracy\", \"precision\", \"recall\", \"especificidad\", \"auc\"]\n"
        "etiquetas = [\"Accuracy\", \"Precisión\", \"Recall\", \"Especificidad\", \"AUC\"]\n"
        "nombres = list(tabla_3.index)\n"
        "x = np.arange(len(metricas))\n"
        "ancho = 0.25\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(10, 4))\n"
        "for i, nombre in enumerate(nombres):\n"
        "    valores = [tabla_3.loc[nombre, m] for m in metricas]\n"
        "    ax.bar(x + (i - 1) * ancho, valores, width=ancho, label=nombre)\n"
        "ax.set_xticks(x)\n"
        "ax.set_xticklabels(etiquetas)\n"
        "ax.set_ylim(0, 1)\n"
        "ax.set_ylabel(\"Valor en test\")\n"
        "ax.set_title(\"Métricas en el conjunto de prueba\")\n"
        "ax.legend()\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    ),
    new_markdown_cell(
        "El accuracy de los tres queda entre 0.700 y 0.768, alto en apariencia porque el 73.3% "
        "de las solicitudes históricas fueron rechazadas. El recall queda bajo en los tres "
        "(0.343, 0.328 y 0.299). Esta figura no elige sola al modelo.\n"
    ),
    new_code_cell(
        "import matplotlib.pyplot as plt\n"
        "from sklearn.metrics import roc_curve\n"
        "\n"
        "plt.figure(figsize=(6, 5))\n"
        "for nombre, modelo in modelos.items():\n"
        "    score = modelo.predict_proba(Xte)[:, 1]\n"
        "    fpr, tpr, _ = roc_curve(yte, score)\n"
        "    auc = tabla_3.loc[nombre, \"auc\"]\n"
        "    plt.plot(fpr, tpr, label=f\"{nombre} (AUC = {auc:.3f})\")\n"
        "plt.plot([0, 1], [0, 1], linestyle=\"--\", color=\"gray\", label=\"Clasificador aleatorio\")\n"
        "plt.xlabel(\"Tasa de falsos positivos (1 - especificidad)\")\n"
        "plt.ylabel(\"Tasa de verdaderos positivos (recall)\")\n"
        "plt.title(\"Curva ROC — conjunto de prueba\")\n"
        "plt.legend()\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    ),
    new_markdown_cell(
        "La curva de la logística queda por encima (AUC 0.805), luego la SVM (0.749) y después el árbol (0.683). "
        "La diagonal es un clasificador que no usa las variables. Esta figura ordena; el corte de `predict` sigue en 0.5.\n"
    ),
]
for desplazamiento, celda in enumerate(barras_y_roc, start=1):
    nb.cells.insert(idx_tabla + desplazamiento, celda)

idx_cm = next(i for i, c in enumerate(nb.cells) if "ConfusionMatrixDisplay" in texto(c))
nb.cells[idx_cm] = new_code_cell(
    "import matplotlib.pyplot as plt\n"
    "from sklearn.metrics import confusion_matrix\n"
    "\n"
    "fig, axes = plt.subplots(1, 3, figsize=(12, 4))\n"
    "for ax, (nombre, pred) in zip(axes, predicciones.items()):\n"
    "    cm = confusion_matrix(yte, pred, labels=[0, 1])\n"
    "    print(nombre, \"TN FP FN TP\", cm.ravel().tolist(), \"suma\", int(cm.sum()))\n"
    "    ax.imshow(cm, cmap=\"Blues\")\n"
    "    ax.set_xticks([0, 1])\n"
    "    ax.set_yticks([0, 1])\n"
    "    ax.set_xticklabels([\"Rechazo (0)\", \"Aprobado (1)\"])\n"
    "    ax.set_yticklabels([\"Rechazo (0)\", \"Aprobado (1)\"])\n"
    "    ax.set_xlabel(\"Predicción\")\n"
    "    ax.set_ylabel(\"Valor real\")\n"
    "    ax.set_title(nombre)\n"
    "    for fila in range(2):\n"
    "        for col in range(2):\n"
    "            ax.text(col, fila, str(int(cm[fila, col])), ha=\"center\", va=\"center\", color=\"black\")\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
)
nb.cells.insert(
    idx_cm + 1,
    new_markdown_cell(
        "FP es aprobar a quien el histórico rechazó. FN es rechazar a quien el histórico aprobó. "
        "En las 250 solicitudes de prueba la logística tiene 14 FP y 45 FN; la SVM, 11 FP y 47 FN; "
        "el árbol, 31 FP y 44 FN. Cada matriz suma 250.\n"
    ),
)

nbformat.write(nb, path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_notebook_estructura.py -v`

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_notebook_estructura.py notebooks/comparacion_modelos_aprobacion_credito.ipynb
git commit -m "$(cat <<'EOF'
feat: compare the three credit models with bars, ROC, and heatmaps

The confusion matrices stay on the test set and now show the counts in each cell.
EOF
)"
```

---

### Task 5: Recomendación, bitácora y guion

**Files:**
- Modify: `tests/test_notebook_estructura.py`
- Modify: `notebooks/comparacion_modelos_aprobacion_credito.ipynb`

**Interfaces:**
- Consumes: los números ya escritos en las celdas de coeficientes y de ROC. No lee `tabla_3` en código nuevo.
- Produces: la celda de recomendación conserva `Recomiendo la regresión logística` y añade el AUC de la ROC y el coeficiente +1.127. La bitácora gana una viñeta de Cursor sobre las figuras. El guion conserva `Guion de 3`.

- [ ] **Step 1: Write the failing test**

Añadir al final de `tests/test_notebook_estructura.py`:

```python
def test_recomendacion_cita_roc_y_coeficiente():
    srcs = _sources()
    idx = next(i for i, s in enumerate(srcs) if "Recomiendo la regresión logística" in s)
    texto = srcs[idx]
    assert "AUC 0.805" in texto
    assert "+1.127" in texto
    assert "interpretabilidad" in texto
    assert "idxmax" not in texto


def test_bitacora_y_guion_mencionan_las_figuras():
    srcs = _sources()
    idx_ia = next(i for i, s in enumerate(srcs) if "## Uso de inteligencia artificial" in s)
    idx_guion = next(i for i, s in enumerate(srcs) if "Guion de 3" in s)
    assert "odds ratios" in srcs[idx_ia]
    assert "Cursor" in srcs[idx_ia]
    assert "ROC" in srcs[idx_guion]
    assert "+1.127" in srcs[idx_guion]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_notebook_estructura.py::test_recomendacion_cita_roc_y_coeficiente tests/test_notebook_estructura.py::test_bitacora_y_guion_mencionan_las_figuras -v`

Expected: FAIL. La recomendación aún no contiene `+1.127`. La bitácora aún no contiene `odds ratios`.

- [ ] **Step 3: Write minimal implementation**

Ejecutar desde la raíz del repo. Reemplaza el source de las tres celdas Markdown finales. No borres la frase `Recomiendo la regresión logística` ni el encabezado `Guion de 3`.

```python
import nbformat

path = "notebooks/comparacion_modelos_aprobacion_credito.ipynb"
nb = nbformat.read(path, as_version=4)

def texto(cell):
    src = cell.source
    return "".join(src) if isinstance(src, list) else src

nb.cells[next(i for i, c in enumerate(nb.cells) if "Recomiendo la regresión logística" in texto(c))].source = """## Recomendación para producción

**Recomiendo la regresión logística** para este proceso de aprobación.

El criterio se declaró antes de entrenar: equilibrar recall, especificidad y AUC; si el desempeño está cercano, preferir el modelo que el equipo de riesgo pueda explicar.

En las 250 solicitudes de prueba (67 que el histórico había aprobado):

- El **árbol** acierta menos en el ranking (AUC 0.683) y deja pasar más aprobaciones dudosas (31 falsos positivos; especificidad 0.831). Sí es el más fácil de dibujar, pero aquí no está cerca: pierde más de 0.03 en AUC y en especificidad frente a la logística. No lo elijo solo por ser interpretable.
- La **SVM** es la más estricta (especificidad 0.940, solo 11 falsos positivos) y su accuracy es 0.768. Su recall es el más bajo (0.299) y su AUC (0.749) queda por debajo de la logística. Además, con kernel RBF no podemos decirle al comité por qué se aprobó una solicitud.
- La **regresión logística** tiene el mejor AUC (0.805), especificidad alta (0.923; 14 falsos positivos) y un recall (0.328) muy parecido al del árbol. Frente a la SVM, recall y especificidad están a menos de 0.03; el AUC favorece a la logística por 0.056. En ese empate práctico gana la interpretabilidad.

En la curva ROC la logística queda por encima (AUC 0.805 frente a 0.749 de la SVM y 0.683 del árbol). El coeficiente estandarizado del score de buró es +1.127 (odds ratio 3.088): sube las chances de aprobación. El DTI va al revés (−0.957).

No uso el accuracy (0.764) como argumento principal. Un modelo que rechace todo acertaría cerca del 73% y sería inútil.

**Límite:** el recall ~0.33 significa que, de cada tres solicitudes que el histórico habría aprobado, el modelo solo recupera una. Es un filtro conservador. No ajustamos `class_weight` ni el umbral 0.5 porque el enunciado pide comparar estos tres modelos en igualdad de condiciones, no redefinir la política de cupos.

En una frase: la logística es el punto medio que el área de riesgo puede defender. Ordena mejor que la SVM, se explica con signos, y no regala el desempeño que perderíamos con el árbol.
"""

nb.cells[next(i for i, c in enumerate(nb.cells) if "## Uso de inteligencia artificial" in texto(c))].source = """## Uso de inteligencia artificial

- **Cursor (agente Grok):** diseño del ejercicio, especificación, plan de implementación y estructura del notebook (secciones, split, pipelines, fórmulas de especificidad).
- **Cursor (agente Grok):** dudas sobre por qué la SVM necesita `StandardScaler` y el árbol no; diferencia de `probability=True` para el AUC.
- **Cursor (agente Grok):** redacción inicial de la justificación de negocio a partir de la tabla de test; yo conservo la decisión (logística) y debo poder explicarla en la oral.
- **Cursor (agente Grok):** figuras nuevas (dibujo del árbol, reglas, importancias, coeficientes con odds ratios, barras de métricas, curva ROC y heatmaps) y los párrafos de lectura. La decisión de recomendar la logística sigue siendo mía.
- El código de carga, split, modelos y métricas sigue el enunciado y la spec; si al implementar se depuró un error de celda con el asistente, anotarlo aquí en una viñeta extra (herramienta + celda + para qué).
"""

nb.cells[next(i for i, c in enumerate(nb.cells) if "Guion de 3" in texto(c))].source = """## Guion de 3 a 5 minutos

1. **Problema (20 s).** 1.000 solicitudes; 26.7% aprobadas. Predigo la decisión histórica, no si el cliente va a pagar.
2. **Método (40 s).** Mismo split de clase (`random_state=11`, `stratify`). Escalo logística y SVM dentro del `Pipeline`; el árbol no, con `max_depth=4` para poder contarlo. Especificidad la calculo a mano con la matriz.
3. **Hallazgo (60 s).** En la ROC la logística queda por encima (AUC 0.805). La SVM es un poco más estricta; el árbol se dibuja, pero ordena peor.
4. **Decisión (90 s).** Recomiendo logística: desempeño cercano a la SVM y se explica con signos. El score de buró tiene coeficiente +1.127, a favor de aprobar. Límite: recall bajo (~0.33), modelo conservador.
5. **IA (20 s).** Usé Cursor para el diseño, las figuras y para ordenar la justificación. Las semillas, el split y las fórmulas las puedo explicar sin el chat.

Preguntas que debo poder responder: por qué 11, por qué `max_depth=4`, por qué `probability=True`, qué es un FP en crédito, por qué no me creo el accuracy, qué signo tiene el score en la logística.
"""

nbformat.write(nb, path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_notebook_estructura.py tests/test_evaluacion_test.py -v`

Expected: all passed. `test_evaluacion_test.py` no se editó.

- [ ] **Step 5: Commit**

```bash
git add tests/test_notebook_estructura.py notebooks/comparacion_modelos_aprobacion_credito.ipynb
git commit -m "$(cat <<'EOF'
docs: tie the credit recommendation to the ROC and the score coefficient

The oral script still fits in a few minutes and points at those two pieces of evidence.
EOF
)"
```

---

### Task 6: Ejecutar el notebook de arriba abajo

**Files:**
- Modify: `notebooks/comparacion_modelos_aprobacion_credito.ipynb` solo si una celda falla al ejecutar.
- Test: `tests/test_notebook_ejecuta.py` (ya existe; no hace falta otro test).

**Interfaces:**
- Consumes: el notebook de las tareas 1 a 5 y el CSV en `data/`.
- Produces: el notebook corre sin celdas en error. Las métricas de `tests/test_evaluacion_test.py` siguen iguales.

- [ ] **Step 1: Run the full suite**

Run: `python3 -m pytest tests/ -v`

Expected: all passed, incluido `test_notebook_corre_de_arriba_abajo`. Puede aparecer `FutureWarning` de `SVC(probability=True)`. No cambiar `probability=True` ni envolver el modelo en `CalibratedClassifierCV`.

Si falla una celda de código, corregir solo ese error (import que falta o nombre mal escrito). No cambiar hiperparámetros, split ni textos de métricas ya fijados.

- [ ] **Step 2: Manual checklist**

1. La celda siguiente al boxplot habla de 267 aprobados.
2. Debajo del árbol, de las reglas, de las importancias, de los coeficientes, de las barras, de la ROC y de los heatmaps hay un párrafo.
3. Cada matriz impresa suma 250.
4. La recomendación nombra la logística, cita AUC 0.805 y el coeficiente +1.127.
5. La bitácora tiene la viñeta de las figuras con Cursor y el para qué.

- [ ] **Step 3: Commit only if step 1 required a fix**

Si no hubo cambios de archivo, no crear un commit vacío.

```bash
git add notebooks/comparacion_modelos_aprobacion_credito.ipynb
git commit -m "$(cat <<'EOF'
fix: make the expanded credit notebook run from top to bottom
EOF
)"
```

---

## Spec coverage

| Spec | Task |
|---|---|
| Frase de contexto sobre árbol, coeficientes y ROC | 1 |
| Párrafo de EDA debajo de la figura, sin duplicarlo | 1 |
| `plot_tree`, profundidad 4, 16 hojas, primer corte 663.50 | 2 |
| `export_text` y dos caminos en prosa | 2 |
| Importancias y aviso de escala distinta | 2 |
| Coeficientes, odds ratios, signos de score y DTI | 3 |
| Párrafo de SVM RBF sin coeficientes | 3 |
| Barras desde `tabla_3`, eje 0 a 1 | 4 |
| ROC con AUC 0.683, 0.805 y 0.749 | 4 |
| Heatmap `imshow` que reemplaza `ConfusionMatrixDisplay` | 4 |
| Recomendación logística citando ROC y +1.127 | 5 |
| Bitácora con herramienta y para qué | 5 |
| Guion de 3 a 5 minutos | 5 |
| Sin GridSearch, sin seaborn, sin tocar `aprobacion_credito.py` | 3 y Global Constraints |
| Notebook ejecuta de arriba abajo | 6 |
