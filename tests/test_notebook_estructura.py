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
