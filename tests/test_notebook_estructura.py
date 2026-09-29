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
