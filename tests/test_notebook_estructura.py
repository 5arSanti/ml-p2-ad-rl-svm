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
