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
