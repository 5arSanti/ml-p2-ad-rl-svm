import pytest
from sklearn.metrics import confusion_matrix

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
        cm = confusion_matrix(yte, pred, labels=[0, 1])
        assert int(cm.sum()) == 250
