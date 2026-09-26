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
