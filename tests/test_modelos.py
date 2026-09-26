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
