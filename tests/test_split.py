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
