# Diseño: ampliación visual de la comparación de aprobación de crédito

Fecha: 2026-09-28  
Curso: Universidad Libre · Machine Learning · Parcial, componente práctico (70/100)  
Estudiante / dataset: Johel Santiago Arias Becerra · `aprobacion_credito_Arias_Becerra_Johel_Santiago_3144.csv`  
Entregable calificado: el mismo notebook `notebooks/comparacion_modelos_aprobacion_credito.ipynb`  
Enfoque aprobado: artefactos de los talleres de clase, insertados en las secciones que ya existen (enfoque 1)

Esta spec **enmienda** `docs/superpowers/specs/2026-09-26-comparacion-modelos-aprobacion-credito-design.md` solo en la capa visual y narrativa. El contrato de modelos, split, métricas y recomendación de esa spec sigue vigente. Donde aquella spec dice «exactamente dos gráficos», esta spec sustituye ese límite por la figura de EDA que ya existe más las figuras de las secciones 5.2, 5.4, 5.7, 5.8 y 5.9.

## 1. Problema

El notebook actual cumple la rúbrica y recomienda la regresión logística, pero se lee corto: dos gráficos de exploración, una tabla y tres matrices. Los talleres de árboles y de SVM muestran el modelo (árbol, reglas, importancias, coeficientes, ROC, barras de métricas) y un párrafo debajo de cada figura. Esta ampliación lleva esa capa al notebook de crédito sin cambiar la comparación oficial.

## 2. Criterio de éxito

La rúbrica de 70 puntos se mantiene en «Excelente». Además, quien abra el notebook ve, en las secciones ya existentes, cada figura y cada tabla nueva de la sección 5 seguidas de un párrafo en español.

La recomendación sigue siendo la regresión logística. El guion oral sigue cabiendo en 3 a 5 minutos: señala la ROC y el signo de un coeficiente; el resto queda como respaldo si el docente pregunta.

## 3. Fuera de alcance

- `GridSearchCV`, `RandomizedSearchCV` u otra búsqueda de hiperparámetros.
- `class_weight`, SMOTE u otro remuestreo.
- Cambiar `max_depth`, el kernel, `probability=True`, el split o el umbral 0.5.
- Un cuarto modelo, ensambles, o elegir el ganador con código (`idxmax` u otro).
- Modificar `aprobacion_credito.py` o `tests/test_evaluacion_test.py`.
- Añadir `seaborn` o cualquier dependencia nueva. Los gráficos usan matplotlib y scikit-learn.
- Gráficos de EDA adicionales (correlación, histogramas de las cinco variables). La figura de EDA sigue siendo la actual.
- Reordenar el notebook al estilo «una idea por celda de punta a punta» ni mover las figuras a un anexo final.

## 4. Arquitectura

Un solo notebook lineal. No importa `aprobacion_credito`. Las funciones de carga, split, métricas y modelos siguen copiadas en la celda de código inicial.

Recorrido. Lo nuevo está marcado.

1. **Contexto.** El texto actual, más una frase: además de la tabla, el notebook muestra cómo se lee cada modelo (dibujo del árbol, coeficientes de la logística y curva ROC).
2. **Carga y EDA.** Carga, tablas y la figura actual (desbalance + boxplot de `score_buro`). El párrafo de lectura que hoy está antes de la figura pasa a la celda Markdown inmediatamente posterior. No se duplica.
3. **Split.** Sin cambios.
4. **Entrenamiento.** El texto de hiperparámetros y el `fit` se quedan. Después del `fit`, en este orden: dibujo del árbol, párrafo, reglas en texto, párrafo, barras de importancia, párrafo, tabla de coeficientes, párrafo, párrafo de la SVM.
5. **Métricas.** El texto de métricas y la tabla de cinco columnas se quedan. Después: barras agrupadas, párrafo, ROC, párrafo, heatmaps de las tres matrices (esta celda **reemplaza** la de `ConfusionMatrixDisplay`), párrafo.
6. **Recomendación.** Misma decisión y mismo criterio. El texto cita la ROC y un coeficiente.
7. **Bitácora de IA y guion.** Una viñeta nueva. El guion, en el hallazgo y en la decisión, señala la ROC y ese coeficiente.

Contrato entre celdas: el split se calcula una vez; los tres modelos se ajustan solo con train y se evalúan solo con test. Las figuras nuevas no llaman `fit` otra vez.

## 5. Componentes

Cada bloque lee modelos ya entrenados o el conjunto de prueba. El párrafo se escribe después de ejecutar la celda, con los números reales de este CSV, y queda congelado en Markdown. El notebook entregado no contiene «TBD», «TODO» ni una frase genérica del tipo «aquí va la variable del primer corte».

Los imports nuevos viven en la celda que los usa por primera vez, igual que hoy `matplotlib.pyplot` y `ConfusionMatrixDisplay`.

### 5.1 Lectura de EDA

- Hace: mueve bajo la figura el párrafo ya escrito (267 aprobados, 26.7 %; score más alto en aprobados; DTI más bajo; ingreso casi no separa; no se fabrican variables).
- Depende: de la figura de dos paneles que ya existe.

### 5.2 Árbol dibujado

- Hace: `plot_tree` del `DecisionTreeClassifier` ya ajustado (`modelos["Árbol"]`).
- Argumentos: `feature_names` de las cinco columnas, `class_names=["Rechazado", "Aprobado"]`, `filled=True`, `rounded=True`, `fontsize=8`, figura de al menos 18 por 10 pulgadas, título que diga `max_depth=4`.
- Imprime en la misma celda `get_depth()` y `get_n_leaves()`.
- Párrafo: nombra la variable del primer corte (la que imprima el árbol en este CSV) y recuerda que la profundidad 4 es el tope elegido para poder contarlo en la oral.
- Depende: del `fit` del árbol. No lleva scaler.

### 5.3 Reglas

- Hace: `export_text` del mismo árbol, con `feature_names` de las cinco columnas.
- Párrafo: traduce a lenguaje de crédito la regla de la raíz y un camino hacia aprobado y un camino hacia rechazado, si ambos existen en el texto exportado. No copia el árbol completo en prosa.
- Depende: del mismo árbol ajustado.

### 5.4 Importancias

- Hace: un `DataFrame` con columnas `Variable` e `Importancia` a partir de `feature_importances_`, ordenado de mayor a menor, y un gráfico de barras horizontales.
- Párrafo: dice cuál variable parte más los nodos. Declara que esas importancias no se comparan en número con los coeficientes de la logística, porque no comparten escala.
- Depende: del árbol ajustado.

### 5.5 Coeficientes de la logística

- Hace: lee `modelos["Regresión logística"].named_steps["clf"].coef_[0]` (escala ya estandarizada por el `StandardScaler` del pipeline). Arma un `DataFrame` con `Variable`, `Coeficiente (escala estandarizada)`, `Odds Ratio` = `exp(coeficiente)` y `Magnitud del efecto` = valor absoluto del coeficiente. Lo ordena por magnitud descendente y lo muestra con tres decimales.
- Párrafo: nombra las dos variables de mayor magnitud y el signo de cada una (a favor o en contra de aprobar). Si un signo no coincide con la media del EDA, el texto reporta el signo del modelo.
- Depende: del pipeline de logística ya ajustado. No se leen coeficientes del árbol ni de la SVM.

### 5.6 Nota de la SVM

- Hace: solo Markdown. Explica que el kernel RBF no ofrece un coeficiente por variable en el espacio original, así que el comité no puede preguntar «¿por qué esta solicitud?» a la SVM. La SVM entra a la comparación por las métricas; la lectura variable por variable queda en el árbol y en la logística.
- Depende: de nada en tiempo de ejecución.

### 5.7 Barras de métricas

- Hace: gráfico de barras agrupadas, cinco métricas (accuracy, precisión, recall, especificidad, AUC) y tres series (Árbol, Regresión logística, SVM) con leyenda. Eje vertical de 0 a 1. Los valores son los de `tabla_3` (tres decimales), para que el gráfico coincida con la tabla.
- Párrafo: el accuracy se ve alto porque el 73.3 % de las solicitudes históricas fueron rechazadas; el recall es bajo en los tres modelos; esta figura no elige sola al ganador.
- Depende: de `tabla_3`, calculada solo en test.

### 5.8 Curva ROC

- Hace: una curva por modelo con `roc_curve` sobre `predict_proba(Xte)[:, 1]`, la diagonal del clasificador aleatorio en línea discontinua, y una leyenda con el AUC a tres decimales.
- AUC que la leyenda debe mostrar, ya fijados para este CSV: árbol 0.683, regresión logística 0.805, SVM 0.749.
- Párrafo: la ROC es el ranking; la logística queda por encima de la SVM y del árbol. El corte de `predict` sigue en 0.5 y no se mueve en esta figura.
- Depende: de los tres modelos ajustados y de `Xte`, `yte`.

### 5.9 Matrices

- Hace: reemplaza la celda de `ConfusionMatrixDisplay` por tres paneles matplotlib. Cada panel dibuja la matriz con `imshow` y escribe el conteo entero dentro de cada celda. Etiquetas «Rechazo (0)» y «Aprobado (1)». `labels=[0, 1]`. Imprime TN, FP, FN, TP y la suma.
- Cada matriz suma 250. Conteos ya fijados: árbol 152, 31, 44, 23; logística 169, 14, 45, 22; SVM 172, 11, 47, 20.
- Párrafo: FP es aprobar a quien el histórico rechazó; FN es rechazar a quien el histórico aprobó.
- Depende: de `predicciones` sobre el test.

### 5.10 Recomendación

Se conserva el criterio y la frase exacta «Recomiendo la regresión logística».

El criterio, ya aplicado en la spec del 2026-09-26, no se recalcula con código:

- Nadie gana 2 de 3 (recall, especificidad, AUC) por al menos 0.03 frente a cada rival.
- El árbol pierde más de 0.03 en AUC y en especificidad frente a la logística.
- Entre logística y SVM el desempeño está cerca; la interpretabilidad elige la logística.

El Markdown añade dos anclas a las figuras nuevas: el AUC de la ROC (0.805 frente a 0.749 y 0.683) y el signo de una de las dos variables de mayor magnitud en la tabla de coeficientes (la que salga al ejecutar; el texto usa ese signo real). Conserva el límite del recall cercano a 0.33 y la advertencia de no usar el accuracy como argumento principal.

Prohibido `tabla.idxmax()` o cualquier celda que asigne un ganador.

### 5.11 Bitácora y guion

Bitácora: se añade una viñeta con herramienta y propósito. Texto mínimo que debe aparecer: Cursor armó las figuras nuevas (árbol, reglas, importancias, coeficientes, barras, ROC, heatmaps) y los párrafos de lectura; la decisión de recomendar la logística sigue siendo del estudiante y tiene que poder explicarla en la oral.

Guion: se mantienen las cinco partes y la duración de 3 a 5 minutos. En el hallazgo se señala que en la ROC la logística queda por encima (AUC 0.805). En la decisión se señala el signo de un coeficiente junto con la interpretabilidad. No se añade un minuto para recitar cada gráfico.

El encabezado sigue conteniendo la cadena «Guion de 3», que el test de estructura ya exige.

## 6. Flujo de datos

```
CSV
 → DataFrame
 → EDA (solo lectura) y párrafo bajo la figura
 → X, y
 → train_test_split (una vez)
 → fit de 3 modelos en train
 → plot_tree, export_text, feature_importances_ del árbol
 → coef_ y odds del clf de la logística
 → predict y predict_proba en test
 → tabla_3
 → barras, ROC y heatmaps
 → Markdown de recomendación
```

El test no ajusta scaler, profundidad ni umbral. Umbral de clasificación = 0.5. Las importancias y los coeficientes no se mezclan en una sola tabla.

## 7. Errores

| Condición | Comportamiento |
|---|---|
| CSV ausente, columnas distintas, nulos o `aprobado` fuera de {0, 1} | Igual que la spec del 2026-09-26: la celda de carga lanza error |
| Figura nueva ejecutada sin el `fit` previo | La celda falla. No hay `try/except` que la deje en verde |
| Coeficientes leídos fuera de `named_steps["clf"]` | No permitido: esa es la única fuente |
| AUC de la leyenda distinto de la tabla a tres decimales | No permitido: ambos salen de `predict_proba` del mismo test |
| Hiperparámetro cambiado para que un gráfico se lea mejor | No permitido |
| Párrafo con placeholder | No permitido en el notebook entregado |

## 8. Verificación

Antes de llamar al notebook listo:

1. Corre de arriba abajo, una vez, vía `tests/test_notebook_ejecuta.py`.
2. `tests/test_evaluacion_test.py` sigue en verde sin editarlo. AUC 0.683, 0.805 y 0.749.
3. `tests/test_notebook_estructura.py` conserva sus asserts actuales y exige además, en el texto del notebook: `plot_tree`, `export_text`, `feature_importances_`, `roc_curve`, `Odds Ratio`.
4. Siguen prohibidos `import aprobacion_credito` e `idxmax`.
5. Siguen presentes: `max_depth=4`, `probability=True`, `test_size=0.25`, `random_state=11`, `stratify=y`, «Recomiendo la regresión logística», «interpretabilidad», «Cursor», «Guion de 3».
6. Revisión manual: cada figura nueva tiene un párrafo Markdown en la celda siguiente; cada matriz suma 250; la recomendación cita la ROC y un coeficiente; la bitácora tiene la viñeta nueva con herramienta y para qué.

No hay tests de píxeles. No hay suite nueva de CI.

## 9. Decisiones cerradas

| Tema | Decisión |
|---|---|
| Dónde viven las figuras | Dentro de EDA, Entrenamiento y Métricas, con el párrafo debajo |
| Qué se añade | Árbol, reglas, importancias, coeficientes con odds, nota de SVM, barras de métricas, ROC, heatmaps |
| Qué no se añade | Búsqueda de hiperparámetros, EDA extra, seaborn, anexo final |
| Modelos y ganador | Los de la spec del 2026-09-26; gana la regresión logística en prosa |
| Módulo Python | `aprobacion_credito.py` no cambia; el notebook sigue autónomo |
| Números de test | Los ya fijados; los párrafos los citan, no los recalculan con otro split |

## 10. Implementación (límite de esta spec)

Esta spec no edita el notebook. El siguiente paso, tras la revisión de este archivo, es el plan de implementación: actualizar el `.ipynb` en el orden de las secciones 4 y 5, ejecutar de arriba abajo, redactar los párrafos con los números reales (raíz del árbol, dos o tres reglas, dos coeficientes de mayor magnitud) y ampliar `tests/test_notebook_estructura.py` como dice la sección 8.
