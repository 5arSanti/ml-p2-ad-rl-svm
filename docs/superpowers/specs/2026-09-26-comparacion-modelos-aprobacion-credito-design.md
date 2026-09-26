# Diseño: comparación de 3 modelos para aprobación de crédito

Fecha: 2026-09-26  
Curso: Universidad Libre · Machine Learning · Parcial, componente práctico (70/100)  
Estudiante / dataset: Johel Santiago Arias Becerra · `aprobacion_credito_Arias_Becerra_Johel_Santiago_3144.csv`  
Entregable calificado: un notebook `.ipynb` (Moodle) + sustentación oral de 3 a 5 minutos  
Enfoque aprobado: notebook único narrativo, alineado a la rúbrica (enfoque 1)

## 1. Problema

En riesgo de una entidad financiera hay que recomendar, con evidencia, cuál de tres clasificadores usar para aprobar o rechazar crédito: árbol de decisión, regresión logística o SVM.

El dataset es individual (1.000 solicitudes históricas). Dos notebooks con las mismas métricas indicarían copia: hay que usar solo este CSV.

Columnas (todas numéricas; objetivo binario):

| Columna | Significado |
|---|---|
| `ingreso_mensual_kcop` | Ingreso mensual (miles de COP) |
| `score_buro` | Score de buró (rango típico 300–850) |
| `antiguedad_laboral_anios` | Años de antigüedad laboral |
| `num_creditos_activos` | Créditos activos |
| `dti` | Deuda / ingreso |
| `aprobado` | 1 = aprobado, 0 = rechazado |

Hechos de este CSV (exploración previa, no hay que “descubrirlos” de nuevo para diseñar):

- 1.000 filas, 0 nulos, 0 columnas extra.
- Desbalance: 267 aprobados (26.7%) y 733 rechazados (73.3%).
- Correlación simple con `aprobado`: `score_buro` +0.39, `dti` −0.28, `num_creditos_activos` −0.14, `antiguedad_laboral_anios` +0.09, `ingreso_mensual_kcop` +0.01.

El objetivo histórico es “¿el proceso habría aprobado?”, no “¿el cliente pagará?”. La justificación habla de solicitudes, no de default real.

## 2. Criterio de éxito

Maximizar la rúbrica de 70 puntos, no el accuracy.

| Criterio | Puntos | Umbral “Excelente” que este diseño obliga |
|---|---|---|
| 1. Preparación y 3 modelos | 20 | Mismo split de clase; 3 modelos; scaler en SVM y logística; árbol sin scaler |
| 2. Comparación de métricas | 15 | Accuracy, precisión, recall, especificidad y AUC de los 3, una tabla correcta |
| 3. Justificación de la decisión | 20 | Un modelo recomendado; desempeño e interpretabilidad; español de negocio |
| 4. Transparencia de IA | 5 | Herramienta + para qué, caso por caso |
| 5. Sustentación oral | 10 | Guion en el notebook; toda decisión del diseño es explicable |

Criterio de negocio declarado **antes** de ver métricas de test (decisión de diseño aprobada):

- Se miran juntas recall, especificidad y AUC (y se contextualiza accuracy por el desbalance).
- Un modelo **domina** si gana en al menos 2 de esas 3 métricas por **≥ 0.03** frente a cada rival.
- Si nadie domina, el desempeño se trata como cercano y gana el más interpretable: árbol > logística > SVM.
- No se elige el ganador en código ni por una sola métrica.

## 3. Fuera de alcance

No entra en el entregable ni en esta spec:

- GridSearch, RandomSearch u otra búsqueda de hiperparámetros.
- `class_weight='balanced'`, SMOTE u otro remuestreo.
- Ingeniería de variables, imputación (no hay nulos) ni descarte de `ingreso_mensual_kcop`.
- Más de tres modelos, ensambles, o elegir el ganador con `if` sobre la tabla.
- App, API, dashboard, módulos `.py` de producto, diapositivas.
- Escalar el árbol o dejar SVM/logística sin scaler.
- Un segundo split o un `random_state` distinto de 11 en el `train_test_split`.

## 4. Arquitectura

Un notebook lineal. El docente abre un archivo y pregunta por celdas.

```
LICENSE
requirements.txt
data/aprobacion_credito_Arias_Becerra_Johel_Santiago_3144.csv
notebooks/comparacion_modelos_aprobacion_credito.ipynb
docs/superpowers/specs/2026-09-26-comparacion-modelos-aprobacion-credito-design.md
```

Dependencias de runtime del notebook: `pandas`, `numpy`, `scikit-learn`, `matplotlib`. `jupyter` (o equivalente) solo para ejecutar.

Secciones del notebook, en este orden:

1. Contexto y reglas del ejercicio (Markdown).
2. Carga + EDA breve.
3. Split fijo de clase.
4. Entrenamiento de los 3 modelos.
5. Métricas de prueba, tabla única y tres matrices de confusión.
6. Recomendación de producción (Markdown, después de la tabla).
7. Bitácora de uso de IA.
8. Guion de sustentación (3–5 minutos).

Contrato entre celdas: el split se calcula una vez; los tres modelos se ajustan solo con `Xtr, ytr` y se evalúan solo con `Xte, yte`. El `StandardScaler` vive dentro de cada `Pipeline` y se ajusta solo con train.

## 5. Componentes

Cada unidad tiene un propósito, una forma de usarse y dependencias explícitas.

### 5.1 Cargador

- Hace: `pd.read_csv` de la ruta del repo.
- Comprueba: 1.000 filas; columnas exactas `ingreso_mensual_kcop`, `score_buro`, `antiguedad_laboral_anios`, `num_creditos_activos`, `dti`, `aprobado`; cero nulos; `aprobado` ⊆ {0, 1}.
- Si falla: la celda lanza error con la ruta o el esquema esperado. No se capturan excepciones para “seguir”.
- Depende: solo del CSV.

### 5.2 EDA breve

- Hace: conteo y tasa de `aprobado`; `describe` de las cinco features; **medias** por clase; mención de las señales ya observadas (score y DTI pesan; ingreso casi no).
- Gráficos: exactamente dos — (1) barras del desbalance de `aprobado`; (2) boxplot de `score_buro` por clase (es la variable con más correlación).
- No hace: transformar, filtrar, ni elegir variables.
- Depende: del DataFrame crudo.

### 5.3 Split de clase

Código obligatorio, sin parámetros distintos:

```python
X = df.drop(columns="aprobado")
y = df["aprobado"]
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=11, stratify=y
)
```

Imprime tamaños (750 / 250) y tasa de aprobación en train y test para mostrar que `stratify` conservó el desbalance.

Nadie vuelve a llamar `train_test_split` después.

### 5.4 Tres entrenadores

Mismos `Xtr, ytr`. Semilla 11 donde el estimador la acepta, para alinear con el split.

| Modelo | Construcción | Reglas |
|---|---|---|
| Árbol | `DecisionTreeClassifier(max_depth=4, random_state=11)` | Sin `StandardScaler`. `max_depth=4` es el único hiperparámetro no default: limita sobreajuste y permite explicar/dibujar el árbol. |
| Logística | `Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=11))])` | El scaler no ve el test. `max_iter=1000` evita no convergencia; no es búsqueda de hiperparámetros. |
| SVM | `Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, random_state=11))])` | Cumple el enunciado. `kernel="rbf"` es el default de `SVC` y se escribe explícito para la oral. `probability=True` habilita `predict_proba` para el AUC. |

No hay `class_weight`. No hay GridSearch. No hay un scaler global compartido por los tres modelos.

### 5.5 Evaluador único

Una función `metricas_prueba(y_true, y_pred, y_score) -> dict` usada por los tres modelos.

Clase positiva = 1 (aprobado). Solo test.

| Clave | Cálculo |
|---|---|
| `accuracy` | `sklearn.metrics.accuracy_score` |
| `precision` | `precision_score(..., pos_label=1, zero_division=0)` |
| `recall` | `recall_score(..., pos_label=1, zero_division=0)` (sensibilidad) |
| `especificidad` | De `confusion_matrix(y_true, y_pred, labels=[0, 1])`: `TN / (TN + FP)`. Si `TN + FP == 0`, se lanza error. |
| `auc` | `roc_auc_score(y_true, y_score)` con `y_score = predict_proba(Xte)[:, 1]` |

Respaldo de AUC: si un estimador no expone `predict_proba`, usar `decision_function`. No inventar scores. Con la construcción de la sección 5.4, los tres exponen probabilidad (el árbol y la logística de forma nativa; SVM por `probability=True`).

### 5.6 Tabla y matrices

- Una `DataFrame` de 3 filas (Árbol, Regresión logística, SVM) y 5 columnas con los nombres del enunciado: accuracy, precisión, sensibilidad (recall), especificidad, AUC.
- Las cinco métricas se muestran con **3 decimales** en las tres filas (`round(..., 3)`).
- Debajo, tres matrices de confusión sobre las 250 filas de test, con `labels=[0, 1]` (orden `[[TN, FP], [FN, TP]]`). Cada una suma 250. Se etiquetan TN, FP, FN, TP en lenguaje de crédito: FP = aprobar a quien el histórico rechazó; FN = rechazar a quien el histórico aprobó.

### 5.7 Recomendación (Markdown)

Se escribe **después** de ver la tabla. No hay celda que asigne `mejor_modelo = tabla.idxmax()`.

Guion obligatorio:

1. Recordar el criterio (recall + especificidad + AUC; interpretabilidad si nadie domina por ≥ 0.03 en al menos 2 de 3).
2. Traducir las matrices a exposición vs negocio perdido.
3. Comparar los tres, no solo el primero.
4. Frase explícita: “Recomiendo X para producción porque…”.
5. Un límite del modelo elegido.

Interpretabilidad fija para el texto (español llano):

- Árbol: reglas visibles (“si el score es alto y el DTI es bajo…”). Útil en comité. Límite: cortes toscos con profundidad 4.
- Logística: signo de cada variable (score a favor, DTI en contra). Punto medio entre número y explicación.
- SVM RBF: puede separar bien; no responde “¿por qué esta solicitud?”. Eso pesa en riesgo.

Prohibido justificar solo con “tuvo el accuracy más alto”. Accuracy se menciona para advertir el sesgo del 73.3% de rechazados.

### 5.8 Bitácora de IA

Celda Markdown, lista de viñetas. Cada ítem: herramienta + para qué. Mínimo incluir este diseño (Cursor) y cualquier generación o depuración de código al implementar. Si una parte se hizo sin IA, se puede decir. No vale una sola frase genérica (“usé IA para el taller”).

### 5.9 Guion oral

Última celda Markdown, no un archivo aparte. Estructura de 3–5 minutos:

1. Problema y desbalance (~20 s).
2. Mismo split; por qué se escala SVM y logística y no el árbol (~40 s).
3. Un hallazgo de la tabla, no un recitado de quince números (~60 s).
4. Recomendación: métricas + interpretabilidad + un límite (~90 s).
5. Qué hizo la IA y qué decidiste tú (~20 s).

Decisiones que el estudiante debe poder explicar de memoria: `random_state=11`, `stratify=y`, `max_depth=4`, `max_iter=1000`, `probability=True`, especificidad a mano, scaler dentro del `Pipeline`.

## 6. Flujo de datos

```
CSV
 → DataFrame
 → EDA (solo lectura)
 → X, y
 → train_test_split (una vez)
 → fit de 3 modelos en train
 → predict y predict_proba en test
 → metricas_prueba (misma función)
 → tabla + 3 matrices
 → Markdown de recomendación
```

El test no ajusta scaler, profundidad ni umbral. Umbral de clasificación = 0.5 (default de `predict`). No se mueve el umbral: el enunciado no lo pide y sería otra decisión oral.

## 7. Errores

| Condición | Comportamiento |
|---|---|
| CSV ausente | Error con la ruta esperada `data/aprobacion_credito_Arias_Becerra_Johel_Santiago_3144.csv` |
| Columnas distintas | Error con el esquema esperado; no se renombra en silencio |
| Nulos o `aprobado` fuera de {0, 1} | Error |
| Split o semillas distintas a las del enunciado/diseño | No permitido: valores cableados, no hay widgets |
| SVM o logística sin scaler; árbol con scaler | No permitido: construcción cableada |
| AUC sin scores | No ocurre con 5.4; respaldo = `decision_function`, no valores inventados |

No hay `try/except` que trague errores y deje el notebook “en verde”.

## 8. Verificación (lista de aceptación)

Antes de llamar al notebook listo:

1. Corre de arriba abajo, una vez, sin depender del orden manual de celdas.
2. `Xtr` tiene 750 filas y `Xte` 250. Con `stratify` sobre 267 aprobados: train 200 positivos (tasa 0.2667) y test 67 positivos (tasa 0.2680).
3. Hay exactamente 3 modelos; el árbol no incluye scaler; logística y SVM sí, vía `Pipeline`.
4. La tabla tiene 3 filas y las 5 métricas del enunciado; ninguna es NaN.
5. Cada matriz de confusión suma 250.
6. El Markdown de recomendación nombra un modelo, al menos dos métricas del criterio, e interpretabilidad.
7. La bitácora de IA tiene herramienta + propósito por caso.

La comprobación de (2)–(5) se hace al ejecutar el notebook (o el mismo código en una sesión Python) durante la implementación. (1), (6) y (7) se revisan a mano. No hay suite de CI ni un segundo archivo de tests como entregable de Moodle.

## 9. Decisiones de diseño (cerradas)

| Tema | Decisión | Por qué |
|---|---|---|
| Alcance del notebook | Obligatorio + EDA breve + matrices + marco FP/FN + guion oral | Cubre rúbricas 3 y 5 sin GridSearch |
| Criterio de negocio | Recall / especificidad / AUC; domina si gana 2 de 3 por ≥ 0.03; si no, interpretabilidad árbol > logística > SVM | El enunciado pide desempeño e interpretabilidad |
| Entrenamiento | Pipelines + scaler en SVM y LR; árbol `max_depth=4`; sin `class_weight` | Cumple rúbrica 1 y se explica en 5 minutos |
| Forma del entregable | Un solo `.ipynb` lineal | Moodle + oral sobre celdas |
| Ganador | Se elige en prosa después de la tabla | Evita “el número más alto” |

## 10. Implementación (límite de esta spec)

Esta spec no implementa el notebook. El siguiente paso, tras aprobación del archivo, es el plan de implementación (skill writing-plans): crear `data/`, `requirements.txt` y el `.ipynb` en el orden de las secciones 4 y 5, copiar el CSV del estudiante, ejecutar de arriba abajo y completar la recomendación con los números reales de test.
