**UNIVERSIDAD LIBRE · MACHINE LEARNING**

**Parcial — Componente Práctico**

_Parte 2 de 2 — trabajo individual con datos reales + sustentación oral_

| **Docente**               | Gustavo Santos                                                                                                                                 |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Actividad**             | Comparación individual de 3 modelos de clasificación (Árbol de Decisión, Regresión Logística, SVM) sobre un caso real de aprobación de crédito |
| **Modalidad**             | Individual · uso libre de computador, notebook y herramientas de inteligencia artificial                                                       |
| **Puntaje de esta parte** | 70 puntos (de los 100 del parcial)                                                                                                             |
| **Dataset**               | Un archivo CSV individual y único por estudiante — aprobacion_credito_\[Su_Nombre\].csv, distribuido por el docente                            |
| **Entrega**               | Notebook (.ipynb) subido a Moodle antes del miércoles 30 de septiembre de 2026                                                                 |
| **Sustentación**          | Individual, 3 a 5 minutos por estudiante, en la clase siguiente a la entrega                                                                   |

# **Contexto**

Esta es la segunda parte del parcial. En la Parte 1 (componente cerrado) se evaluaron los fundamentos conceptuales sin herramientas. Aquí se evalúa si puede aplicar esos fundamentos con datos reales y las mismas herramientas que va a usar en su vida profesional — incluyendo inteligencia artificial. Trabaja en el área de riesgo de una entidad financiera y debe decidir, con evidencia, qué modelo de clasificación recomendar para aprobar o rechazar solicitudes de crédito: un árbol de decisión, una regresión logística, o una SVM.

# **Uso de inteligencia artificial: permitido, con dos condiciones**

Puede usar ChatGPT, Copilot, Claude o cualquier otro asistente de IA libremente para escribir código, resolver dudas o generar ideas. A cambio de esa libertad, se le pide lo siguiente:

1. Documente en una celda de texto (Markdown) qué herramienta usó y para qué — por ejemplo: "usé ChatGPT para depurar un error de sintaxis en la celda 5" o "le pedí a Claude que me explicara la diferencia entre gamma='scale' y gamma='auto'".
2. En la sustentación oral debe poder explicar y justificar CUALQUIER línea de su notebook, así la haya escrito la IA. Si no puede explicar una decisión (por qué ese hiperparámetro, qué significa esa métrica, por qué ese modelo y no otro), esa parte de la rúbrica se califica como insuficiente — independientemente de que el código funcione.

# **El dataset**

Cada estudiante recibe un archivo distinto, aprobacion_credito_\[Su_Nombre\].csv, con 1.000 solicitudes de crédito históricas y las mismas columnas y relaciones entre variables para todos — pero con valores únicos por persona. Use únicamente el archivo con su propio nombre: como cada dataset es distinto, dos notebooks con exactamente las mismas métricas son evidencia de que no se trabajó de forma individual. Columnas:

| **Columna**              | **Descripción**                                |
| ------------------------ | ---------------------------------------------- |
| ingreso_mensual_kcop     | Ingreso mensual del solicitante (miles de COP) |
| score_buro               | Score de buró de crédito (300–850)             |
| antiguedad_laboral_anios | Años de antigüedad laboral                     |
| num_creditos_activos     | Número de créditos activos                     |
| dti                      | Deuda / ingreso (debt-to-income)               |
| aprobado (objetivo)      | 1 = aprobado, 0 = rechazado                    |

# **Tareas**

1. Cargue el dataset y realice el mismo split usado en clase: train_test_split(X, y, test_size=0.25, random_state=11, stratify=y).

```
X = df.drop(columns="aprobado"); y = df["aprobado"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                       random_state=11, stratify=y)
```

1. Entrene los 3 modelos sobre el mismo split: un Árbol de Decisión, una Regresión Logística y una SVM. Recuerde que SVM (y en general los modelos basados en distancia) requiere StandardScaler; los árboles no lo necesitan.
2. Para cada modelo, reporte sobre el conjunto de prueba: accuracy, precisión, sensibilidad (recall), especificidad y AUC. Presente los resultados de los 3 modelos en una sola tabla comparativa.
3. Elija el modelo que recomendaría para producción y justifique su decisión en texto plano (Markdown), en español, sin jerga innecesaria — como si se lo estuviera explicando al equipo de riesgo. La justificación debe considerar tanto el desempeño (las métricas) como la interpretabilidad de cada modelo, no solo cuál dio el número más alto.
4. Documente, en una celda aparte, qué herramientas de IA usó durante el ejercicio y para qué (ver sección anterior).

# **Entrega y sustentación**

Suba el notebook (.ipynb) a esta tarea en la plataforma antes de las 2 pm.. En la clase siguiente, cada estudiante sustenta individualmente su trabajo ante el docente (3 a 5 minutos): qué hizo, qué encontró, y por qué eligió el modelo final. El docente puede preguntar sobre cualquier parte del código, las métricas, o las decisiones tomadas — incluidas las que se apoyaron en IA.

# **Rúbrica de evaluación (70 puntos)**

| **Criterio**                                                       | **Excelente**                                                                                                                     | **Aceptable**                                                                                           | **Insuficiente**                                                                |
| ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 1\. Preparación de datos y entrenamiento de los 3 modelos (20 pts) | 20 pts — entrena correctamente los 3 modelos sobre el mismo split, con escalamiento donde corresponde (SVM).                      | 12 pts — entrena los 3 modelos pero con errores menores (split distinto, falta de escalamiento en SVM). | 0–6 pts — falta uno o más modelos, o el código no ejecuta.                      |
| 2\. Comparación de métricas (15 pts)                               | 15 pts — reporta accuracy, precisión, recall, especificidad y AUC de los 3 modelos, en una tabla clara y correcta.                | 9 pts — reporta las métricas pero incompletas, sin tabla clara, o con algún error de cálculo.           | 0–4 pts — faltan varias métricas o están mal calculadas.                        |
| 3\. Justificación de la decisión final (20 pts)                    | 20 pts — recomienda un modelo con justificación sólida que considera desempeño E interpretabilidad, en lenguaje de negocio claro. | 12 pts — recomienda un modelo pero la justificación es superficial o solo mira una métrica.             | 0–6 pts — no toma una decisión clara o no la justifica.                         |
| 4\. Transparencia sobre el uso de IA (5 pts)                       | 5 pts — documenta con claridad qué herramientas de IA usó y para qué, en cada caso relevante.                                     | 3 pts — menciona el uso de IA de forma genérica, sin detalle de para qué se usó.                        | 0 pts — no documenta el uso de IA, aunque sea evidente que se usó.              |
| 5\. Sustentación oral individual (10 pts)                          | 10 pts — explica con claridad y seguridad cualquier parte de su notebook y responde bien las preguntas del docente.               | 6 pts — explica lo general pero duda o no puede justificar decisiones específicas.                      | 0–3 pts — no puede explicar su propio trabajo, evidenciando que no lo entiende. |