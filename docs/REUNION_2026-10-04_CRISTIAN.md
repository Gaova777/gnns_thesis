# Reunión con Cristian, 4-oct-2026: acta y plan de trabajo

> Fuente: transcripción `jwc-sszk-tnx (2026-10-04 17_01 GMT-5)`. Asistentes: Cristian Rosero
> Arias, Alejandro Gómez Huertas y Juan Diego Garzón Ovalle. Duró 2 h 17 min y quedó
> pendiente continuarla al día siguiente (5-oct) desde H1.
>
> Cristian separó la conversación en dos planos: **objetivo 1, enviar la tesis** (10-oct) y
> **objetivo 2, que quede bien**. Lo del objetivo 2 se puede completar después de enviar, pero
> «se debe corregir obligatoriamente» y tiene que quedar planteado desde ya.

## Lo que pidió para la reunión siguiente

Para H1, H2 y H3, cada lámina de resultados debe responder seis preguntas:

1. ¿Qué se esperaba inicialmente?
2. ¿Por qué se esperaba eso?
3. ¿Qué resultado dio?
4. ¿Era lo que se esperaba?
5. Si no lo era, ¿por qué dio ese resultado?
6. ¿En qué impacta?

Casos concretos que mencionó:

- **H1 (desbalance):** se esperaba que más desbalance diera menos estabilidad, pero falta
  decir por qué. Posibles lecturas: el rango 1:10 a 1:40 no es tan distinto, o el
  submuestreo y el sobremuestreo estuvieron bien hechos.
- **H2 (arquitectura):** GCN no pasa la compuerta. ¿Era de esperarse? ¿En qué consiste GCN
  para quedar atrás? ¿Sorprende que GraphSAGE, TAGCN y GAT queden parecidas?
- **H3 (balanceo):** PGExplainer da ~0 con class weighting y mejor con focal loss que sin
  ajuste. Eso es «rarísimo» y no puede quedar así. Hay que decir si es culpa de PGExplainer,
  de class weighting, de ambos o un bug, y explicar la naturaleza de cada explicador.
- **Definir la estabilidad** antes de mostrarla. Hoy no se define en ninguna lámina.
- **Plan de estructura** de las láminas y del documento con los cambios ya acordados.

## Plan de trabajo

Estado: ✅ hecho · 🔄 en curso · ⏳ pendiente · ❓ requiere decisión

Tareas en Linear:

| Tarea | Linear |
|---|---|
| A1 | THE-30 |
| A2 | THE-35 |
| A3 | THE-36 |
| A4 | THE-37 |
| A5 | THE-34 |
| A6 | THE-21 |
| A7 | THE-31 |
| A9 | THE-32 |
| A10 | THE-33 |
| Las seis preguntas de H1, H2 y H3 | THE-38 |
| Sección B (presentación) | lista de verificación en THE-17 |

### A. Metodología y experimentos (obligatorio, objetivo 2)

| # | Qué pidió | Qué hacemos | Estado |
|---|---|---|---|
| A1 | Las semillas no son modelos: son incertidumbre. La unidad es la **configuración** (48 = 4 arquitecturas × 4 escenarios × 3 pérdidas) con media ± incertidumbre, y la compuerta se aplica a la configuración, no a cada semilla. | Recalcular la tabla de 48 configuraciones, aplicar la compuerta sobre la media y repetir H1, H2 y H3 con esa unidad. | 🔄 |
| A2 | El 1:1 no tiene sentido con las tres pérdidas: con clases balanceadas, class weighting da peso 1 y focal converge a sin ajuste. Además ningún 1:1 pasó la compuerta. | Sacar el 1:1 de la discusión principal y llevarlo a un anexo. Reemplazarlo por otro nivel de desbalance (Cristian sugirió 1:5, 1:20 o similar). | ❓ |
| A3 | Optuna de 8 trials no es una optimización real («es como entrenar 10 épocas»). | La matriz pasa a ser la **fase de selección**. Con ella se escoge un candidato top (idealmente una arquitectura con una pérdida) y se corre Optuna completo (100 a 200 trials) en los escenarios. En las láminas no se menciona Optuna: se dice que se usó la misma configuración de hiperparámetros y 3 semillas para la variación estadística. | ⏳ depende de A1 |
| A4 | Extra para el remate: comparar contra un modelo de la literatura (preentrenado o con la configuración publicada). | Buscar un modelo GNN publicado sobre Elliptic con pesos o configuración reproducible. | ⏳ opcional |
| A5 | El 15,6 % de sin etiqueta que marca el modelo B no prueba que haya fraude escondido. Puede ser drift, que B aprendió mal porque las lícitas eran pocas, o estructura del grafo. | Análisis de población (PSI, PCA), cobertura de las lícitas, estructura del grafo (grado, vecinos), drift por timestep y un explicador sencillo sobre esos casos. Cerrar con 2 o 3 argumentos. | 🔄 |
| A6 | Falta decir qué técnica de submuestreo se usó y por qué. | Ya está en el código: submuestreo aleatorio de negativos, estratificado por origen (lícita / sin etiqueta), con semilla de datos fija (2026) e independiente de la semilla del modelo. Falta redactarlo con su justificación y cita (THE-21). | ⏳ redacción |
| A7 | PR-AUC solo no sirve. Se reporta con ROC-AUC, y F1 con KS. Hay que mostrar las curvas, no solo el número: la forma dice si un modelo domina, si se cruzan o si hay picos de suerte. | Curvas ROC y PR por configuración con banda de las 3 semillas, en validación y test, más la tabla de ROC-AUC, PR-AUC, F1 y KS. | 🔄 |
| A8 | No promediar escenarios sin decir de qué modelo se habla. Siempre con incertidumbre. | Se resuelve con A1 y A7. | 🔄 |
| A9 | PGExplainer con class weighting ~0. | Investigar si hay un bug o un colapso del explicador y explicarlo. | 🔄 |
| A10 | ¿Por qué GCN queda atrás? | Explicación con evidencia (hiperparámetros, forma de agregar, literatura, modos A y B). | 🔄 |

### B. Presentación (y figuras que pasan a la tesis)

| # | Cambio | Estado |
|---|---|---|
| B1 | Lámina de entrada para público general: Bitcoin, economía, banca, transacciones, uso de redes neuronales, con referencias. Luego las «patologías económicas» (fraude, lavado) y después «el lavado es un patrón de red». | ⏳ |
| B2 | En la lámina de la GNN, la alerta debe tener etiqueta visible (lícito / ilícito). «Caja negra» se menciona después, no ahí. | ⏳ |
| B3 | Una cita en cada lámina, en la línea naranja de abajo. El pie de figura se queda donde está. | ⏳ |
| B4 | Antes de «una explicación que cambia cada vez…», una lámina que introduzca qué es una explicación y un explicador, y por qué hace falta, en contexto empresarial (banco que cambia de modelo y empieza a alertar distinto; ejemplo de robo de identidad y edad). | ⏳ |
| B5 | «Corrida» pasa a «ejecución» (o «run»). | ⏳ |
| B6 | Título de sección arriba en cada lámina (Introducción, Marco teórico, Metodología, Objetivos, Resultados), y lo actual queda como subtítulo. | ⏳ |
| B7 | En la lámina de objetivos, un desbalance genérico (≈1:40) en vez de 1:38,4. El caption «qué esperaría ver si la hipótesis…» se quita o se reemplaza por «Figura 4, a la izquierda». | ⏳ |
| B8 | Lámina de Elliptic: qué es, por qué se usa, cuándo, cómo está compuesto y por qué lo escogimos, con una imagen. | ⏳ |
| B9 | Ordenar A/B/C. Quitar «C es el caso real: el fraude que nadie detectó no puede estar etiquetado» porque lleva a ideas raras. Decir que los sin etiqueta se toman como lícitos por efectos prácticos: la clase es desbalanceada, los negativos aportan ~40 veces más información y en su mayoría deben ser lícitos, igual que haría un banco. | ⏳ |
| B10 | La lámina de sobremuestreo necesita un título que diga que es la sección de sobremuestreo, por qué se sobremuestrea y por qué SMOTE. Reescribir frases como «se reducen los negativos, se crean ilícitas», que no transmiten nada. | ⏳ |
| B11 | Definir la estabilidad antes de usarla. | ⏳ |
| B12 | Las láminas de resultados no pueden ser promedios sin decir de qué modelo. Llevar incertidumbre y las otras métricas. | ⏳ depende de A1 y A7 |
| B13 | Las figuras que le gustaron (grafo, GNN por capas) pasan a la tesis. | ⏳ |

### C. Documento

- La estructura de la tesis sigue la historia de «campeones» (THE-18), ahora con la matriz
  como fase de selección y el candidato top como cierre.
- El 1:1 y lo que no aporta va a anexo, para que no se pierda pero no estorbe.
- «Explicar no es trivial» es una premisa de toda la tesis: no se afirma nada que solo diga
  el modelo sin una explicación que lo respalde.

## Decisiones que necesitan respuesta

1. **Escenario que reemplaza al 1:1 (A2).** Propuesta: 1:20, que queda entre 1:10 y el
   nativo (≈1:38) y es otro nivel real de desbalance que no quita ilícitas. Costo: 36 modelos
   nuevos (4 arquitecturas × 3 pérdidas × 3 semillas) más su explicación.
2. **Candidato para Optuna completo (A3).** Sale del análisis de las 48 configuraciones.
3. **Si A3 y A2 entran antes del 10-oct o en la ventana de correcciones.**
