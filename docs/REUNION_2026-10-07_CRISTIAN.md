# Reunión con Cristian, 7-oct-2026: acta, análisis y plan de trabajo

> Fuente: transcripción `sdg-rkzv-gev (2026-10-07 21_12 GMT-5)`. Asistentes: Cristian Rosero
> Arias, Alejandro Gómez Huertas y Juan Diego Garzón Ovalle. Duró 1 h 35 min. Se presentó el
> deck de hipótesis (versión con el escenario 1:20). La transcripción es automática: las citas
> están depuradas y llevan el minuto aproximado.
>
> Cierre de Cristian (min. 93): con **un explicador extra** y **el caso 1:100** «cerramos el
> tema, con sus imperfecciones» y se puede enviar. El fin de semana (10 y 11-oct) se dedica a
> la comunicación: que en el documento queden claros los pasos.

## Resumen

1. **El resultado central no está cerrado.** Todo dio estable, y la tesis quería argumentar
   alrededor de la inestabilidad. Si la estabilidad no cae en el rango 1:10 a 1:38, hay que
   forzarla: **1:100 y 1:200**. Cristian lo llamó «obligatorio».
2. **Falta un explicador.** PGExplainer no sirve en Elliptic. Con dos explicadores no se puede
   concluir sobre el efecto del explicador. Hay que agregar uno que lo compense.
3. **Los números no bastan.** «Esto no está terminado si ustedes no me pueden decir por qué
   pasó eso.» Cada hipótesis necesita su porqué con ecuaciones y con el funcionamiento de la
   arquitectura y del explicador.
4. **El relato tiene que ser coherente de principio a fin.** La lámina de curvas mostró a GCN
   por encima de GAT en test, después de haber dicho que GCN no pasa la compuerta. No supimos
   explicarlo. El orden correcto es: tabla completa, filtro, candidatos.
5. **Simplificar la comparación de pérdidas.** Pesos por clase en las cuatro arquitecturas, con
   «sin ajuste» como referencia. Nada de «la mejor pérdida de cada una».
6. **Tener un motivo para cada decisión.** Sobre todo para dos: por qué la compuerta está en
   validación y por qué se siguió con un test que se sabe que no es representativo.

**Aviso (10-oct):** al validar los resultados apareció un problema de medición que cambia la
lectura de todo lo anterior. Está en la sección G y conviene leerla primero.

La sección A recoge cada pedido con su tarea. La sección B es el banco de respuestas a lo que
no supimos contestar. La sección C adelanta el porqué de cada hipótesis. La sección D lista las
correcciones de láminas. La sección E trae los hallazgos que salieron al preparar este
documento y la sección G la validación de la medida de estabilidad.

## A. Lo que pidió Cristian y qué hacemos

Estado: ✅ hecho · 🔄 en curso · ⏳ pendiente · ❓ requiere decisión

| # | Qué pidió (minuto) | Qué hacemos | Estado | Linear |
|---|---|---|---|---|
| C1 | Forzar el desbalance a **1:100 o 1:200** para ver caer la estabilidad. «Va a ser obligatorio» (min. 76 y 91). | Dos escenarios de estrés con el mismo protocolo de la matriz. Ver A.1. | 🔄 corriendo desde el 10-oct 07:41 | THE-39 |
| C2 | **Un explicador extra** que compense a PGExplainer (min. 86 a 91). | Integrated Gradients, con la variante de línea base aleatoria para el protocolo de 5 réplicas. Ver A.2. | ⏳ | THE-40 |
| C3 | **El porqué** de H1, H2 y H3 con ecuaciones, arquitectura, puntos fuertes y débiles (min. 82). | Sección C de este documento y redacción para la tesis. | 🔄 | THE-41 |
| C4 | **Pesos por clase para todas** las arquitecturas, «sin ajuste» como referencia, nativo como escenario de referencia (min. 46 a 53). | Rehacer las láminas de rendimiento. Antes hay que resolver la compuerta de GAT (sección E.1). | ❓ | THE-42 |
| C5 | Respuesta a «¿por qué seguir con un test que no es representativo?» y, más adelante, **correr también solo hasta antes del cierre** (min. 0 a 4). | Respuesta en B.12. La corrida con la partición alterna queda para después del envío. | ⏳ posterior | THE-44 |
| C6 | **Coherencia de GCN**: si no pasa la compuerta no puede aparecer como la mejor en test sin explicación (min. 6 a 44). | Lámina explícita y orden «tabla completa, filtro, candidatos». Respuestas en B.1 a B.3. | ⏳ | THE-43 |
| C7 | Láminas: leyenda de colores, qué es la banda, qué son los puntos, redacción de H1, H2 y H3, Shapley mal graficado (min. 14, 45, 77 a 85). | Lista de verificación en la sección D. | ⏳ | THE-43 |
| C8 | Las hipótesis «así están un poco raras»; hay que ver cómo se plantean (min. 79). | Sigue abierta la tarea de pasar a una sola hipótesis. | ⏳ | THE-16 |
| C9 | Saber argumentar cada decisión. «No debería suceder que sea: ni se me ocurrió» (min. 3). | Banco de respuestas de la sección B. | 🔄 | THE-45 |

### A.1 Escenarios 1:100 y 1:200 (C1)

Lo que dijo Cristian (min. 91, depurado): «Si la estabilidad sigue bien, entonces hay que
romperla. No se va a romper submuestreando, por lo que ya vimos. Entonces hay que romperla
sobremuestreando, o incluso, ya con este conocimiento, tocando a los fraudes, sabiendo que
estamos forzando a que la explicación caiga. Pero bueno, primero sobremuestrear.»

En el modo C el escenario nativo (1:38,4) ya usa **todos** los negativos de entrenamiento
(132.803). Submuestrear negativos solo acerca las clases. Para pasar de 1:38 a 1:100 hay dos
caminos, y son los dos que mencionó:

| Camino | Cómo queda el entrenamiento | Costo | Qué confunde |
|---|---|---|---|
| **Quitar ilícitas** («tocar a los fraudes») | 1:100 = 1.328 ilícitas / 132.803 negativos. 1:200 = 664 / 132.803. | El mismo de un escenario normal (1 h 45 min cada uno). Sin datos sintéticos. | Cambian a la vez la razón y la cantidad de fraude (queda el 38 % y el 19 %). |
| **Sobremuestrear negativos** con SMOTE | 1:100 = 3.462 / 346.200 (213.397 negativos sintéticos). 1:200 = 3.462 / 692.400. | El grafo pasa de 203 mil a 417 mil y 763 mil nodos. Es probable que GAT no quepa en 8 GB. Requiere código nuevo. | Más de la mitad de los negativos sería sintética. |

**Decisión tomada para arrancar:** se corre primero el camino de quitar ilícitas, que no
requiere código nuevo ni datos sintéticos y cabe en la GPU. Cristian dijo «primero
sobremuestrear»; se invierte el orden por costo y se le informa. Si la estabilidad no cae ni
con 664 ilícitas, la variante con negativos sintéticos pierde interés. Si cae, se corre la
segunda para separar el efecto de la razón del efecto de tener menos fraude.

Estos dos escenarios rompen a propósito la regla de diseño del 28-sep («ningún escenario quita
ilícitas»). Por eso van aparte, como escenarios de estrés, y no se mezclan con los cuatro
principales en las pruebas de H1.

- Código: `src/data/imbalance.py` (`1:100_subil`, `1:200_subil`, `V4_STRESS_SCENARIOS`),
  `configs/experiment_v4.yaml`, `tests/test_data_v4.py::test_b2b_stress_scenarios`.
- Corrida: `scripts/v4/run_estres.sh`. Registros en `runs_v4/EST_*.log`.

### A.2 Explicador adicional (C2)

Lo que dijo (min. 90, depurado): «Cada vez que hay un explicador que funciona mal y el motivo
es que no se debió haber escogido, debería haber uno que lo compense. La idea era verlo a
través de distintos explicadores y la conclusión es que no se puede usar PGExplainer; entonces
no están pudiendo concluir el efecto del explicador.» Aclaró que todo lo complementario es
válido, en el cuerpo o en anexos, mientras no cambie el flujo.

**Propuesta: Integrated Gradients** (Sundararajan et al. 2017).

| Criterio | Por qué lo cumple |
|---|---|
| Familia distinta | GNNExplainer optimiza una máscara, PGExplainer entrena una red, Shapley muestrea coaliciones. Falta un método de gradientes. |
| Funciona en Elliptic | Atribuye sobre las 165 variables del nodo, donde está la señal. No depende de las aristas (el 93 % de las ilícitas tiene grado ≤ 2). |
| Respaldo | Está en los bancos de prueba de explicabilidad en grafos (Agarwal et al. 2023). |
| Costo | Sin dependencias nuevas. Unos 50 pasos hacia atrás por nodo sobre el subgrafo exacto que ya usa el pipeline. |

Punto a cuidar: con una línea base fija, Integrated Gradients es determinista y su estabilidad
entre réplicas vale 1 por construcción. Para que las 5 réplicas midan algo, la línea base se
muestrea en cada réplica de las transacciones lícitas de entrenamiento (Expected Gradients;
Erion et al. 2021). Se reportan las dos versiones y, para la determinista, la estabilidad
entre semillas del modelo.

Se descartaron: GraphMaskExplainer y PGM-Explainer (trabajan sobre aristas o nodos vecinos,
con el mismo problema de PGExplainer), GraphLIME (necesita vecindarios grandes) y SubgraphX
(costo alto y otra dependencia).

## B. Banco de respuestas: lo que no supimos contestar

Cada punto trae la pregunta, qué pasó en la reunión y la respuesta que se debe poder dar sin
leer. Las cifras son del escenario nativo, media de 3 semillas.

### B.1 ¿Cuál arquitectura no pasa la compuerta?

**Qué pasó.** Se respondió que era GAT, por la memoria. Cristian tuvo que insistir en que era
GCN (min. 7 a 12).

**Respuesta.** Es **GCN**: ninguna de sus 12 configuraciones pasa. Lo de GAT es otro asunto:
por el límite de 8 GB corre con 4 cabezas y 2 capas, pero aprende y pasa la compuerta.

### B.2 ¿Por qué la compuerta se evalúa en validación y no en test?

**Qué pasó.** Se respondió con datos del dataset (periodos, conteos, el cierre del mercado).
Cristian pidió el principio general, válido para cualquier dataset (min. 24 a 37).

**Respuesta.** Por sesgo de selección.

1. El test es la única prueba que nadie ha tocado. Si se usa para escoger modelos, deja de ser
   una prueba honesta y lo que se reporta incluye la suerte de los que salieron bien.
2. Escoger en validación también tiene sesgo, pero deja el test intacto. Que un modelo sea el
   mejor en validación no implica que lo sea en test, y eso es lo que hace justo el reporte.
3. La compuerta pregunta si el modelo aprendió algo que valga la pena explicar. Eso se
   contesta en el periodo que el modelo vio. El test contesta otra cosa: si lo aprendido sigue
   sirviendo cuando la población cambia.

En este dataset hay una razón adicional (el cierre del mercado oscuro en t = 43), pero el
argumento principal no depende de ella.

### B.3 Si GCN no pasa, ¿por qué en test aparece por encima de GAT?

**Qué pasó.** No se supo responder. Alejandro dio la respuesta correcta al final (min. 39) y
Cristian la completó con la incertidumbre.

**Respuesta.** Porque en test todas quedan cerca del azar, y cerca del azar el orden lo decide
la semilla.

| Arquitectura (pérdida de la lámina) | ROC-AUC val | PR-AUC val | ROC-AUC test | PR-AUC test |
|---|---|---|---|---|
| GCN (focal) | 0,719 ± 0,038 | 0,105 ± 0,026 | 0,601 ± **0,132** | 0,012 ± 0,012 |
| GraphSAGE (pesos por clase) | 0,923 ± 0,003 | 0,462 ± 0,020 | 0,711 ± 0,013 | 0,010 ± 0,000 |
| GAT (focal) | 0,885 ± 0,001 | 0,332 ± 0,030 | 0,476 ± 0,085 | 0,005 ± 0,001 |
| TAGCN (pesos por clase) | 0,869 ± 0,064 | 0,299 ± 0,139 | 0,667 ± 0,032 | 0,008 ± 0,001 |

Azar: PR-AUC 0,024 en validación y 0,0057 en test; ROC-AUC 0,5.

- En validación GCN es la peor por un margen claro en las dos métricas.
- En test el ROC-AUC de GCN (0,60) supera al de GAT (0,48), pero su desviación entre semillas
  es la más grande de la tabla (0,13): va del peor modelo al mejor según la semilla. Eso no es
  aprendizaje.
- Un modelo que no aprendió el periodo de entrenamiento no tiene nada que perder cuando la
  población cambia. Los que sí aprendieron pierden lo aprendido.
- La incertidumbre grande de GCN en test confirma que filtrar en validación fue lo correcto.

Con pesos por clase en las cuatro (lo que pidió en C4) el cruce casi desaparece: en test GCN
0,635 ± 0,067 y GAT 0,611 ± 0,044.

### B.4 ¿Qué periodo es validación, cuál es test y dónde cae el cierre?

**Qué pasó.** Hubo confusión sobre cuál era «el periodo raro» (min. 30 a 35).

**Respuesta.** Entrenamiento t = 1 a 34. Validación t = 35 a 42 (914 ilícitas). Test t = 43 a
49 (169 ilícitas). El cierre del mercado oscuro ocurre en t = 43 (Weber et al. 2019), o sea
que **el periodo que cambia es el test**. Validación es anterior al cierre.

### B.5 ¿Qué es la banda de las curvas y qué son los puntos?

**Qué pasó.** Se dudó (min. 13 a 16).

**Respuesta.** La línea es la curva media de las 3 semillas. La banda va del mínimo al máximo
de las 3 semillas en cada punto del eje. Cada punto marcado es el **umbral calibrado** de una
semilla: el punto de operación con el que se reportan F1, precisión y recall. La curva no es
un modelo: son todos los clasificadores que salen de mover el umbral. F1 y KS solo existen
después de fijar un umbral.

### B.6 ¿Por qué cada arquitectura aparece con una pérdida distinta?

**Qué pasó.** Se dijo que a cada arquitectura «se le enfoca más el peso por la forma en que
entrena» (min. 5). No es así.

**Respuesta.** La lámina mostraba, para cada arquitectura, la pérdida con mejor PR-AUC de
validación. Es una elección de presentación, no una propiedad de la arquitectura. Cristian
pidió cambiarla: pesos por clase en todas (C4).

### B.7 ¿Cómo se lee una curva ROC?

**Qué pasó.** Cristian tuvo que explicarlo (min. 55 a 72).

**Respuesta.**
- El modelo entrega una probabilidad entre 0 y 1. Al fijar un umbral se vuelve clasificador.
- Cada umbral da una tasa de verdaderos positivos (recall) y una tasa de falsos positivos. La
  curva ROC es el recorrido de ese par al mover el umbral.
- La **diagonal** es el azar: las dos tasas suben a la par y el modelo no separa las clases.
- Una curva que queda por encima de otra en todos los umbrales es mejor que la otra. Eso no
  dice si es buena: para eso hay que escoger un umbral y mirar F1 o recall.
- Un ROC-AUC **menor que 0,5** no es «peor que el azar»: el modelo ordena al revés. Si se
  invirtieran sus etiquetas quedaría por encima de 0,5. En el test nos pasa en 11 de los 36
  modelos del 1:20, porque la relación entre las variables y el fraude cambió con el cierre.
- El clasificador perfecto llega a la esquina superior izquierda (área 1).

### B.8 ¿Por qué se reportan juntas ROC y PR? ¿Qué mide KS?

**Respuesta.** La ROC mide la separación global entre las dos clases, pero no ve el
desbalance: con muchos negativos, una tasa pequeña de falsos positivos son muchos casos. La
curva PR sí lo ve, porque la precisión cae con cada falso positivo. Por eso el ROC-AUC de
validación es 0,92 y el PR-AUC 0,46 para el mismo modelo. KS es la distancia máxima entre la
tasa de verdaderos positivos y la de falsos positivos: dice qué tan separadas quedan las dos
clases en el mejor punto de corte. Cristian aclaró que un PR-AUC por debajo de los de la
literatura de frontera no es un problema para este trabajo, porque el interés es la
explicación y no la marca de rendimiento.

### B.9 ¿Por qué pesos por clase rinde mejor que focal loss?

**Qué pasó.** Cristian dijo que no era lo esperado y que para discutirlo había que mirar la
matemática (min. 47).

**Respuesta.** Por cuánto compensa cada una.
- **Pesos por clase** usa w_c = N / (2 N_c). En el nativo, w de ilícita = 19,7 y w de negativo
  = 0,51. La razón entre los dos es 38,4, exactamente el desbalance. La pérdida resultante es
  ½ · (promedio sobre ilícitas) + ½ · (promedio sobre negativos): cada clase pesa lo mismo.
- **Focal loss** es FL = −α_t (1 − p_t)^γ log p_t, con α = 0,75 y γ = 2. El factor α da a la
  ilícita 3 veces el peso del negativo (0,75 / 0,25), no 38. El resto lo debe hacer el factor
  (1 − p_t)^γ, que baja el peso de los ejemplos fáciles pero no distingue clases.
- En el nativo la diferencia entre compensar 38 a 1 y compensar 3 a 1 es grande, y pesos por
  clase gana. En 1:10 la diferencia es 10 a 1 contra 3 a 1, y las curvas se acercan. Es lo que
  Cristian vio en la lámina.
- Focal loss se propuso para detección de objetos con α = 0,25 y γ = 2 ajustados juntos (Lin
  et al. 2017). Aquí α quedó fijo y no se buscó: es una limitación que hay que declarar.

### B.10 ¿Por qué GNNExplainer no se mueve con la pérdida y PGExplainer sí?

**Qué pasó.** Se respondió «por cómo explica cada explicador». Cristian: «eso no lo pueden
responder así» (min. 81). La explicación de Juan Diego sobre PGExplainer (aristas, nodos con
dos aristas) sí le gustó (min. 84).

**Respuesta.** Por el objeto sobre el que se mide el orden.
- GNNExplainer ordena **165 variables**. Dos réplicas con ruido distinto mueven poco un orden
  de 165 elementos, y la pérdida con la que se entrenó el modelo no cambia qué variables
  llevan la señal.
- PGExplainer ordena **aristas**. El 83 % de los nodos explicados tiene 2 aristas o menos. Con
  2 aristas el Spearman entre réplicas solo puede valer +1 o −1 (pasa en el 93 % de los
  pares). Además las máscaras se saturan en 0 o en 1, y cuál arista queda arriba lo decide el
  ruido numérico.
- La prueba de que es el instrumento y no la pérdida: el mismo modelo explicado dos veces dio
  0,04 y 0,42.

Detalle completo en `results_v4/reunion_0410/pgexpl_gcn/REPORTE.md`.

### B.11 ¿Qué mide Shapley aquí y por qué da 0,99?

**Qué pasó.** No quedó claro por qué no estaba bien en la lámina ni qué mide (min. 85).

**Respuesta.** Mide la contribución de cada una de las 165 variables del nodo a la predicción,
promediando sobre coaliciones de variables muestreadas al azar. No es el GNNShap publicado, que
atribuye sobre aristas. Da 0,99 porque el promedio sobre muchas coaliciones reduce el ruido por
construcción.

### B.12 ¿Por qué se siguió con un test que se sabe que no es representativo?

**Qué pasó.** Fue la pregunta con la que abrió Cristian (min. 0 a 4) y pidió tener la
respuesta lista.

**Respuesta.**
1. La partición temporal es la de la literatura de Elliptic y la única honesta para datos
   transaccionales: se entrena con el pasado y se prueba con el futuro.
2. La estabilidad de las explicaciones se mide en validación (t = 35 a 42), que es posterior al
   entrenamiento y anterior al cierre. El estudio ya mide la explicación en el periodo
   consistente.
3. El test se conserva y se reporta porque es lo que le habría pasado a un modelo en
   producción: funcionaba, el mercado cambió y dejó de servir. Esa caída no depende de la
   arquitectura, de la pérdida ni de cuánto se optimice (el candidato con 150 trials cae igual).
4. Lo que falta: una partición completa antes del cierre, para tener una prueba intacta en el
   periodo consistente (THE-44). Cristian anticipó tres desenlaces: las mismas conclusiones;
   las mismas con bandas más estrechas; o un resultado distinto, y en ese caso la historia es
   cómo se refleja el cambio de mercado en la explicación.

## C. El porqué de cada hipótesis

Cristian (min. 82, depurado): «Estas tres hipótesis no pueden progresar si eso no está claro,
y el porqué tiene que estar muy bien argumentado y explicado con las ecuaciones, la misma
arquitectura, su funcionalidad, sus puntos débiles y sus puntos fuertes. No puede ser solo un
número.»

### C.1 H1: por qué el desbalance no movió la estabilidad

> Esta sección se corrigió el 10-oct después de validar cada afirmación
> (`results_v4/reunion_0710/VALIDACION.md`). La primera versión daba por hecho un mecanismo que
> los datos solo respaldan en parte.

**Lo que se sostiene.** Con pesos por clase la función de pérdida es la misma en todos los
escenarios principales: ½ · promedio(ilícitas) + ½ · promedio(negativos). Se comprobó
numéricamente que es exactamente lo que calcula PyTorch. Al submuestrear negativos solo cambia
la muestra con la que se estima el segundo promedio. Como además los modelos de una misma
semilla parten de los mismos pesos iniciales y usan los mismos hiperparámetros, terminan casi
idénticos: el coseno entre sus pesos es de 0,90 a 1,00 (entre dos semillas es de 0,02) y sus
puntajes sobre validación tienen correlación de 0,96 a 0,99. Por eso la estabilidad coincide al
tercer decimal entre escenarios:

| Arquitectura | Nativo | 1:10 | 1:10 con SMOTE | 1:20 |
|---|---|---|---|---|
| GraphSAGE | 0,964 | 0,964 | 0,965 | 0,964 |
| TAGCN | 0,937 | 0,936 | 0,937 | 0,937 |
| GAT | 0,939 | 0,939 | 0,941 | 0,939 |

**La evidencia directa.** Se comparó la explicación del mismo nodo entre modelos
(GNNExplainer; GraphSAGE, GAT y TAGCN):

| Comparación | Pesos por clase | Sin ajuste | Focal loss |
|---|---|---|---|
| Mismo escenario, otra semilla (referencia) | 0,940 | 0,936 | 0,924 |
| Otro escenario y otra semilla | 0,941 | 0,918 | 0,915 |
| Otro escenario, misma semilla | 0,987 | 0,924 | 0,925 |

Cambiar el escenario no mueve la explicación más de lo que ya la mueve cambiar la semilla. El
0,99 de la última fila no es una propiedad del desbalance: es que esos modelos son casi el
mismo modelo.

**Lo que no se sostiene.** La teoría dice que submuestrear negativos al azar solo desplaza el
logit del clasificador óptimo en una constante, log(1/β) (Elkan 2001; Dal Pozzolo et al. 2015).
En los modelos sin ajuste eso **no se observa**: entre el nativo y el 1:10 el desplazamiento
mediano es de −0,4 a 0,2 (la teoría predice +1,34) y la correlación de los puntajes es de 0,40
a 0,80. Esos modelos se detienen entre las épocas 17 y 59 y están lejos del óptimo. El
argumento del desplazamiento se puede citar como expectativa teórica, no como lo que pasó.

**Qué implica.** Con pesos por clase, H1 no podía fallar: los escenarios principales cambian
algo que casi no altera el modelo. Para que el desbalance afecte la explicación tiene que
cambiar lo que el modelo aprende del fraude. Es el argumento para los escenarios 1:100 y
1:200, que quitan ilícitas.

**Advertencia de diseño.** El 1:10 con SMOTE y el 1:20 usan exactamente los mismos negativos
reales (69.240, mismo sorteo). Se diferencian solo en las 3.462 ilícitas sintéticas.

**Cómo lo leyó Cristian** (min. 78): no como hipótesis, sino como decisión de diseño. Se
escogió no tocar el fraude porque es lo más representativo, y el resultado respalda que se
escogió bien. No se puede afirmar nada sobre un desbalance «más severo» con este rango.

### C.2 H2: por qué las tres arquitecturas que aprenden explican igual

La diferencia entre las cuatro está en cómo entra el nodo propio en cada capa:

| Arquitectura | Capa | ¿Peso propio para el nodo? |
|---|---|---|
| GCN | h_i = σ( Σ_{j ∈ N(i) ∪ {i}} W h_j / √(d_i d_j) ) | No. El nodo y sus vecinos comparten W y se promedian. |
| GraphSAGE | h_i = σ( W_1 h_i + W_2 · media_{j ∈ N(i)} h_j ) | Sí: W_1. |
| GAT | h_i = σ( Σ_{j ∈ N(i) ∪ {i}} α_ij W h_j ) | Sí, por la atención: α_ii se aprende. |
| TAGCN | h_i = σ( Σ_{k=0..K} (A^k H)_i W_k ) | Sí: el término k = 0 es el propio nodo con W_0. |

En Elliptic la señal está en las variables del nodo y los vecinos casi no ayudan: solo el 12 %
de los vecinos de una ilícita es ilícito, y un MLP sin grafo rinde igual que GraphSAGE (0,485
frente a 0,481). GCN mezcla al nodo con vecinos que no se le parecen y diluye la señal. Al
agregarle un peso propio sube de 0,214 a 0,464.

Las tres que conservan al nodo llegan al mismo techo de rendimiento. La arquitectura decide
**si el modelo aprende**.

**Lo que no se puede afirmar** (validación del 10-oct): que las tres «se apoyan en las mismas
variables». Entre arquitecturas, con el mismo escenario, la misma pérdida y la misma semilla,
el Jaccard de las 10 variables más importantes es de 0,20 a 0,50 (comparten de 3 a 7 de las
10; corregido el 10-oct, antes decía «de 2 a 5»). El
Spearman sobre las 165 variables da 0,84 a 0,97, pero ese valor está inflado por la razón que
se explica en la sección G. Tampoco es cierto que «expliquen igual» con una medida que no se
infle: en GNNExplainer la estabilidad sobre las variables distintas de 0 es 0,84 en GraphSAGE,
0,73 en GAT y 0,68 en TAGCN.

Cristian encontró interesante la frase de GCN (min. 80) pero dijo que así no se entiende:
necesita la ecuación. Detalle en `results_v4/reunion_0410/pgexpl_gcn/REPORTE.md`, parte B.

### C.3 H3: por qué la pérdida no mueve a GNNExplainer y sí a PGExplainer

Ver B.10. En una frase para la lámina: la pérdida cambia el umbral y la calibración del
modelo, no las variables que usa; GNNExplainer y Shapley miden variables y no lo notan;
PGExplainer mide aristas en nodos con dos aristas y lo que registra es ruido del instrumento.

Cristian leyó la lámina al revés (min. 80): entendió que el efecto estaba en GNNExplainer. Hay
que reescribirla.

## D. Correcciones de láminas (THE-43)

| Lámina | Qué dijo Cristian | Corrección |
|---|---|---|
| Resumen nativo (ROC de las cuatro) | «¿Por qué pusieron esas cuatro?» «¿Esto es a una semilla?» | Pesos por clase en las cuatro. Decir en la lámina: media de 3 semillas, banda mínimo a máximo. |
| Tabla de métricas | Esperaba no ver a GCN después del filtro. | Orden: tabla completa, compuerta, candidatos. La tabla con GCN es la previa al filtro o va a anexo. |
| Curvas por escenario | «En la slide no está la información más importante» (los colores). | Leyenda grande dentro de la lámina. «Sin ajuste» presentado como referencia. |
| Curvas 1:10 | «Sin balanceo» no es exacto: el submuestreo ya es un balanceo. | Renombrar a «sin ajuste de la pérdida». Explicar por qué las curvas se acercan. |
| Curvas de test | El test «es su resultado de verdad». | Mantenerlas. Agregar la nota sobre ROC-AUC bajo 0,5. |
| H1 | No entendió «El rango es estrecho: se conserva todo el fraude y el modelo aprende casi lo mismo». | Reescribir como decisión de diseño respaldada por el resultado. Sin afirmar nada sobre desbalance «más severo». |
| H2 | «Esperábamos que la arquitectura cambiara la estabilidad» está raro. La frase de GCN necesita detalle. | Reescribir el enunciado. Agregar la tabla de capas de C.2. |
| H3 | Mal redactada: se entiende al revés. Shapley no está bien graficado. | Reescribir. Corregir la gráfica para que se vean los tres explicadores. |
| Nueva | Pidió el caso 1:100 y el explicador extra. | Una lámina por cada uno cuando haya resultados. |

## E. Dos hallazgos al preparar este documento

### E.1 GAT con pesos por clase no pasa la compuerta actual

Cristian pidió usar pesos por clase en todas las arquitecturas. Con la compuerta actual (F1 ≥
0,30 y MCC ≥ 0,15 con el umbral 0,5) eso deja a GAT fuera:

| Configuración (pesos por clase) | PR-AUC val | Pasa con umbral 0,5 | Pasa con umbral calibrado |
|---|---|---|---|
| GraphSAGE | 0,44 a 0,46 | 4 de 4 escenarios | 4 de 4 |
| TAGCN | 0,29 a 0,30 | 4 de 4 | 4 de 4 |
| GAT | 0,32 a 0,35 | **0 de 4** | 4 de 4 |
| GCN | 0,10 a 0,17 | 0 de 4 | 0 de 4 |

La causa es el umbral, no el modelo. Con pesos por clase las probabilidades quedan desplazadas
hacia arriba y el umbral calibrado de GAT está en 0,87. Con 0,5 marca demasiados positivos y
el F1 se hunde, aunque el modelo ordena bien (ROC-AUC 0,90).

**Recomendación:** evaluar la compuerta con el F1 y el MCC del umbral calibrado, que es el
punto que ya se marca en las curvas y con el que se reporta el F1. Queda la pista de pesos por
clase con las tres arquitecturas que aprenden y GCN fuera. La variante ya está calculada
(`cfg_calibrada_sin1:1` en `results_v4/reunion_0410/configs48/`). Es una decisión que deben
tomar los autores con Cristian (THE-42), porque cambia el conteo de configuraciones que pasan.

### E.2 Con pesos por clase la estabilidad no cambia entre escenarios al tercer decimal

Es la tabla de C.1. Refuerza el mecanismo de H1 y explica por qué Cristian tiene razón al decir
que submuestreando no se va a romper.

### E.3 La medida de estabilidad está inflada por las variables en cero

Ver la sección G. Es el hallazgo más importante de la validación.

## F. Orden de ejecución

| Orden | Tarea | Depende de | Estado |
|---|---|---|---|
| 1 | Corrida de estrés 1:100 y 1:200 (THE-39) | GPU | 🔄 |
| 2 | Explicador adicional: código y prueba (THE-40) | CPU, mientras corre la 1 | ⏳ |
| 3 | Explicador adicional sobre todos los modelos (THE-40) | GPU libre | ⏳ |
| 4 | Análisis de los seis niveles de desbalance y acuerdo entre escenarios (THE-39, THE-41) | 1 y 3 | ⏳ |
| 5 | Láminas corregidas y dos láminas nuevas (THE-43) | 4 y decisión E.1 | ⏳ |
| 6 | Texto del porqué para la tesis (THE-41) | 4 | ⏳ |
| 7 | Partición antes del cierre (THE-44) | después del envío | ⏳ |

## G. Validación del 10-oct: la medida de estabilidad

Detalle completo en `results_v4/reunion_0710/VALIDACION.md`.

1. **El 46 % de las variables de los 30 nodos explicados vale exactamente 0.** Son variables
   iguales a su mediana, que el escalado robusto lleva a 0. Un explicador casi no puede
   atribuirle nada a una variable en 0, así que queda al fondo del orden en cualquier modelo.
2. **Solo ese bloque de ceros ya da un Spearman de 0,74 a 0,82** entre dos órdenes al azar.
3. **Una red sin entrenar da casi la misma estabilidad** que la entrenada (control de pesos al
   azar, la prueba de Adebayo et al. 2018). La explicación del modelo entrenado coincide en
   0,88 a 0,96 con la de la misma arquitectura con pesos al azar.
4. **Con medidas que no se inflan el nivel baja.** En GNNExplainer, de 0,95 a 0,75 (Spearman
   sobre las variables distintas de 0) y a 0,62 (coincidencia de las 10 variables más
   importantes). Entre semillas del modelo, las 10 más importantes coinciden en 0,39.
5. **Qué conclusiones aguantan.** H1 aguanta con las tres medidas: el escenario no agrega nada
   a la semilla. H2 y H3, con GNNExplainer y la medida sobre variables distintas de 0, dejan
   de verse planas (por arquitectura 0,68 a 0,84; por pérdida 0,59 a 0,75).

Esto explica de raíz por qué todo salió estable, que era la pregunta de Cristian, y a la vez
obliga a decidir con qué medida se reportan los resultados.

## Decisiones que necesitan respuesta

1. **Compuerta con umbral calibrado** para poder converger a pesos por clase (E.1).
2. **Orden de los caminos al 1:100.** Se arrancó quitando ilícitas. ¿Se corre también la
   variante con negativos sobremuestreados o basta con declararla?
3. **Deck de hipótesis.** Otra sesión publicó las versiones 6 y 7. Hay que confirmar que ya no
   se está editando antes de aplicar la sección D.
4. **Medida de estabilidad** (sección G). ¿Se mantiene el Spearman sobre las 165 variables
   declarando su piso, o se reportan también la medida sobre variables distintas de 0 y la
   coincidencia de las 10 más importantes?

## Referencias

- Adebayo, J. et al. (2018). Sanity checks for saliency maps. NeurIPS.
- Agarwal, C. et al. (2023). Evaluating explainability for graph neural networks. Scientific Data, 10, 144.
- Dal Pozzolo, A. et al. (2015). Calibrating probability with undersampling for unbalanced classification. IEEE Symposium Series on Computational Intelligence.
- Du, J. et al. (2017). Topology adaptive graph convolutional networks. arXiv:1710.10370.
- Elkan, C. (2001). The foundations of cost-sensitive learning. IJCAI.
- Erion, G. et al. (2021). Improving performance of deep learning models with axiomatic attribution priors and expected gradients. Nature Machine Intelligence, 3, 620-631.
- Hamilton, W. et al. (2017). Inductive representation learning on large graphs. NeurIPS.
- Kipf, T. y Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR.
- Lin, T.-Y. et al. (2017). Focal loss for dense object detection. ICCV.
- Sundararajan, M. et al. (2017). Axiomatic attribution for deep networks. ICML.
- Veličković, P. et al. (2018). Graph attention networks. ICLR.
- Weber, M. et al. (2019). Anti-money laundering in Bitcoin: experimenting with graph convolutional networks for financial forensics. KDD Workshop on Anomaly Detection in Finance. arXiv:1908.02591.
