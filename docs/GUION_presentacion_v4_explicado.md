# Guion explicado de las presentaciones v4

Este documento acompaña los dos decks publicados:

- **Deck principal** («Estabilidad XAI en GNNs», 12 láminas): contexto, marco teórico y metodología.
- **Deck de hipótesis** («Hipótesis v4»): rendimiento, H1, H2, H3, calidad y plan. Ya tiene 26
  láminas: entraron dos de curvas del escenario 1:20 (láminas 13 y 14) y de ahí en adelante la
  numeración de este guion corre dos lugares.

Para cada lámina hay tres partes:

1. **Qué decir.** Un discurso para leer en voz alta, de 45 segundos a 1 minuto y medio.
2. **Qué hay detrás.** La explicación desde la raíz: de dónde sale lo que se ve, cómo se hizo,
   qué es cada técnica y por qué se escogió. No hay que decirlo todo; es para entenderlo.
3. **Si preguntan.** Las preguntas probables del profesor o del jurado, con una respuesta corta.

Las cifras salen de `results_v4/reunion_0410/` (configs48, curvas, pgexpl_gcn y sin_etiqueta),
de `results_phase1_v4/` y del código en `src/` y `configs/experiment_v4.yaml`.

---

> **Actualización del 10-oct.** Después de la reunión del 7 de octubre se agregaron dos niveles
> extremos de desbalance (1:100 y 1:200), dos explicadores de gradientes y una prueba sobre la
> propia medida de estabilidad. Las láminas 10, 11 y 12 de la Parte 1 traen abajo un bloque
> «Actualización» con lo que hay que añadir al discurso. La **Parte 2 quedó reemplazada** por
> `docs/GUION_hipotesis_v8_explicado.md`, que sigue el deck de hipótesis de 36 láminas.

## La historia completa en un minuto

Si solo se entiende esto, se entiende la tesis:

1. Los bancos usan modelos para marcar transacciones sospechosas. Cuando un modelo marca una,
   alguien tiene que justificar por qué. Para eso existen los **explicadores**: algoritmos que
   dicen qué información pesó en la decisión.
2. Los explicadores tienen una parte aleatoria. Si al repetirlos dan una razón distinta cada vez,
   esa razón no sirve ante un auditor. Que la razón se repita es lo que llamamos **estabilidad**.
3. En el lavado de dinero las transacciones ilícitas son muy pocas (el **desbalance**). La
   pregunta de la tesis es si ese desbalance vuelve inestables las explicaciones.
4. Usamos Elliptic, un grafo real de transacciones de Bitcoin. Entrenamos 4 tipos de red
   neuronal de grafos, con 4 niveles de desbalance y 3 formas de compensarlo: 48
   **configuraciones**, cada una repetida 3 veces (3 **semillas**).
5. Solo explicamos las configuraciones que aprenden de verdad (pasan una **compuerta**): 20.
   Las explicamos con 3 explicadores, 5 veces cada uno, y medimos cuánto coinciden.
6. Resultados:
   - El desbalance **no** cambia la estabilidad.
   - Entre las arquitecturas que aprenden, la arquitectura **tampoco**.
   - La forma de compensar el desbalance **tampoco**, en GNNExplainer y en Shapley. En
     PGExplainer la estabilidad no se puede medir bien en este grafo.
   - Lo más importante: **estable no es lo mismo que correcto**. Un explicador puede repetir
     siempre la misma razón y que esa razón sea mala. Por eso la estabilidad hay que reportarla
     junto con el rendimiento, la plausibilidad y la fidelidad.

---

## Diccionario rápido

| Término | En palabras simples |
|---|---|
| Transacción (nodo) | Un pago de Bitcoin. En el grafo es un punto. |
| Flujo (arista) | La flecha entre dos transacciones: el dinero que sale de una entra a la otra. |
| Grafo | El conjunto de puntos y flechas: la red de pagos. |
| Variables (features) | Los datos que describe cada transacción. Elliptic trae 166 columnas: el paso de tiempo y 165 descriptivas. Están anonimizadas: no sabemos qué significa cada una. |
| GNN (red neuronal de grafos) | Un modelo que decide sobre cada punto mirando sus datos y los de sus vecinos. |
| Etiqueta | Lo que se sabe de la transacción: ilícita, lícita o sin etiqueta (nadie la revisó). |
| Clase positiva / negativa | Lo que buscamos (ilícitas) / todo lo demás. |
| Desbalance 1:40 | Por cada ilícita hay unas 40 negativas. |
| Submuestreo | Quitar negativos al azar para que la proporción sea menos extrema. |
| Sobremuestreo, SMOTE | Agregar ilícitas. SMOTE las crea «a mitad de camino» entre dos ilícitas reales parecidas. |
| Función de pérdida | La regla con la que el modelo mide qué tan mal lo hizo y aprende a corregirse. |
| Pesos por clase (class weighting) | Que equivocarse con una ilícita cueste más que equivocarse con una lícita. |
| Focal loss | Que los casos fáciles pesen poco y el modelo se concentre en los difíciles. |
| Semilla | El número que fija el azar del entrenamiento. Otra semilla, otro entrenamiento con los mismos datos. |
| Configuración | Una combinación de arquitectura, escenario y pérdida. Se entrena con 3 semillas. |
| Hiperparámetros | Los ajustes del modelo que no aprende solo (tamaño, velocidad de aprendizaje). |
| Época | Una pasada completa del modelo por los datos de entrenamiento. |
| Entrenamiento / validación / test | Pasado con el que se aprende / período siguiente para ajustar y elegir / futuro que se usa solo para evaluar. |
| Umbral | El puntaje desde el cual el modelo alerta. |
| ROC-AUC | Probabilidad de que una ilícita al azar reciba un puntaje más alto que una lícita al azar. 0,5 es tirar una moneda. |
| PR-AUC | Qué tan limpias son las alertas cuando se intenta detectar más fraude. El azar es la proporción de ilícitas (2,4 % en validación). |
| F1 | Equilibrio entre detectar muchas ilícitas y no dar falsas alarmas, en el umbral elegido. |
| MCC | Una medida de acierto que no se deja engañar por el desbalance. 0 es azar. |
| KS | La mayor distancia entre el porcentaje de ilícitas detectadas y el de lícitas alertadas. |
| Compuerta | El filtro para explicar solo lo que aprendió: F1 ≥ 0,30 y MCC ≥ 0,15 en validación. |
| Explicador | Algoritmo que señala qué variables y qué flujos pesaron en una decisión. |
| Máscara | La lista de «perillas» de 0 a 1 que un explicador asigna a cada variable o flujo: cuánto importó. |
| Estabilidad | Cuánto coinciden dos ejecuciones del explicador sobre la misma alerta. |
| Spearman | Compara dos órdenes: 1 es el mismo orden, 0 nada que ver, −1 el orden al revés. |
| Jaccard | Compara dos listas: elementos en común sobre el total de elementos distintos. |
| Plausibilidad | Si el explicador señala el patrón de lavado verdadero (solo medible en el grafo sintético). |
| Fidelidad | Si quitar lo que el explicador señala cambia de verdad la decisión del modelo. |
| TOST | Prueba de equivalencia: demuestra que una diferencia es menor que un margen fijado antes (±0,05). |
| Friedman | Prueba que compara tres o más grupos medidos sobre los mismos casos. |
| p-valor | La probabilidad de ver una diferencia así por puro azar. Si es menor que 0,05 se considera que hay efecto. |

---

# Parte 1. Deck principal

## Lámina 1. Portada

**Qué decir**

> Buenos días. Somos Juan Diego Garzón y Alejandro Gómez, y presentamos nuestra tesis de
> maestría, dirigida por el profesor Cristian Rosero. Estudiamos qué tan confiables son las
> explicaciones que dan las redes neuronales de grafos cuando se usan para detectar lavado de
> dinero, y en particular si esa confianza se rompe cuando el fraude es muy escaso en los datos.
> Empecemos por el problema.

**Qué hay detrás**

El título tiene cuatro piezas, y conviene tenerlas claras desde el principio:

- **Estabilidad:** que la explicación se repita.
- **Métodos de explicabilidad (XAI):** los explicadores.
- **Graph Neural Networks:** el tipo de modelo.
- **Desbalance extremo:** muy pocas ilícitas frente a muchas negativas.

Toda la presentación es el camino que une esas cuatro piezas.

---

## Lámina 2. Contexto: la economía se mueve en transacciones

**Qué decir**

> Toda la economía se mueve en transacciones: pagos entre cuentas de bancos, de pasarelas de
> pago o de criptomonedas como Bitcoin. En Bitcoin el registro es público: se ve quién le paga a
> quién, aunque no se sepa quién es. Sobre esas transacciones aparecen lo que llamamos patologías
> económicas: el fraude, como el robo de identidad, y el lavado de dinero, que es hacer pasar
> dinero ilícito por legítimo. Según Naciones Unidas, en 2009 se lavaron cerca de 1,6 billones de
> dólares, casi el 3 % del producto mundial. Para vigilar esto, bancos y empresas usan
> aprendizaje automático, y entre esos modelos las redes neuronales de grafos se estudian cada vez
> más porque leen la red de pagos completa.

**Qué hay detrás**

- Esta lámina existe porque Cristian pidió una entrada para público general antes de hablar de
  GNN: primero el mundo (economía, banca, Bitcoin) y después el problema técnico.
- **Por qué Bitcoin:** es el único sistema de pagos grande donde la red de transacciones es
  pública. En un banco esos datos son privados; en Bitcoin cualquiera puede descargarlos. Por eso
  existen conjuntos de datos como Elliptic.
- **Fraude frente a lavado:** en el fraude alguien pierde dinero (lo engañan). En el lavado no
  hay víctima directa: el dinero ya es ilícito y se busca disfrazar su origen.
- La cifra de la ONU (UNODC, 2011) es la estimación clásica y se cita como orden de magnitud.

**Si preguntan**

- *¿Por qué redes de grafos y no un modelo normal?* Porque el lavado se reconoce en cómo se
  mueve el dinero entre transacciones, no en una transacción aislada. Se ve en la lámina
  siguiente.

---

## Lámina 3. El lavado es un patrón de red

**Qué decir**

> Esto es una red de transacciones: cada punto es una transacción y cada flecha, dinero que pasa
> de una a otra. A simple vista nada parece sospechoso. Pero si seguimos el dinero aparece un
> patrón: una transacción se reparte en cuatro, y esas cuatro vuelven a reunirse en una sola.
> Ninguna es rara por sí sola; lo que delata el lavado es la forma que dibujan juntas. Para
> detectar esas formas se usan redes neuronales de grafos: reciben la red completa y clasifican
> cada transacción como lícita o ilícita. Aquí la marca como ilícita: es una alerta. Y en un banco
> el analista tiene que justificar cada alerta. Para eso hay que ver cómo decide la red.

**Qué hay detrás**

- El dibujo es ilustrativo, no son datos reales. Muestra la técnica de lavado más conocida:
  **dispersar y reagrupar**. El dinero se parte en muchos pagos pequeños para no llamar la
  atención y después se vuelve a juntar.
- **La idea clave:** una transacción vista sola puede parecer normal (monto común, comisión
  común). El patrón solo aparece cuando se mira la red. Por eso se usan modelos que leen la red.
- La alerta lleva su etiqueta visible («ilícita»), porque Cristian pidió que el público entienda
  qué produce el modelo: no un número, sino una decisión.
- **Weber et al. (2019)** es el artículo que publicó Elliptic y probó por primera vez una GNN
  sobre esos datos.

---

## Lámina 4. Cómo decide una GNN

**Qué decir**

> Para saber qué puede explicar una red de grafos, primero hay que ver cómo decide. Queremos
> clasificar la transacción del centro. En la primera capa, la transacción recibe un resumen de
> sus vecinas directas y lo combina con sus propias variables. A eso se le llama paso de mensajes:
> los puntos que se mueven son esa información. En la segunda capa le llega también lo que sus
> vecinas recibieron de las suyas. Con dos capas, la decisión depende de lo que está a dos pasos;
> lo que queda más lejos no influye. Con esa mezcla el modelo calcula un puntaje y la marca como
> ilícita. Por eso explicar una decisión es señalar dos cosas: qué flujos de la red pesaron más y
> cuáles de sus 165 variables pesaron más.

**Qué hay detrás**

Cómo funciona una GNN, paso a paso:

1. Cada transacción empieza con sus 165 variables: números que la describen, como montos,
   comisiones, cantidad de entradas y salidas, y resúmenes de sus vecinas. En Elliptic vienen
   anonimizadas.
2. **Capa 1:** cada transacción junta la información de sus vecinas directas (las que le mandan
   o reciben dinero), la mezcla con la suya y obtiene una nueva descripción.
3. **Capa 2:** repite lo mismo. Como las vecinas ya traían información de sus propias vecinas,
   ahora la transacción «ve» hasta dos pasos de distancia.
4. Al final, la descripción pasa por una capa de decisión que da un puntaje de 0 a 1: qué tan
   ilícita parece.

**Por qué dos capas:** es lo habitual en la literatura y en nuestros modelos. Más capas hacen que
todo se mezcle con todo y la información del nodo se diluye.

Lo que se explica sale directo de este mecanismo. Como la decisión usa **flujos** (aristas) y
**variables**, un explicador puede decir «pesaron estos flujos» o «pesaron estas variables».

**Si preguntan**

- *¿166 o 165 variables?* Elliptic trae 166 columnas. Una es el paso de tiempo, que sirve para
  dividir los datos y no se usa para decidir. Quedan 165.

---

## Lámina 5. Qué es explicar

**Qué decir**

> Antes de medir explicaciones, qué es explicar. Imaginen que un banco cambia su modelo y empieza
> a alertar distinto: antes de bloquear clientes, alguien tiene que entender por qué. Tomemos dos
> modelos igual de buenos. El primero explica sus alertas así: clientes mayores con pagos
> inusuales, que es el patrón del robo de identidad. Eso tiene sentido de negocio y se puede
> defender ante un auditor. El segundo acierta igual, pero sus alertas dependen de una mezcla de
> variables sin relación aparente. No sabemos si acierta por la razón correcta. Lo que nos deja
> ver esa diferencia es un explicador: un algoritmo que señala qué variables y qué conexiones
> pesaron en cada decisión. Sin él, la red es una caja negra.

**Qué hay detrás**

- Es el ejemplo que dio Cristian en la reunión: **explicar no es dar una métrica** («este modelo
  tiene mejor ROC-AUC»). Es dar una razón que tenga sentido para el negocio.
- **Caja negra:** una red neuronal tiene miles de números internos (pesos). Ningún humano puede
  leerlos y entender la decisión. El explicador traduce eso a «estas variables, estos flujos».
- **Por qué importa en la práctica:**
  - *Auditoría:* un regulador puede exigir la razón de un bloqueo.
  - *Confianza:* si la razón no tiene sentido, el modelo puede estar acertando por casualidad.
  - *Venta del producto:* un modelo que se puede explicar se puede vender.

**Si preguntan**

- *¿Cuál es la diferencia entre explicar y que el modelo acierte?* Dos modelos pueden acertar lo
  mismo y decidir por razones distintas. La explicación dice cuál razón usó cada uno.

---

## Lámina 6. Qué es la estabilidad

**Qué decir**

> Ahora pidámosle a un explicador que justifique una alerta. Mismo modelo, misma transacción,
> mismo explicador. En una primera ejecución señala estos flujos como importantes. Pero los
> explicadores tienen una parte aleatoria, y si lo ejecutamos otra vez señala otros. Para medir
> cuánto coinciden contamos los flujos en común sobre el total: aquí coinciden 2 de 10, un 0,2.
> Esa coincidencia entre ejecuciones es lo que llamamos estabilidad: 1 si señalan lo mismo, 0 si
> no tienen nada en común. Ante un auditor, la misma alerta tiene que tener la misma razón.

**Qué hay detrás**

- **De dónde viene lo aleatorio en cada explicador:**
  - *GNNExplainer* arranca con perillas en valores al azar y las va ajustando.
  - *PGExplainer* entrena una pequeña red, que empieza con números al azar.
  - *Shapley* prueba combinaciones de variables escogidas al azar.

  Cambiar la semilla cambia ese punto de partida.
- **La medida del ejemplo es Jaccard:** elementos en común sobre elementos distintos en total.
  Sirve para listas de flujos («los 20 más importantes»).
- **Ojo con la coherencia:** las cifras de resultados (0,93 y similares) se miden con
  **Spearman**, que compara el **orden** completo de las 165 variables. Las dos son medidas de
  coincidencia. Jaccard se usa para flujos y Spearman para el orden de variables, como se explica
  en la lámina 12. Al presentar, decir «coincidencia» en general y dejar el detalle para esa
  lámina.

**Si preguntan**

- *¿Por qué no fijar la semilla y listo?* Porque fijarla esconde el problema, no lo resuelve.
  Si la razón cambia con la semilla, depende del azar, y otra persona con otra semilla obtendría
  otra razón.

---

## Lámina 7. Pregunta, hipótesis y objetivo

**Qué decir**

> De ahí sale nuestra pregunta: ¿cambia la estabilidad de las explicaciones según el desbalance
> con que se entrena el modelo? En lavado, las ilícitas son muy pocas, y eso cambia lo que el
> modelo aprende. Nuestra hipótesis es que sí: a mayor desbalance, menos estables las
> explicaciones. La alternativa es que no haya efecto: una línea plana. El diseño está hecho para
> distinguir entre las dos. El objetivo es medirlo sobre datos reales, Elliptic, con cuatro
> arquitecturas de red y tres explicadores.

**Qué hay detrás**

- **Por qué esperábamos que el desbalance importara:** con muy pocas ilícitas, el modelo tiene
  pocos ejemplos para aprender dónde está la frontera entre lícito e ilícito. Una frontera
  aprendida con pocos ejemplos es frágil, y una explicación de algo frágil debería moverse más al
  repetirla.
- Además hay antecedentes. Alarab y Prakoonwit (2022) mostraron en Elliptic, con un bosque
  aleatorio, que algunas técnicas de remuestreo cambian qué variables parecen importantes (no
  todas: en el conjunto de entrenamiento ninguna lo hizo de forma significativa).
- El dibujo es un **esquema**, no datos. Muestra las dos posibilidades antes de ver resultados:
  la línea que baja (la hipótesis) y la línea plana (sin efecto).
- Se usa «≈1:40» y no «1:38,4» porque Cristian pidió un número genérico en esta lámina: el
  decimal no aporta y genera una pregunta innecesaria.

**Si preguntan**

- *¿Por qué una sola hipótesis aquí si después hay H1, H2 y H3?* La pregunta central es el
  desbalance (H1). La arquitectura (H2) y la forma de compensar (H3) son los otros factores del
  diseño, y hay que descartarlos para que la respuesta a H1 sea limpia.

---

## Lámina 8. El conjunto de datos Elliptic

**Qué decir**

> Para medirlo usamos Elliptic: un grafo de transacciones reales de Bitcoin unidas por los flujos
> de pago, que publicaron en 2019 la empresa Elliptic, IBM y el MIT. Es el conjunto de referencia
> para detectar lavado con modelos de grafos, y lo escogimos porque trae lo que necesitamos: datos
> reales, la red, el orden en el tiempo y etiquetas. Son unas 200 mil transacciones. Solo el 2 %
> está marcado como ilícito y el 21 % como lícito; el 77 % no tiene etiqueta, nadie lo revisó.
> Respetamos el tiempo: entrenamos con el pasado, ajustamos con el período siguiente y evaluamos
> con el futuro, como pasaría en un banco.

**Qué hay detrás**

- **Composición:** 203.769 transacciones, 234.355 flujos y 49 pasos de tiempo (cada paso es más
  o menos dos semanas). Etiquetas: 4.545 ilícitas, 42.019 lícitas y 157.205 sin etiqueta.
- **División temporal:**

  | Parte | Pasos de tiempo | Ilícitas |
  |---|---|---|
  | Entrenamiento | 1 a 34 | 3.462 |
  | Validación | 35 a 42 | 914 |
  | Test | 43 a 49 | 169 |

- **Por qué dividir por tiempo y no al azar:** si se mezclan, el modelo aprende con datos del
  futuro y los resultados salen inflados. En un banco el modelo siempre se entrena con lo que ya
  pasó.
- **Dato que se usa después:** en el paso 43 cerró un mercado oscuro grande (lo reportan Weber
  y colegas en 2019) y el fraude cambió de forma. Eso explica por qué todos los modelos caen en test.

**Si preguntan**

- *¿Por qué no otro conjunto, como AMLSim?* AMLSim es simulado. Elliptic es el único grafo real,
  público y etiquetado de este tamaño. La parte simulada la cubrimos con nuestro propio grafo
  sintético, que va en el anexo.

---

## Lámina 9. Qué cuenta como negativo (A, B y C)

**Qué decir**

> Primera decisión de diseño. La clase positiva siempre son las ilícitas. La pregunta es contra
> qué las comparamos. Opción A: solo contra las sin etiqueta. Opción B: solo contra las lícitas
> revisadas; es lo más limpio, pero deja fuera cuatro de cada cinco negativos y su desbalance no
> es el real. Opción C: lícitas y sin etiqueta juntas. No sabemos si todas las sin etiqueta son
> lícitas, pero como el fraude es la clase rara, la gran mayoría deben serlo. Por efectos
> prácticos las tratamos como lícitas, que es lo que haría un banco. Y lo comprobamos: C rinde
> igual que B, y lo que B marca entre las sin etiqueta no se parece al fraude conocido.

**Qué hay detrás**

**Los números en entrenamiento:** 3.462 ilícitas, 26.432 lícitas y 106.371 sin etiqueta.

| Modo | Negativos | Desbalance |
|---|---|---|
| A | solo sin etiqueta | 1:30,7 |
| B | solo lícitas | 1:7,6 |
| C | lícitas + sin etiqueta | 1:38,4 (≈1:40) |

**Por qué C:**

1. Es lo que pasa en un banco: lo que nadie reportó se asume normal.
2. Usa todos los datos.
3. Su desbalance es el real.

**La duda de Cristian.** El modelo B, aplicado a las sin etiqueta, marca el 15,6 %. ¿Eso es
fraude escondido? Lo analizamos (`results_v4/reunion_0410/sin_etiqueta/REPORTE.md`):

- **Las lícitas revisadas no se parecen a las sin etiqueta.** Un clasificador las separa casi
  perfecto (ROC-AUC 0,984). B nunca vio ese tipo de transacciones como negativas y, al
  encontrarlas, adivina. Es la explicación que sugirió Cristian: B aprendió con un grupo pequeño
  de lícitas y aprendió algo incompleto.
- **Lo que B marca no se parece al fraude conocido.** Otro clasificador separa esas marcadas de
  las ilícitas reales con ROC-AUC 0,994. Con un argumento de mezcla, a lo sumo el 1,2 % de las
  marcadas podría ser del tipo del fraude etiquetado.
  - En palabras simples: si una parte se pareciera al fraude, el clasificador no podría
    separarlas tan bien.
- **No es la estructura del grafo:** las marcadas tienen menos conexiones, no más.
- **No es el cambio en el tiempo:** la tasa de marcado es parecida en las tres particiones.
- **Los modelos B no se ponen de acuerdo** en a quién marcar. Si fuera fraude real, coincidirían.

**Si preguntan**

- *¿Están diciendo que no hay fraude entre las sin etiqueta?* No. Decimos que el 15,6 % no es
  evidencia de fraude: es un efecto de con qué se entrenó B. Seguramente hay algo de fraude, pero
  poco, y es razonable tratarlas como lícitas para entrenar.

---

## Lámina 10. Submuestreo y sobremuestreo

**Qué decir**

> Con C el desbalance real es de 1 a 40. Para estudiar su efecto construimos escenarios con una
> regla: nunca quitar ilícitas, porque son la señal que el modelo tiene que aprender. Primera
> técnica, el submuestreo: para llegar a 1 a 10 tomamos al azar una parte de los negativos,
> manteniendo la misma mezcla de lícitas y sin etiqueta. Lo mismo para 1 a 20. Segunda técnica,
> el sobremuestreo: en vez de quitar negativos, agregamos ilícitas con SMOTE, que crea una ilícita
> nueva a mitad de camino entre una real y otra parecida, y le copia sus conexiones. El 1 a 1 lo
> pasamos al anexo.

**Qué hay detrás**

**Los escenarios** (solo cambian el entrenamiento; validación y test son siempre los mismos):

| Escenario | Ilícitas | Negativos | Cómo se construye |
|---|---|---|---|
| Nativo (≈1:40) | 3.462 | 132.803 | todo, sin tocar |
| 1:10 | 3.462 | 34.620 | submuestreo |
| 1:20 | 3.462 | 69.240 | submuestreo (en ejecución) |
| 1:10 con SMOTE | 6.924 | 69.240 | 3.462 reales + 3.462 sintéticas |

**Submuestreo aleatorio estratificado.** Se sacan negativos al azar, respetando la proporción
de lícitas y sin etiqueta (≈19,9 % lícitas).

- *Por qué al azar:* cualquier criterio inteligente para escoger qué negativos quitar cambiaría
  también qué aprende el modelo, y ya no sabríamos si el efecto es del desbalance o del criterio.
- *Por qué estratificado:* para que todos los escenarios tengan la misma mezcla.
- Usa una semilla de datos fija (2026) e independiente de la semilla del modelo. Así, las tres
  semillas de un escenario ven exactamente los mismos negativos.

**SMOTE** (Chawla et al. 2002), paso a paso:

1. Se toma una ilícita real.
2. Se buscan sus 5 ilícitas más parecidas (por sus variables).
3. Se escoge una de ellas.
4. Se crea un punto nuevo en algún lugar de la línea que las une.

- *Por qué no copiar la ilícita tal cual:* el modelo memorizaría casos repetidos.
- *En un grafo:* la nueva transacción necesita vecinas. Se le copian las conexiones de la
  original, que es la variante usada como referencia en GraphSMOTE (Zhao et al. 2021).

**1:10 con y sin SMOTE** tienen la misma proporción. Compararlos aísla el efecto de la técnica:
quitar negativos o agregar ilícitas.

**Por qué el 1:1 va al anexo:** con clases iguales, las tres formas de compensar se vuelven
casi la misma.

- Los pesos por clase valen 1 para las dos.
- Focal loss se acerca a no ajustar.

Además ningún modelo 1:1 pasó la compuerta. Lo pidió Cristian.

**Si preguntan**

- *¿Por qué 1:20?* Es el punto intermedio entre 1:10 y el nativo. Reemplaza al 1:1 en el
  análisis principal. Van 11 de sus 36 modelos.

---

**Actualización del 10-oct: el caso extremo**

Añadir al discurso, después de la regla de no quitar ilícitas:

> Esa regla tiene una excepción que hicimos a propósito. Nos pidieron forzar el desbalance
> hasta 1 por cada 100 y 1 por cada 200. Como el nivel real ya usa todos los negativos, la
> única forma de ir más allá sin inventar datos es quitar fraude: dejamos 1.328 y 664 de las
> 3.462 ilícitas. Son dos niveles de estrés, y los reportamos aparte.

- **Conteos de entrenamiento:** 1:10 = 3.462 ilícitas y 34.620 negativos; 1:20 = 3.462 y
  69.240; real = 3.462 y 132.803; 1:100 = 1.328 y 132.803; 1:200 = 664 y 132.803.
- **Por qué aparte:** cambian a la vez la proporción y la cantidad de fraude (queda el 38 % y
  el 19 %). No entran en las pruebas estadísticas de los cuatro niveles principales.
- **Si preguntan por qué no crear negativos sintéticos:** el grafo pasaría de 203 mil a 417
  mil y 763 mil transacciones, más de la mitad inventadas, y no cabe en los 8 GB de la tarjeta
  gráfica.

---

## Lámina 11. La matriz experimental

**Qué decir**

> Sobre esos datos entrenamos cuatro arquitecturas de red de grafos muy usadas: GCN, GraphSAGE,
> GAT y TAGCN, en los cuatro escenarios y con tres formas de compensar el desbalance en la
> pérdida: sin ajuste, con pesos por clase y con focal loss. Son 48 configuraciones, y cada una se
> entrena con tres semillas: las semillas no son modelos distintos, son la incertidumbre de la
> configuración. No tiene sentido explicar un modelo que no aprendió, sería explicar ruido. Por
> eso hay una compuerta de calidad en validación, sobre el promedio de las tres semillas. Pasan 20
> configuraciones, y de esta fase sale el candidato que luego se optimiza a fondo.

**Qué hay detrás**

**Las cuatro arquitecturas** se diferencian en cómo mezclan a la transacción con sus vecinas:

| Arquitectura | Cómo mezcla | Analogía |
|---|---|---|
| GCN (Kipf y Welling, 2017) | promedia a la transacción con sus vecinas, todos con el mismo peso | «soy el promedio de mi barrio» |
| GraphSAGE (Hamilton et al. 2017) | guarda por separado lo propio y lo de las vecinas, cada cosa con su peso | «soy yo, más lo que dice mi barrio» |
| GAT (Veličković et al. 2018) | aprende cuánto escuchar a cada vecina (atención) | «a algunos vecinos les creo más» |
| TAGCN (Du et al. 2017) | mira uno, dos y tres pasos a la vez | «miro el barrio y el barrio del barrio» |

- *Por qué estas cuatro:* son las más usadas en la literatura de grafos y en Elliptic, y cubren
  las cuatro formas básicas de mezclar.
- GAT corre con 4 cabezas y 2 capas por la memoria de la tarjeta gráfica.

**Las tres pérdidas** (cómo se castigan los errores al entrenar):

- **Sin ajuste:** todos los errores pesan igual. Con 40 negativas por cada ilícita, el modelo
  puede bajar su error diciendo casi siempre «lícita». Es la referencia.
- **Pesos por clase:** equivocarse con una ilícita cuesta tantas veces más como el desbalance.
  Es lo estándar.
- **Focal loss** (Lin et al. 2017): baja el peso de los casos que el modelo ya clasifica bien y
  se concentra en los difíciles. Usamos α = 0,75 (peso de la clase rara) y γ = 2 (cuánto se
  ignoran los fáciles).

Lo que se esperaba antes de ver datos, según Cristian: sin ajuste debería ir peor, pesos por
clase es lo normal y focal loss a veces iguala y a veces mejora a pesos por clase.

**Las semillas.**

- *Por qué 3:* una sola semilla puede salir buena por suerte. Con tres se ve cuánto varía el
  resultado. Cristian insistió en que la unidad es la configuración, no la semilla.
- Las semillas son 42, 43 y 44.

**La compuerta:** F1 ≥ 0,30 y MCC ≥ 0,15 en validación, sobre el promedio de las 3 semillas.
Son los mismos umbrales de la versión anterior, fijados antes de ver estos resultados.

**Cómo se entrenó:**

1. Con la semilla 42 se hizo una búsqueda corta y automática de ajustes con Optuna: 8 intentos
   de 50 épocas.
2. Las semillas 43 y 44 reutilizan lo encontrado.
3. Cada modelo se entrena hasta 150 épocas y se detiene si no mejora en 20 seguidas.
4. El umbral de alerta se calibra solo con validación.

**Si preguntan**

- *¿Usaron los mismos hiperparámetros para todos?* La lámina lo dice así, pero no es literal.
  Cada configuración tuvo su propia búsqueda corta. Lo igual para todas es el procedimiento y el
  presupuesto. Respuesta sugerida: «Todas recibieron el mismo procedimiento y el mismo
  presupuesto de ajuste; es una fase de selección, no de optimización. La optimización completa,
  con 100 o más intentos, se hace sobre el candidato: GraphSAGE con pesos por clase».
- *¿Por qué no optimizar todas a fondo?* Son 144 modelos. Ocho intentos dan un ajuste
  razonable con un costo igual para todas. Cristian propuso exactamente esto: usar la matriz para
  escoger el candidato y optimizar solo ese.

---

**Actualización del 10-oct: la matriz creció**

- A los cuatro niveles principales se suman los dos de estrés: 72 modelos más (2 niveles, 4
  arquitecturas, 3 pérdidas, 3 semillas).
- Con pesos por clase, GAT solo pasa el filtro de calidad si se evalúa con el umbral ajustado
  (0 de 12 modelos con el umbral 0,5; 12 de 12 con el ajustado). Es una decisión pendiente.
- El candidato que salió de la matriz, GraphSAGE con pesos por clase, se volvió a entrenar con
  una búsqueda de parámetros 19 veces más larga: sube de 0,46 a 0,50 en validación.

---

## Lámina 12. Cómo se mide la estabilidad

**Qué decir**

> Ahora, cómo medimos la estabilidad. Todos los modelos explican las mismas 30 transacciones
> ilícitas de validación; si cada uno explicara las suyas, compararíamos cosas distintas. Primera
> medida: mismo modelo, misma transacción, el explicador se corre cinco veces y comparamos cada par.
> Segunda medida: la misma configuración entrenada con tres semillas; si la explicación depende de
> con qué semilla se entrenó, tampoco es reproducible. El orden de las 165 variables se compara
> con Spearman y los flujos más importantes, con Jaccard. Y si un explicador no produce lo que se
> mide, se reporta «no aplica», nunca cero.

**Qué hay detrás**

- **Por qué verdaderos positivos de validación:** son ilícitas que el modelo detectó bien.
  Explicar un error no informa nada, y en test los modelos fallan (lámina 8 del deck de
  hipótesis). Se escogen con una semilla fija (1234), así que son las mismas en todos los modelos.
- **Cinco réplicas** dan 10 pares (1 con 2, 1 con 3, …, 4 con 5). La estabilidad es el promedio
  de esos 10.
- **Qué produce cada explicador:**

  | Explicador | Cómo funciona | Qué produce |
  |---|---|---|
  | GNNExplainer (Ying et al. 2019) | Para cada transacción ajusta perillas sobre variables y flujos, buscando el mínimo de información que mantiene la decisión | variables y flujos |
  | PGExplainer (Luo et al. 2020) | Entrena una vez una pequeña red que, para cualquier transacción, predice qué flujos importan. Se reentrena en cada réplica | solo flujos |
  | Shapley (GNNShap) | Teoría de juegos: el aporte de una variable es cuánto cambia la decisión al agregarla, promediado sobre muchas combinaciones al azar (50 muestras) | solo variables |

- **Spearman sobre las 165 variables:** compara el orden completo. Que dos variables de la mitad
  de la lista cambien de lugar mueve poco el resultado; que el orden se invierta da −1.
- **«No aplica» y no cero:** Shapley no produce flujos. Si pusiéramos 0, parecería inestable
  cuando en realidad no mide eso.

---

**Actualización del 10-oct: dos explicadores más y un límite de la medida**

Añadir al discurso:

> Como PGExplainer no sirve en este grafo, agregamos un explicador de otra familia, la de
> gradientes, en dos versiones. Y una advertencia sobre la medida: casi la mitad de las
> variables de una transacción vale cero, y los explicadores siempre las dejan al fondo del
> orden. Eso sube la correlación sin que el modelo tenga nada que ver. Por eso reportamos
> también dos medidas que no se inflan.

- **Integrated Gradients** (Sundararajan et al. 2017): reparte la decisión entre las variables
  siguiendo el gradiente desde una transacción de referencia. Es determinista: repetirlo da lo
  mismo, y su estabilidad entre repeticiones vale 1 por construcción.
- **Expected Gradients** (Erion et al. 2021): lo mismo, con la referencia tomada al azar de
  transacciones no ilícitas. Tiene ruido, así que las cinco repeticiones sí miden algo.
- **El límite de la medida.** El 46 % de las variables de las 30 transacciones vale 0 después
  del escalado. Solo ese bloque común ya da un Spearman de 0,74 a 0,82. Y una red sin entrenar
  da casi la misma estabilidad que la entrenada: GNNExplainer 0,95 y 0,93; Shapley 0,97 y 0,96
  (prueba de Adebayo et al. 2018). Adebayo ya había visto que, al comparar la importancia en
  valor absoluto (como hacemos), la correlación sigue alta aunque los pesos sean al azar, y
  Agarwal et al. (2023) que la estabilidad de las explicaciones sobre variables se parece a la de
  explicaciones aleatorias.
- **Las dos medidas que no se inflan:** el Spearman solo sobre las variables distintas de 0, y
  la coincidencia de las 10 variables más importantes (índice de Jaccard). Con ellas
  GNNExplainer baja de 0,95 a 0,75 y a 0,63.
- **Si preguntan si eso invalida el estudio:** no. La conclusión sobre el desbalance se
  sostiene con las tres medidas. Lo que cambia es cuánto se puede concluir de una estabilidad
  alta, que es la tesis de fondo: estable no es lo mismo que correcto.

Referencias de esta actualización: Adebayo, J. et al. (2018). Sanity checks for saliency maps.
NeurIPS. Agarwal, C. et al. (2023). Evaluating explainability for graph neural networks.
Scientific Data, 10, 144. Erion, G. et al. (2021). Improving performance of deep learning models with axiomatic
attribution priors and expected gradients. Nature Machine Intelligence, 3, 620-631.
Sundararajan, M. et al. (2017). Axiomatic attribution for deep networks. ICML.

---

> **Esta parte describe la versión de 26 láminas y quedó reemplazada** por
> `docs/GUION_hipotesis_v8_explicado.md`. Se conserva como registro.

# Parte 2. Deck de hipótesis

## Lámina 1. Portada

**Qué decir**

> Profe, retomamos donde quedamos. Para cada hipótesis respondemos las seis preguntas que nos
> pidió: qué esperábamos, por qué, qué dio, si era lo esperado, por qué dio eso y en qué impacta.
> Y aplicamos sus cambios: las semillas como incertidumbre, la configuración como unidad y el 1:1
> fuera del análisis principal.

---

## Lámina 2. Qué es la estabilidad

Si se presenta después del deck principal, esta lámina repite la 6 y la 12. Se puede pasar
rápido: «recordemos la definición».

**Qué decir**

> Recordemos la definición. Tomamos una alerta, se la damos al explicador y nos devuelve qué
> variables pesaron más, en orden. Lo repetimos cambiando solo la semilla del explicador y
> comparamos los dos órdenes con Spearman: 1 es el mismo orden, 0 nada en común. GNNExplainer da
> 0,93 en promedio: los órdenes casi coinciden, solo se intercambian posiciones cercanas.

**Qué hay detrás**

En el ejemplo, la variable 12 y la 3 se intercambian el primer y segundo lugar, y el resto queda
igual. Eso da un Spearman alto: el orden general se conserva. Un 0,93 significa que las
explicaciones casi coinciden.

---

## Lámina 3. Cómo respondemos las hipótesis

**Qué decir**

> Así vamos a leer cada hipótesis: seis preguntas. Y sobre qué datos: ya no hablamos de 144
> modelos sino de 48 configuraciones con tres semillas cada una. Sacamos el 1:1, quedan 36. Y
> solo explicamos las que aprenden, decidido con el promedio de las tres semillas: quedan 20.
> GraphSAGE y TAGCN aprenden en 8 de 9 configuraciones, GAT en 4 y GCN en ninguna. Para decir que
> dos cosas son iguales no basta con no encontrar diferencia: usamos una prueba de equivalencia,
> con un margen de cinco centésimas fijado antes de ver los datos.

**Qué hay detrás**

- **El embudo:** 48 configuraciones. Sin el 1:1 quedan 36. Con la compuerta sobre el promedio
  quedan 20, repartidas así:

  | | Configuraciones que pasan |
  |---|---|
  | GraphSAGE | 8 |
  | TAGCN | 8 |
  | GAT | 4 |
  | GCN | 0 |
  | Nativo | 5 |
  | 1:10 | 7 |
  | 1:10 con SMOTE | 8 |

- **Por qué el promedio de semillas y no semilla por semilla:** Cristian lo pidió. Si una
  semilla pasa y otra no, quedarse con la que pasó es quedarse con la que tuvo suerte.
- **La prueba de equivalencia (TOST)** se entiende con una analogía. Decir «no encontré
  diferencia» es como decir «no vi al ladrón»: puede ser porque no estaba o porque no miré bien.
  TOST cambia la pregunta: demuestra que la diferencia es **menor que 0,05**. Ese margen se fijó
  antes de los datos: en una escala de 0 a 1, cinco centésimas es una diferencia que no cambia
  ninguna decisión práctica.

**Si preguntan**

- *¿La compuerta sobre el promedio no deja pasar semillas que no aprendieron?* Sí. En 5 de las
  20 configuraciones una de las tres semillas queda bajo el umbral; por ejemplo, TAGCN con pesos
  por clase en el nativo tiene una semilla con F1 0,18. Con un criterio más estricto (promedio
  menos una desviación) quedan 15. Escogimos el promedio porque es lo que pidió: la configuración
  como unidad. La lectura de H1, H2 y H3 no cambia con 15.

---

## Lámina 4. Cómo leer las curvas ROC y PR

**Qué decir**

> Antes de los resultados, cómo se leen las curvas. El modelo no dice sí o no: da un puntaje de
> sospecha, y el umbral decide desde dónde se alerta. Cada punto de la curva es el mismo modelo
> con otro umbral, y el modelo que usaríamos en el banco es un solo punto. La curva ROC dice si el
> modelo separa lícitas de ilícitas; la diagonal es tirar una moneda. La curva PR dice si aprende
> la clase rara: con 2,4 % de ilícitas, alertar al azar da 2,4 % de acierto. Se leen juntas. Si una
> curva va arriba en todo el recorrido, ese modelo es mejor; si se cruzan, depende del umbral. Y
> una curva suave indica aprendizaje; los picos, suerte.

**Qué hay detrás**

- **El umbral, con una analogía:** un detector de humo. Muy sensible, suena con el pan tostado
  (muchas falsas alarmas). Poco sensible, no suena en un incendio pequeño (se escapa el fraude).
  Cada sensibilidad es un punto de la curva.
- **Curva ROC.**
  - Eje horizontal: de las lícitas, cuántas alerto por error.
  - Eje vertical: de las ilícitas, cuántas detecto.
  - Arriba a la izquierda es lo ideal.
  - **Limitación:** con muy pocas ilícitas puede verse bien aunque las alertas estén llenas de
    falsas alarmas.
- **Curva PR.**
  - Eje horizontal: cuántas ilícitas detecto.
  - Eje vertical: de las alertas, cuántas son ilícitas de verdad.
  - Mira solo la clase rara, por eso es la que importa con desbalance (Saito y Rehmsmeier,
    2015).
- **Por qué juntas:** lo pidió Cristian. La ROC dice si el modelo ordena bien en general; la PR,
  si sus alertas sirven. Una sola puede engañar.
- **Suave frente a picos:** los picos aparecen cuando unas pocas transacciones deciden el
  resultado, y eso cambia de una semilla a otra. Una curva suave muestra un comportamiento que
  se repite.
- **Cómo leer las figuras que siguen:**
  - La línea es la media de las 3 semillas.
  - La sombra va del mínimo al máximo de las 3; si es ancha, las semillas no coinciden.
  - Los puntos son el umbral elegido en cada semilla.

---

## Lámina 5. Rendimiento: la curva resumen

**Qué decir**

> Primero el resumen: el escenario con el desbalance real y, para cada arquitectura, la forma de
> compensar que mejor le fue en validación. En validación, GraphSAGE con pesos por clase está
> arriba en todo el recorrido y su sombra es angosta: las tres semillas coinciden. En test todo se
> aplasta contra el azar, y GAT incluso queda bajo la diagonal: ordena al revés. Como en test
> nada se sostiene, medimos la estabilidad en validación, donde el modelo sí aprendió.

**Qué hay detrás**

- **Cómo se eligió «la mejor pérdida»:** por el PR-AUC de validación, nunca por test. Escoger
  mirando test sería hacer trampa.
- **Por qué los modelos caen en test** (pasos 43 a 49): en el paso 43 cerró un mercado oscuro
  grande y el fraude cambió de forma (Weber et al. 2019).
  - Lo que el modelo aprendió como fraude en el pasado ya no describe el fraude del futuro.
  - Le pasa a todos los modelos publicados sobre Elliptic. No es un error nuestro.
- **«Ordena al revés»:** una ROC bajo la diagonal significa que el modelo les da más puntaje a
  las lícitas que a las ilícitas. Lo que aprendió como señal apunta al lado contrario después del
  cambio.
- **Por qué medir la estabilidad en validación:** explicar un acierto informa; explicar un
  modelo que en ese período funciona como una moneda no informa.

**Si preguntan**

- *Entonces, ¿los modelos no sirven?* Sirven en el período donde el fraude se parece al del
  entrenamiento. Lo que muestra el test es que un banco tendría que reentrenar cuando el fraude
  cambia. Esa es otra pregunta de investigación; la nuestra es sobre las explicaciones.

---

## Lámina 6. La tabla de cuatro métricas

La lámina 5 ya mostró esto con curvas. Esta tabla pone los números de esas mismas curvas. No hay
que leerla fila por fila: basta con señalar el patrón.

**Qué decir**

> Estas son las curvas de la lámina anterior puestas en números. Con clases desbalanceadas una
> sola métrica no basta, así que reportamos cuatro, en pares: ROC-AUC con PR-AUC, y F1 con KS.
> *(clic)* En validación, GraphSAGE, GAT y TAGCN están muy por encima del azar en las cuatro.
> *(clic)* GCN apenas se separa del azar, y por eso no pasa la compuerta.
> *(clic)* En test el F1 es casi cero y el PR-AUC queda en el azar. El KS de test sigue alto, pero
> se consigue con muchas falsas alarmas, así que no sirve para operar.

**Qué hay detrás**

Cómo leer cada métrica:

| Métrica | Qué mide en palabras simples | Azar | Mejor valor (GraphSAGE, validación) |
|---|---|---|---|
| ROC-AUC | si una ilícita al azar recibe más puntaje que una lícita al azar | 0,50 | 0,92 |
| PR-AUC | qué tan limpias son las alertas cuando se busca más fraude | 0,024 | 0,46 (unas 19 veces el azar) |
| F1 | el equilibrio entre detectar y no dar falsas alarmas, en el umbral elegido | 0,05 | 0,49 |
| KS | la mayor separación entre ilícitas detectadas y lícitas alertadas | 0 | 0,73 |

- **Por qué en pares:** Cristian lo pidió así. La ROC-AUC mira la separación general y la PR-AUC
  la clase rara. El F1 mide el punto elegido y el KS la mejor separación posible.
- **Qué significa «± 0,00»:** las tres semillas dieron casi lo mismo. Es buena señal.
- **Por qué el KS de test (0,35) no sirve aunque sea alto:** esa separación se alcanza alertando
  entre el 47 y el 80 % de las lícitas, y ningún banco opera así.
- **Qué modelos son:** escenario nativo y, para cada arquitectura, la pérdida que mejor le fue en
  validación:

  | Arquitectura | Pérdida |
  |---|---|
  | GraphSAGE | pesos por clase |
  | GAT | focal loss |
  | TAGCN | pesos por clase |
  | GCN | focal loss |

---

## Lámina 7. Curvas en validación, desbalance real

**Qué decir**

> Ahora todas las combinaciones del desbalance real, en validación: arriba la ROC, abajo la PR, una
> columna por arquitectura, y el color es la forma de compensar. La curva azul, sin compensar, va
> abajo en todo el recorrido en casi todos los casos. Con este desbalance, compensar ayuda, y sin
> ambigüedad. GCN tiene sombras anchas: sus semillas dan curvas distintas, y por eso ninguno de sus
> modelos pasa la compuerta. Los puntos caen en el codo de la curva, donde se detecta mucho con
> pocas falsas alarmas: el umbral está bien puesto.

**Qué hay detrás**

- **«Domina en 7 de 8 pares»:**
  - Se compara la curva sin ajuste contra cada una de las otras dos, en las 4 arquitecturas
    (8 pares).
  - En 7, la curva sin ajuste queda debajo en todo el recorrido.
  - La excepción es GCN, donde sin ajuste y pesos por clase se cruzan.
- **Por qué sin ajuste va peor:** con 40 negativas por ilícita, decir «lícita» casi siempre ya
  da un error bajo. El modelo aprende poco de la clase rara si nadie le dice que esos errores
  importan más. Coincide con lo que Cristian esperaba.
- **El codo:** el punto donde la curva deja de subir rápido. Más allá, cada ilícita extra se
  paga con muchas falsas alarmas.

---

## Lámina 8. Curvas en test, desbalance real

**Qué decir**

> El mismo escenario en test, que son los meses posteriores. Las curvas PR quedan pegadas a la
> línea del azar. Dos modelos ordenan al revés: lo que aprendieron como fraude apunta al lado
> contrario. Weber y colegas reportan este quiebre tras el cierre de un mercado oscuro. Y los
> puntos del umbral caen cerca del origen: el umbral que funcionaba en validación casi no alerta
> en test.

**Qué hay detrás**

- **La escala logarítmica** en la precisión de test: como el azar es 0,0057 (solo 169 ilícitas
  en casi 30 mil transacciones), en escala normal todas las curvas se verían como una línea en
  el piso.
- **Las curvas PR van a lo sumo al doble del azar.** Es casi nada.
- **Los modelos que ordenan al revés** son GraphSAGE y TAGCN sin balanceo, con ROC-AUC de test de
  0,39 y 0,31.

---

## Láminas 9 a 12. Curvas de 1:10 y 1:10 con SMOTE

**Recomendación:** son cuatro láminas que repiten el patrón. En una reunión con Cristian se
pueden mostrar. En la sustentación conviene dejar en el flujo solo la 9 (la del cruce de
curvas, que es la idea nueva) y llevar 10, 11 y 12 al anexo.

### Lámina 9. 1:10 en validación

**Qué decir**

> Con 10 lícitas por cada ilícita pasa algo distinto: compensar deja de ayudar y las curvas se
> cruzan. En GAT, no compensar incluso supera a las otras dos. La razón es que el submuestreo ya
> corrigió buena parte del desbalance. GraphSAGE con pesos por clase sigue arriba. Y en TAGCN las
> curvas se cruzan: según el umbral gana una u otra, así que el área sola no basta para decidir.

**Qué hay detrás**

- **Por qué compensar deja de ayudar:** compensar en la pérdida y quitar negativos atacan el
  mismo problema. Si ya se quitaron negativos, compensar encima puede pasarse de la raya y hacer
  que el modelo alerte de más.
- **Los números:** en GAT 1:10, el PR-AUC de validación es 0,40 sin ajuste, frente a 0,32 con
  pesos por clase y 0,30 con focal loss.
- **Que las curvas se crucen** es exactamente lo que Cristian advirtió. El área es un promedio
  sobre todos los umbrales, y en un umbral concreto puede ganar la otra curva.

### Lámina 10. 1:10 en test

**Qué decir**

> En test se repite la caída al azar. Pesos por clase es la que mejor resiste en GraphSAGE. GCN
> sin balanceo tiene el área PR más alta, pero su sombra es enorme: es suerte de una semilla.

**Qué hay detrás:** es el ejemplo perfecto de por qué Cristian pidió ver la sombra. El número
solo diría que GCN es el mejor en test, y la sombra muestra que lo da una sola semilla.

### Lámina 11. 1:10 con SMOTE en validación

**Qué decir**

> Con SMOTE las curvas quedan casi iguales a las de 1:10 sin SMOTE: GraphSAGE arriba, con pesos
> por clase y focal loss casi empatados, y GCN abajo. La sombra ancha de TAGCN con pesos por clase
> es una semilla que no aprendió.

**Qué hay detrás:** que 1:10 con y sin SMOTE se parezcan quiere decir que agregar ilícitas
sintéticas no aporta información nueva en validación. Inventar ilícitas a partir de las
existentes no enseña patrones que el modelo no hubiera visto ya.

### Lámina 12. 1:10 con SMOTE en test

**Qué decir**

> Y en test, SMOTE tampoco evita la caída. Lo poco que se separa del azar está donde hay muchas
> falsas alarmas, que no sirve para operar.

---

## Lámina 13. H1: qué esperábamos y qué dio

**Qué decir**

> Primera hipótesis, el desbalance. Esperábamos que, a más desbalance, las explicaciones fueran
> menos estables. Porque con menos fraude para aprender el modelo traza una frontera más frágil,
> y una explicación de algo frágil debería moverse más. Pero la estabilidad sale igual en los tres
> escenarios: 0,93 en los tres, y todo cabe en el margen de cinco centésimas. No era lo esperado:
> la prueba de equivalencia confirma que son iguales, y con Shapley pasa lo mismo.

**Qué hay detrás**

- **Las cifras:** 0,928 (1:10, 7 configuraciones), 0,928 (1:10 con SMOTE, 8) y 0,931 (nativo, 5).
- **TOST con p ≤ 0,003:** la probabilidad de ver diferencias tan pequeñas si de verdad fueran
  mayores que 0,05 es menor de 3 en mil. Por eso decimos «iguales», no solo «sin diferencia
  detectada».
- **El eje empieza en 0,80:** para que se vean las barras. Si empezara en 0 las tres se verían
  idénticas, que de hecho lo son.
- **«Frontera frágil», con una analogía:** si aprendo a distinguir perros de gatos con dos fotos
  de gato, mi idea de «gato» es pobre y cualquier detalle me hace cambiar de opinión.

**Actualización del 14-oct: H1 depende de la forma de compensar.** Lo anterior vale para pesos
por clase. Al repetir la prueba para cada forma de compensar (GraphSAGE, GAT y TAGCN; 3 semillas;
6 niveles; prueba de Friedman), GNNExplainer da:

| GNNExplainer | Sin el cierre (corrida principal) | Partición original |
|---|---|---|
| Pesos por clase | no cambia (p ≥ 0,12 con las tres medidas) | no cambia (p ≥ 0,22) |
| Focal loss | baja: 0,923 → 0,909 del real a 1:200 (p = 0,010); top-10 0,623 → 0,561 (p = 0,035) | no cambia (p ≥ 0,29) |
| Sin ajuste | baja: 0,942 en 1:10 → 0,889 en 1:200 (p < 0,001); top-10 0,589 → 0,502 | baja: 0,930 → 0,901 (p < 0,001); top-10 0,590 → 0,505 |

Shapley y Expected Gradients casi no responden (diferencias de milésimas). Lo que se repite en
las dos particiones es la caída **sin ajuste**; la de focal loss solo aparece sin el cierre.

*Por qué:* sin ajuste, a más desbalance el fraude pesa menos en la pérdida, el modelo se degrada
hasta marcar casi todo como no fraude, y GNNExplainer, que optimiza una máscara sobre la salida de
ese modelo, pierde estabilidad con él. Pesos por clase saca la proporción de la pérdida (cada
clase pesa la mitad) y corta ese efecto. El desbalance no actúa sobre la explicación de forma
directa, sino a través de cuánto se degrada el modelo.

*Cómo decirlo:* «H1 se sostiene en parte. Con pesos por clase, el desbalance no cambia la
estabilidad, ni con 1 por cada 200. Sin compensarlo, la estabilidad de GNNExplainer baja junto
con el rendimiento del modelo, en las dos particiones.»


---

## Lámina 14. H1: por qué dio eso y en qué impacta

**Qué decir**

> ¿Por qué no cambió? Miren qué ve el modelo al entrenar. Las ilícitas, en naranja, son siempre
> las mismas 3.462. Lo único que cambia es cuántos negativos le mostramos. Y el rendimiento en
> validación es parecido en los tres. Como el fraude es el mismo y el modelo aprende casi lo
> mismo, explica casi lo mismo. Además el rango es estrecho: de 1 a 10 hasta 1 a 40, solo un
> factor cuatro. Por eso agregamos el 1:20. El impacto: un banco puede submuestrear negativos
> para entrenar más rápido sin perder estabilidad en las explicaciones.

**Qué hay detrás**

- **La clave es la regla de no quitar ilícitas.** Toda la información sobre el fraude está en
  las ilícitas, y nunca la tocamos. Quitar negativos cambia la proporción, pero no lo que el
  modelo sabe del fraude.
- **Rendimiento parecido:** PR-AUC de validación 0,38, 0,38 y 0,34; ROC-AUC 0,89 a 0,90. Un
  modelo que aprende lo mismo explica lo mismo.
- **Las dos causas que Cristian sugirió** aparecen aquí:
  1. El rango es estrecho.
  2. El muestreo está bien hecho (no se perdió fraude).
- **El 1:20** completa la escalera 1:10, 1:20 y 1:40. Van 11 de 36 modelos.

**Si preguntan**

- *¿Y con un desbalance mucho más extremo, 1:100?* No lo probamos en la v4. Para llegar a 1:100
  con estos datos habría que quitar ilícitas, que es justo lo que no queremos hacer. Lo
  declaramos como límite.

---

## Lámina 15. H2: qué esperábamos y qué dio

**Qué decir**

> Segunda hipótesis, la arquitectura. Esperábamos que cambiara la estabilidad. Cada una combina a
> los vecinos de otra forma: GCN promedia, GraphSAGE guarda un peso para la propia transacción,
> GAT usa atención y TAGCN mira varios pasos. Si deciden por caminos distintos, sus explicaciones
> deberían variar distinto. Pero las tres que aprenden quedan juntas, dentro del margen. GCN no
> entra porque ninguna de sus configuraciones aprende. Para las tres no era lo esperado; que GCN
> quedara atrás sí era esperable.

**Qué hay detrás**

- **Las cifras:** GraphSAGE 0,938 (8 configuraciones), GAT 0,932 (4) y TAGCN 0,919 (8).
- **La hipótesis original, la del manuscrito,** era más concreta: «TAGCN, por mirar varios
  pasos, daría las explicaciones más estables». Conviene decirla así, porque la respuesta es más
  clara: TAGCN quedó la más baja de las tres, aunque equivalente. Frase sugerida:
  «esperábamos que TAGCN fuera la más estable, porque al mirar varios pasos tiene una visión más
  completa de la red».
- **La versión anterior (v3)** mostraba dos grupos de arquitecturas. Esa conclusión no sobrevivió
  a la corrección de las etiquetas.

---

## Lámina 16. H2: por qué GCN queda atrás

**Qué decir**

> ¿Por qué GCN queda atrás? Hicimos un experimento de control. Un modelo que solo mira las
> variables de la propia transacción, sin el grafo, ya llega a 0,49. GraphSAGE llega a lo mismo.
> GCN se queda en 0,21. Y si a GCN le agregamos un peso propio para la transacción, sube a 0,46.
> La señal del fraude está en la propia transacción. Casi todas las ilícitas tienen una o dos
> vecinas, y casi ninguna es ilícita. GCN promedia a la transacción con esas vecinas y diluye la
> señal. El impacto: la arquitectura decide si el modelo aprende, no qué tan estable explica.

**Qué hay detrás**

**El experimento de control.** Se entrenaron cuatro modelos con los mismos ajustes, el mismo
procedimiento y el mismo escenario (nativo), con las semillas 42 y 43:

| Modelo | Qué es | PR-AUC de validación |
|---|---|---|
| MLP | red neuronal común: solo ve las 165 variables de la transacción, no ve el grafo | 0,485 |
| GraphSAGE | | 0,481 |
| GCN con peso propio | | 0,464 |
| GCN normal | | 0,214 |

**La explicación, paso a paso:**

1. El 93 % de las ilícitas tiene 1 o 2 vecinas.
2. Solo el 12 % de esas vecinas es ilícita.
3. GCN calcula «yo = promedio de mí y mis vecinas».
4. Si tengo 2 vecinas lícitas, mi descripción queda dos tercios lícita: la señal se diluye.
5. GraphSAGE, GAT y TAGCN tienen una forma de conservar lo propio, y GCN no.

**Era esperable:** Weber et al. (2019) ya vieron que modelos sobre las variables superan a GCN en
Elliptic, y propusieron un GCN con un «atajo» (Skip-GCN), que es lo mismo que el peso propio.

**Por qué GraphSAGE, GAT y TAGCN quedan parecidas:** las tres conservan la información propia, y
esa es la que importa aquí.

**Si preguntan**

- *¿No será que a GCN le faltó ajuste?* El control usa los mismos ajustes para todos. Con un
  solo cambio de diseño, el peso propio, GCN pasa de 0,21 a 0,46. El problema es la forma de
  mezclar, no el ajuste.

---

## Lámina 17. H3: qué esperábamos y qué dio

**Qué decir**

> Tercera hipótesis: la forma de compensar el desbalance en la pérdida. Esperábamos que no moviera
> la estabilidad, porque solo cambia cuánto pesa cada error al entrenar: cambia cómo aprende el
> modelo, no cómo se lo explica. Con la configuración como unidad, como nos pidió, GNNExplainer no
> muestra efecto. Antes habíamos reportado un efecto, pero venía de promediar solo las semillas que
> pasaban, justo lo de la semilla con suerte. Lo que sí se ve grande es la caída de PGExplainer
> con pesos por clase: casi cero. Para GNNExplainer y Shapley era lo esperado; para PGExplainer
> no, y lo explicamos en la siguiente.

**Qué hay detrás**

**Las cifras:**

| Explicador | Sin ajuste | Pesos por clase | Focal loss |
|---|---|---|---|
| GNNExplainer | 0,93 | 0,95 | 0,91 |
| PGExplainer | 0,56 | ≈ 0 (0,034) | 0,24 |
| Shapley | 0,990 a 0,996 | | |

**Friedman exacto.** La prueba de Friedman compara las tres pérdidas dentro de cada «bloque»
(misma arquitectura y escenario). Con tan pocos bloques (4), el p-valor aproximado no es
confiable, así que se calculó el exacto: se prueban todas las permutaciones posibles.

- GNNExplainer da p = 0,27: sin efecto.
- PGExplainer da p = 0,07: no alcanza significancia, aunque la caída sea grande.

**Por qué el antiguo 0,0498 era un artefacto:**

- La semilla 44 de TAGCN con pesos por clase se detuvo en la época 3 y no pasó la compuerta.
- Al promediar solo las semillas que pasaban, esa configuración quedó con dos semillas «buenas».
- Además era el p aproximado con 4 bloques (el exacto daba 0,069).

**Esto responde la expectativa de Cristian.** Él dijo que esperaba que el balanceo moviera un
poco más la estabilidad. Nuestra expectativa era que no la moviera.

- Las dos posturas son defendibles.
- Hay que decidir una y sostenerla; la lámina usa «sin efecto».
- Si Cristian insiste, la respuesta es que en GNNExplainer no se movió, y que lo que se movía
  venía de cómo se promediaba.

---

## Lámina 18. H3: por qué PGExplainer da casi cero

**Qué decir**

> ¿Por qué PGExplainer da casi cero? PGExplainer le pone un peso a cada flujo. En Elliptic casi
> todas las transacciones explicadas tienen uno o dos flujos, y PGExplainer empuja esos pesos a 0 o
> a 1. Con dos flujos solo hay dos órdenes posibles: o queda arriba el primero o el segundo. Entre
> ejecuciones eso se invierte casi al azar, así que la comparación salta entre más uno y menos uno,
> y el promedio tiende a cero. No es un error de cálculo: recalculamos todo y da exacto. Y no es
> culpa de los pesos por clase: el mismo modelo, explicado dos veces, da 0,04 y 0,42. La diferencia
> entre pérdidas cabe en ese ruido. Lo declaramos como limitación de PGExplainer en vecindarios tan
> pequeños.

Las notas actuales de la lámina dicen «todavía lo estamos verificando». Ya está verificado
(`results_v4/reunion_0410/pgexpl_gcn/REPORTE.md`), así que conviene usar el discurso de arriba.

**Qué hay detrás**

**Lo que pidió Cristian:** decir si era culpa de PGExplainer, de los pesos por clase, de los dos
o un error de programación. Respuesta: **es de PGExplainer en este grafo.**

**No es un error:**

- Ninguna de las 144 ejecuciones falló.
- Recalculando desde las máscaras guardadas se obtiene exactamente el mismo valor.

**Por qué satura:**

- El 83 % de las transacciones explicadas tiene 2 flujos o menos.
- Con 2 flujos, Spearman solo puede dar +1 o −1. El 93 % de los pares da exactamente ±1.
- Las perillas quedan casi iguales entre flujos (diferencia menor que 0,001), así que el orden
  lo decide el ruido numérico.

**La prueba de repetición (test-retest):** el modelo GraphSAGE nativo con pesos por clase,
explicado otra vez con el mismo código, los mismos nodos y las mismas semillas, dio 0,42 frente
al 0,04 original. La tarjeta gráfica no es determinista: el ruido de PGExplainer es mayor que
las diferencias que estamos comparando.

**Con los 144 modelos** la diferencia existe, pero es moderada:

| Pérdida | Estabilidad de PGExplainer |
|---|---|
| Sin ajuste | 0,45 |
| Focal loss | 0,32 |
| Pesos por clase | 0,24 |

- Se concentra en GraphSAGE y TAGCN.
- En GAT no hay diferencia.

**La naturaleza de cada explicador:**

- **GNNExplainer** ajusta perillas para cada transacción y compara el orden de 165 variables.
  Un cambio pequeño mueve poco el resultado.
- **PGExplainer** entrena una red para todas las transacciones y solo mira flujos. Con 2 flujos,
  todo o nada.

**Si preguntan**

- *¿Por qué los pesos por clase empeoran más a PGExplainer?* Hay un mecanismo plausible, no
  probado: con pesos por clase el modelo marca como ilícitas muchas más transacciones (50 % frente
  a 5 %) en los nodos con que se entrena PGExplainer, y eso cambia lo que aprende. Lo decimos como
  hipótesis, no como hallazgo.

---

## Lámina 19. Estable frente a correcto: qué esperábamos y qué dio

**Qué decir**

> La cuarta pregunta, la de fondo: ¿ser estable es lo mismo que ser correcto? Esperábamos que sí.
> Es la intuición natural: si siempre señala lo mismo, debe estar señalando lo correcto. En
> Elliptic no lo podemos comprobar porque no sabemos cuál es el patrón real; por eso construimos un
> grafo sintético donde sí lo sabemos. Shapley, los cuadros, es el más estable, pero acierta al
> nivel del azar. GNNExplainer, los círculos, es menos estable y es el que más acierta. No era lo
> esperado: estable no es lo mismo que correcto.

**Qué hay detrás**

- **El grafo sintético:** un grafo artificial que construimos nosotros, con transacciones
  normales y un patrón de lavado «sembrado» a propósito. Como lo sembramos, sabemos cuáles
  variables y flujos forman el patrón. Es la única forma de medir si un explicador acierta.
- **Plausibilidad:** de las variables que el explicador pone como más importantes, qué fracción
  pertenece al patrón sembrado. Si se escogieran al azar, saldría 0,15.
- **Las cifras:**

  | Explicador | Estabilidad | Plausibilidad |
  |---|---|---|
  | Shapley | 0,95 a 0,98 | 0,11 a 0,16 (en el azar) |
  | GNNExplainer | 0,85 a 0,96 | 0,24 a 0,45 (hasta tres veces el azar) |

- **Esta es la hipótesis 3 del manuscrito original** («un explicador más estable señala mejor el
  patrón»). En el deck se llama «calidad» para no confundirla con H3 (balanceo). Hay que alinear
  la numeración al reescribir el documento.

---

## Lámina 20. Estable no es correcto: por qué y en qué impacta

**Qué decir**

> ¿Por qué el más estable no es el más correcto? Shapley promedia cientos de combinaciones de
> variables, y ese promedio baja el ruido por construcción, haya encontrado el patrón o no. Por eso
> se repite igual sin acertar. En Elliptic pasa lo mismo por otro lado: GCN casi no aprende, y aun
> así sus explicaciones son casi tan estables como las del mejor modelo. El impacto, y es el aporte
> central de la tesis: la estabilidad es necesaria pero no suficiente. Para auditar una alerta hay
> que reportarla junto con el rendimiento, la plausibilidad y la fidelidad.

**Qué hay detrás**

- **Por qué promediar estabiliza.** Una analogía: si pregunto la estatura promedio de un salón
  midiendo a 5 personas al azar, cada vez sale algo distinto. Si mido a 500, sale casi igual
  siempre. Pero si estoy midiendo mal (con la regla equivocada), sale el mismo error siempre:
  estable e incorrecto.
- **El ejemplo de Elliptic.** Este análisis usa los 144 modelos, sin compuerta (análisis de
  sensibilidad):

  | Arquitectura | Estabilidad de GNNExplainer | PR-AUC de validación |
  |---|---|---|
  | GCN | 0,916 | 0,10 (casi no aprende) |
  | GraphSAGE | 0,946 | 0,31 |

  Un modelo que no aprende puede dar explicaciones que se repiten.
- **Necesaria pero no suficiente:**
  - Una explicación inestable seguro no sirve.
  - Una estable puede servir o no.
  - Hay que mirar también si el modelo aprendió (rendimiento), si señala lo correcto
    (plausibilidad) y si lo que señala de verdad mueve la decisión (fidelidad).

---

## Lámina 21. Las cuatro preguntas juntas

**Qué decir**

> Las cuatro preguntas juntas. Con pesos por clase el desbalance no cambia la estabilidad,
> porque el fraude es el mismo y el modelo aprende parecido; sin compensarlo, la de GNNExplainer
> baja junto con el rendimiento (actualización del 14-oct en la lámina 13). La arquitectura no la cambia entre las que aprenden; GCN no
> aprende porque diluye la señal de la propia transacción. La forma de compensar no la cambia; el
> casi cero de PGExplainer es una limitación del explicador. Y estable no es correcto: eso es lo
> que la tesis aporta.

**Qué hay detrás:** esta lámina es una tabla de palabras, no de números, y funciona como
resumen. Si se quiere menos texto, se pueden dejar solo las columnas «Resultó» e «Impacto».

---

## Lámina 22. Plan de trabajo

**Qué decir**

> Para cerrar, cómo queda todo ordenado. Una entrada para público general: Bitcoin, banca y
> transacciones; el lavado como patrón de red, la red que alerta y por qué hace falta explicar.
> Luego Elliptic, las etiquetas con C justificado, el muestreo con su porqué y la definición de
> estabilidad. Después el rendimiento con las curvas, cada hipótesis con sus seis preguntas y el
> cierre de estable frente a correcto. Al final, el candidato con la optimización completa y la
> recomendación. El 1:1 y el sintético van al anexo.

**Qué hay detrás**

**Lo que ya está:**

- Las curvas de las 48 configuraciones.
- La compuerta por configuración.
- El análisis de los sin etiqueta: ya hay reporte, así que se puede marcar como hecho.
- El diagnóstico de PGExplainer.

**Lo que falta:**

- El 1:20 (11 de 36 modelos).
- La optimización completa del candidato (GraphSAGE con pesos por clase, unas 0,4 horas de
  GPU).
- La comparación con un modelo de la literatura.

---

## Láminas 23 y 24. Anexo: escenario 1:1

**Qué decir**

> El 1:1 queda en el anexo, como acordamos. Sin balanceo y pesos por clase dan la misma curva,
> porque con clases iguales el peso es el mismo. Y rinde menos que 1:10: al ver tan pocos
> negativos, pierde precisión frente a la proporción real. En test, lo mismo que los demás.

**Qué hay detrás**

- **Por qué rinde menos:**
  - Con 1:1 el modelo entrena con 3.462 negativos, frente a 132.803 en el nativo: ve muy poca
    variedad de transacciones normales.
  - Al evaluarlo con la proporción real, confunde muchas normales con fraude.
- **Por qué las curvas coinciden:** el peso por clase es «negativos sobre positivos», que en
  1:1 da 1. Es lo mismo que no ajustar.

---

# Correcciones sugeridas antes de presentar

1. **Deck de hipótesis, lámina 6:** se queda con la tabla; la lámina 5 la explica con curvas.
2. **Deck de hipótesis, láminas 10 a 12:** para la sustentación, al anexo.
3. **Deck de hipótesis, lámina 18:** cambiar las notas («todavía lo estamos verificando») por
   el discurso de este documento: ya está verificado.
4. **Deck de hipótesis, lámina 22:** marcar el análisis de los sin etiqueta como hecho.
5. **Deck principal, lámina 11:** cambiar «Todas usan la misma configuración de
   hiperparámetros» por «Todas usan el mismo procedimiento y presupuesto de ajuste».
6. **Deck principal, lámina 8 frente a la 4 y la 12:** dice 166 variables en una y 165 en las
   otras. Aclarar «166 columnas: el paso de tiempo y 165 variables».
7. **Deck de hipótesis, lámina 15:** usar la hipótesis registrada («TAGCN sería la más estable»)
   como «qué esperábamos».
8. **Si se presentan los dos decks seguidos:** la lámina 2 del deck de hipótesis repite la 6 y
   la 12 del principal.
