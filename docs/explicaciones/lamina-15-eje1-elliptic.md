# Lámina 15 — Eje 1: Elliptic Bitcoin Dataset

> Bloque: Metodología
> Voz: Alejandro · Página PDF 19 (15/40) · Tiempo objetivo 60 s

## 🎯 Objetivo de la lámina

Describir con números concretos el dataset real que se usa como primer eje del estudio, y adelantar
el dato más importante que va a condicionar todo el análisis sobre este eje: la dispersión extrema de
su vecindario.

## 📋 Qué dice, item por item

- **"203.769 nodos · 234.355 aristas · 166 features · 49 pasos."** — El tamaño del grafo: más de 200
  mil transacciones, conectadas por más de 200 mil aristas, cada una con 166 atributos, distribuidas a
  lo largo de 49 pasos de tiempo.
- **"Ilícitas ~2,2% (razón ~1:9 en el subconjunto etiquetado)."** — De las transacciones que sí tienen
  etiqueta (lícita/ilícita), solo el 2,2% son ilícitas; dentro de esa parte etiquetada, la proporción
  es de aproximadamente una ilícita por cada nueve lícitas.
- **"Partición temporal causal: entrenamiento en los pasos 1 a 34, validación de 35 a 42 y test de 43
  a 49."** — Los datos se dividen respetando el orden del tiempo: se entrena con los pasos más
  antiguos, se valida con los siguientes, y se evalúa (test) con los más recientes. Esto imita cómo
  operaría un sistema real, que aprende del pasado y se enfrenta al futuro.
- **"Campo receptivo mediana ~2 nodos: solo el Spearman discrimina."** — El dato más importante de la
  lámina: el vecindario típico de una transacción es minúsculo (mediana de solo 2 nodos), por lo que
  las métricas basadas en aristas no sirven para distinguir estabilidad; solo la correlación de
  Spearman sobre el ranking de atributos discrimina.
- **Figura 3:** un gráfico que muestra la dispersión de la topología de Elliptic (grado y campo
  receptivo).

## 🔑 Conceptos y técnicas que aparecen

- **Grado de un nodo:** el número de aristas (conexiones) que tiene ese nodo. Un grado bajo significa
  pocas conexiones.
- **Partición temporal causal:** dividir los datos en entrenamiento/validación/test respetando el
  orden cronológico, en vez de mezclar todo al azar. Es la única forma honesta de evaluar un modelo
  que en el mundo real tendría que predecir el futuro con datos del pasado.
- **Dispersión de la topología:** qué tan "vacío" o "lleno" es el grafo en términos de conexiones. En
  Elliptic, la dispersión es extrema: la mayoría de los nodos tiene muy pocos vecinos.
- **Spearman como única métrica que discrimina:** cuando el vecindario es tan pequeño, medir
  "estabilidad de la máscara de aristas" no tiene sentido (casi no hay aristas entre las que elegir),
  así que la única forma de medir estabilidad que aporta información real es comparar el ranking de
  los 166 atributos con la correlación de Spearman. Ver
  [glosario](README.md#correlación-de-spearman).

## 📈 Cómo leer la figura / tabla

La Figura 3 muestra la dispersión de la topología de Elliptic: probablemente un histograma o gráfico
de distribución del grado de los nodos y del tamaño del campo receptivo. La lectura clave es que la
gran mayoría de los nodos se concentra en valores muy bajos (pocos vecinos), con una mediana de
aproximadamente 2 nodos en el campo receptivo. Esto confirma visualmente por qué las métricas basadas
en aristas se "saturan" (casi todos los valores posibles son extremos, 0 o 1, sin espacio para
distinguir diferencias reales).

## 🎤 El discurso (como se dice en voz alta)

> El primer eje en detalle. Elliptic tiene doscientos tres mil setecientos sesenta y nueve nodos,
> doscientas treinta y cuatro mil aristas, ciento sesenta y seis atributos por nodo y cuarenta y nueve
> pasos temporales. Las transacciones ilicitas son apenas el dos coma dos por ciento, una razon cercana a
> uno a nueve en la parte etiquetada. *(pausa)* Hicimos una particion temporal causal: entrenamos con el
> pasado y evaluamos con el futuro, que es como opera un sistema real. Y hay un dato que gobierna todo lo
> demas: el vecindario tipico de un nodo tiene una mediana de unos dos nodos. Es un grafo extremadamente
> disperso. Por eso, como veran, la unica metrica de estabilidad que discrimina bien aqui es la
> correlacion de Spearman entre rankings de atributos. Las metricas de aristas se saturan.

### Versión ampliada y explicada

*"El primer eje en detalle. Elliptic tiene doscientos tres mil setecientos sesenta y nueve nodos,
doscientas treinta y cuatro mil aristas, ciento sesenta y seis atributos por nodo y cuarenta y nueve
pasos temporales."* — Da la escala exacta del dataset: es un grafo grande, con muchos atributos por
nodo, distribuido en el tiempo.

*"Las transacciones ilicitas son apenas el dos coma dos por ciento, una razon cercana a uno a nueve en
la parte etiquetada."* — Cuantifica el desbalance real del dataset original: incluso sin manipular
nada, el fraude ya es muy raro.

*"Hicimos una particion temporal causal: entrenamos con el pasado y evaluamos con el futuro, que es
como opera un sistema real."* — Justifica la partición por tiempo como la opción metodológicamente
correcta: un sistema de detección real siempre está prediciendo sobre transacciones que ocurren
**después** de las que usó para entrenar, así que evaluarlo de otra forma sería poco realista.

*"Y hay un dato que gobierna todo lo demas: el vecindario tipico de un nodo tiene una mediana de unos
dos nodos. Es un grafo extremadamente disperso."* — Marca este dato como el más importante de la
lámina, porque va a condicionar decisiones metodológicas en varias láminas posteriores.

*"Por eso, como veran, la unica metrica de estabilidad que discrimina bien aqui es la correlacion de
Spearman entre rankings de atributos. Las metricas de aristas se saturan."* — Explica la consecuencia
directa: con vecindarios tan pequeños, cualquier métrica basada en aristas (por ejemplo, el índice de
Jaccard) tiende a dar siempre valores extremos (casi todo se repite o casi nada se repite), sin
distinguir realmente si el explicador es o no consistente. Por eso el Spearman sobre atributos se
convierte en la métrica primaria para todo el eje real.

## 🧑‍⚖️ Preguntas de jurado probables

- **"¿Por qué el vecindario de Elliptic es tan pequeño?"** → Es una característica propia de la
  estructura real de las transacciones de Bitcoin en este dataset, documentada en el trabajo original
  de Weber y colaboradores (2019); no es una decisión de los autores de esta tesis, es un dato del
  dominio.
- **"¿Cómo afecta esta dispersión a la validez de los resultados sobre Elliptic?"** → Limita lo que se
  puede medir ahí (solo estabilidad, no plausibilidad ni fidelidad), pero no invalida la estabilidad
  medida con Spearman sobre atributos, que sigue siendo una medición legítima y es la que se usa como
  eje de validez externa.

## 🧠 En una frase

Elliptic es un grafo real, grande y muy disperso (vecindarios de apenas 2 nodos), lo que hace que la
única métrica de estabilidad que funciona ahí sea la correlación de Spearman sobre atributos.
