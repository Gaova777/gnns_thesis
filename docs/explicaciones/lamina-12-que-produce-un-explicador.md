# Lámina 12 — Qué produce, en concreto, un explicador

> Bloque: Metodología
> Voz: Alejandro · Página PDF 16 (12/40) · Tiempo objetivo 60 s

## 🎯 Objetivo de la lámina

Antes de entrar en el diseño metodológico, aterrizar de forma muy concreta qué **objeto exacto**
produce un explicador, porque toda la tesis mide propiedades sobre ese objeto. Es la lámina "puente"
entre el marco conceptual (láminas 08-11) y el diseño experimental que viene después.

## 📋 Qué dice, item por item

- **Diagrama (izquierda):** un nodo objetivo (resaltado) rodeado de varios vecinos; algunas aristas
  están marcadas en un color destacado ("aristas relevantes") y otras en gris ("aristas
  irrelevantes"). Ilustra que el explicador no marca todo el vecindario por igual, solo una parte.
- **"Ante una transacción marcada como ilícita, el explicador devuelve dos cosas":**
  1. **Una máscara de aristas:** qué conexiones del vecindario sostienen la predicción.
  2. **Un ranking de features:** qué atributos del nodo pesaron más, de los 166 disponibles.
- **Recuadro "Las tres preguntas":** se repiten aquí, aplicadas directamente a este objeto:
  estabilidad (¿sale lo mismo al repetir?), plausibilidad (¿coincide con el patrón real?) y fidelidad
  (¿es lo que el modelo usó?).
- **Nota final:** en Elliptic, el vecindario tiene una mediana de aproximadamente 2 nodos, así que ahí
  solo el ranking de features discrimina (la máscara de aristas es casi trivial, porque casi no hay
  aristas entre las que elegir).

## 🔑 Conceptos y técnicas que aparecen

- **Máscara de aristas y ranking de features:** ver [glosario](README.md#explicador-post-hoc-máscara-y-ranking).
  Aquí se ve de forma visual: en el diagrama, la máscara son las aristas resaltadas; el ranking son
  los 166 atributos del nodo, ordenados de más a menos importante.
- **166 features:** el número exacto de atributos que Elliptic asigna a cada transacción (datos como
  montos, tiempos, y estadísticas agregadas del vecindario). El ranking de importancia se calcula
  sobre estos 166 valores.
- **Dispersión del vecindario (adelanto):** el hecho de que en Elliptic los vecindarios sean
  minúsculos (mediana de ~2 nodos) es un dato que va a explicar muchas cosas más adelante, en
  particular por qué la estabilidad ahí se mide con Spearman sobre atributos y no sobre aristas. Se
  detalla en la [lámina 15](lamina-15-eje1-elliptic.md).

## 📈 Cómo leer la figura / tabla

El diagrama (Figura 2) muestra un nodo objetivo en el centro (marcado en un color distinto), rodeado
de vecinos conectados por líneas. Algunas líneas están resaltadas en un color fuerte (las aristas que
el explicador considera relevantes) y otras quedan en un tono apagado (irrelevantes). La lectura es:
de todas las conexiones posibles alrededor del nodo objetivo, el explicador solo destaca un
subconjunto pequeño como "lo que sostuvo la predicción". Esa selección es exactamente la "máscara de
aristas" mencionada en el texto.

## 🎤 El discurso (como se dice en voz alta)

> Antes de entrar en la metodologia quiero aterrizar que es, en concreto, una explicacion, porque toda la
> tesis mide propiedades de este objeto. *(pausa)* Cuando el modelo marca una transaccion como ilicita y
> le pedimos al explicador que justifique esa decision, lo que devuelve son dos cosas. Primero, una
> mascara de aristas: de todas las conexiones del vecindario de esa transaccion, cuales sostienen la
> prediccion, que son las que ven resaltadas en el diagrama. Y segundo, un ranking de atributos: de las
> ciento sesenta y seis features que tiene cada nodo, cuales pesaron mas. *(pausa)* Sobre ese objeto se
> definen las tres preguntas de la lamina anterior. La estabilidad pregunta si al repetir el calculo sale
> la misma mascara y el mismo ranking. La plausibilidad, si eso coincide con el patron real de lavado. Y
> la fidelidad, si es de verdad lo que el modelo uso. *(pausa)* Un detalle que va a explicar varias cosas
> mas adelante: en Elliptic el vecindario tipico tiene una mediana de unos dos nodos, asi que la mascara
> de aristas es casi trivial y solo el ranking de atributos discrimina. Por eso ahi medimos estabilidad
> con Spearman sobre features.

### Versión ampliada y explicada

*"Antes de entrar en la metodologia quiero aterrizar que es, en concreto, una explicacion, porque toda
la tesis mide propiedades de este objeto."* — Justifica por qué esta lámina existe justo aquí, antes
del diseño experimental: sin tener claro qué es exactamente el "objeto" que se va a medir, el diseño
metodológico no tendría sentido.

*"Cuando el modelo marca una transaccion como ilicita y le pedimos al explicador que justifique esa
decision, lo que devuelve son dos cosas."* — Plantea el escenario concreto: hay una predicción, y se
le pide al explicador que la justifique.

*"Primero, una mascara de aristas: de todas las conexiones del vecindario de esa transaccion, cuales
sostienen la prediccion, que son las que ven resaltadas en el diagrama."* — Conecta directamente con
la Figura 2, dando una referencia visual inmediata.

*"Y segundo, un ranking de atributos: de las ciento sesenta y seis features que tiene cada nodo,
cuales pesaron mas."* — Fija el número exacto (166), que va a reaparecer en varias láminas más
adelante (por ejemplo, al explicar el bug del truncamiento en la lámina 20).

*"Sobre ese objeto se definen las tres preguntas de la lamina anterior."* — Conecta con la lámina 11:
las tres propiedades no son abstractas, se aplican concretamente sobre esta máscara y este ranking.

*"Un detalle que va a explicar varias cosas mas adelante: en Elliptic el vecindario tipico tiene una
mediana de unos dos nodos, asi que la mascara de aristas es casi trivial y solo el ranking de
atributos discrimina."* — Es un adelanto deliberado de un dato que va a ser clave para entender por
qué, en Elliptic, la estabilidad se mide con Spearman sobre atributos (y no, por ejemplo, con el
índice de Jaccard sobre aristas, que se satura porque casi no hay aristas que comparar).

## 🧑‍⚖️ Preguntas de jurado probables

- **"¿Por qué la máscara de aristas es 'casi trivial' cuando el vecindario es tan pequeño?"** → Porque
  con solo 1 o 2 aristas disponibles, casi cualquier selección coincide entre corridas distintas: no
  hay suficiente variedad como para que la máscara discrimine si el explicador es o no estable.
- **"¿Se pueden medir estabilidad, plausibilidad y fidelidad sobre la máscara y sobre el ranking a la
  vez?"** → Sí, en principio ambas se pueden medir sobre cada objeto, pero el eje real (Elliptic) solo
  permite medir bien la estabilidad, y solo sobre el ranking de atributos, por la dispersión del grafo.

## 🧠 En una frase

Un explicador entrega dos objetos concretos —una máscara de aristas y un ranking de atributos—, y
sobre esos dos objetos exactos se miden la estabilidad, la plausibilidad y la fidelidad.
