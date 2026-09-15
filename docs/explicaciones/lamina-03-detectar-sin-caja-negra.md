# Lámina 03 — El problema: detectar sin caja negra

> Bloque: Problema y motivación
> Voz: Alejandro · Página PDF 5 (3/40) · Tiempo objetivo 65 s

## 🎯 Objetivo de la lámina

Dimensionar el problema con cifras concretas, mostrar por qué los sistemas actuales (reglas fijas)
no son suficientes, y presentar a las redes de grafos como la alternativa poderosa que, sin embargo,
trae un problema nuevo: son cajas negras.

## 📋 Qué dice, item por item

- **"El lavado mueve entre el 2 y el 5% del PIB mundial."** — Una cifra que dimensiona el problema:
  es un fenómeno económico enorme, no un caso aislado.
- **"Los sistemas por reglas generan entre el 95 y el 98% de falsos positivos: reglas rígidas que los
  criminales aprenden a esquivar."** — Los sistemas tradicionales usan reglas fijas ("alerta si una
  transacción supera tal monto"), y de cada 100 alertas que generan, entre 95 y 98 resultan ser
  ruido: transacciones normales marcadas por error. Además, como las reglas son conocidas y fijas,
  los criminales aprenden a evitarlas (por ejemplo, fraccionando montos para no cruzar el umbral).
- **"Las GNN modelan la red de flujos y superan a lo tabular, que mira cada transacción aislada."** —
  Las redes neuronales de grafos sí consideran las conexiones entre cuentas, y por eso les va mejor
  que a los métodos que tratan cada transacción como una fila independiente en una tabla.
- **"Pero son cajas negras: no dicen por qué marcan una transacción."** — El precio de ese mejor
  rendimiento es que estos modelos son opacos: aciertan, pero no explican su razonamiento.
- **Figura 1:** un esquema que compara el enfoque tradicional basado en reglas (un embudo que filtra
  transacciones con muchos falsos positivos) contra el enfoque relacional basado en grafos (una red
  de transacciones con ciclos y clústeres sospechosos).

## 🔑 Conceptos y técnicas que aparecen

- **Sistema por reglas:** un conjunto de condiciones fijas escritas por humanos ("si el monto supera
  X, marcar"). Es fácil de entender pero fácil de esquivar y genera muchísimo ruido.
- **Caja negra:** un modelo que da una respuesta (aquí, "esta transacción es sospechosa") sin explicar
  el razonamiento detrás. El problema no es que se equivoque más; el problema es que **no se sabe
  por qué** decide lo que decide.
- **Modelo tabular vs. relacional:** un modelo tabular mira cada fila (transacción) por separado,
  ignorando cómo se conecta con las demás; un modelo relacional (como una GNN) sí usa esas conexiones.

## 📈 Cómo leer la figura / tabla

La Figura 1 contrasta dos enfoques lado a lado. A la izquierda, el enfoque **tradicional basado en
reglas**: un embudo que recibe muchas transacciones y filtra, pero deja pasar (o marca de más) según
reglas fijas — de ahí el altísimo porcentaje de falsos positivos y el costo de cumplimiento. A la
derecha, el enfoque **relacional basado en grafos**: se ve la red de transacciones con sus ciclos y
agrupaciones (clústeres), que es justo la información que un modelo tabular no puede ver. La lectura
clave es que el problema no es solo "qué tan bueno es el filtro", sino "qué tipo de información puede
usar el filtro".

## 🎤 El discurso (como se dice en voz alta)

> Con eso claro, dimensionemos el problema. El lavado mueve entre el dos y el cinco por ciento del
> producto interno bruto mundial. *(pausa)* Los sistemas tradicionales funcionan por reglas fijas, y ese
> enfoque genera entre el noventa y cinco y el noventa y ocho por ciento de falsos positivos: de cada
> cien alertas, casi todas son ruido que un analista revisa a mano. Y son reglas rigidas, que los
> criminales aprenden a esquivar. *(pausa)* Las redes de grafos ofrecen una alternativa poderosa, porque
> modelan las transacciones como lo que son, una red de flujos, y superan a los metodos tabulares que
> miran cada transaccion aislada. El problema es que estas redes son cajas negras: aciertan, pero no
> dicen por que marcaron una transaccion. Y ahi es donde entra la explicabilidad.

### Versión ampliada y explicada

*"Con eso claro, dimensionemos el problema."* — Frase de enlace: ya se explicó qué es el lavado
(lámina anterior), ahora se ponen cifras que muestran su tamaño real.

*"El lavado mueve entre el dos y el cinco por ciento del producto interno bruto mundial."* — Un dato
de escala macroeconómica: no es un problema marginal, es comparable al tamaño de economías enteras.

*"Los sistemas tradicionales funcionan por reglas fijas, y ese enfoque genera entre el noventa y
cinco y el noventa y ocho por ciento de falsos positivos..."* — Aquí está el costo operativo real:
un analista humano tiene que revisar manualmente casi todas las alertas, y casi todas resultan ser
ruido. Es ineficiente y costoso.

*"Y son reglas rigidas, que los criminales aprenden a esquivar."* — Es el segundo problema de las
reglas: no solo generan ruido, también son predecibles y, por tanto, evadibles.

*"Las redes de grafos ofrecen una alternativa poderosa, porque modelan las transacciones como lo que
son, una red de flujos, y superan a los metodos tabulares..."* — Se presenta la solución candidata:
un modelo que sí ve las conexiones entre cuentas.

*"El problema es que estas redes son cajas negras: aciertan, pero no dicen por que marcaron una
transaccion. Y ahi es donde entra la explicabilidad."* — El giro de la lámina: la solución al primer
problema (reglas rígidas) trae un problema nuevo (opacidad), que es justo el que aborda esta tesis.

## 🧑‍⚖️ Preguntas de jurado probables

- **"¿De dónde salen esas cifras del 2-5% del PIB y del 95-98% de falsos positivos?"** → De informes
  de organismos internacionales (UNODC y GAFI/FATF), citados en el Capítulo 2 del manuscrito.
- **"¿Por qué no simplemente mejorar las reglas en vez de usar un modelo de IA?"** → Porque el
  problema de fondo es estructural: una regla fija sobre atributos individuales no puede capturar un
  patrón que solo existe en la **red de conexiones**, sin importar qué tan bien calibrada esté.

## 🧠 En una frase

Las redes de grafos superan a los métodos tradicionales para detectar lavado, pero a cambio de ser
cajas negras, y ahí nace la necesidad de la explicabilidad.
