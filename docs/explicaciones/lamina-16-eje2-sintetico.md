# Lámina 16 — Eje 2: grafo sintético con ground-truth

> Bloque: Metodología
> Voz: Alejandro · Página PDF 20 (16/40) · Tiempo objetivo 65 s

## 🎯 Objetivo de la lámina

Presentar el segundo eje del estudio: un grafo construido por el propio equipo, con el propósito
específico de que sí se pueda saber, de antemano, cuál es el patrón de lavado verdadero. Sin este
eje, la plausibilidad y la fidelidad simplemente no serían medibles.

## 📋 Qué dice, item por item

- **"¿Por qué? Medir plausibilidad exige saber cuál subgrafo es el patrón: Elliptic no lo da."** — La
  motivación central: no existe forma de medir si una explicación "acierta" sin conocer la respuesta
  correcta de antemano, y Elliptic no la ofrece.
- **"4 tipologías: structuring, layering, fan-in, fan-out."** — Se inyectan a propósito las cuatro
  tipologías clásicas del lavado de dinero, ya presentadas en la lámina 02, como patrones concretos
  dentro del grafo.
- **"Aristas distractoras + firma atenuada: la métrica discrimina."** — Dos decisiones de diseño
  (explicadas a fondo en la siguiente lámina) que evitan que la tarea sea trivialmente fácil de
  resolver.
- **"Ground-truth ciego al explicador y 3 realizaciones del grafo."** — El patrón verdadero se define
  antes de correr ningún explicador (así ningún método tiene ventaja de fábrica), y se generan tres
  versiones distintas del grafo para comprobar que los hallazgos no dependen de una sola instancia.
- **Figura 4:** una representación del grafo sintético mostrando las tipologías de lavado plantadas y
  el ground-truth por arista.

## 🔑 Conceptos y técnicas que aparecen

- **Grafo sintético:** un grafo generado artificialmente por los propios investigadores, con reglas
  conocidas y controladas, en vez de recolectado del mundo real.
- **Ground-truth ciego al explicador:** que la respuesta correcta se defina en la construcción del
  grafo, **antes** de que se corra ningún método de explicación sobre él, para que no exista ninguna
  posibilidad (ni intencional ni accidental) de que el diseño favorezca a un explicador en particular.
- **Aristas distractoras y firma atenuada:** dos técnicas que "endurecen" la prueba, para que acertar
  el patrón no sea trivial. Se explican en detalle en la
  [lámina 17](lamina-17-como-se-construyo-sintetico.md).
- **Realización del grafo:** ver [glosario](README.md#ground-truth-y-validez-interna-vs-externa) y la
  explicación en la [lámina 13](lamina-13-dos-ejes.md). Aquí se concreta en el número exacto: 3
  realizaciones distintas.

## 📈 Cómo leer la figura / tabla

La Figura 4 muestra el grafo sintético con sus tipologías de lavado resaltadas dentro de la red más
amplia de transacciones "normales" de fondo. La lectura clave es que las tipologías (structuring,
layering, fan-in, fan-out) aparecen como **formas reconocibles** dentro del grafo (agrupaciones,
cadenas, estrellas que convergen o se abren), y que cada arista de esas formas tiene una etiqueta de
ground-truth: se sabe con certeza cuáles pertenecen al patrón de lavado y cuáles no.

## 🎤 El discurso (como se dice en voz alta)

> El segundo eje, nuestro grafo sintetico, responde a una necesidad concreta: para medir si una
> explicacion es plausible, hay que saber de antemano cual es el subgrafo correcto, y Elliptic no lo da.
> Asi que lo construimos. *(pausa)* Inyectamos cuatro tipologias de lavado reconocidas: structuring,
> layering, fan-in y fan-out. Y tomamos tres decisiones para que la medicion no quede servida: simetrizamos
> las aristas para que el vecindario tenga estructura suficiente, anadimos aristas distractoras para que
> acertar no sea trivial, y atenuamos la firma de los atributos para que la tarea no se resuelva sola.
> Un detalle importante que nos van a preguntar: el patron verdadero es ciego al explicador, se define
> en la construccion del grafo, antes de correr ningun metodo. No lo ajustamos para favorecer a nadie.

### Versión ampliada y explicada

*"El segundo eje, nuestro grafo sintetico, responde a una necesidad concreta: para medir si una
explicacion es plausible, hay que saber de antemano cual es el subgrafo correcto, y Elliptic no lo
da. Asi que lo construimos."* — Empieza justificando la necesidad, no presentando el grafo como una
elección arbitraria: se construyó porque era la única forma de llenar el vacío identificado en la
lámina 07.

*"Inyectamos cuatro tipologias de lavado reconocidas: structuring, layering, fan-in y fan-out."* —
Conecta directamente con la lámina 02: estas no son formas inventadas, son las tipologías estándar y
documentadas en la literatura de lavado de dinero.

*"Y tomamos tres decisiones para que la medicion no quede servida: simetrizamos las aristas para que
el vecindario tenga estructura suficiente, anadimos aristas distractoras para que acertar no sea
trivial, y atenuamos la firma de los atributos para que la tarea no se resuelva sola."* — Anticipa,
en una sola frase, las tres decisiones de diseño que se desarrollan a fondo en la siguiente lámina;
el mensaje aquí es que el grafo se diseñó deliberadamente para que fuera **difícil**, no para que
cualquier explicador acertara con facilidad.

*"Un detalle importante que nos van a preguntar: el patron verdadero es ciego al explicador, se
define en la construccion del grafo, antes de correr ningun metodo. No lo ajustamos para favorecer a
nadie."* — Este es el punto de mayor importancia frente a un jurado exigente: se anticipa
explícitamente la objeción de circularidad (que el grafo se haya diseñado para que gane cierto
explicador) y se responde de frente, antes de que se pregunte.

## 🧑‍⚖️ Preguntas de jurado probables

- **"¿No es circular construir el propio dataset y luego medir sobre él?"** → No, porque el
  ground-truth se fija en la construcción del generador, antes de correr ningún explicador, y la
  mejor prueba de que no hay sesgo confirmatorio es que el grafo terminó **refutando** la hipótesis
  central de la tesis (el "puente" entre estabilidad y plausibilidad), algo que nadie diseñaría a
  propósito. Ver el desarrollo completo en `docs/DEFENSA_R2_evidencia_sintetica.md`.
- **"¿Por qué estas cuatro tipologías y no otras?"** → Son las tipologías canónicas y más citadas en
  la literatura de detección de lavado de dinero, lo que les da relevancia general más allá de este
  estudio particular.

## 🧠 En una frase

El grafo sintético se construyó para llenar exactamente el vacío que Elliptic no puede cubrir: tener
una respuesta correcta conocida de antemano, fijada antes de correr cualquier explicador.
