# Lámina 11 — Tres propiedades que no son lo mismo

> Bloque: Marco conceptual
> Voz: Alejandro · Página PDF 14 (11/40) · Tiempo objetivo 65 s

## 🎯 Objetivo de la lámina

Esta es, en palabras del propio guion, "la tesis central": establecer que cuando se dice que una
explicación es "buena", en realidad se están mezclando tres preguntas distintas, y que esta tesis va
a demostrar con datos que hay que medirlas por separado, porque **no se implican entre sí**.

## 📋 Qué dice, item por item

- **Estabilidad — "¿La explicación se reproduce entre semillas y perturbaciones?"** — Si se repite el
  cálculo (con otra semilla de azar, o con una pequeña perturbación de la entrada), ¿sale lo mismo?
- **Plausibilidad — "¿Señala el patrón real de lavado (ground-truth)?"** — ¿La explicación apunta a lo
  que un experto humano reconocería como el patrón verdadero de fraude?
- **Fidelidad — "¿Refleja de verdad la decisión interna del modelo?"** — ¿Lo que la explicación dice
  que importó es, de verdad, lo que el modelo usó internamente para decidir?
- **"Tesis central: las tres son empíricamente independientes."** — Frase destacada en el centro de la
  lámina: no basta con medir una y asumir que las otras dos van de la mano.
- **Nota final:** un explicador puede ser estable y equivocado, como un reloj parado, que siempre
  marca lo mismo y siempre está mal.

## 🔑 Conceptos y técnicas que aparecen

- **Estabilidad, plausibilidad, fidelidad:** ver el desarrollo completo en el
  [glosario](README.md#las-tres-propiedades-estabilidad-plausibilidad-fidelidad). Vale la pena
  repetir aquí la diferencia central: estabilidad pregunta "¿es consistente?", plausibilidad pregunta
  "¿es lo que un humano reconocería como correcto?", y fidelidad pregunta "¿es lo que el modelo
  realmente usó?". Son tres preguntas distintas, con respuestas potencialmente distintas.
- **Independencia empírica:** que dos (o tres) cosas no estén relacionadas entre sí, tal como lo
  muestran los datos, aunque intuitivamente uno esperaría que sí lo estuvieran. Esta tesis pone a
  prueba precisamente si la estabilidad "implica" acierto (plausibilidad), y la respuesta es que no
  (ver [lámina 25](lamina-25-resultado3-puente-nulo.md)).

## 📈 Cómo leer la figura / tabla

No aplica: son tres recuadros (uno por propiedad) más una frase destacada en el centro, sin figura
ni tabla numérica.

## 🎤 El discurso (como se dice en voz alta)

> Esta lamina contiene la tesis central, asi que me detengo. Cuando decimos que una explicacion es
> "buena", en realidad mezclamos tres preguntas distintas. La estabilidad pregunta si la explicacion se
> reproduce cuando cambio la semilla o perturbo un poco la entrada. La plausibilidad pregunta si senala el
> patron real de lavado, lo que un experto reconoceria. Y la fidelidad pregunta si refleja de verdad lo
> que el modelo uso para decidir. *(pausa)* La tesis central es que estas tres son dimensiones distintas
> que no se implican entre si. Una explicacion puede ser muy estable y aun asi apuntar al patron
> equivocado, como un reloj parado, que siempre marca la misma hora y siempre esta mal. Puede ser
> plausible para un humano y no reflejar el mecanismo del modelo. Gran parte de la literatura reporta una
> sola de estas y la llama calidad. Nosotros vamos a mostrar, con datos, que hay que medir las tres por
> separado.

### Versión ampliada y explicada

*"Esta lamina contiene la tesis central, asi que me detengo."* — Un aviso explícito de que lo que
sigue es lo más importante conceptualmente de toda la presentación; conviene que el orador haga una
pausa notable aquí y hable más despacio.

*"Cuando decimos que una explicacion es 'buena', en realidad mezclamos tres preguntas distintas."* —
Empieza señalando un error común (mezclar tres ideas en una sola palabra, "buena"), que es justamente
lo que esta tesis viene a desenredar.

*"La estabilidad pregunta si la explicacion se reproduce cuando cambio la semilla o perturbo un poco
la entrada."* — Define estabilidad con dos formas concretas de ponerla a prueba: cambiar la semilla
de azar del cálculo, o alterar levemente la entrada.

*"La plausibilidad pregunta si senala el patron real de lavado, lo que un experto reconoceria."* —
Define plausibilidad conectándola con un criterio humano y externo: lo que un experto validaría como
correcto.

*"Y la fidelidad pregunta si refleja de verdad lo que el modelo uso para decidir."* — Define fidelidad
como una pregunta **interna** al modelo, distinta de la plausibilidad (que es una pregunta externa,
sobre el mundo real).

*"La tesis central es que estas tres son dimensiones distintas que no se implican entre si."* — Es la
afirmación que estructura toda la tesis: no asumir que una implica las otras.

*"Una explicacion puede ser muy estable y aun asi apuntar al patron equivocado, como un reloj parado,
que siempre marca la misma hora y siempre esta mal."* — La analogía central de toda la defensa: algo
perfectamente consistente puede, al mismo tiempo, estar sistemáticamente equivocado. Consistencia y
acierto son cosas distintas.

*"Puede ser plausible para un humano y no reflejar el mecanismo del modelo."* — El otro lado de la
independencia: algo que "le hace sentido" a una persona no necesariamente es lo que el modelo usó de
verdad (esto se vuelve muy concreto en la disociación de la lámina 24).

*"Gran parte de la literatura reporta una sola de estas y la llama calidad. Nosotros vamos a mostrar,
con datos, que hay que medir las tres por separado."* — Cierra posicionando el aporte de la tesis
frente al estado del arte: muchos trabajos previos hablan de "calidad" de una explicación como si
fuera una sola cosa, cuando en realidad son tres preguntas distintas que pueden dar respuestas
distintas.

## 🧑‍⚖️ Preguntas de jurado probables

- **"¿Cómo se puede probar que estas tres propiedades son realmente independientes, y no que
  simplemente no las midieron bien?"** → Se mide cada una con su propia métrica sobre el mismo grafo
  sintético con ground-truth, y se calcula la correlación entre pares; el resultado más contundente es
  que la correlación entre estabilidad y plausibilidad es prácticamente cero, con un intervalo de
  confianza que incluye el cero (ver lámina 25).
- **"¿No sería más simple reportar solo la que más le importa al usuario final?"** → No, porque según
  el propósito de la auditoría (reconocer el patrón, o confiar en que refleja el mecanismo del
  modelo), la propiedad relevante cambia; reducirlo a una sola escondería esa diferencia crítica (ver
  la matriz de recomendación, lámina 32).

## 🧠 En una frase

Estabilidad, plausibilidad y fidelidad son tres preguntas distintas sobre una explicación, y una
explicación puede cumplir una sin cumplir las otras dos.
