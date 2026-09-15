# Lámina 06 — Tres hipótesis falsables

> Bloque: Problema y motivación
> Voz: Alejandro · Página PDF 8 (6/40) · Tiempo objetivo 55 s

## 🎯 Objetivo de la lámina

Comprometer al equipo, en público y por escrito, con tres predicciones concretas **antes** de haber
visto los resultados. Esto es lo que le da fuerza científica a lo que viene después: cuando esas
predicciones se pongan a prueba en los resultados, su refutación (o confirmación) va a significar
algo real, no una interpretación acomodada después de ver los números.

## 📋 Qué dice, item por item

- **H1 — "La estabilidad se degradará al agravarse el desbalance."** — La predicción es que, cuanto
  más raro sea el fraude, más inestables (menos consistentes) serán las explicaciones.
- **H2 — "TAGCN será la más estable, por su alcance multi-salto."** — La predicción es que, de las
  cuatro arquitecturas, TAGCN va a ganar en estabilidad, porque su diseño le permite "ver" más lejos
  en la red (varios saltos de distancia) de una sola vez.
- **H3 — "Un explicador más estable señalará mejor el patrón real (plausibilidad)."** — La predicción
  es que si un explicador es consistente (siempre da la misma respuesta), esa respuesta también va a
  ser más acertada sobre el patrón real de lavado.
- **Nota final:** comprometerse con predicciones concretas es lo que hace que refutarlas signifique
  algo. El desenlace de las tres se cuenta en la sección de Resultados.

## 🔑 Conceptos y técnicas que aparecen

- **Hipótesis falsable:** una predicción que se puede **comprobar que es falsa** con datos. Si una
  afirmación no se puede refutar con ningún experimento posible, no es una hipótesis científica útil.
  Las tres de esta lámina sí lo son: cada una se puede confirmar o tumbar con un número concreto.
- **Comprometerse antes de ver los datos:** es una práctica de rigor científico. Si uno formula sus
  predicciones **después** de ver los resultados, siempre puede "acomodarlas" para que encajen (eso
  se llama, informalmente, hacer trampa con el análisis). Formularlas antes evita esa tentación.
- **Alcance multi-salto (adelanto de TAGCN):** significa que, en una sola operación, el modelo puede
  usar información de varios pasos de distancia en la red (no solo del vecino inmediato). Se explica
  a fondo en la [lámina 09](lamina-09-cuatro-arquitecturas.md).

## 📈 Cómo leer la figura / tabla

No aplica: es una lista de tres hipótesis con una nota final, sin figura ni tabla.

## 🎤 El discurso (como se dice en voz alta)

> Y para no medir a ciegas, nos comprometimos con tres hipotesis falsables antes de ver los datos.
> *(pausa)* La primera, que la estabilidad se degradaria al agravarse el desbalance: cuanto mas raro el
> fraude, mas inestables las explicaciones. La segunda, que TAGCN seria la arquitectura mas estable, por
> su alcance multi-salto. Y la tercera, que un explicador mas estable senalaria tambien mejor el patron
> real de lavado, es decir, que la consistencia implicaria acierto. *(pausa)* Comprometerse con
> predicciones concretas antes de mirar los datos es lo que hace que refutarlas signifique algo. Adelanto
> que las tres se matizaron o se cayeron, y que contarlo con honestidad es parte del aporte. El desenlace
> de cada una lo vera el jurado en la seccion de resultados.

### Versión ampliada y explicada

*"Y para no medir a ciegas, nos comprometimos con tres hipotesis falsables antes de ver los datos."*
— La frase "no medir a ciegas" es clave: sin una predicción previa, cualquier resultado podría
interpretarse como "el hallazgo esperado". Con una predicción escrita antes, el resultado se juzga
contra algo concreto.

*"La primera, que la estabilidad se degradaria al agravarse el desbalance: cuanto mas raro el fraude,
mas inestables las explicaciones."* — H1, explicada con la intuición que la motiva: menos ejemplos de
fraude para aprender, se esperaría, debería significar explicaciones menos confiables.

*"La segunda, que TAGCN seria la arquitectura mas estable, por su alcance multi-salto."* — H2: la
intuición es que ver más lejos en la red (varios saltos) da más contexto, y por tanto más consistencia.

*"Y la tercera, que un explicador mas estable senalaria tambien mejor el patron real de lavado, es
decir, que la consistencia implicaria acierto."* — H3: la intuición, muy extendida, es que si algo es
repetible, probablemente sea porque "encontró algo real". Esta tesis va a poner esa intuición a
prueba directamente.

*"Comprometerse con predicciones concretas antes de mirar los datos es lo que hace que refutarlas
signifique algo."* — Es la justificación metodológica de por qué esta lámina existe como tal, antes
de cualquier resultado.

*"Adelanto que las tres se matizaron o se cayeron, y que contarlo con honestidad es parte del
aporte."* — Un adelanto deliberado y valiente: se avisa desde ya que las tres hipótesis no se
confirmaron, y se presenta eso como una fortaleza, no como una debilidad.

## 🧑‍⚖️ Preguntas de jurado probables

- **"Si las tres hipótesis se cayeron, ¿qué valor tiene haberlas planteado?"** → Justamente ese: sin
  haberlas planteado por adelantado, no se podría distinguir un hallazgo real de una lectura
  acomodada después de ver los datos. El valor científico de un no-resultado depende de que la
  predicción existiera antes.
- **"¿Por qué se esperaba que TAGCN fuera la más estable?"** → Por su capacidad de usar filtros que
  alcanzan varios saltos en la red de una sola vez, lo que en teoría le daría más contexto para
  generar explicaciones consistentes (se explica en la lámina 09).

## 🧠 En una frase

Antes de ver un solo dato, el equipo se comprometió con tres predicciones concretas y refutables,
para que su desenlace tuviera valor científico real.
