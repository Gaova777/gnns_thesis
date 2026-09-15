# Lámina 25 — Resultado 3: el puente que no existe

> Bloque: Resultados
> Voz: Juan Diego · Página PDF 30 (25/40) · Tiempo objetivo 60 s

## 🎯 Objetivo de la lámina

Cerrar la tercera hipótesis (H3) mostrando que una creencia muy intuitiva resulta falsa: que un
explicador más **estable** también sea más **plausible** (que la consistencia implique acierto). Es
un no-resultado, y se reporta con honestidad precisamente porque contradice lo que se esperaba.

## 📋 Qué dice, item por item

- **"Nuestra hipótesis central: una explicación más estable señalaría mejor el patrón real."** —
  Recuerda H3 tal como se planteó en la lámina 06.
- **"El dato dice que no. Correlación entre estabilidad y plausibilidad r=−0,01, IC 95% de −0,038 a
  +0,011."** — El número central de la lámina: la correlación es prácticamente cero.
- **"El intervalo incluye el cero: no hay relación."** — La lectura honesta de ese intervalo de
  confianza: como va de un valor negativo a uno positivo pasando por el cero, no hay evidencia de
  ninguna relación (ni positiva ni negativa).
- **Frase en cursiva:** "Es un resultado nulo, y lo reportamos porque contradice lo que esperábamos."
- **Figura 8:** un gráfico de dispersión que muestra la relación entre estabilidad y plausibilidad
  para GNNExplainer, con una línea ajustada casi plana, y r=−0,01 con el intervalo de confianza
  incluyendo el cero.

## 🔑 Conceptos y técnicas que aparecen

- **Correlación cercana a cero:** ver [glosario](README.md#correlación-de-spearman). Un valor de
  −0,01 está prácticamente en el punto neutro entre −1 y +1: no hay ninguna tendencia detectable, ni a
  favor ni en contra.
- **Intervalo de confianza que incluye el cero:** ver [glosario](README.md#bootstrap-e-intervalo-de-confianza).
  Cuando el rango de valores posibles para la correlación real abarca tanto números negativos como
  positivos, la conclusión honesta es que no se puede afirmar que exista relación alguna.
- **No-resultado (resultado nulo):** un hallazgo donde se buscaba una relación y no se encontró. Su
  valor científico depende de que la hipótesis se haya planteado **antes** de ver los datos (ver
  lámina 06): sin ese compromiso previo, un no-resultado no dice mucho, pero con él, refutar una
  hipótesis concreta es tan valioso como confirmarla.
- **Analogía del reloj parado:** un reloj parado es perfectamente estable (siempre marca la misma
  hora) y siempre está mal. Estabilidad y acierto son propiedades **independientes**: ser consistente
  no garantiza tener razón. Ver [banco de analogías](README.md#banco-de-analogías-para-dar-coherencia-entre-láminas).

## 📈 Cómo leer la figura / tabla

La Figura 8 muestra, en el eje horizontal, la estabilidad de la explicación (Spearman), y en el eje
vertical, la plausibilidad (recuerdo de la tipología). Cada punto es una configuración distinta. La
lectura clave es la **línea roja ajustada**: si el puente existiera, esta línea subiría claramente de
izquierda a derecha (más estabilidad → más plausibilidad). En cambio, la línea aparece **casi
horizontal**, confirmando visualmente que moverse hacia una mayor estabilidad no se traduce en una
mayor plausibilidad.

## 🎤 El discurso (como se dice en voz alta)

> Esta lamina cierra nuestra hipotesis central, y tambien se cae. *(pausa)* Esperabamos que una explicacion
> mas estable fuera tambien mas plausible, es decir que la consistencia entre ejecuciones implicara acierto
> sobre el patron real. Es una intuicion que esta implicita en buena parte de la literatura y nunca se
> habia contrastado de frente, porque para contrastarla hace falta medir las dos cosas a la vez sobre las
> mismas explicaciones, y eso exige conocer de antemano cual es el patron verdadero, algo que los datos reales no dan y el grafo sintetico si. *(pausa)* Los datos dicen que no. La correlacion entre
> estabilidad y plausibilidad es de menos cero coma cero uno, con un intervalo de confianza que va de menos
> cero coma cero treinta y ocho a mas cero coma cero once, o sea que incluye el cero. Es un puente nulo. Un
> explicador estable no es por ello mas acertado sobre el patron real, y ambas propiedades hay que medirlas
> por separado. *(pausa)* Y hay un detalle que refuerza la conclusion: cuando desagregamos por tipologia, el
> signo de la relacion cambia segun cual mires, positivo en structuring y en fan-out, negativo en layering.
> No hay una ley que ligue las dos dimensiones, ni siquiera dentro del mismo grafo. Es un resultado nulo, y lo
> reportamos porque contradice lo que esperabamos. Si hubieramos disenado el
> experimento para lucirnos, habriamos forzado una correlacion bonita, y no lo hicimos.

### Versión ampliada y explicada

*"Esta lamina cierra nuestra hipotesis central, y tambien se cae."* — Marca el desenlace de H3, la
tercera y última de las hipótesis planteadas en la lámina 06.

*"Esperabamos que una explicacion mas estable fuera tambien mas plausible... Es una intuicion que esta
implicita en buena parte de la literatura y nunca se habia contrastado de frente, porque para
contrastarla hace falta medir las dos cosas a la vez sobre las mismas explicaciones, y eso exige
conocer de antemano cual es el patron verdadero..."* — Explica por qué esta pregunta, tan intuitiva,
nunca se había puesto a prueba realmente: hacía falta exactamente el tipo de entorno controlado que
ofrece el grafo sintético (con ground-truth), y por eso la disponibilidad de ese eje es lo que hace
posible este contraste por primera vez.

*"Los datos dicen que no. La correlacion entre estabilidad y plausibilidad es de menos cero coma cero
uno, con un intervalo de confianza que va de menos cero coma cero treinta y ocho a mas cero coma cero
once, o sea que incluye el cero."* — Presenta el número central con su margen de error completo, y
explica de inmediato qué implica que ese margen cruce el cero.

*"Es un puente nulo. Un explicador estable no es por ello mas acertado sobre el patron real, y ambas
propiedades hay que medirlas por separado."* — Formula la conclusión de la forma más directa posible.

*"Y hay un detalle que refuerza la conclusion: cuando desagregamos por tipologia, el signo de la
relacion cambia segun cual mires, positivo en structuring y en fan-out, negativo en layering. No hay
una ley que ligue las dos dimensiones, ni siquiera dentro del mismo grafo."* — Añade una capa más de
evidencia: no solo el promedio general es nulo, sino que ni siquiera hay un patrón consistente al
desglosarlo por tipo de fraude. Esto descarta la posibilidad de que el resultado nulo global esconda
una relación real que solo aparece en ciertos casos.

*"Es un resultado nulo, y lo reportamos porque contradice lo que esperabamos. Si hubieramos disenado el
experimento para lucirnos, habriamos forzado una correlacion bonita, y no lo hicimos."* — Cierra con
el argumento de honestidad científica: reportar un resultado que contradice la propia hipótesis, sin
manipular el análisis para "salvarla", es evidencia de que el diseño experimental fue genuino y no
sesgado hacia un resultado deseado de antemano.

## 🧑‍⚖️ Preguntas de jurado probables

- **"¿No podría ser que su métrica de correlación simplemente no captó la relación real?"** → El
  intervalo de confianza cruza el cero en la medición global y el signo cambia de forma inconsistente
  por tipología, lo que descarta que exista una relación sistemática escondida; además, se reporta
  como no-resultado precisamente porque contradice la hipótesis de partida, no porque se buscara ese
  desenlace.
- **"¿Este resultado invalida el uso de la estabilidad como criterio de calidad?"** → No; significa
  que la estabilidad y la plausibilidad son criterios **independientes** que deben evaluarse por
  separado, cada uno según el propósito de la auditoría, tal como se plantea desde la lámina 11.

## 🧠 En una frase

Esperábamos que una explicación más estable fuera también más acertada; no lo es, la correlación es
prácticamente nula, y lo reportamos porque contradice lo que esperábamos.
