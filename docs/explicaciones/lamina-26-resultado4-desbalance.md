# Lámina 26 — Resultado 4: el desbalance no gobierna la estabilidad

> Bloque: Resultados
> Voz: Juan Diego · Página PDF 31 (26/40) · Tiempo objetivo 55 s

## 🎯 Objetivo de la lámina

Cerrar la primera hipótesis (H1) demostrando que otra creencia muy lógica también resulta falsa: que
cuanto más raro sea el fraude, más inestables serían las explicaciones. Los datos muestran que no solo
no hay degradación, sino que la poca variación que existe **va en la dirección contraria** a la
esperada, y que la técnica de balanceo tampoco influye.

## 📋 Qué dice, item por item

- **"Esperábamos degradación al agravarse el desbalance. No la hay."** — El resumen directo del
  desenlace de H1.
- **"Rango estrecho: de 0,696 (1:1) a 0,765 (1:50), amplitud de siete centésimas, menor que las once
  que separan a las arquitecturas."** — El dato numérico: entre el escenario más equilibrado y el más
  desbalanceado que sí muestra variación, la estabilidad se mueve solo siete centésimas; para
  comparar, entre arquitecturas (un factor que sí importa) la diferencia es de once centésimas.
- **"Kruskal-Wallis p=0,18: no se rechaza la igualdad entre los cinco escenarios. Tamaño de efecto
  η²=0,05, pequeño frente al 0,13 de la arquitectura."** — La prueba estadística formal: con un
  p-valor de 0,18 (alto), no hay evidencia de que los cinco escenarios sean realmente distintos entre
  sí, y el tamaño de efecto del escenario (0,05) es mucho más pequeño que el de la arquitectura (0,13).
- **"El valor más bajo es 1:1, el más equilibrado: la variación no tiene dirección."** — El detalle
  más contundente: si la hipótesis fuera cierta, el valor más bajo debería estar en el escenario más
  desbalanceado, pero en realidad está en el más equilibrado.
- **"El balanceo tampoco pesa: η² ≤ 0,01 en las tres dimensiones."** — La técnica usada para
  compensar el desbalance (distinta del desbalance mismo) tiene un efecto casi nulo sobre estabilidad,
  plausibilidad y fidelidad.
- **Recuadro final:** implicación práctica: el balanceo puede elegirse por **rendimiento**, sin temer
  que degrade la interpretabilidad.
- **Figura 9:** un gráfico de línea que muestra la estabilidad media por escenario de desbalance sobre
  Elliptic, promediada sobre las 3 semillas de modelo.

## 🔑 Conceptos y técnicas que aparecen

- **Kruskal-Wallis con p=0,18:** ver [glosario](README.md#el-p-valor). Un p-valor de 0,18 es bastante
  alto: significa que, si en realidad los cinco escenarios fueran idénticos, obtener una diferencia
  como la observada sería bastante probable por puro azar (nada raro). Por eso no se puede afirmar que
  la diferencia sea real.
- **Tamaño de efecto η²=0,05 vs. 0,13:** ver [glosario](README.md#tamaño-de-efecto-η).
  Comparar estos dos números da una vara de medida honesta: el escenario de desbalance pesa
  **menos de la mitad** de lo que pesa la arquitectura, un factor que sí se sabe que importa.
- **Dirección de la variación:** no basta con mirar cuánto varía un número; también importa **hacia
  dónde** varía. Aquí, lo poco que varía va justo al revés de lo que predecía la hipótesis (el mínimo
  está en el escenario más fácil, no en el más difícil), lo que refuerza que no hay una relación causal
  real entre desbalance y estabilidad.
- **Balanceo vs. desbalance:** recordatorio importante, ver
  [glosario](README.md#balanceo-la-técnica-frente-a-desbalance-el-problema). Esta lámina prueba que
  **ninguno de los dos** (ni la rareza del fraude, ni la técnica para compensarla) mueve
  significativamente la calidad de las explicaciones.

## 📈 Cómo leer la figura / tabla

La Figura 9 muestra la estabilidad media (eje vertical) para cada escenario de desbalance (eje
horizontal: 1:1, 1:10, 1:50, 1:100, nativo), promediada sobre las 3 semillas de modelo. La lectura
correcta no es solo "¿la línea sube o baja?", sino "¿el movimiento es mayor que el ruido esperado?".
Visualmente la línea se mueve poco y sin una tendencia clara, y el punto más bajo está en el extremo
izquierdo (1:1, el escenario más equilibrado), justo lo opuesto de lo que predeciría la hipótesis de
que el desbalance degrada la estabilidad.

## 🎤 El discurso (como se dice en voz alta)

> Cierro la primera hipotesis, y tambien se cae. *(pausa)* Esperabamos que la estabilidad se degradara a
> medida que el desbalance se agravara, y que hubiera algun punto de quiebre. No ocurre ninguna de las dos
> cosas. Sobre las tres semillas, la estabilidad media recorre un rango estrecho, de cero coma sesenta y
> nueve seis en el escenario uno a uno a cero coma setenta y seis cinco en el uno a cincuenta: siete
> centesimas en total, menos que las once centesimas que separan a las arquitecturas entre si. Y lo
> decisivo es que esa variacion no pasa la prueba estadistica: la diferencia entre los cinco escenarios es
> perfectamente compatible con el azar. Ademas, el escenario explica apenas un cinco por ciento de la
> variacion, mientras que la arquitectura explica el trece por ciento sobre la misma medida. *(pausa)* Y conviene mirar hacia donde apunta lo poco que se mueve, porque
> apunta al reves de lo que esperabamos: el valor mas bajo esta en el escenario uno a uno, que es el mas
> equilibrado de todos, y los escenarios de desbalance acentuado quedan por encima. No hay deterioro
> monotono, no hay pico en el escenario uno a cincuenta, y el escenario nativo no se comporta de forma
> anomala. *(pausa)* Y en la misma linea, la estrategia de balanceo, que suele recibir mucha atencion en
> la literatura, resulta practicamente irrelevante: su tamano de efecto es de cero coma cero uno en las
> tres dimensiones. *(pausa)* Esto tiene una implicacion practica que me parece la mas util de toda la
> tesis para un equipo de cumplimiento: el balanceo pueden elegirlo por rendimiento predictivo, sin temer
> que al hacerlo esten degradando la interpretabilidad. Son decisiones que se pueden tomar por separado.

### Versión ampliada y explicada

*"Cierro la primera hipotesis, y tambien se cae."* — Marca el desenlace de H1, la última de las tres
hipótesis que aún quedaba por resolver (H2 y H3 ya se cerraron en las láminas 21 y 25).

*"Esperabamos que la estabilidad se degradara a medida que el desbalance se agravara, y que hubiera
algun punto de quiebre. No ocurre ninguna de las dos cosas."* — Aclara que se descartan dos
predicciones a la vez: ni hay degradación gradual, ni un punto crítico donde todo empeore de golpe.

*"Sobre las tres semillas, la estabilidad media recorre un rango estrecho, de cero coma sesenta y
nueve seis en el escenario uno a uno a cero coma setenta y seis cinco en el uno a cincuenta: siete
centesimas en total, menos que las once centesimas que separan a las arquitecturas entre si."* — Da
los números precisos y, de inmediato, los pone en perspectiva comparándolos con un factor que sí es
relevante (la arquitectura), para que el jurado tenga una vara de medida clara.

*"Y lo decisivo es que esa variacion no pasa la prueba estadistica: la diferencia entre los cinco
escenarios es perfectamente compatible con el azar."* — Traduce el resultado de Kruskal-Wallis
(p=0,18) a lenguaje llano: esa pequeña variación podría explicarse enteramente por azar.

*"Ademas, el escenario explica apenas un cinco por ciento de la variacion, mientras que la
arquitectura explica el trece por ciento sobre la misma medida."* — Complementa el p-valor con el
tamaño de efecto (η²), dando una segunda confirmación de que el desbalance pesa poco, esta vez en
términos de magnitud práctica, no solo de significancia estadística.

*"Y conviene mirar hacia donde apunta lo poco que se mueve, porque apunta al reves de lo que
esperabamos: el valor mas bajo esta en el escenario uno a uno, que es el mas equilibrado de todos..."*
— Este es el argumento más contundente de la lámina: no solo el movimiento es pequeño y no
significativo, sino que además va en la dirección **opuesta** a la hipótesis, lo que hace muy difícil
sostener que exista algún efecto real del desbalance sobre la estabilidad.

*"No hay deterioro monotono, no hay pico en el escenario uno a cincuenta, y el escenario nativo no se
comporta de forma anomala."* — Descarta explícitamente tres formas alternativas en que la hipótesis
podría haberse manifestado (un deterioro gradual, un pico anómalo en un escenario intermedio, o un
comportamiento raro en el escenario nativo), cerrando todas las salidas posibles.

*"Y en la misma linea, la estrategia de balanceo... resulta practicamente irrelevante: su tamano de
efecto es de cero coma cero uno en las tres dimensiones."* — Extiende la conclusión al segundo factor
relacionado (el balanceo, distinto del desbalance): tampoco influye de forma relevante.

*"Esto tiene una implicacion practica que me parece la mas util de toda la tesis para un equipo de
cumplimiento: el balanceo pueden elegirlo por rendimiento predictivo, sin temer que al hacerlo esten
degradando la interpretabilidad."* — Cierra con la aplicación práctica: como ninguno de los dos
factores afecta la calidad de las explicaciones, se pueden optimizar por separado, según el criterio
que más convenga (en este caso, rendimiento predictivo).

## 🧑‍⚖️ Preguntas de jurado probables

- **"El perfil por escenario, ¿es plano de verdad o solo se ve así en el gráfico?"** → No se deja en
  la impresión visual: se contrasta formalmente con Kruskal-Wallis, que da p=0,18 (no se rechaza la
  igualdad entre los cinco escenarios), y con un tamaño de efecto de 0,05, pequeño frente al 0,13 de
  la arquitectura sobre la misma métrica. Además, la variación no tiene la dirección que predecía la
  hipótesis: el mínimo está en el escenario más equilibrado.
- **"¿Y si con más escenarios (más granularidad) sí aparecería un efecto?"** → Los cinco escenarios ya
  cubren un rango amplio (de 1:1 hasta 1:100, más el nativo), y la ausencia de una tendencia monótona
  o de un punto de quiebre en ninguno de ellos hace poco probable que un muestreo más fino revele un
  patrón oculto; es un límite reconocido explícitamente en las limitaciones de la tesis.
- **"¿Por qué el balanceo, que recibe tanta atención en la literatura, resulta tan poco relevante
  aquí?"** → Porque esta tesis mide su efecto específicamente sobre la **interpretabilidad** (las tres
  dimensiones: estabilidad, plausibilidad, fidelidad), no sobre el rendimiento predictivo del
  clasificador, que es donde la literatura suele reportar sus beneficios; son preguntas distintas.

## 🧠 En una frase

Ni el desbalance ni la técnica de balanceo afectan de forma significativa la calidad de las
explicaciones, y lo poco que varía va, de hecho, en contra de lo que se esperaba.
