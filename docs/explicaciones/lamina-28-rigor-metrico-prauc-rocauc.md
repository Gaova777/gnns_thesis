# Lámina 28 — Rigor métrico: por qué PR-AUC y no ROC-AUC

> Bloque: Resultados
> Voz: Juan Diego · Página PDF 33 (28/40) · Tiempo objetivo 55 s

## 🎯 Objetivo de la lámina

Desarrollar en detalle, con una tabla comparativa, el punto metodológico introducido en la lámina
anterior: mostrar cómo el ROC-AUC y el PR-AUC, calculados sobre exactamente los mismos modelos, cuentan
historias completamente distintas, y explicar la razón estructural detrás de esa discrepancia.

## 📋 Qué dice, item por item

- **Tabla — Métricas de ordenamiento en validación y test (mismos modelos):**
  - ROC-AUC: 0,884 en validación → 0,653 en test.
  - **PR-AUC: 0,367 en validación → 0,017 en test.**
  - Precisión en los primeros 50: 0,657 en validación → 0,020 en test.
  - Precisión en los primeros 100: 0,640 en validación → 0,022 en test.
- **Texto de apoyo:** el ROC-AUC de 0,884 sugeriría un clasificador casi excelente; el PR-AUC de 0,367,
  sobre exactamente los mismos modelos, revela la dificultad real. Bajo desbalance extremo, el eje de
  falsos positivos del ROC queda dominado por la clase mayoritaria y permanece bajo aunque el modelo
  no distinga la clase rara.

## 🔑 Conceptos y técnicas que aparecen

- **Métrica de ordenamiento:** una métrica que evalúa qué tan bien un modelo **ordena** los casos de
  más a menos sospechoso, sin depender de fijar un único umbral de decisión. Tanto ROC-AUC como PR-AUC
  son de este tipo, pero se comportan de forma muy distinta bajo desbalance extremo.
- **Por qué el eje de falsos positivos del ROC se satura bajo desbalance:** la tasa de falsos positivos
  del ROC se calcula como (falsos positivos) dividido entre (todos los casos negativos reales). Como
  los casos negativos (transacciones lícitas) son la inmensa mayoría, incluso si el modelo comete
  bastantes errores en términos absolutos, esa fracción se mantiene pequeña, y eso hace que la curva
  ROC —y su área bajo la curva— se vea artificialmente favorable.
- **Por qué el PR-AUC no se deja engañar:** la precisión (uno de los dos ejes del PR-AUC) se calcula
  sobre las predicciones **positivas** del modelo (los casos que marcó como sospechosos), así que si el
  modelo se equivoca mucho al marcar falsos positivos, esa fracción de precisión cae de inmediato,
  reflejando la verdadera dificultad de la tarea.

## 📈 Cómo leer la figura / tabla

La tabla contrasta cuatro métricas de ordenamiento, cada una con su valor en validación y en test. La
lectura clave es fijarse en la **magnitud de la caída** entre columnas para cada fila: el ROC-AUC cae
de 0,884 a 0,653 (una caída moderada, que todavía "parece" un modelo mediocre pero funcional), mientras
que PR-AUC y las dos precisiones en el top de la lista caen de forma mucho más drástica, casi hasta
cero. Esa diferencia en la magnitud de la caída, sobre los mismos modelos exactos, es la evidencia
central de por qué el ROC-AUC no captura bien la dificultad real del problema.

## 🎤 El discurso (como se dice en voz alta)

> Esta lamina desarrolla el punto metodologico que acabo de mencionar, porque creo que merece detenerse.
> *(pausa)* Miren la tabla. Sobre validacion, el ROC-AUC da cero coma ochenta y ocho. Si reportaramos solo
> esa cifra, cualquiera concluiria que tenemos un clasificador casi excelente. Pero sobre exactamente los
> mismos modelos, el area de precision y exhaustividad da cero coma treinta y siete, y la precision en los
> primeros cincuenta nodos cero coma sesenta y seis. Es una tarea mucho mas dificil de lo que el ROC-AUC
> insinua. *(pausa)* Y sobre test la disociacion se vuelve extrema: el ROC-AUC se mantiene en cero coma
> sesenta y cinco, que parecerian un modelo mediocre pero funcional, mientras que el area de precision y
> exhaustividad se desploma a cero coma cero dos. *(pausa)* La razon es estructural: bajo desbalance
> extremo, el eje de tasa de falsos positivos del ROC queda dominado por la enorme clase mayoritaria y
> permanece bajo aunque el modelo no distinga la clase rara. Por eso nuestras metricas primarias son el
> area de precision y exhaustividad y la precision en los primeros nodos, que ademas son las que gobiernan
> el trabajo real de un analista, que revisa una lista acotada de alertas y no todo el universo de
> transacciones.

### Versión ampliada y explicada

*"Esta lamina desarrolla el punto metodologico que acabo de mencionar, porque creo que merece
detenerse."* — Justifica por qué se dedica una lámina completa a este contraste, en vez de dejarlo
como una mención de pasada en la lámina anterior.

*"Miren la tabla. Sobre validacion, el ROC-AUC da cero coma ochenta y ocho. Si reportaramos solo esa
cifra, cualquiera concluiria que tenemos un clasificador casi excelente."* — Plantea el problema de
forma directa: mirar solo el ROC-AUC llevaría a una conclusión errónea sobre la calidad real del
modelo.

*"Pero sobre exactamente los mismos modelos, el area de precision y exhaustividad da cero coma treinta
y siete, y la precision en los primeros cincuenta nodos cero coma sesenta y seis. Es una tarea mucho
mas dificil de lo que el ROC-AUC insinua."* — Repite, deliberadamente, "los mismos modelos": el
contraste no viene de comparar modelos distintos, viene de mirar el **mismo** resultado con dos
métricas diferentes.

*"Y sobre test la disociacion se vuelve extrema: el ROC-AUC se mantiene en cero coma sesenta y cinco,
que parecerian un modelo mediocre pero funcional, mientras que el area de precision y exhaustividad se
desploma a cero coma cero dos."* — Muestra que el problema se agrava en test: ahí la brecha entre lo
que sugiere el ROC-AUC (mediocre pero utilizable) y la realidad (casi inútil, según PR-AUC) se hace
todavía más grande.

*"La razon es estructural: bajo desbalance extremo, el eje de tasa de falsos positivos del ROC queda
dominado por la enorme clase mayoritaria y permanece bajo aunque el modelo no distinga la clase
rara."* — Explica el mecanismo matemático de fondo: no es un accidente ni una casualidad de este
dataset en particular, es una propiedad estructural de cómo se calcula el ROC-AUC cuando una de las
dos clases es rarísima.

*"Por eso nuestras metricas primarias son el area de precision y exhaustividad y la precision en los
primeros nodos, que ademas son las que gobiernan el trabajo real de un analista, que revisa una lista
acotada de alertas y no todo el universo de transacciones."* — Cierra justificando la elección no solo
en términos estadísticos, sino también prácticos: precisión en el top de la lista es exactamente lo
que le importa a un analista humano que revisa un número limitado de casos.

## 🧑‍⚖️ Preguntas de jurado probables

- **"¿Por qué no reportar ambas métricas y dejar que el lector decida?"** → Se reportan ambas (están
  en la misma tabla), pero se es explícito sobre cuál es la primaria y por qué, precisamente para
  evitar que alguien se quede solo con el ROC-AUC y saque una conclusión equivocada sobre la
  dificultad real de la tarea.
- **"¿Este problema del ROC-AUC bajo desbalance es específico de este trabajo o es conocido en la
  literatura?"** → Es un fenómeno estructural bien conocido en la literatura de clasificación bajo
  desbalance extremo; esta tesis lo documenta con datos concretos de su propio experimento, mostrando
  la magnitud exacta de la discrepancia en este dominio.

## 🧠 En una frase

Sobre los mismos modelos, el ROC-AUC sugiere un clasificador aceptable mientras el PR-AUC revela un
colapso casi total, y la razón es que el ROC-AUC se deja dominar por la enorme clase mayoritaria bajo
desbalance extremo.
