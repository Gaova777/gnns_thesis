# Lámina 27 — Rendimiento y colapso validación → test

> Bloque: Resultados
> Voz: Juan Diego · Página PDF 32 (27/40) · Tiempo objetivo 70 s

## 🎯 Objetivo de la lámina

Declarar con total transparencia un resultado de rendimiento que **enmarca todo lo anterior**: los
modelos aprenden bien sobre los datos de validación pero se derrumban sobre los datos de test, por un
cambio en el tiempo (no por un defecto del método), y explicar por qué eso obliga a medir la
estabilidad sobre validación y no sobre test.

## 📋 Qué dice, item por item

- **Figura 10 (gráfico de barras):** compara PR-AUC, precisión@50 y ROC-AUC en validación (azul) y en
  test (rojo). PR-AUC y precisión@50 colapsan de validación a test, mientras que el ROC-AUC cae mucho
  menos y aparenta un modelo aceptable.
- **"El modelo aprende en validación y colapsa en test: es el shift temporal del dataset (los
  patrones de lavado cambian con el tiempo). Propiedad del dato, no del método."** — La causa: el
  desplazamiento temporal, no un error del enfoque de la tesis.
- **"El ROC-AUC engaña bajo desbalance (0,88): tomamos PR-AUC y precisión@k como primarias."** — El
  punto metodológico central: si solo se mirara el ROC-AUC, se concluiría erróneamente que el modelo
  es casi excelente.
- **"La estabilidad se estudia sobre verdaderos positivos de validación (encuadre declarado)."** — La
  consecuencia práctica: dado que el modelo colapsa en test, medir la estabilidad ahí no aportaría
  información útil, así que se mide donde el modelo sí discrimina.

## 🔑 Conceptos y técnicas que aparecen

- **Shift temporal (desplazamiento temporal):** ver [glosario](README.md#shift-temporal-y-el-colapso-validacióntest).
  Los patrones de lavado de dinero no son estáticos: cambian con el tiempo, así que un modelo
  entrenado con datos del pasado puede encontrarse, en el futuro, con un "terreno de juego" distinto.
- **Colapso validación → test:** el fenómeno concreto de que el rendimiento cae drásticamente al pasar
  de los datos de validación (más cercanos en el tiempo a los de entrenamiento) a los de test (los más
  recientes, del "futuro" respecto al entrenamiento).
- **Por qué el ROC-AUC engaña bajo desbalance:** ver
  [glosario](README.md#pr-auc-roc-auc-y-precisiónk) y la analogía del
  [guardia del pueblo](README.md#banco-de-analogías-para-dar-coherencia-entre-láminas). En resumen, el
  ROC-AUC le da mucho crédito al modelo por acertar en la enorme mayoría de casos fáciles (las
  transacciones normales), aunque falle en detectar los pocos casos de fraude, que es justo lo que
  importa.
- **Encuadre declarado:** decir, de forma explícita y proactiva, dónde y por qué se decidió medir algo
  de una manera específica, en vez de dejar que el jurado lo descubra o lo cuestione sin explicación
  previa.

## 📈 Cómo leer la figura / tabla

La Figura 10 muestra tres pares de barras (una azul para validación, una roja para test), uno por
cada métrica: PR-AUC, precisión@50 y ROC-AUC. La lectura clave es comparar **cuánto cae** cada barra
roja respecto a su azul: PR-AUC y precisión@50 caen dramáticamente (casi a cero), mientras que el
ROC-AUC cae mucho menos y se mantiene en un valor que, de forma aislada, parecería razonable. Ese
contraste entre "cae mucho" (las métricas primarias) y "cae poco" (el ROC-AUC) es precisamente la
evidencia visual de por qué el ROC-AUC engaña bajo este tipo de desbalance.

## 🎤 El discurso (como se dice en voz alta)

> Un resultado de rendimiento que debemos declarar con transparencia, porque enmarca todo lo anterior.
> Los modelos aprenden en validacion, con un PR-AUC medio de cero coma treinta y siete, pero colapsan en
> test, donde cae a cero coma cero dos. *(pausa)* La causa es el desplazamiento temporal del dataset: los
> patrones de lavado cambian entre los primeros y los ultimos pasos, y un modelo entrenado con el pasado
> encuentra en el futuro una distribucion distinta. Es una propiedad del dato, no un defecto de nuestro
> metodo. Aqui hay un punto metodologico que quisimos remarcar: el ROC-AUC se ve enganosamente alto bajo
> desbalance extremo, cero coma ochenta y ocho en validacion, y por eso no lo usamos como metrica
> principal. Usamos el area de precision y exhaustividad y la precision en los primeros de la lista, que no se dejan enganar. *(pausa)* Como consecuencia, la
> estabilidad la estudiamos sobre los verdaderos positivos de validacion, donde el modelo si discrimina,
> y lo declaramos de forma abierta. No es esconder el colapso, es medir donde la pregunta tiene sentido.

### Versión ampliada y explicada

*"Un resultado de rendimiento que debemos declarar con transparencia, porque enmarca todo lo
anterior."* — Anuncia que esta lámina, aunque hable de rendimiento (no de explicabilidad), es
fundamental para entender el encuadre de todos los resultados de estabilidad ya presentados.

*"Los modelos aprenden en validacion, con un PR-AUC medio de cero coma treinta y siete, pero colapsan
en test, donde cae a cero coma cero dos."* — Presenta el contraste central con los dos números más
importantes: el modelo funciona razonablemente bien en validación y casi deja de funcionar en test.

*"La causa es el desplazamiento temporal del dataset: los patrones de lavado cambian entre los
primeros y los ultimos pasos, y un modelo entrenado con el pasado encuentra en el futuro una
distribucion distinta. Es una propiedad del dato, no un defecto de nuestro metodo."* — Es crucial la
distinción que se hace aquí: el colapso no es culpa del diseño del experimento ni de las arquitecturas
usadas, es una característica inherente al dominio (el lavado de dinero evoluciona con el tiempo).

*"Aqui hay un punto metodologico que quisimos remarcar: el ROC-AUC se ve enganosamente alto bajo
desbalance extremo, cero coma ochenta y ocho en validacion, y por eso no lo usamos como metrica
principal."* — Introduce el segundo tema de la lámina: por qué se eligió PR-AUC en vez de ROC-AUC como
métrica principal, un tema que se desarrolla a fondo en la siguiente lámina (28).

*"Como consecuencia, la estabilidad la estudiamos sobre los verdaderos positivos de validacion, donde
el modelo si discrimina, y lo declaramos de forma abierta."* — Conecta el colapso con una decisión
metodológica que ya se había mencionado en láminas anteriores (12 y 18): medir la estabilidad sobre
validación, no sobre test, porque ahí es donde el modelo realmente distingue fraude de no-fraude.

*"No es esconder el colapso, es medir donde la pregunta tiene sentido."* — Cierra con la frase que
resume la postura ética de la tesis frente a este resultado: no ocultarlo, sino declararlo
abiertamente y explicar por qué se tomó la decisión metodológica correspondiente.

## 🧑‍⚖️ Preguntas de jurado probables

- **"Si el modelo colapsa en test, ¿qué sentido tiene medir la estabilidad de sus explicaciones?"**
  (la pregunta más filosa, según el propio guion) → La estabilidad y el rendimiento son preguntas
  distintas. El colapso en test es un desplazamiento temporal del dato, documentado en la literatura
  de Elliptic, no un defecto del método. Se mide la estabilidad donde la pregunta tiene sentido, sobre
  los verdaderos positivos de validación, donde el modelo sí discrimina; explicar una predicción
  equivocada no aportaría información. Además, la coherencia con el eje sintético (donde no hay
  colapso) respalda que lo medido en validación no es un artefacto del colapso.
- **"¿Por qué no simplemente reentrenar incluyendo datos más recientes hasta que el colapso
  desaparezca?"** → Porque el objetivo del estudio es evaluar la propiedad de estabilidad bajo el
  protocolo estándar de evaluación temporal (entrenar con el pasado, evaluar con el futuro), que es
  como opera un sistema real; el colapso en sí mismo es un hallazgo relevante sobre el dominio, no un
  problema a "arreglar" artificialmente.

## 🧠 En una frase

Los modelos colapsan de validación a test por un cambio real en los patrones de lavado a través del
tiempo, así que la estabilidad se mide donde el modelo sí funciona: sobre los verdaderos positivos de
validación.
