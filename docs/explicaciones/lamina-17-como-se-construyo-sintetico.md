# Lámina 17 — Cómo se construyó el grafo sintético

> Bloque: Metodología
> Voz: Alejandro · Página PDF 21 (17/40) · Tiempo objetivo 70 s

## 🎯 Objetivo de la lámina

Detallar, con total transparencia, las decisiones concretas de construcción del grafo sintético,
porque —como dice el propio guion— "es donde se juega la credibilidad de todo el eje sintético". La
estrategia es exponer estos detalles antes de que el jurado los pregunte.

## 📋 Qué dice, item por item

- **Decisiones que endurecen la prueba:**
  - **Simetrización de aristas:** sin ella, el campo receptivo caería a ~2 nodos y la plausibilidad de
    subgrafo no sería medible.
  - **Aristas distractoras (n=3 por nodo de patrón):** sin ellas, cualquier selección "top-k"
    acertaría automáticamente.
  - **Firma de features atenuada, de +4 a +1,5:** evita que la tarea se resuelva sola (que sea
    demasiado obvia).
- **Nota:** el ground-truth se fija en la construcción, **antes** de correr ningún explicador.
- **Escala y replicación:**
  - Aproximadamente 9.500 nodos, 1.530 ilícitos, 31.000 aristas.
  - Las cuatro tipologías canónicas plantadas dentro.
  - Grafo g0: factorial completo × 3 semillas (540 filas de resultados).
  - Grafos g1 y g2: un corte de verificación más reducido (144 filas).
- **Nota:** el contraste principal se calcula sobre la unión de los tres grafos.

## 🔑 Conceptos y técnicas que aparecen

- **Simetrización de aristas:** convertir un grafo dirigido (donde una conexión "A envía a B" es
  distinta de "B envía a A") en uno donde las conexiones se tratan como bidireccionales. Sin esto, el
  campo receptivo (cuántos nodos "ve" cada nodo) se reduce drásticamente, porque el paso de mensajes
  solo puede fluir en una dirección.
- **Arista distractora:** una conexión añadida a propósito que **no** forma parte del patrón de
  lavado, pero que está cerca de él en la red. Sin distractores, cualquier método que simplemente
  eligiera "las primeras k aristas del vecindario" acertaría siempre, porque el subgrafo sería
  prácticamente 100% patrón real.
- **Firma de features atenuada:** reducir la diferencia entre los valores de los atributos de los
  nodos ilícitos y los lícitos (de una diferencia grande, +4, a una más sutil, +1,5), para que el
  patrón no sea tan obvio que cualquier método lo detecte trivialmente con solo mirar los atributos,
  sin necesidad de usar la estructura de la red.
- **Realizaciones g0, g1, g2:** tres versiones distintas del grafo sintético, generadas con semillas
  distintas. g0 es la más completa (todo el diseño factorial, repetido 3 veces); g1 y g2 son un corte
  más reducido, usado como verificación adicional.

## 📈 Cómo leer la figura / tabla

No aplica: esta lámina presenta la información en dos columnas de texto (decisiones de diseño, y
escala/replicación), sin figura ni tabla numérica.

## 🎤 El discurso (como se dice en voz alta)

> Me detengo en la construccion, porque es donde se juega la credibilidad de todo el eje sintetico y
> prefiero exponerla nosotros antes de que se pregunte. *(pausa)* El grafo tiene unos nueve mil quinientos
> nodos, de los cuales unos mil quinientos son ilicitos, y unas treinta y un mil aristas, con las cuatro
> tipologias canonicas plantadas dentro. Pero lo importante no es el tamano, son tres decisiones que
> tomamos deliberadamente para que la prueba fuera dificil. *(pausa)* La primera, simetrizar las aristas.
> Con el grafo dirigido el campo receptivo cae a unos dos nodos y los patrones de estrella y de cadena
> quedan invisibles al paso de mensajes, con lo que la plausibilidad de subgrafo no seria medible. La
> segunda, anadir aristas distractoras desde cada nodo de patron hacia el fondo licito. Sin ellas el
> subgrafo seria cien por cien patron y cualquier seleccion acertaria, con lo que la metrica no
> discriminaria nada. Y la tercera, atenuar la firma de los atributos de mas cuatro a mas uno coma cinco,
> para que el problema no se resolviera solo. *(pausa)* Las tres decisiones endurecen la prueba, no la
> inflan. Y el punto que quiero dejar fijado: el patron verdadero lo fija el generador, que es
> completamente ciego a que explicador se va a evaluar despues. Que PGExplainer gane en plausibilidad no
> esta cableado en ninguna parte.

### Versión ampliada y explicada

*"Me detengo en la construccion, porque es donde se juega la credibilidad de todo el eje sintetico y
prefiero exponerla nosotros antes de que se pregunte."* — Una decisión de estrategia de defensa: es
mejor exponer proactivamente los detalles más vulnerables a objeciones que esperar a que el jurado los
descubra y pregunte.

*"El grafo tiene unos nueve mil quinientos nodos, de los cuales unos mil quinientos son ilicitos, y
unas treinta y un mil aristas, con las cuatro tipologias canonicas plantadas dentro."* — Da la escala
concreta antes de entrar en las decisiones de diseño, para que el jurado tenga una imagen mental clara
del tamaño del grafo.

*"La primera, simetrizar las aristas. Con el grafo dirigido el campo receptivo cae a unos dos nodos y
los patrones de estrella y de cadena quedan invisibles al paso de mensajes..."* — Explica el "por
qué", no solo el "qué": sin simetrizar, el problema de campo receptivo minúsculo (el mismo que aqueja
a Elliptic) haría que el sintético no sirviera para su propósito.

*"La segunda, anadir aristas distractoras desde cada nodo de patron hacia el fondo licito. Sin ellas
el subgrafo seria cien por cien patron y cualquier seleccion acertaria..."* — Explica por qué,
paradójicamente, hacer el grafo "más fácil de resolver sin distractores" haría la métrica **inútil**:
si acertar es automático, no se puede distinguir un buen explicador de uno malo.

*"Y la tercera, atenuar la firma de los atributos de mas cuatro a mas uno coma cinco, para que el
problema no se resolviera solo."* — Explica la tercera decisión: si los atributos por sí solos ya
delatan claramente el patrón, ni siquiera hace falta mirar la estructura de la red, y el experimento
dejaría de poner a prueba lo que realmente interesa (el uso de la información relacional).

*"Las tres decisiones endurecen la prueba, no la inflan."* — Frase de cierre muy deliberada: reafirma
que cada decisión hace el experimento más exigente, no más fácil de aprobar.

*"Y el punto que quiero dejar fijado: el patron verdadero lo fija el generador, que es completamente
ciego a que explicador se va a evaluar despues. Que PGExplainer gane en plausibilidad no esta
cableado en ninguna parte."* — Cierra respondiendo, de forma explícita y con nombre propio (PGExplainer,
que va a resultar el más plausible en los resultados), la objeción de circularidad antes de que
aparezca.

## 🧑‍⚖️ Preguntas de jurado probables

- **"¿Por qué elegir justo +1,5 como firma atenuada y no otro valor?"** → Es un valor calibrado en una
  ronda de ajuste previa para que la tarea sea difícil pero no imposible; el criterio de éxito fue que
  ningún explicador alcanzara un desempeño trivial (cercano al 100%) solo con los atributos, sin usar
  la estructura del grafo.
- **"¿Tres aristas distractoras por nodo de patrón es suficiente?"** → Es la cantidad usada de forma
  consistente en las tres realizaciones del grafo (g0, g1, g2), y los resultados de plausibilidad
  muestran que la tarea sigue siendo medible y discriminante (ningún explicador alcanza el 100%), lo
  que confirma que el nivel de dificultad es adecuado.

## 🧠 En una frase

El grafo sintético se construyó con tres decisiones deliberadas —simetrizar, añadir distractores y
atenuar la señal— para que la prueba fuera difícil de verdad, con el patrón verdadero fijado antes de
correr ningún explicador.
