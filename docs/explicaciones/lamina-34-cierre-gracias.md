# Lámina 34 — Gracias / Preguntas (cierre)

> Bloque: Conclusiones
> Voz: Juan Diego · Página PDF 40 · Tiempo objetivo 80 s

## 🎯 Objetivo de la lámina

Cerrar la defensa respondiendo, de frente y sin rodeos, la pregunta de investigación original (lámina
05), dejar tres mensajes finales que resuman el espíritu de todo el trabajo, mencionar el trabajo
futuro, y agradecer al jurado antes de abrir la ronda de preguntas.

## 📋 Qué dice, item por item

- **Título de la lámina (destacada, tipo "standout"):** "Gracias · Preguntas".
- **"Estabilidad ≠ plausibilidad ≠ fidelidad: el mejor explicador depende del objetivo."** — El
  resumen más comprimido posible del mensaje central de toda la tesis.
- **"Trabajo futuro: Elliptic2, AMLSim, GNNs temporales, GraphSMOTE."** — Cuatro líneas concretas de
  continuación: un dataset más reciente (Elliptic2), un simulador alternativo (AMLSim, si se
  desbloquean sus limitaciones), arquitecturas que modelan explícitamente el paso del tiempo (GNNs
  temporales), y una técnica específica de balanceo mencionada como pendiente de explorar
  (GraphSMOTE).
- **Nombres de los autores** al pie: Alejandro Gómez Huertas y Juan Diego Garzón Ovalle.

## 🔑 Conceptos y técnicas que aparecen

- **Elliptic2:** una versión más reciente o extendida del dataset Elliptic (mencionada como trabajo
  futuro, no usada en esta tesis).
- **AMLSim:** un simulador de transacciones para detección de lavado de dinero que se consideró como
  alternativa, pero que quedó bloqueado por limitaciones técnicas (una dependencia de software
  inexistente) y de datos (no ofrece ground-truth por arista). Documentado como trabajo futuro
  explícito.
- **GNNs temporales:** arquitecturas de redes de grafos diseñadas específicamente para modelar cómo
  cambia una red a través del tiempo, en vez de tratarla como una foto fija. Podrían abordar de forma
  más directa el fenómeno de shift temporal (lámina 27).
- **GraphSMOTE:** una técnica de balanceo específica para grafos (mencionada en el manuscrito como
  parte del código, aunque no estuvo activa en los experimentos de esta versión de la tesis, "no
  cableado en v3").

## 📈 Cómo leer la figura / tabla

No aplica: es una lámina de cierre tipo "standout" (fondo destacado, texto centrado), sin figura ni
tabla.

## 🎤 El discurso (como se dice en voz alta)

> Para cerrar, respondo de frente nuestra pregunta de investigacion y dejo tres mensajes. *(pausa)* La
> respuesta directa a la pregunta es que no existe una combinacion unica optima de arquitectura,
> explicador y balanceo, y que la eleccion depende del proposito de la auditoria. Para estabilidad y
> auditabilidad, GAT o GCN con un explicador consistente son el mejor punto de partida. *(pausa)* De ahi,
> tres mensajes. Primero: estabilidad, plausibilidad y fidelidad no son lo mismo, y por eso el mejor
> explicador depende del objetivo de quien audita. Segundo: con la medicion corregida, los datos reales y
> los sinteticos cuentan una historia coherente, con GAT y GCN como las arquitecturas mas estables.
> Y tercero, quiza el mas transversal: la estabilidad de una explicacion depende del protocolo de
> evaluacion, no solo del metodo, y por eso corregir artefactos y retractar conclusiones apoyadas en ellos
> no debilito la tesis, la hizo mas solida. *(pausa)* Como trabajo futuro, extender el eje real a datasets
> con atributos no anonimizados, incorporar desplazamiento temporal al grafo sintetico, y explorar
> arquitecturas temporales. *(pausa)* Con esto cerramos. Agradecemos al director y al jurado, y quedamos
> atentos a sus preguntas.

### Versión ampliada y explicada

*"Para cerrar, respondo de frente nuestra pregunta de investigacion y dejo tres mensajes."* — Anuncia
la estructura del cierre: primero la respuesta directa a la pregunta original (planteada en la lámina
05), luego tres mensajes de síntesis.

*"La respuesta directa a la pregunta es que no existe una combinacion unica optima de arquitectura,
explicador y balanceo, y que la eleccion depende del proposito de la auditoria."* — Responde
literalmente la pregunta de investigación, sin rodeos, aunque la respuesta sea "depende" en vez de un
único ganador — y explica por qué eso es la respuesta correcta y no una evasiva.

*"Para estabilidad y auditabilidad, GAT o GCN con un explicador consistente son el mejor punto de
partida."* — Aterriza la respuesta general en una recomendación concreta, para no dejar la pregunta
solo en abstracto.

*"Primero: estabilidad, plausibilidad y fidelidad no son lo mismo, y por eso el mejor explicador
depende del objetivo de quien audita."* — El primer mensaje repite, por última vez, el hallazgo
central de toda la tesis.

*"Segundo: con la medicion corregida, los datos reales y los sinteticos cuentan una historia
coherente, con GAT y GCN como las arquitecturas mas estables."* — El segundo mensaje resalta la
concordancia entre los dos ejes como una de las piezas más sólidas del trabajo.

*"Y tercero, quiza el mas transversal: la estabilidad de una explicacion depende del protocolo de
evaluacion, no solo del metodo, y por eso corregir artefactos y retractar conclusiones apoyadas en
ellos no debilito la tesis, la hizo mas solida."* — El tercer mensaje, calificado por el propio orador
como "el más transversal", es la lección metodológica que trasciende este estudio específico: cómo se
mide algo importa tanto como el método que se está midiendo, y ser transparente sobre errores
corregidos fortalece la credibilidad en vez de debilitarla.

*"Como trabajo futuro, extender el eje real a datasets con atributos no anonimizados, incorporar
desplazamiento temporal al grafo sintetico, y explorar arquitecturas temporales."* — Da tres líneas
concretas de continuación, cada una atacando directamente una de las limitaciones reconocidas en la
lámina 33 (dataset anonimizado, ausencia de tiempo en el sintético, y el fenómeno de shift temporal).

*"Con esto cerramos. Agradecemos al director y al jurado, y quedamos atentos a sus preguntas."* —
Cierre formal y cortés, que abre la puerta a la ronda de preguntas que sigue a la defensa.

## 🧑‍⚖️ Preguntas de jurado probables

- **"En una frase, ¿cuál es el mensaje que quieren que se recuerde de esta tesis?"** → Que estabilidad,
  plausibilidad y fidelidad no son la misma cosa, y que la calidad de una explicación de IA no se
  resume en un solo número: depende de qué se le esté pidiendo a esa explicación.
- **"¿Por qué GraphSMOTE se menciona como trabajo futuro si ya está en el código?"** → Porque, aunque
  existe en el código base del proyecto, no estuvo activo (cableado) en los experimentos reportados en
  esta versión de la tesis; evaluarlo de forma sistemática junto a las otras estrategias de balanceo es
  una extensión natural pendiente.

## 🧠 En una frase

No existe una combinación única óptima de arquitectura, explicador y balanceo: la elección depende del
propósito de la auditoría, y esa dependencia es, en sí misma, el hallazgo más importante de la tesis.
