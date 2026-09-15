# Lámina 05 — Pregunta de investigación y objetivos

> Bloque: Problema y motivación
> Voz: Alejandro · Página PDF 7 (5/40) · Tiempo objetivo 60 s

## 🎯 Objetivo de la lámina

Formalizar, en una sola pregunta y cuatro objetivos concretos, todo lo que la tesis se propone medir.
Después de haber justificado por qué importa (láminas anteriores), aquí se convierte esa motivación
en algo que se puede investigar y responder con datos.

## 📋 Qué dice, item por item

- **La pregunta (recuadro):** "¿Cómo se comporta la estabilidad de los métodos XAI sobre GNNs para
  detección de lavado bajo desbalance, y qué combinación de arquitectura, explicador y balanceo
  produce la interpretación más robusta y auditable?" — Une en una sola frase los tres "ingredientes"
  que se van a combinar y comparar: **arquitectura** (el tipo de red neuronal de grafos), **explicador**
  (el método que genera la explicación) y **balanceo** (la técnica para compensar el desbalance).
- **O1 — "¿Se degrada la estabilidad al agravarse el desbalance?"** — El primer objetivo pregunta si,
  a medida que el fraude se vuelve más raro, las explicaciones se vuelven menos consistentes.
- **O2 — "¿Qué tan resilientes son GCN, GraphSAGE, GAT y TAGCN?"** — El segundo objetivo compara las
  cuatro arquitecturas de red de grafos entre sí, en términos de qué tan estables son sus
  explicaciones.
- **O3 — "¿Influyen las estrategias de balanceo?"** — El tercer objetivo pregunta si la técnica que se
  usa para compensar el desbalance (no confundir con el desbalance mismo) afecta la calidad de las
  explicaciones.
- **O4 — "Condensar todo en una matriz de recomendación."** — El cuarto objetivo es práctico: traducir
  todos los hallazgos en una guía de qué combinación usar según lo que se necesite auditar.

## 🔑 Conceptos y técnicas que aparecen

- **Arquitectura:** el tipo específico de red neuronal de grafos (GCN, GraphSAGE, GAT o TAGCN). Se
  explican en detalle en la [lámina 09](lamina-09-cuatro-arquitecturas.md).
- **Explicador:** el método que, después de entrenar el modelo, intenta decir por qué tomó una
  decisión (GNNExplainer, PGExplainer, GNNShap). Se explican en la
  [lámina 10](lamina-10-tres-explicadores.md).
- **Balanceo vs. desbalance:** son cosas distintas que conviene no confundir. El **desbalance** es el
  problema (el fraude es raro); el **balanceo** es la **técnica** para compensarlo al entrenar. Ver
  [glosario](README.md#balanceo-la-técnica-frente-a-desbalance-el-problema).

## 📈 Cómo leer la figura / tabla

No aplica: es texto en un recuadro y una lista de cuatro objetivos en dos columnas, sin figura ni
tabla numérica.

## 🎤 El discurso (como se dice en voz alta)

> Con ese marco, esta es nuestra pregunta: como se comporta la estabilidad de los metodos de
> explicabilidad sobre redes de grafos para deteccion de lavado, cuando el dato esta fuertemente
> desbalanceado, y que combinacion de arquitectura, explicador y estrategia de balanceo produce la
> interpretacion mas robusta y auditable. *(pausa)* De ahi se desprenden cuatro objetivos. El primero,
> medir si la estabilidad se degrada a medida que el desbalance se agrava. El segundo, comparar la
> resiliencia de cuatro arquitecturas: GCN, GraphSAGE, GAT y TAGCN. El tercero, evaluar si las
> estrategias de balanceo afectan la calidad de las explicaciones. Y el cuarto, condensar todo en una
> matriz de recomendacion que diga que usar segun el objetivo de la auditoria.

### Versión ampliada y explicada

*"Con ese marco, esta es nuestra pregunta..."* — Conecta con lo ya justificado (la caja negra
inaceptable, la necesidad de estabilidad) y lo convierte en una pregunta de investigación formal.

*"...como se comporta la estabilidad de los metodos de explicabilidad sobre redes de grafos para
deteccion de lavado, cuando el dato esta fuertemente desbalanceado, y que combinacion de
arquitectura, explicador y estrategia de balanceo produce la interpretacion mas robusta y
auditable."* — Esta frase es densa a propósito: contiene los tres factores que se van a cruzar en el
experimento (arquitectura, explicador, balanceo) y las dos condiciones bajo las que se van a evaluar
(desbalance, y el objetivo de que la interpretación sea robusta y auditable).

*"De ahi se desprenden cuatro objetivos."* — Traduce la pregunta general en tareas concretas y
medibles.

*"El primero, medir si la estabilidad se degrada a medida que el desbalance se agrava."* — Objetivo
1: es literalmente probar si, cuando el fraude es más raro, las explicaciones se vuelven más
inconsistentes.

*"El segundo, comparar la resiliencia de cuatro arquitecturas: GCN, GraphSAGE, GAT y TAGCN."* —
Objetivo 2: no basta con un solo tipo de red neuronal; hay que ver si el comportamiento cambia según
la arquitectura elegida.

*"El tercero, evaluar si las estrategias de balanceo afectan la calidad de las explicaciones."* —
Objetivo 3: la técnica de compensación del desbalance podría, en principio, tener un efecto secundario
sobre la interpretabilidad; hay que comprobarlo.

*"Y el cuarto, condensar todo en una matriz de recomendacion que diga que usar segun el objetivo de la
auditoria."* — Objetivo 4: el resultado final debe ser útil en la práctica, no solo académico.

## 🧑‍⚖️ Preguntas de jurado probables

- **"¿Por qué estos cuatro objetivos y no otros?"** → Porque cubren, de forma ordenada, las tres
  variables que definen el experimento (arquitectura, explicador, balanceo) frente al desbalance, y
  culminan en una recomendación práctica, que es el objetivo general de la tesis.
- **"¿La pregunta de investigación se responde al final?"** → Sí, de frente, en la última lámina de
  conclusiones y cierre (lámina 34): no existe una combinación única óptima, y la elección depende del
  propósito de la auditoría.

## 🧠 En una frase

La tesis pregunta cómo se comporta la estabilidad de las explicaciones bajo desbalance, según la
arquitectura, el explicador y el balanceo, y busca condensar la respuesta en una guía práctica.
