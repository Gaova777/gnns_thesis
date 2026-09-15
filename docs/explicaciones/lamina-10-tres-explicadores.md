# Lámina 10 — Tres explicadores post-hoc

> Bloque: Marco conceptual
> Voz: Alejandro · Página PDF 13 (10/40) · Tiempo objetivo 50 s

## 🎯 Objetivo de la lámina

Presentar los tres métodos de explicabilidad que se van a comparar en toda la tesis, con la
intuición mínima de cómo funciona cada uno por dentro. Junto con la lámina anterior (las cuatro
arquitecturas), esto completa el "elenco" de la comparación: 4 arquitecturas × 3 explicadores.

## 📋 Qué dice, item por item

- **Definición de "post-hoc" (recuadro superior):** significa que estos métodos explican un modelo
  **ya entrenado**, sin modificarlo. Cada uno produce una máscara de aristas y un ranking de features.
- **GNNExplainer — "optimiza una máscara por instancia — aprende qué importa para ese nodo."** — Para
  cada transacción que se quiere explicar, este método hace un pequeño proceso de ajuste (optimización)
  específico para ese caso, buscando qué aristas y atributos, si se les quita importancia, más cambian
  la predicción.
- **PGExplainer — "entrena una red amortizada que aprende a explicar en todo el grafo (generaliza
  entre nodos)."** — En vez de repetir el proceso de ajuste para cada nodo por separado, entrena una
  sola red que aprende un patrón general de qué hace importante a una arista, y luego aplica ese
  patrón aprendido a cualquier nodo.
- **GNNShap — "reparte el crédito con valores de Shapley (teoría de juegos); muy consistente."** —
  Usa una técnica matemática (valores de Shapley) que viene de la teoría de juegos, pensada
  originalmente para repartir de forma justa las ganancias entre varios jugadores que colaboran; aquí
  se usa para repartir "el crédito" de la predicción entre los distintos elementos (aristas o
  atributos).
- **Nota final:** cuatro arquitecturas por tres explicadores es el núcleo de toda la comparación.

## 🔑 Conceptos y técnicas que aparecen

- **Explicador post-hoc:** ver [glosario](README.md#explicador-post-hoc-máscara-y-ranking). La idea
  clave es que no cambia ni reentrena el modelo original: solo lo "interroga" después de que ya tomó
  su decisión.
- **Máscara de aristas:** un valor (de 0 a 1, en la práctica) asignado a cada arista del vecindario,
  que indica qué tan importante fue esa conexión para la predicción.
- **Ranking de features (atributos):** un orden de los atributos del nodo (de los 166 disponibles),
  de más a menos importante para la decisión.
- **Optimización por instancia:** hacer un cálculo de ajuste específico para cada caso individual (el
  método de GNNExplainer), en vez de usar una regla general aprendida de antemano.
- **Red amortizada (PGExplainer):** un modelo entrenado una sola vez sobre muchos ejemplos, que
  después puede aplicarse rápidamente a nuevos casos sin tener que "reoptimizar" desde cero cada vez.
  "Amortizar" aquí significa repartir el costo de aprender entre todos los casos futuros.
- **Valores de Shapley:** una forma matemática de repartir "el mérito" de un resultado entre varios
  factores que contribuyeron juntos, mirando cómo cambia el resultado según qué combinaciones de esos
  factores están presentes o ausentes. Viene de la teoría de juegos (cómo repartir ganancias entre
  jugadores de forma justa).

## 📈 Cómo leer la figura / tabla

No aplica: es una lista de tres viñetas, sin figura ni tabla.

## 🎤 El discurso (como se dice en voz alta)

> Y sobre esas redes aplicamos tres explicadores post-hoc. Post-hoc significa que explican un modelo ya
> entrenado, sin modificarlo. Cada uno entrega dos cosas: una mascara de aristas y un ranking de
> atributos. *(pausa)* GNNExplainer optimiza una mascara por cada instancia, es decir, aprende que importa
> para ese nodo en particular. PGExplainer, en cambio, entrena una red amortizada que aprende a explicar
> en todo el grafo, y por eso generaliza entre nodos. Y GNNShap reparte el credito entre los elementos con
> valores de Shapley, que vienen de la teoria de juegos, lo que lo hace muy consistente. *(pausa)* Cuatro
> arquitecturas por tres explicadores es el nucleo de la comparacion.

### Versión ampliada y explicada

*"Y sobre esas redes aplicamos tres explicadores post-hoc. Post-hoc significa que explican un modelo
ya entrenado, sin modificarlo."* — Define el término técnico en la misma frase, para que no quede
ambiguo: el modelo predictivo (la GNN) ya terminó su entrenamiento; el explicador solo lo analiza
después.

*"Cada uno entrega dos cosas: una mascara de aristas y un ranking de atributos."* — Es importante fijar
esto: los tres explicadores, aunque funcionan distinto por dentro, producen el mismo **tipo** de
salida, lo que hace posible compararlos entre sí de forma justa.

*"GNNExplainer optimiza una mascara por cada instancia, es decir, aprende que importa para ese nodo en
particular."* — Se resalta su rasgo distintivo: es un proceso "a medida" para cada caso, lo que en
principio lo hace muy preciso para ese caso concreto, pero también más costoso de calcular.

*"PGExplainer, en cambio, entrena una red amortizada que aprende a explicar en todo el grafo, y por
eso generaliza entre nodos."* — Se contrasta con GNNExplainer: en vez de "empezar de cero" con cada
nodo, aprende un patrón general una sola vez.

*"Y GNNShap reparte el credito entre los elementos con valores de Shapley, que vienen de la teoria de
juegos, lo que lo hace muy consistente."* — Se anticipa, en esta misma frase, uno de los hallazgos más
importantes de los resultados: GNNShap va a resultar ser el más estable internamente de los tres (ver
[lámina 24](lamina-24-resultado2-disociacion.md)).

*"Cuatro arquitecturas por tres explicadores es el nucleo de la comparacion."* — Cierra recordando la
matriz completa que se va a poner a prueba: 4 × 3 = 12 combinaciones básicas, antes de multiplicar por
los escenarios de desbalance y las estrategias de balanceo.

## 🧑‍⚖️ Preguntas de jurado probables

- **"¿Por qué elegir estos tres explicadores y no otros?"** → Porque representan tres familias
  distintas de enfoque en la literatura de XAI para GNNs: optimización por instancia (GNNExplainer),
  aprendizaje amortizado (PGExplainer) y valores de Shapley (GNNShap), lo que da una comparación
  representativa del estado del arte.
- **"Si los tres dan una máscara y un ranking, ¿por qué no siempre se pueden comparar entre sí de
  forma directa?"** → Porque, aunque la salida tiene la misma forma, cada uno puede degenerar o
  saturarse de forma distinta según el tipo de grafo (ver, por ejemplo, la lámina 21 y el respaldo
  sobre por qué el ranking usa solo GNNExplainer).

## 🧠 En una frase

Los tres explicadores post-hoc entregan lo mismo (una máscara de aristas y un ranking de atributos),
pero llegan a esa respuesta por caminos muy distintos: optimización caso por caso, aprendizaje
generalizado, o reparto de crédito estilo teoría de juegos.
