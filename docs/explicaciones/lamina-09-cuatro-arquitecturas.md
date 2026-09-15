# Lámina 09 — Cuatro arquitecturas GNN

> Bloque: Marco conceptual
> Voz: Alejandro · Página PDF 12 (9/40) · Tiempo objetivo 50 s

## 🎯 Objetivo de la lámina

Presentar, con una intuición simple para cada una, las cuatro arquitecturas de red de grafos que se
van a comparar en toda la tesis. La idea no es memorizar fórmulas, sino entender **qué hace distinta**
a cada una en la forma de "escuchar" a sus vecinos.

## 📋 Qué dice, item por item

- **GCN — "promedia a los vecinos (convolución espectral de un salto). Simple y sólida."** — Es la
  arquitectura más básica: junta la información de todos los vecinos dándoles, en esencia, el mismo
  peso, como un promedio.
- **GraphSAGE — "muestrea vecinos y los agrega; inductiva, escala a grafos grandes."** — En vez de
  mirar a todos los vecinos, elige una muestra (un subconjunto al azar) y agrega solo esos. Esto la
  hace más eficiente en grafos muy grandes, y "inductiva" significa que puede generalizar a nodos que
  nunca vio durante el entrenamiento.
- **GAT — "pondera a los vecinos con atención; no todos pesan igual."** — En vez de tratar a todos los
  vecinos por igual (como GCN), GAT aprende a darle más peso a los vecinos que considera más
  relevantes para cada nodo, usando un mecanismo llamado "atención".
- **TAGCN — "filtros polinómicos de orden K que alcanzan varios saltos de una vez."** — En vez de
  necesitar varias capas para llegar a varios saltos de distancia, TAGCN usa un tipo de filtro
  matemático que puede combinar información de varios saltos en un solo paso.
- **Nota final:** se eligieron porque cubren las familias dominantes de agregación de vecinos, y
  compararlas es precisamente el segundo objetivo de la tesis (O2).

## 🔑 Conceptos y técnicas que aparecen

- **Convolución espectral:** una forma matemática (tomada del procesamiento de señales) de combinar
  la información de un nodo con la de sus vecinos de forma sistemática. En GCN, en la práctica,
  termina pareciéndose a un promedio ponderado por la estructura del grafo.
- **Muestreo (sampling):** en vez de usar todos los vecinos de un nodo (que en un grafo grande pueden
  ser miles), se elige aleatoriamente un número fijo de ellos. Esto ahorra memoria y tiempo de
  cómputo.
- **Modelo inductivo:** un modelo que puede aplicarse a nodos o grafos que no vio durante el
  entrenamiento, sin tener que volver a entrenarse desde cero. Es lo contrario de un modelo
  "transductivo", que solo funciona sobre el grafo exacto con el que fue entrenado.
- **Mecanismo de atención:** una forma de que el modelo aprenda, por sí solo, qué vecinos merecen más
  peso para una decisión concreta, en vez de tratarlos a todos igual.
- **Filtro polinómico de orden K:** una operación matemática que, en una sola pasada, combina
  información de hasta K saltos de distancia (en vez de necesitar K capas separadas para lograr lo
  mismo).

## 📈 Cómo leer la figura / tabla

No aplica: es una lista con cuatro viñetas, una por arquitectura, sin figura ni tabla.

## 🎤 El discurso (como se dice en voz alta)

> Sobre esa base trabajamos con cuatro arquitecturas, que se diferencian justamente en como cada nodo
> agrega a sus vecinos. *(pausa)* GCN simplemente los promedia, con una convolucion espectral de un salto;
> es la mas simple y solida. GraphSAGE muestrea un subconjunto de vecinos y los agrega; es inductiva y
> escala a grafos grandes. GAT no trata a todos los vecinos por igual: les asigna pesos con un mecanismo
> de atencion. Y TAGCN usa filtros polinomicos de orden K, que alcanzan varios saltos de una sola vez.
> *(pausa)* Las elegimos porque cubren las familias dominantes de agregacion, y compararlas es
> precisamente nuestro segundo objetivo.

### Versión ampliada y explicada

*"Sobre esa base trabajamos con cuatro arquitecturas, que se diferencian justamente en como cada nodo
agrega a sus vecinos."* — Conecta directamente con la lámina anterior (el paso de mensajes): estas
cuatro arquitecturas son cuatro formas distintas de implementar exactamente esa idea.

*"GCN simplemente los promedia, con una convolucion espectral de un salto; es la mas simple y
solida."* — Se presenta a GCN como el punto de referencia más básico: no discrimina entre vecinos,
solo combina su información de forma pareja.

*"GraphSAGE muestrea un subconjunto de vecinos y los agrega; es inductiva y escala a grafos
grandes."* — Se destacan sus dos ventajas prácticas: eficiencia (no necesita mirar a todos los
vecinos) y capacidad de generalizar a nodos nuevos.

*"GAT no trata a todos los vecinos por igual: les asigna pesos con un mecanismo de atencion."* — Se
resalta su diferencia clave respecto a GCN: en vez de un promedio simple, aprende qué vecinos
importan más para cada decisión concreta.

*"Y TAGCN usa filtros polinomicos de orden K, que alcanzan varios saltos de una sola vez."* — Se
explica su rasgo distintivo: no necesita apilar muchas capas para "ver lejos"; lo logra en una sola
operación matemática. Esta es la razón por la que, en la hipótesis H2, se esperaba que fuera la más
estable.

*"Las elegimos porque cubren las familias dominantes de agregacion, y compararlas es precisamente
nuestro segundo objetivo."* — Justifica por qué estas cuatro y no otras: representan los enfoques más
usados en la literatura, así que la comparación tiene relevancia general, no solo para casos
particulares.

## 🧑‍⚖️ Preguntas de jurado probables

- **"¿Por qué se esperaría que la arquitectura influya en la estabilidad de las explicaciones?"** →
  Porque cada arquitectura agrega la información del vecindario de forma distinta, y esa diferencia
  en cómo se construye la representación interna del nodo podría, en teoría, hacer que las
  explicaciones basadas en esas representaciones sean más o menos consistentes.
- **"¿Cuál de las cuatro resultó ser la más estable?"** → No hay una única ganadora: los resultados
  muestran una partición en dos grupos (GAT y GCN en el grupo alto; GraphSAGE y TAGCN en el grupo
  bajo), no un ranking de cuatro posiciones (ver [lámina 21](lamina-21-resultado1-dos-grupos.md)).

## 🧠 En una frase

Las cuatro arquitecturas se diferencian en cómo cada nodo escucha a sus vecinos: promediando (GCN),
muestreando (GraphSAGE), ponderando con atención (GAT) o alcanzando varios saltos de una vez (TAGCN).
