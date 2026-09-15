# Lámina 08 — Cómo funciona una GNN: paso de mensajes

> Bloque: Marco conceptual
> Voz: Alejandro · Página PDF 11 (8/40) · Tiempo objetivo 60 s

## 🎯 Objetivo de la lámina

Explicar, desde cero, el mecanismo básico que hace funcionar a **cualquier** red neuronal de grafos:
el paso de mensajes. Esta lámina es la base sin la cual no se entiende nada de lo que viene después
(las arquitecturas, los explicadores, ni por qué "explicar" en un grafo significa señalar aristas y
atributos).

## 📋 Qué dice, item por item

- **"Cada nodo actualiza su representación agregando la de sus vecinos."** — La idea central: un nodo
  (una transacción) no se queda con su información original; la va mezclando con la de los nodos
  conectados a él.
- **"Se repite capa por capa: con L capas, un nodo 've' hasta L saltos de distancia (su campo
  receptivo)."** — Este proceso de "mezclar con los vecinos" no ocurre una sola vez, se repite varias
  veces (una vez por cada capa del modelo). Con dos capas, un nodo ya está usando información de sus
  vecinos y de los vecinos de sus vecinos.
- **"La predicción sobre una transacción depende de su vecindario, no solo de ella."** — Consecuencia
  directa: cuando el modelo dice "esta transacción es sospechosa", esa decisión está influida por
  toda la red que la rodea, no solo por los datos propios de esa transacción.
- **Recuadro "Por eso, explicar ="** — señalar **qué** del vecindario (qué aristas, qué atributos)
  sostuvo la predicción. Esta frase define, de una vez, qué es "explicar" en el contexto de esta
  tesis.
- **Diagrama:** un nodo central (v) rodeado de cuatro vecinos, cada uno con una flecha que apunta
  hacia v, representando que "envían mensajes" hacia el nodo central.

## 🔑 Conceptos y técnicas que aparecen

- **Paso de mensajes (message passing):** el mecanismo por el cual cada nodo combina su propia
  información con la de sus vecinos, en rondas sucesivas (una por capa). Es la operación fundamental
  detrás de todas las arquitecturas de GNN. Ver también el
  [glosario](README.md#gnn-y-paso-de-mensajes).
- **Capa:** cada "ronda" de paso de mensajes. Más capas significa que la información puede viajar más
  lejos en la red antes de llegar al nodo que se está prediciendo.
- **Campo receptivo:** el conjunto de nodos que, directa o indirectamente, influyen en la predicción
  de un nodo dado, según el número de capas (saltos) del modelo. Con dos capas, el campo receptivo
  llega hasta dos saltos de distancia.

## 📈 Cómo leer la figura / tabla

El diagrama muestra un nodo central resaltado (v) con cuatro vecinos alrededor, cada uno conectado
por una flecha que apunta **hacia** v. La lectura es literal: cada vecino "envía un mensaje" al nodo
central, y v combina esos cuatro mensajes con su propia información para actualizar su
representación. Este es el paso básico que, repetido capa por capa, define cómo "piensa" una GNN.

## 🎤 El discurso (como se dice en voz alta)

> Un marco minimo antes de los metodos, empezando por como funciona una red neuronal de grafos. *(pausa)*
> La idea central es el paso de mensajes: cada nodo actualiza su representacion agregando la informacion
> de sus vecinos. Y eso se repite capa por capa: con dos capas, un nodo ve a sus vecinos y a los vecinos
> de sus vecinos, es decir, hasta dos saltos de distancia; a eso lo llamamos su campo receptivo. *(pausa)*
> La consecuencia importante para nosotros es esta: la prediccion sobre una transaccion no depende solo de
> ella, sino de todo su vecindario. Y por eso explicar, en una red de grafos, significa senalar que parte
> de ese vecindario, que aristas y que atributos, sostuvo la prediccion. Ese es el objeto que toda la
> tesis va a medir.

### Versión ampliada y explicada

*"Un marco minimo antes de los metodos, empezando por como funciona una red neuronal de grafos."* —
Anuncia que se va a construir, paso a paso, el vocabulario técnico mínimo necesario, empezando desde
el mecanismo más básico.

*"La idea central es el paso de mensajes: cada nodo actualiza su representacion agregando la
informacion de sus vecinos."* — Es la definición central. "Representación" aquí quiere decir el
conjunto de números internos con los que el modelo describe a ese nodo; esos números se van
actualizando al mezclarse con los de los vecinos.

*"Y eso se repite capa por capa: con dos capas, un nodo ve a sus vecinos y a los vecinos de sus
vecinos, es decir, hasta dos saltos de distancia; a eso lo llamamos su campo receptivo."* — Aquí se
explica por qué el número de capas importa: cada capa añade un salto más de alcance en la red.

*"La consecuencia importante para nosotros es esta: la prediccion sobre una transaccion no depende
solo de ella, sino de todo su vecindario."* — Esta es la idea que se necesita para entender por qué
el vecindario, y no solo la transacción aislada, es relevante para la decisión del modelo.

*"Y por eso explicar, en una red de grafos, significa senalar que parte de ese vecindario, que aristas
y que atributos, sostuvo la prediccion. Ese es el objeto que toda la tesis va a medir."* — Cierra
definiendo, con toda precisión, qué es una "explicación" en este contexto: no es una frase en
lenguaje natural, es señalar concretamente qué aristas y qué atributos del vecindario pesaron en la
decisión. Sobre ese objeto exacto (una selección de aristas más un ranking de atributos) se va a medir
estabilidad, plausibilidad y fidelidad durante el resto de la tesis.

## 🧑‍⚖️ Preguntas de jurado probables

- **"¿Por qué el número de capas no puede ser muy alto, para que el campo receptivo sea enorme?"** →
  Porque en la práctica, con demasiadas capas, la información de todos los nodos empieza a mezclarse
  y se pierde la capacidad de distinguir vecindarios (un fenómeno conocido en la literatura como
  "over-smoothing"); por eso los modelos suelen usar pocas capas.
- **"¿Todas las arquitecturas de GNN usan exactamente el mismo paso de mensajes?"** → El principio
  general es el mismo, pero cada arquitectura agrega a los vecinos de forma distinta (promedio,
  muestreo, atención, filtros polinómicos), como se ve en la siguiente lámina.

## 🧠 En una frase

Una GNN decide sobre un nodo mezclando su información con la de sus vecinos, capa por capa, y por eso
explicar su decisión significa señalar qué parte de ese vecindario la sostuvo.
