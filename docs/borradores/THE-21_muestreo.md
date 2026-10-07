# THE-21: justificación del muestreo (borrador para el capítulo de Metodología)

Pedido de Cristian (28-sep y 4-oct): una técnica, con su porqué y un paper, sin listar siete. Hay que decir qué técnica de submuestreo se usó y por qué se eligió.

Abajo van el párrafo listo para LaTeX y las entradas de bibliografía que faltan. Los datos del código salen de `src/data/imbalance.py`.

## Párrafo (LaTeX)

```latex
\subsection{Cómo se construyen los escenarios de desbalance}

El criterio que guía todos los escenarios es proteger la señal del fraude: las transacciones
ilícitas son la clase más escasa, de modo que ningún escenario principal descarta una sola de
ellas. El nivel de desbalance se modifica por dos vías, y únicamente en el conjunto de
entrenamiento; validación y prueba conservan su composición real.

La primera vía es el \emph{submuestreo aleatorio de la clase negativa}. Para obtener una razón
de 1:10 o 1:20 se toma al azar, sin reemplazo, el número necesario de negativos, estratificando
por su origen (lícitas etiquetadas y transacciones sin etiqueta) para que la mezcla de ambos
grupos sea la misma que en el escenario nativo. El sorteo usa una semilla de datos fija,
independiente de la semilla del modelo, de forma que las tres semillas de una configuración ven
exactamente el mismo conjunto de entrenamiento y la variación entre ellas refleja solo el
entrenamiento. Se eligió el submuestreo aleatorio y no una variante informada, como el vecino
más cercano editado o los enlaces de Tomek, porque estas eliminan precisamente los negativos
cercanos a la frontera de decisión. Con ellas el escenario no solo cambiaría la proporción de
clases, sino también qué tipo de negativos ve el modelo, y el efecto del desbalance quedaría
confundido con una limpieza de los datos. Además, el submuestreo aleatorio no introduce
información que no esté en el conjunto original \parencite{He2009LearningData}. Las
transacciones descartadas no se eliminan del grafo: siguen enviando mensajes a sus vecinas y
solo dejan de contar en la función de pérdida.

La segunda vía es el \emph{sobremuestreo de las ilícitas con SMOTE}
\parencite{Chawla2002SMOTE}, que se usa en el escenario 1:10 con SMOTE para duplicar el número
de ilícitas sin repetir ninguna. Cada ilícita sintética interpola las variables de una ilícita
real y de una de sus cinco ilícitas más cercanas; a diferencia de la copia exacta, esto evita
que el modelo memorice casos repetidos. Como el modelo es una red sobre grafos, el nodo
sintético necesita vecinos: hereda las aristas de la ilícita que le dio origen, que es la línea
base con copia de aristas de GraphSMOTE \parencite{Zhao2021Graphsmote:Networks}. Los nodos
sintéticos viven solo en los pasos de tiempo de entrenamiento y nunca tocan validación ni
prueba.

Estas técnicas se usan en el mismo conjunto de datos: \textcite{Alarab2022Resampling} compararon
métodos de sobremuestreo y submuestreo sobre Elliptic y mostraron que el remuestreo cambia la
importancia que el modelo asigna a cada variable, es decir, su explicación. Ese hallazgo es el
punto de partida de la primera hipótesis de esta tesis: si remuestrear cambia qué explica el
modelo, cabe preguntar si también cambia qué tan estable es esa explicación.
```

## Entradas de bibliografía que faltan en `tesis_latex/bibliografia.bib`

`Zhao2021Graphsmote:Networks` ya está. Las tres siguientes son reales; Alarab y Prakoonwit (2022) se verificó en el PDF de la revista.

```bibtex
@article{Chawla2002SMOTE,
    title = {{SMOTE: Synthetic minority over-sampling technique}},
    year = {2002},
    journal = {Journal of Artificial Intelligence Research},
    author = {Chawla, N V and Bowyer, K W and Hall, L O and Kegelmeyer, W P},
    volume = {16},
    pages = {321-357},
    doi = {10.1613/jair.953}
}

@article{He2009LearningData,
    title = {{Learning from imbalanced data}},
    year = {2009},
    journal = {IEEE Transactions on Knowledge and Data Engineering},
    author = {He, H and Garcia, E A},
    number = {9},
    volume = {21},
    pages = {1263-1284},
    doi = {10.1109/TKDE.2008.239}
}

@article{Alarab2022Resampling,
    title = {{Effect of data resampling on feature importance in imbalanced blockchain data: Comparison studies of resampling techniques}},
    year = {2022},
    journal = {Data Science and Management},
    author = {Alarab, I and Prakoonwit, S},
    volume = {5},
    pages = {66-76},
    doi = {10.1016/j.dsm.2022.04.003}
}
```

## Si el jurado pregunta

- **¿Por qué no submuestrear las ilícitas?** Porque son la señal más débil: quitarlas reduce lo que el modelo puede aprender del fraude y mezcla el efecto del desbalance con el de tener menos ejemplos. Las versiones que quitaban ilícitas (diseño del 23-sep) se descartaron por eso.
- **¿Por qué el 1:1 va al anexo?** Con clases balanceadas los pesos por clase valen 1 y focal loss queda igual que la entropía cruzada, así que no sirve para comparar formas de balanceo. Además ninguna configuración 1:1 pasó la compuerta.
- **¿Por qué no GraphSMOTE completo?** Su generador de aristas aprendido añade un modelo más que entrenar y explicar. La copia de aristas es la línea base que el mismo artículo de GraphSMOTE reporta.
