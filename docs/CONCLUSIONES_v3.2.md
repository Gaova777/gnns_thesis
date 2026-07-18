# CONCLUSIONES v3.2 — Estabilidad de la explicabilidad en GNNs sobre Elliptic

Fecha: 2026-07-15
Estado: reemplaza los hallazgos cuantitativos de CONCLUSIONES_v3.1.md, que quedaron
invalidados tras la corrección de los bugs de medición, la resolución del OOM de GAT y
la reescopación de la explicación al receptive field (k-hop subgraph). Todos los
números provienen de la corrida verificada de forma independiente (GNNExplainer,
nodes_per_class=30, k-hop subgraph, GAT completo en GPU).

## 1. Encuadre del estudio

Los modelos evaluados aprenden a discriminar la clase ilícita sobre el conjunto de
validación pero no transfieren ese desempeño al conjunto de prueba, cuyo período
temporal corresponde a los timesteps posteriores al cierre de un mercado de la darknet.
Esta caída entre validación y prueba es una manifestación del cambio de distribución
temporal propio del Elliptic dataset y no un defecto del pipeline de entrenamiento.
En consecuencia, el estudio de estabilidad de la explicabilidad se realiza sobre los
verdaderos positivos del conjunto de validación, es decir, sobre los nodos ilícitos
que cada modelo clasifica correctamente en el régimen donde efectivamente discrimina.
Este encuadre se declara de forma explícita como una limitación del alcance: las
conclusiones sobre estabilidad son válidas en el régimen de validación y no deben
extrapolarse al régimen de prueba bajo cambio de distribución.

## 2. La topología de Elliptic condiciona qué estabilidad es medible

El grafo de Elliptic es extremadamente disperso alrededor de los nodos ilícitos. El
receptive field de dos saltos de un nodo ilícito de validación contiene, en la mediana,
entre dos y tres nodos y entre uno y dos edges, según la arquitectura. Esta dispersión
tiene una consecuencia directa sobre las métricas de estabilidad estructural: el índice
de Jaccard sobre los top-k edges no aporta información discriminante, porque el número
de edges disponibles en el vecindario es muy inferior al valor de k solicitado por la
configuración. Por esta razón, la estabilidad del ranking de features, medida mediante
la correlación de Spearman sobre las 165 features de entrada, se adopta como la métrica
primaria de estabilidad, mientras que el Jaccard se reporta únicamente para documentar
por qué la estabilidad estructural no es evaluable de forma informativa en este dataset.

## 3. Resultado principal: ranking de estabilidad por arquitectura

Sobre la métrica primaria de estabilidad del ranking de features, el orden de las
arquitecturas es el siguiente, con el número de celdas medibles indicado entre
paréntesis sobre un total de quince por arquitectura:

| Arquitectura | Spearman medio | Spearman mediano | Celdas medibles |
|---|---|---|---|
| GraphSAGE | 0.630 | 0.656 | 15/15 |
| GAT | 0.486 | 0.476 | 15/15 |
| GCN | 0.468 | 0.639 | 15/15 |
| TAGCN | 0.270 | 0.154 | 12/12 |

GraphSAGE es la arquitectura cuyas explicaciones son más estables, y este resultado se
mantiene a través de las tres versiones sucesivas del pipeline, incluyendo las dos que
resultaron contaminadas por bugs de medición o por fallos de memoria. Su robustez frente
al método de medición lo convierte en el hallazgo más sólido del estudio. TAGCN produce
las explicaciones menos estables, con una separación clara respecto a las demás
arquitecturas tanto en media como en mediana. GAT y GCN ocupan posiciones intermedias y
no son distinguibles entre sí, dado que la diferencia entre sus medias es menor que la
dispersión interna de sus celdas; por tanto se reportan como comparables y no se afirma
un orden entre ellos.

## 4. Hallazgos previos que quedan retractados

Dos hallazgos que la versión anterior del documento presentaba como contribuciones
centrales quedan retractados por no sostenerse sobre datos corregidos. El primero es el
supuesto tradeoff entre desempeño predictivo y estabilidad de la explicación, que
sostenía que la arquitectura más precisa producía las explicaciones menos estables; con
la medición corregida GraphSAGE encabeza la estabilidad, de modo que el tradeoff no se
observa. El segundo es el patrón de pico y colapso de la estabilidad en función del
desbalance, según el cual la estabilidad alcanzaba un máximo en el escenario 1:50 y
caía en 1:100; con la medición sobre el receptive field real este patrón desaparece y la
estabilidad por escenario no muestra una tendencia monótona pronunciada. Ambos patrones
eran artefactos de medir sobre el conjunto de prueba contaminado por el cambio de
distribución temporal y sobre el grafo completo en lugar del vecindario relevante.

## 5. Estado de los explicadores

GNNExplainer es el único explicador que produce una señal de estabilidad con variación
informativa en este dataset, con correlaciones de Spearman que cubren un rango amplio
según la arquitectura y el escenario. PGExplainer presenta una degeneración total, con
correlación de Spearman nula en la totalidad de las celdas medibles, atribuible a un
comportamiento del método en la versión 2.7 de la librería utilizada; este resultado se
reporta como un hallazgo metodológico y no como una medida de estabilidad, y los ajustes
de hiperparámetros ensayados mitigan el problema en el dataset Cora pero no lo resuelven
en Elliptic. GNNShap ofrece una señal parcial y se considera una métrica secundaria.

## 6. Soporte estadístico y sus límites

La caracterización se apoya en treinta nodos por celda, de los cuales cincuenta y dos de
las cincuenta y siete celdas alcanzan soporte pleno y solo cuatro quedan por debajo de
diez nodos medibles, lo que representa una mejora sustancial respecto a la configuración
previa de tres nodos por celda. No obstante, el diseño mantiene un único modelo entrenado
por celda, de modo que la comparación entre celdas descansa sobre modelos individuales y
no sobre réplicas de entrenamiento. Por esta razón el análisis se limita a marginales por
factor acompañadas de tamaños de efecto, y no a un análisis de varianza factorial con
pretensión de potencia estadística; una verificación de degeneración del diseño precede a
cualquier prueba y descarta las respuestas de varianza nula, como la del índice de Jaccard.
Esta limitación se declara de forma explícita para que las conclusiones se interpreten
como una caracterización descriptiva y no como una inferencia poblacional.

## 7. Síntesis

El estudio caracteriza la estabilidad de las explicaciones de cuatro arquitecturas de
GNN sobre el Elliptic dataset bajo desbalance, en el régimen de validación donde los
modelos discriminan. GraphSAGE produce las explicaciones más estables de forma robusta al
método de medición, TAGCN las menos estables, y GAT y GCN quedan en un rango intermedio
indistinguible entre sí. La dispersión estructural del dataset impide evaluar la
estabilidad a nivel de subgrafo, lo que restringe el análisis a la estabilidad del
ranking de features y motiva el uso de un dataset sintético con estructura de tipología
controlada para el estudio de plausibilidad previsto en la segunda fase de la
investigación.
