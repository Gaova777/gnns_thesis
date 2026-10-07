# PGExplainer con pesos por clase y GCN que queda atrás (reunión del 4-oct, THE-32 y THE-33)

## Parte A (THE-32): ¿por qué PGExplainer da ≈ 0 con class weighting?

### La pregunta

En el análisis por configuración (las que pasan la compuerta, sin 1:1), la estabilidad de PGExplainer por pérdida fue:

| Pérdida | Estabilidad |
|---|---|
| Sin ajuste | 0,558 |
| Class weighting | 0,034 |
| Focal loss | 0,245 |

Con GNNExplainer las tres quedan entre 0,91 y 0,96. Cristian pidió decir si es culpa de PGExplainer, de class weighting, de ambos o de un bug, y explicar la naturaleza de cada explicador.

### Respuesta corta

**No es un bug ni un efecto de class weighting: es una limitación de PGExplainer en Elliptic.** En este grafo la métrica de estabilidad de PGExplainer se calcula sobre unos 16 nodos con 2 o 3 aristas. Sobre esa base es tan ruidosa que el mismo modelo, explicado dos veces, pasa de 0,04 a 0,42. El ≈ 0 venía de un subconjunto pequeño (4 bloques). Con los 144 modelos el orden se mantiene, pero la diferencia es moderada y se concentra en dos arquitecturas.

### Evidencia

1. **No hay error de cálculo.**
   - Ninguna de las 144 ejecuciones tuvo réplicas fallidas (`pg_failed_replicas` = 0).
   - Los pares de réplicas con máscara constante se descartan (quedan como NaN) en vez de contarse como 0.
   - Recalculando desde las máscaras guardadas (`results_v4/rankings/*__PGExplainer.npz`) se obtiene exactamente el valor del pipeline: correlación 1,0 y diferencia máxima 0.
2. **La métrica se apoya en muy pocos nodos y casi solo toma los valores ±1.**
   - PGExplainer explica aristas, no variables. El 83 % de los nodos explicados tiene 2 aristas o menos en su subgrafo, y solo unos 16 por modelo tienen 2 o más (mediana 16, mínimo 8).
   - Con 2 aristas, el Spearman entre dos réplicas solo puede ser +1 o −1. El **93 %** de los pares toma exactamente ±1.
3. **Las máscaras se saturan y el orden entre aristas lo decide el ruido numérico.**
   - Los valores se van a 0 o a 1.
   - En el 82 % (class weighting) al 98 % (focal) de los nodos, la diferencia entre las aristas de un mismo nodo es menor que 0,001.
   - Cuando una réplica sale «alta» y otra «baja», el Spearman medio de ese par es −0,59 a −0,69. Esos pares son el 54 % del total, lo que arrastra el promedio hacia 0.
4. **El mismo modelo da valores muy distintos al repetirlo.** El modelo nativo de GraphSAGE con pesos por clase, explicado de nuevo con el mismo código, los mismos nodos y las mismas semillas, dio **0,422** frente a **0,038** de la ejecución original. La GPU no es determinista.
5. **El intervalo de cada modelo es más ancho que las diferencias entre pérdidas.** El intervalo bootstrap del 95 % (remuestreando nodos) tiene un ancho mediano de **0,45** (de ±0,23) por modelo.
6. **Con los 144 modelos la diferencia existe, pero es moderada y no general:**

   | Pérdida | Estabilidad media (error estándar) |
   |---|---|
   | Sin ajuste | 0,446 (0,036) |
   | Focal loss | 0,324 (0,051) |
   | Class weighting | 0,243 (0,044) |

   La caída con class weighting viene de GraphSAGE (0,433 a 0,056) y TAGCN (0,470 a 0,120). En GAT no hay diferencia: 0,585, 0,606 y 0,541.

   Un mecanismo plausible, que no está probado: con pesos por clase el modelo predice ilícita para más nodos (en el entrenamiento de PGExplainer, 50 % frente a 5 % sin ajuste). Además, el acuerdo entre réplicas del mismo tipo baja (0,43 frente a 0,74 en los pares «bajos»).

### Naturaleza de cada explicador

| Explicador | Qué explica | Cómo | Por qué es más o menos estable aquí |
|---|---|---|---|
| GNNExplainer | Variables (165) y aristas | Optimiza una máscara por nodo explicado | El orden se mide sobre 165 variables, así que una réplica distinta mueve poco el Spearman |
| PGExplainer | Solo aristas | Entrena una red (MLP) que predice la máscara de cada arista para todos los nodos a la vez (amortizado) | Con 2 o 3 aristas por nodo el orden es ±1, y con máscaras saturadas lo decide el ruido |
| Shapley (GNNShap) | Variables | Promedia contribuciones sobre muchas coaliciones muestreadas | El promedio reduce el ruido por construcción (0,99 de estabilidad) |

### Qué cambia en la tesis

- H3 se responde con GNNExplainer y Shapley. Las dos dan el resultado esperado: el balanceo no mueve la estabilidad (GNNExplainer p exacto 0,27; Shapley equivalente por TOST).
- PGExplainer se reporta como **limitación**: en un grafo tan disperso como Elliptic (el 93 % de las ilícitas tiene grado ≤ 2), una medida de estabilidad sobre aristas no tiene resolución. Se informa el promedio de los 144 modelos con su error y el resultado del test-retest, no el ≈ 0 del subconjunto.

## Parte B (THE-33): ¿por qué GCN queda atrás?

### La pregunta

Ninguna configuración de GCN pasa la compuerta. Su PR-AUC de validación queda entre 0,08 y 0,19, mientras GraphSAGE, GAT y TAGCN llegan a 0,46, 0,40 y 0,41 en su mejor configuración (azar 0,024). Cristian preguntó:

- ¿Era esperado?
- ¿En qué consiste GCN para quedar atrás?
- ¿Por qué GraphSAGE, TAGCN y GAT quedan parecidas?

### Respuesta corta

**En Elliptic la señal del fraude está en las variables del propio nodo, y GCN es la única de las cuatro arquitecturas que no tiene un peso propio para el nodo.** GCN promedia al nodo con sus vecinos usando el mismo peso, y como casi ningún vecino de una ilícita es ilícito, la señal se diluye. Basta agregarle un peso propio al nodo para que rinda como GraphSAGE.

### Evidencia

1. **Experimento de control con los mismos hiperparámetros, el mismo bucle y 300 épocas.** Escenario nativo, sin ajuste y pesos por clase, semillas 42 y 43, PR-AUC de validación media (desviación estándar):

   | Modelo | Qué es | PR-AUC val |
   |---|---|---|
   | MLP | Sin grafo, solo las variables del nodo | 0,485 (0,014) |
   | GraphSAGE | Peso propio + media de los vecinos | 0,481 (0,020) |
   | GCN con peso propio | GCN + una capa lineal para el nodo en cada capa | 0,464 (0,071) |
   | GCN | Kipf y Welling: D^-1/2 (A+I) D^-1/2 | **0,214** (0,038) |

   Con los hiperparámetros que eligió Optuna para GCN en la matriz se repite: GCN 0,236, GCN con peso propio 0,495 y MLP 0,468. No es un problema de hiperparámetros.
2. **El grafo no ayuda a detectar ilícitas** (modo C, train):
   - El 93 % de las ilícitas tiene grado ≤ 2.
   - Solo el 12 % de los vecinos de una ilícita es ilícito.
   - La homofilia de las ilícitas en aristas etiquetadas es 0,37, contra 0,95 de las lícitas.

   Un MLP sin grafo rinde igual que GraphSAGE: el aporte del grafo es pequeño. Lo que importa es no perder las variables del nodo.
3. **Por qué las otras tres quedan parecidas:** las tres conservan el nodo con su propio peso.
   - GraphSAGE tiene un peso raíz separado.
   - TAGCN incluye el término k = 0 del filtro polinomial, que es la identidad con su propio peso.
   - GAT puede dar más atención al propio nodo.

   Al conservar la señal del nodo, las tres llegan al techo que marcan las variables (un MLP) y por eso rinden parecido.
4. **Es esperable y está en la literatura.** Weber et al. (2019, Tabla 1), sobre las etiquetadas de Elliptic:

   | Modelo | F1 de la clase ilícita |
   |---|---|
   | Random Forest con todas las variables | 0,788 |
   | MLP | 0,653 |
   | GCN | 0,628 |
   | Skip-GCN | 0,705 |

   Skip-GCN es GCN con una conexión directa desde las variables de entrada, que es lo mismo que nuestro GCN con peso propio. Los autores reportan la ganancia de Skip-GCN sobre GCN y que Random Forest, sin grafo, es el mejor. Nuestro resultado reproduce ese patrón en el modo C.

### Qué cambia en la tesis

- **H2:** la arquitectura decide *si el modelo aprende*, no qué tan estable explica: las tres que aprenden explican igual.
- GCN queda atrás por cómo agrega (sin peso propio para el nodo) en un grafo donde la señal está en el nodo, no por falta de optimización.

## Archivos

| Qué | Ruta |
|---|---|
| Scripts | `scripts/v4/reunion_0410/{inspect_pg_masks,pg_pairs_by_saturation,repro_pg_stability,diag_pgexplainer,gcn_vs_baselines}.py` |
| Máscaras de PGExplainer por modelo | `pg_masks_summary.csv` |
| Pares de réplicas por tipo de saturación | `pg_pairs_by_saturation.csv` |
| Intervalo bootstrap por modelo | `pg_ic_por_modelo.csv` |
| Test-retest | `repro_pg_stability.csv` |
| Diagnóstico instrumentado (GraphSAGE nativo y 1:10) | `diag_native_*`, `diag_1-10_*` |
| Control de GCN | `gcn_vs_baselines_common.csv`, `gcn_vs_baselines_gcn_optuna.csv`, `gcn_graph_stats.json` |
| Meta de los 144 modelos | `meta_all_models.csv` |

## Referencias

- Weber, M. et al. (2019). Anti-money laundering in Bitcoin: experimenting with graph convolutional networks for financial forensics. KDD Workshop on Anomaly Detection in Finance. arXiv:1908.02591.
- Kipf, T. y Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR.
- Hamilton, W. et al. (2017). Inductive representation learning on large graphs. NeurIPS.
- Du, J. et al. (2017). Topology adaptive graph convolutional networks. arXiv:1710.10370.
- Ying, R. et al. (2019). GNNExplainer: generating explanations for graph neural networks. NeurIPS.
- Luo, D. et al. (2020). Parameterized explainer for graph neural network. NeurIPS.
