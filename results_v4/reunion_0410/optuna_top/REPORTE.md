# Optimización completa del candidato top (THE-36, 7-oct)

Pedido de Cristian (reunión del 4-oct, punto A3): la matriz de 48 configuraciones con 8 trials de Optuna es la **fase de selección**; al candidato que sale de ella se le da una optimización completa y se vuelve a explicar.

- **Candidato:** GraphSAGE con pesos por clase. Es 1.º en val PR-AUC en native, 1:10 y 1:20 y 2.º en 1:10 con SMOTE (`../configs48/REPORTE.md`, sección 7-oct; Kendall W = 0,80).
- **Búsqueda:** 150 trials de Optuna por escenario (native, 1:10, 1:10_os, 1:20) con la semilla 42, el mismo espacio de búsqueda y las mismas 50 épocas por trial que la matriz. Las semillas 43 y 44 usan los hiperparámetros de la 42. Se explicaron los 12 modelos sobre los mismos 30 nodos de la matriz.
- **Scripts:** `scripts/v4/run_optuna_top.sh` (búsqueda), `scripts/v4/run_top_prauc.sh` (reentrenamiento) y `scripts/v4/reunion_0410/optuna_top.py` (este análisis).
- **Salidas:** este directorio. Los modelos están en `results_models_v4_top*/` y `results_models_v4_mat_prauc/`, y las explicaciones en `results_v4_top*/` y `results_v4_mat_prauc/` (no se versionan).

## Resumen

1. **La optimización completa sube el PR-AUC de validación de 0,44 a 0,46 hasta 0,50 a 0,51 en los cuatro escenarios**, con las tres semillas de acuerdo (sd ≤ 0,023) y las 12 configuraciones sobre la compuerta. El ROC-AUC de validación pasa de 0,92 a 0,93.
2. **En test no cambia nada de fondo.** PR-AUC de 0,011 a 0,012, unas dos veces la prevalencia (0,0057), y ROC-AUC de 0,71 a 0,73. El modelo optimizado cae igual que el de la matriz con el cambio temporal.
3. **La estabilidad de las explicaciones no cambia.**
   - GNNExplainer: 0,95 a 0,96, frente a 0,96 en la matriz. La diferencia es de una centésima, dentro del margen de equivalencia de ±0,05.
   - Shapley: 0,99 en las dos versiones.
   - PGExplainer: sigue bajo (0,16 a 0,22) por la saturación de máscaras explicada en `../pgexpl_gcn/REPORTE.md`.

   **Las conclusiones de H1 a H3 se sostienen con el candidato optimizado.**
4. **Hubo que alinear la detención temprana con la métrica que optimiza Optuna.**
   - **El problema:** la primera pasada usó la detención de la matriz, F1 en argmax con paciencia 20. Con las tasas de aprendizaje altas que eligió la búsqueda (0,004 a 0,010), 4 de los 12 modelos se detuvieron en la época 21 con su mejor F1 en la época 1, antes de aprender (val PR-AUC 0,13 a 0,16).
   - **La corrección:** se reentrenaron los 12 modelos deteniendo por PR-AUC, sin repetir la búsqueda.
   - **El control:** se reentrenaron también los hiperparámetros de la matriz con la misma regla. Su val PR-AUC queda igual que en la matriz (0,44 a 0,47), así que **la mejora viene de los hiperparámetros, no del cambio de detención**.
5. **Convergencia.** En native y 1:20 el mejor trial es el 21 y no mejora en los 129 siguientes. En 1:10 y 1:10 con SMOTE el mejor aparece en los trials 143 y 144. En esos dos escenarios la búsqueda no convergió del todo, pero lo que gana después del trial 50 es poco: de 0,46 a 0,51 en 1:10 y de 0,46 a 0,50 con SMOTE.

## Rendimiento (media ± sd de 3 semillas)

| Escenario | Versión | val PR-AUC | val ROC-AUC | test PR-AUC | test ROC-AUC | Compuerta |
|---|---|---|---|---|---|---|
| native | matriz (8 trials) | 0,462 ± 0,020 | 0,923 | 0,010 | 0,711 | 3/3 |
| native | **optimizado (150 trials)** | **0,505 ± 0,010** | 0,930 | 0,011 | 0,720 | 3/3 |
| 1:10 | matriz | 0,455 ± 0,008 | 0,926 | 0,011 | 0,713 | 3/3 |
| 1:10 | **optimizado** | **0,499 ± 0,006** | 0,927 | 0,011 | 0,728 | 3/3 |
| 1:10 con SMOTE | matriz | 0,438 ± 0,015 | 0,918 | 0,011 | 0,715 | 3/3 |
| 1:10 con SMOTE | **optimizado** | **0,507 ± 0,023** | 0,920 | 0,011 | 0,708 | 3/3 |
| 1:20 | matriz | 0,457 ± 0,032 | 0,923 | 0,011 | 0,722 | 3/3 |
| 1:20 | **optimizado** | **0,509 ± 0,020** | 0,929 | 0,012 | 0,732 | 3/3 |

Azar: PR-AUC 0,024 en validación y 0,0057 en test; ROC-AUC 0,5. Las dos versiones intermedias están en `comparacion_config.csv`:
- optimizado con detención por F1: 4 modelos sin aprender;
- matriz con detención por PR-AUC: igual rendimiento, pero 9 de 12 modelos bajo la compuerta de F1 en argmax.

La compuerta por F1 en argmax es degenerada bajo desbalance. Es la misma advertencia que ya trae CLAUDE.md para la v3.

## Estabilidad (media de los 4 escenarios)

| Explicador | Matriz | Optimizado | Entre semillas, matriz | Entre semillas, optimizado |
|---|---|---|---|---|
| GNNExplainer | 0,964 | 0,954 | 0,962 | 0,960 |
| ShapleyFeatures | 0,993 | 0,992 | 0,962 | 0,962 |
| PGExplainer | 0,018 | 0,191 | 0,540 | 0,590 |

Por escenario, GNNExplainer del modelo optimizado da 0,957 (native), 0,952 (1:10), 0,952 (1:10 con SMOTE) y 0,956 (1:20). Los cuatro caben en ±0,05, igual que H1 en la matriz.

## Hiperparámetros elegidos

Los cuatro escenarios eligen 64 neuronas ocultas y 3 capas, con dropout de 0,46 a 0,50. La matriz había elegido en los cuatro escenarios la misma red de 2 capas (64 neuronas, dropout 0,49, lr 0,0046, weight decay 2,7×10⁻⁵): el tercer trial de la búsqueda, que es el mismo en todos los escenarios porque el muestreador usa la semilla 42. La diferencia de fondo es la tercera capa.

| Escenario | lr | weight decay |
|---|---|---|
| native | 0,0036 | 7,3×10⁻⁵ |
| 1:10 | 0,0079 | 1,6×10⁻⁵ |
| 1:10 con SMOTE | 0,0100 | 1,4×10⁻⁵ |
| 1:20 | 0,0040 | 2,9×10⁻⁵ |

## Figura

`convergencia.png`: mejor PR-AUC de validación acumulado por trial, en escala log. La línea punteada marca los 8 trials de la matriz.

## Qué decir en la tesis

La matriz selecciona; el candidato optimizado confirma. Con una búsqueda 19 veces más larga, el candidato rinde mejor en validación (+0,04 a +0,07 de PR-AUC), y sus explicaciones son igual de estables y responden igual a los escenarios. El colapso en test no depende de lo bien optimizado que esté el modelo.

Como limitación va la detención temprana. El protocolo de la matriz detiene por F1 en argmax. Al optimizar a fondo hubo que detener por PR-AUC, la misma métrica de la búsqueda, y el control muestra que ese cambio por sí solo no mejora el rendimiento.
