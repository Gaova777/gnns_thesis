# Curvas ROC y PR de la corrida v4 (modo C): reunión con Cristian del 04-oct

Pedido del director: reportar PR-AUC junto con ROC-AUC y F1 junto con KS, mostrar las curvas completas y expresar la incertidumbre con las 3 semillas (media y banda), con el umbral calibrado marcado como un punto de cada curva.

## Qué se hizo

- Se reconstruyeron los 144 modelos de `results_models_v4/` (48 configuraciones por 3 semillas; no faltó ningún checkpoint) con la misma ruta del pipeline: `load_elliptic`, `apply_label_mode` (C), `preprocess`, `create_v4_scenario` con la semilla de datos 2026 y `build_model` con los `best_params` del meta.json. Una sola pasada hacia adelante por modelo, sin reentrenar.
- Verificación: el PR-AUC recalculado coincide con `val_metrics.pr_auc` y `test_metrics.pr_auc` del meta.json (diferencia máxima 9×10⁻⁹) y el F1 en el umbral calibrado coincide exactamente.
- Validación: 37.820 nodos, 914 ilícitos (prevalencia 2,42 %). Test: 29.684 nodos, 169 ilícitos (prevalencia 0,57 %). El test es el mismo para los 144 modelos.
- PR-AUC es el del pipeline (trapecio sobre la curva PR); el CSV trae además la average precision. KS = máx (TPR menos FPR) sobre todos los umbrales, que es el estadístico de Kolmogorov-Smirnov de dos muestras entre los scores de ilícitas y de negativas (el p-valor bilateral de `ks_2samp` está en el CSV por modelo).
- F1, precision, recall y MCC se evalúan en el `calibrated_threshold` del meta.json (calibrado solo con validación).
- Curvas medias: ROC interpolada en una rejilla de 501 valores de FPR; PR tomada, para cada recall de la rejilla, en el umbral más alto que lo alcanza, **sin** la envolvente de "precisión interpolada", para que los picos sigan visibles. Banda = mínimo a máximo de las 3 semillas (con 3 semillas una banda de ±sd sería poco informativa). Cada punto marcado es el umbral calibrado de una semilla.
- La PR de test se dibuja con eje de precisión logarítmico: con prevalencia 0,0057 la escala lineal aplasta todas las curvas contra el eje.

## Archivos

| Qué | Ruta |
|---|---|
| Script | `scripts/v4/reunion_0410/curvas.py` (`--stage scores` usa GPU; `metrics` y `figures` solo CPU) |
| Scores por nodo (npz, 35 MB) | `results_v4/reunion_0410/curvas/scores/{run_id}.npz` y `_labels.npz` |
| Métricas por modelo (288 filas) | `results_v4/reunion_0410/curvas/curvas_metricas_por_modelo.csv` |
| Métricas por configuración (media, sd) | `results_v4/reunion_0410/curvas/curvas_metricas_por_config.csv` |
| Dominancia y cruces entre pérdidas | `results_v4/reunion_0410/curvas/curvas_dominancia.csv` |
| Figuras por escenario (val y test) | `figuras/curvas_{native,1-10,1-10_os}_{val,test}.{png,pdf}` |
| Figura resumen (nativo) | `figuras/resumen_native_mejor_por_arquitectura.{png,pdf}` |
| Anexo 1:1 | `figuras/anexo_curvas_1-1_{val,test}.{png,pdf}` |

## Cifras clave: escenario nativo, mejor pérdida de cada arquitectura

La pérdida se eligió por PR-AUC medio de **validación** (nunca por test). Media ± sd de 3 semillas.

| Arquitectura | Pérdida | ROC-AUC val | PR-AUC val | KS val | ROC-AUC test | PR-AUC test | KS test | TPR con FPR 1 % (val / test) |
|---|---|---|---|---|---|---|---|---|
| GCN | focal loss | 0,719 ± 0,038 | 0,105 ± 0,026 | 0,325 ± 0,057 | 0,601 ± 0,132 | 0,012 ± 0,012 | 0,263 ± 0,134 | 0,107 / 0,034 |
| GraphSAGE | pesos por clase | 0,923 ± 0,003 | 0,462 ± 0,020 | 0,725 ± 0,003 | 0,711 ± 0,013 | 0,010 ± 0,000 | 0,346 ± 0,023 | 0,433 / 0,014 |
| GAT | focal loss | 0,885 ± 0,001 | 0,332 ± 0,030 | 0,610 ± 0,010 | 0,476 ± 0,085 | 0,005 ± 0,001 | 0,144 ± 0,066 | 0,313 / 0,010 |
| TAGCN | pesos por clase | 0,869 ± 0,064 | 0,299 ± 0,139 | 0,602 ± 0,107 | 0,667 ± 0,032 | 0,008 ± 0,001 | 0,344 ± 0,003 | 0,314 / 0,004 |

Azar: ROC-AUC 0,5; PR-AUC igual a la prevalencia (0,024 en validación, 0,0057 en test); TPR con FPR 1 % igual a 0,01.

## Lectura de las curvas en validación

1. **Sin balanceo queda dominada en el escenario nativo.** En las cuatro arquitecturas, la curva PR de `none` está por debajo de pesos por clase y de focal loss en todo el rango de recall (dominancia completa en 7 de los 8 pares con `none` en la PR nativa; la excepción es GCN, donde `none` y pesos por clase se cruzan a recall bajo). Es el único régimen donde la pérdida ordena las curvas sin ambigüedad.
2. **GraphSAGE con pesos por clase domina a sus alternativas en el nativo** en todo el recall (es la curva superior en el 100 % de la rejilla PR) y tiene la banda más angosta de las cuatro arquitecturas (ancho medio de la banda ROC 0,020, frente a 0,129 de GCN).
3. **Al submuestrear (1:10 y 1:10 SMOTE) la ventaja del balanceo desaparece y las curvas se cruzan.** En 1:10, GAT sin balanceo domina a sus dos variantes balanceadas en la PR (PR-AUC 0,404 frente a 0,323 y 0,304); en TAGCN, focal loss va arriba a recall bajo y las tres convergen desde recall 0,5. Sumando los tres escenarios principales, solo en el 47 % de los pares de pérdidas una curva PR de validación domina a la otra en todo el recall (tolerancia del 5 %); en el resto hay cruces. Es coherente con que el submuestreo ya corrige la razón de clases y la pérdida ponderada deja de aportar.
4. **Curvas ruidosas.** GCN es la más ruidosa: bandas anchas en ROC y PR, picos a recall bajo y ninguno de sus 36 modelos pasa la compuerta. TAGCN con pesos por clase tiene una semilla que no aprende (sd de PR-AUC 0,139 en nativo, 1:10 y 1:10 SMOTE), lo que se ve como una banda que abarca casi todo el panel. Todas las PR muestran dientes de sierra para recall menor que 0,05: ahí cada punto lo deciden unos pocos nodos y no debe leerse como diferencia entre modelos.
5. **Umbral calibrado.** En validación los puntos caen en el codo de la ROC (FPR alrededor de 0,02, TPR de 0,4 a 0,6) y, en la PR, en la zona de recall 0,4 a 0,6 con precisión 0,3 a 0,5: el umbral está bien puesto para validación.

## Lectura de las curvas en test: qué muestran sobre el colapso

1. **El ROC-AUC de test no se mantiene alto: cae a la franja 0,31 a 0,72.** La media de test por pérdida es 0,649 (pesos por clase), 0,568 (focal loss) y 0,558 (sin balanceo). **31 de los 144 modelos quedan por debajo de 0,5** (17 sin balanceo, 12 focal loss, 2 pesos por clase). No es solo pérdida de discriminación: es una **inversión del orden**. En GraphSAGE y TAGCN sin balanceo (nativo), la curva ROC va por debajo de la diagonal hasta FPR alrededor de 0,7 (ROC-AUC 0,391 y 0,312) y la mediana del score de las ilícitas de test es menor que la de las negativas (GraphSAGE: 0,005 frente a 0,016). Lo que el modelo aprendió como señal de fraude en train y validación apunta en sentido contrario en los timesteps 43 a 49.
2. **El ROC-AUC que sobrevive no sirve operativamente.** GraphSAGE con pesos por clase conserva 0,711 en test, pero la curva sube entre FPR 0,2 y 0,6; en la zona útil para AML (FPR ≤ 1 %) su TPR es 0,014, igual al azar. Lo mismo vale para el KS: en test el máximo de TPR menos FPR se alcanza con FPR entre 0,47 y 0,80 (en validación, entre 0,10 y 0,20). El KS de test es estadísticamente distinto de cero (p < 0,05 en 140 de 144 modelos, bilateral, y en los modelos invertidos lo que detecta es la inversión), pero no es un punto de operación.
3. **Las PR de test son planas y pegadas a la prevalencia.** Todas recorren el recall completo con precisión entre 0,004 y 0,013, a lo sumo dos veces el azar (0,0057), y las de los modelos invertidos van por debajo de la línea de azar. Con la escala logarítmica se ve que no hay región de recall en la que algún modelo despegue de forma estable entre semillas.
4. **El umbral calibrado se desploma al origen.** Los puntos de test caen cerca de (0, 0): FPR mediana 0,014 (el 75 % por debajo de 0,026) y, salvo en GCN, recall entre 0 y 0,09. El umbral de validación casi no marca ilícitas en test. Las excepciones son semillas sueltas de GCN (por ejemplo nativo con focal loss, semilla 42: recall 0,48 con precisión 0,028 y FPR 0,095), que marcan más nodos pero con precisión de apenas cinco veces el azar. Por eso el F1 de test queda entre 0,000 y 0,028 en las 48 configuraciones.
5. **Pesos por clase es la pérdida más robusta al cambio temporal.** Es la curva superior de la ROC de test en GraphSAGE (los tres escenarios), GAT (1:10 y 1:10 SMOTE) y TAGCN (nativo y 1:10), y la que menos veces cae bajo la diagonal. No cambia la conclusión de fondo (en test nada se separa del azar en la zona de FPR baja), pero es el patrón de forma más consistente del test.
6. **El ranking de validación no se conserva en test.** GAT con focal loss, segunda en PR-AUC de validación en el nativo, es la peor en test (ROC-AUC 0,476); GCN, la peor en validación, tiene el PR-AUC de test más alto (0,012), con una banda que va de 0,004 a 0,029 entre semillas: es suerte de semilla, no una ventaja. Esto respalda medir la estabilidad sobre validación y no sobre test.

## Anexo 1:1

Con 1:1 los pesos por clase valen lo mismo para las dos clases, así que `none` y `class_weighting` dan curvas prácticamente idénticas (ROC-AUC de validación 0,856 y 0,856 en GCN; 0,924 y 0,924 en GraphSAGE). Ninguno de los 36 modelos de 1:1 pasa la compuerta. El ROC-AUC de validación es alto (0,73 a 0,92), pero el PR-AUC es menor que en 1:10 para GraphSAGE y GAT (0,364 y 0,347 frente a 0,455 y 0,404): al ver tan pocos negativos el modelo pierde precisión a la prevalencia real. En test se repite el colapso (ROC-AUC 0,55 a 0,70; PR-AUC 0,006 a 0,014).

## Tablas por configuración (media ± sd de 3 semillas)

F1 en el umbral calibrado del meta.json. "Compuerta" = semillas que pasan la compuerta de calidad de validación (F1 ≥ 0,30 y MCC ≥ 0,15 en argmax).

### Validación (prevalencia 2,42 %)

| Escenario | Arquitectura | Pérdida | ROC-AUC | PR-AUC | F1 (umbral calibrado) | KS | Compuerta |
|---|---|---|---|---|---|---|---|
| nativo | GCN | sin balanceo | 0,677 ± 0,097 | 0,083 ± 0,046 | 0,154 ± 0,082 | 0,262 ± 0,122 | 0/3 |
| nativo | GCN | pesos por clase | 0,720 ± 0,024 | 0,097 ± 0,012 | 0,180 ± 0,015 | 0,317 ± 0,021 | 0/3 |
| nativo | GCN | focal loss | 0,719 ± 0,038 | 0,105 ± 0,026 | 0,189 ± 0,033 | 0,325 ± 0,057 | 0/3 |
| nativo | GraphSAGE | sin balanceo | 0,843 ± 0,034 | 0,177 ± 0,060 | 0,279 ± 0,072 | 0,540 ± 0,062 | 0/3 |
| nativo | GraphSAGE | pesos por clase | 0,923 ± 0,003 | 0,462 ± 0,020 | 0,492 ± 0,022 | 0,725 ± 0,003 | 3/3 |
| nativo | GraphSAGE | focal loss | 0,885 ± 0,004 | 0,305 ± 0,041 | 0,407 ± 0,036 | 0,613 ± 0,018 | 3/3 |
| nativo | GAT | sin balanceo | 0,807 ± 0,108 | 0,202 ± 0,095 | 0,277 ± 0,093 | 0,462 ± 0,162 | 2/3 |
| nativo | GAT | pesos por clase | 0,900 ± 0,015 | 0,320 ± 0,040 | 0,401 ± 0,044 | 0,660 ± 0,042 | 0/3 |
| nativo | GAT | focal loss | 0,885 ± 0,001 | 0,332 ± 0,030 | 0,394 ± 0,010 | 0,610 ± 0,010 | 3/3 |
| nativo | TAGCN | sin balanceo | 0,763 ± 0,031 | 0,097 ± 0,050 | 0,168 ± 0,059 | 0,406 ± 0,014 | 0/3 |
| nativo | TAGCN | pesos por clase | 0,869 ± 0,064 | 0,299 ± 0,139 | 0,398 ± 0,123 | 0,602 ± 0,107 | 2/3 |
| nativo | TAGCN | focal loss | 0,865 ± 0,037 | 0,290 ± 0,092 | 0,379 ± 0,053 | 0,580 ± 0,069 | 2/3 |
| 1:10 | GCN | sin balanceo | 0,757 ± 0,039 | 0,121 ± 0,034 | 0,214 ± 0,046 | 0,382 ± 0,040 | 0/3 |
| 1:10 | GCN | pesos por clase | 0,790 ± 0,116 | 0,162 ± 0,104 | 0,224 ± 0,112 | 0,469 ± 0,194 | 0/3 |
| 1:10 | GCN | focal loss | 0,751 ± 0,092 | 0,146 ± 0,055 | 0,227 ± 0,061 | 0,390 ± 0,166 | 0/3 |
| 1:10 | GraphSAGE | sin balanceo | 0,897 ± 0,010 | 0,373 ± 0,064 | 0,447 ± 0,031 | 0,668 ± 0,023 | 3/3 |
| 1:10 | GraphSAGE | pesos por clase | 0,926 ± 0,004 | 0,455 ± 0,008 | 0,500 ± 0,013 | 0,735 ± 0,004 | 3/3 |
| 1:10 | GraphSAGE | focal loss | 0,908 ± 0,019 | 0,426 ± 0,083 | 0,477 ± 0,038 | 0,686 ± 0,058 | 3/3 |
| 1:10 | GAT | sin balanceo | 0,908 ± 0,017 | 0,404 ± 0,026 | 0,459 ± 0,027 | 0,681 ± 0,048 | 3/3 |
| 1:10 | GAT | pesos por clase | 0,900 ± 0,015 | 0,323 ± 0,041 | 0,403 ± 0,044 | 0,660 ± 0,044 | 0/3 |
| 1:10 | GAT | focal loss | 0,883 ± 0,014 | 0,304 ± 0,025 | 0,394 ± 0,022 | 0,608 ± 0,039 | 0/3 |
| 1:10 | TAGCN | sin balanceo | 0,886 ± 0,016 | 0,320 ± 0,054 | 0,420 ± 0,017 | 0,621 ± 0,038 | 3/3 |
| 1:10 | TAGCN | pesos por clase | 0,874 ± 0,068 | 0,294 ± 0,135 | 0,399 ± 0,126 | 0,620 ± 0,119 | 2/3 |
| 1:10 | TAGCN | focal loss | 0,879 ± 0,018 | 0,408 ± 0,021 | 0,484 ± 0,009 | 0,658 ± 0,030 | 3/3 |
| 1:10 SMOTE | GCN | sin balanceo | 0,758 ± 0,038 | 0,124 ± 0,037 | 0,216 ± 0,046 | 0,385 ± 0,039 | 0/3 |
| 1:10 SMOTE | GCN | pesos por clase | 0,791 ± 0,116 | 0,162 ± 0,104 | 0,223 ± 0,113 | 0,469 ± 0,196 | 0/3 |
| 1:10 SMOTE | GCN | focal loss | 0,704 ± 0,008 | 0,108 ± 0,011 | 0,184 ± 0,020 | 0,298 ± 0,043 | 0/3 |
| 1:10 SMOTE | GraphSAGE | sin balanceo | 0,891 ± 0,006 | 0,347 ± 0,037 | 0,434 ± 0,022 | 0,655 ± 0,019 | 3/3 |
| 1:10 SMOTE | GraphSAGE | pesos por clase | 0,918 ± 0,005 | 0,438 ± 0,015 | 0,466 ± 0,018 | 0,710 ± 0,014 | 3/3 |
| 1:10 SMOTE | GraphSAGE | focal loss | 0,911 ± 0,006 | 0,442 ± 0,019 | 0,476 ± 0,015 | 0,694 ± 0,015 | 3/3 |
| 1:10 SMOTE | GAT | sin balanceo | 0,900 ± 0,013 | 0,385 ± 0,034 | 0,439 ± 0,017 | 0,659 ± 0,032 | 3/3 |
| 1:10 SMOTE | GAT | pesos por clase | 0,909 ± 0,010 | 0,345 ± 0,026 | 0,424 ± 0,036 | 0,685 ± 0,041 | 0/3 |
| 1:10 SMOTE | GAT | focal loss | 0,866 ± 0,042 | 0,294 ± 0,102 | 0,360 ± 0,100 | 0,568 ± 0,103 | 2/3 |
| 1:10 SMOTE | TAGCN | sin balanceo | 0,894 ± 0,006 | 0,397 ± 0,031 | 0,459 ± 0,028 | 0,626 ± 0,012 | 3/3 |
| 1:10 SMOTE | TAGCN | pesos por clase | 0,870 ± 0,067 | 0,292 ± 0,135 | 0,390 ± 0,122 | 0,603 ± 0,113 | 2/3 |
| 1:10 SMOTE | TAGCN | focal loss | 0,884 ± 0,006 | 0,402 ± 0,028 | 0,465 ± 0,019 | 0,656 ± 0,013 | 3/3 |
| 1:1 (anexo) | GCN | sin balanceo | 0,856 ± 0,003 | 0,185 ± 0,013 | 0,265 ± 0,015 | 0,587 ± 0,011 | 0/3 |
| 1:1 (anexo) | GCN | pesos por clase | 0,856 ± 0,004 | 0,185 ± 0,013 | 0,266 ± 0,016 | 0,587 ± 0,011 | 0/3 |
| 1:1 (anexo) | GCN | focal loss | 0,726 ± 0,077 | 0,079 ± 0,029 | 0,149 ± 0,041 | 0,333 ± 0,113 | 0/3 |
| 1:1 (anexo) | GraphSAGE | sin balanceo | 0,924 ± 0,003 | 0,364 ± 0,015 | 0,438 ± 0,015 | 0,743 ± 0,011 | 0/3 |
| 1:1 (anexo) | GraphSAGE | pesos por clase | 0,924 ± 0,003 | 0,364 ± 0,015 | 0,438 ± 0,016 | 0,742 ± 0,011 | 0/3 |
| 1:1 (anexo) | GraphSAGE | focal loss | 0,877 ± 0,013 | 0,273 ± 0,080 | 0,353 ± 0,073 | 0,621 ± 0,037 | 0/3 |
| 1:1 (anexo) | GAT | sin balanceo | 0,905 ± 0,017 | 0,347 ± 0,012 | 0,414 ± 0,018 | 0,671 ± 0,057 | 0/3 |
| 1:1 (anexo) | GAT | pesos por clase | 0,905 ± 0,017 | 0,347 ± 0,012 | 0,414 ± 0,018 | 0,671 ± 0,057 | 0/3 |
| 1:1 (anexo) | GAT | focal loss | 0,860 ± 0,008 | 0,204 ± 0,064 | 0,280 ± 0,059 | 0,563 ± 0,014 | 0/3 |
| 1:1 (anexo) | TAGCN | sin balanceo | 0,871 ± 0,065 | 0,200 ± 0,051 | 0,316 ± 0,045 | 0,622 ± 0,121 | 0/3 |
| 1:1 (anexo) | TAGCN | pesos por clase | 0,870 ± 0,064 | 0,205 ± 0,057 | 0,318 ± 0,046 | 0,617 ± 0,117 | 0/3 |
| 1:1 (anexo) | TAGCN | focal loss | 0,864 ± 0,046 | 0,229 ± 0,074 | 0,338 ± 0,077 | 0,606 ± 0,100 | 0/3 |

### Test (prevalencia 0,57 %)

| Escenario | Arquitectura | Pérdida | ROC-AUC | PR-AUC | F1 (umbral calibrado) | KS | Compuerta |
|---|---|---|---|---|---|---|---|
| nativo | GCN | sin balanceo | 0,561 ± 0,106 | 0,008 ± 0,005 | 0,016 ± 0,025 | 0,227 ± 0,059 | 0/3 |
| nativo | GCN | pesos por clase | 0,635 ± 0,067 | 0,008 ± 0,002 | 0,014 ± 0,009 | 0,279 ± 0,042 | 0/3 |
| nativo | GCN | focal loss | 0,601 ± 0,132 | 0,012 ± 0,012 | 0,019 ± 0,029 | 0,263 ± 0,134 | 0/3 |
| nativo | GraphSAGE | sin balanceo | 0,391 ± 0,023 | 0,004 ± 0,000 | 0,001 ± 0,002 | 0,102 ± 0,026 | 0/3 |
| nativo | GraphSAGE | pesos por clase | 0,711 ± 0,013 | 0,010 ± 0,000 | 0,012 ± 0,005 | 0,346 ± 0,023 | 3/3 |
| nativo | GraphSAGE | focal loss | 0,551 ± 0,029 | 0,006 ± 0,000 | 0,002 ± 0,003 | 0,201 ± 0,011 | 3/3 |
| nativo | GAT | sin balanceo | 0,529 ± 0,044 | 0,005 ± 0,000 | 0,001 ± 0,001 | 0,209 ± 0,059 | 2/3 |
| nativo | GAT | pesos por clase | 0,611 ± 0,044 | 0,008 ± 0,001 | 0,022 ± 0,019 | 0,221 ± 0,083 | 0/3 |
| nativo | GAT | focal loss | 0,476 ± 0,085 | 0,005 ± 0,001 | 0,009 ± 0,009 | 0,144 ± 0,066 | 3/3 |
| nativo | TAGCN | sin balanceo | 0,312 ± 0,041 | 0,004 ± 0,000 | 0,000 ± 0,000 | 0,055 ± 0,012 | 0/3 |
| nativo | TAGCN | pesos por clase | 0,667 ± 0,032 | 0,008 ± 0,001 | 0,006 ± 0,001 | 0,344 ± 0,003 | 2/3 |
| nativo | TAGCN | focal loss | 0,624 ± 0,079 | 0,008 ± 0,002 | 0,008 ± 0,006 | 0,262 ± 0,051 | 2/3 |
| 1:10 | GCN | sin balanceo | 0,581 ± 0,101 | 0,013 ± 0,013 | 0,025 ± 0,029 | 0,247 ± 0,091 | 0/3 |
| 1:10 | GCN | pesos por clase | 0,526 ± 0,034 | 0,006 ± 0,000 | 0,007 ± 0,003 | 0,106 ± 0,038 | 0/3 |
| 1:10 | GCN | focal loss | 0,577 ± 0,014 | 0,008 ± 0,002 | 0,017 ± 0,024 | 0,182 ± 0,082 | 0/3 |
| 1:10 | GraphSAGE | sin balanceo | 0,501 ± 0,041 | 0,006 ± 0,001 | 0,002 ± 0,003 | 0,091 ± 0,034 | 3/3 |
| 1:10 | GraphSAGE | pesos por clase | 0,713 ± 0,019 | 0,011 ± 0,001 | 0,010 ± 0,009 | 0,349 ± 0,033 | 3/3 |
| 1:10 | GraphSAGE | focal loss | 0,666 ± 0,015 | 0,010 ± 0,001 | 0,015 ± 0,005 | 0,276 ± 0,030 | 3/3 |
| 1:10 | GAT | sin balanceo | 0,557 ± 0,093 | 0,006 ± 0,001 | 0,004 ± 0,001 | 0,177 ± 0,060 | 3/3 |
| 1:10 | GAT | pesos por clase | 0,609 ± 0,044 | 0,008 ± 0,000 | 0,023 ± 0,021 | 0,210 ± 0,075 | 0/3 |
| 1:10 | GAT | focal loss | 0,482 ± 0,036 | 0,005 ± 0,000 | 0,005 ± 0,006 | 0,113 ± 0,020 | 0/3 |
| 1:10 | TAGCN | sin balanceo | 0,608 ± 0,081 | 0,008 ± 0,002 | 0,008 ± 0,001 | 0,205 ± 0,102 | 3/3 |
| 1:10 | TAGCN | pesos por clase | 0,651 ± 0,038 | 0,008 ± 0,001 | 0,006 ± 0,002 | 0,322 ± 0,025 | 2/3 |
| 1:10 | TAGCN | focal loss | 0,411 ± 0,060 | 0,004 ± 0,000 | 0,000 ± 0,000 | 0,070 ± 0,026 | 3/3 |
| 1:10 SMOTE | GCN | sin balanceo | 0,590 ± 0,085 | 0,013 ± 0,012 | 0,023 ± 0,030 | 0,248 ± 0,083 | 0/3 |
| 1:10 SMOTE | GCN | pesos por clase | 0,577 ± 0,099 | 0,007 ± 0,003 | 0,007 ± 0,003 | 0,197 ± 0,147 | 0/3 |
| 1:10 SMOTE | GCN | focal loss | 0,587 ± 0,019 | 0,008 ± 0,003 | 0,018 ± 0,021 | 0,205 ± 0,064 | 0/3 |
| 1:10 SMOTE | GraphSAGE | sin balanceo | 0,459 ± 0,016 | 0,005 ± 0,000 | 0,000 ± 0,000 | 0,058 ± 0,027 | 3/3 |
| 1:10 SMOTE | GraphSAGE | pesos por clase | 0,715 ± 0,017 | 0,011 ± 0,001 | 0,013 ± 0,009 | 0,353 ± 0,018 | 3/3 |
| 1:10 SMOTE | GraphSAGE | focal loss | 0,676 ± 0,012 | 0,010 ± 0,000 | 0,011 ± 0,007 | 0,302 ± 0,031 | 3/3 |
| 1:10 SMOTE | GAT | sin balanceo | 0,517 ± 0,090 | 0,006 ± 0,001 | 0,003 ± 0,002 | 0,144 ± 0,058 | 3/3 |
| 1:10 SMOTE | GAT | pesos por clase | 0,647 ± 0,046 | 0,009 ± 0,001 | 0,014 ± 0,021 | 0,261 ± 0,069 | 0/3 |
| 1:10 SMOTE | GAT | focal loss | 0,554 ± 0,092 | 0,006 ± 0,002 | 0,003 ± 0,005 | 0,203 ± 0,102 | 2/3 |
| 1:10 SMOTE | TAGCN | sin balanceo | 0,683 ± 0,014 | 0,010 ± 0,001 | 0,002 ± 0,004 | 0,314 ± 0,020 | 3/3 |
| 1:10 SMOTE | TAGCN | pesos por clase | 0,671 ± 0,040 | 0,008 ± 0,001 | 0,007 ± 0,002 | 0,347 ± 0,019 | 2/3 |
| 1:10 SMOTE | TAGCN | focal loss | 0,452 ± 0,108 | 0,005 ± 0,001 | 0,000 ± 0,000 | 0,087 ± 0,058 | 3/3 |
| 1:1 (anexo) | GCN | sin balanceo | 0,652 ± 0,050 | 0,010 ± 0,002 | 0,013 ± 0,004 | 0,253 ± 0,098 | 0/3 |
| 1:1 (anexo) | GCN | pesos por clase | 0,651 ± 0,049 | 0,010 ± 0,002 | 0,013 ± 0,005 | 0,254 ± 0,098 | 0/3 |
| 1:1 (anexo) | GCN | focal loss | 0,676 ± 0,063 | 0,014 ± 0,008 | 0,028 ± 0,013 | 0,277 ± 0,083 | 0/3 |
| 1:1 (anexo) | GraphSAGE | sin balanceo | 0,698 ± 0,029 | 0,010 ± 0,001 | 0,012 ± 0,006 | 0,345 ± 0,049 | 0/3 |
| 1:1 (anexo) | GraphSAGE | pesos por clase | 0,698 ± 0,028 | 0,010 ± 0,001 | 0,012 ± 0,006 | 0,345 ± 0,049 | 0/3 |
| 1:1 (anexo) | GraphSAGE | focal loss | 0,548 ± 0,070 | 0,006 ± 0,001 | 0,000 ± 0,000 | 0,210 ± 0,066 | 0/3 |
| 1:1 (anexo) | GAT | sin balanceo | 0,614 ± 0,087 | 0,009 ± 0,002 | 0,018 ± 0,005 | 0,235 ± 0,101 | 0/3 |
| 1:1 (anexo) | GAT | pesos por clase | 0,614 ± 0,087 | 0,009 ± 0,002 | 0,018 ± 0,005 | 0,235 ± 0,101 | 0/3 |
| 1:1 (anexo) | GAT | focal loss | 0,575 ± 0,054 | 0,007 ± 0,001 | 0,009 ± 0,005 | 0,184 ± 0,057 | 0/3 |
| 1:1 (anexo) | TAGCN | sin balanceo | 0,684 ± 0,045 | 0,009 ± 0,002 | 0,009 ± 0,003 | 0,346 ± 0,009 | 0/3 |
| 1:1 (anexo) | TAGCN | pesos por clase | 0,693 ± 0,051 | 0,009 ± 0,002 | 0,009 ± 0,003 | 0,364 ± 0,023 | 0/3 |
| 1:1 (anexo) | TAGCN | focal loss | 0,635 ± 0,053 | 0,007 ± 0,001 | 0,003 ± 0,004 | 0,271 ± 0,068 | 0/3 |
