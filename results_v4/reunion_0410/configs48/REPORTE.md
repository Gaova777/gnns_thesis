# Configuraciones como unidad de análisis (reunión con Cristian, 04-oct-2026)

Eje Elliptic v4, modo de etiqueta C (negativos = lícitas + sin etiqueta). Script: `scripts/v4/reunion_0410/configs48.py`. Salidas: este directorio. Solo CPU; no se modificó ningún archivo de `results_v4/`, `src/`, `phase1/` ni ningún script existente.

## Resumen

1. **Compuerta por configuración.** Con la media de las 3 semillas (val F1 ≥ 0,30 y val MCC ≥ 0,15) pasan **20 de 48 configuraciones** (GraphSAGE 8, TAGCN 8, GAT 4, GCN 0; native 5, 1:10 7, 1:10_os 8, 1:1 0). El análisis v4 vigente usaba, sin decirlo así, el criterio «alguna semilla pasa» (21 configuraciones, promediando solo las semillas que pasan). Las variantes conservadoras (media - sd y límite inferior del IC 95 % t) dejan **15**, las mismas que pasan con las 3 semillas. Hay **6 configuraciones** con desacuerdo entre semillas (todas con 2 de 3).
2. **Hipótesis (unidad = configuración, compuerta de configuración, sin 1:1, n = 20).** H1 y H2 **no cambian**: equivalencia para GNNExplainer y ShapleyFeatures, sin detección para PGExplainer. **H3 de GNNExplainer cambia**: de «efecto significativo» (Friedman p = 0,0498) a «no se detectó con esta potencia» (p = 0,174; exacto 0,273). Además, el p = 0,0498 vigente es la aproximación χ² con 4 bloques; **el p exacto por permutación de esos mismos datos es 0,069**. Todas las pruebas ómnibus de la versión principal tienen 4 bloques completos: la potencia es mínima.
3. **Candidato para el Optuna completo:** **GraphSAGE + class_weighting** (1.º, 1.º y 2.º en val PR-AUC en native, 1:10 y 1:10_os; 0,452 [0,439; 0,464] con los 9 modelos; 9 de 9 pasan la compuerta). Suplente: **GraphSAGE + focal_loss** (0,391 [0,329; 0,453]; 9 de 9). Si se prefiere un suplente de otra arquitectura: TAGCN + focal_loss (0,367 [0,309; 0,425]).
4. **Costo:** 100 trials × 3 escenarios + 3 semillas de entrenamiento final del candidato cuestan unas **0,42 h de GPU** (cota superior 0,47 h) en la RTX 4060, con el presupuesto de épocas de la v4. La explicación de los 9 modelos agrega unos 17 min.

## 1. Métricas de las 48 configuraciones

### 1.1 Qué métrica usa la compuerta actual

| Bloque del meta.json | Decisión | Uso actual |
|---|---|---|
| `val_metrics` | argmax (umbral 0,5) en la mejor época | **Compuerta** (`quality_passed`) y columnas `val_*` de `elliptic_v4_stability.csv` |
| `val_metrics_calibrated` | umbral que maximiza F1 en validación | Solo informativo |
| `test_metrics` | umbral calibrado en validación | Columnas `test_*` del análisis v4 |
| `test_metrics_argmax` | argmax | Solo informativo |

En `scripts/train_matrix.py` (`run_one_config`): `quality_passed = val_argmax["f1"] >= f1_min and val_argmax["mcc"] >= mcc_min`, con 0,30 y 0,15 de `configs/experiment_v4.yaml` (`analysis.quality_gate`). El script verifica que recalcular la compuerta desde `val_metrics` reproduce `quality_passed` en los 144 modelos. **Sí existe `val_metrics_calibrated`**, pero la compuerta no lo usa; se reporta solo como sensibilidad, porque el umbral se elige con los mismos datos de validación y es optimista.

**ROC-AUC** no está en los meta.json. Se recalculó en CPU desde los 144 checkpoints (inferencia de grafo completo, mismo preprocesamiento que `train_matrix.py`; `roc_auc_cpu.csv`). Control: el PR-AUC recalculado coincide con el del meta.json con diferencia máxima de 1,9 × 10⁻⁶ (validación) y 4,0 × 10⁻⁸ (test).

### 1.2 Tabla

Media ± sd sobre las 3 semillas (n = 3); para val PR-AUC, IC 95 % t (t₀,₉₇₅;₂ = 4,30). Los IC de todas las métricas están en `configs48_metricas.csv`. F1 y MCC de validación: argmax (los de la compuerta); de test: umbral calibrado en validación. «Pasan»: semillas que pasan la compuerta por modelo. «Comp. media»: compuerta de configuración; «Comp. cons.»: media - sd. CW = class_weighting; focal = focal_loss. Prevalencia ilícita: 0,024 en validación y 0,0057 en test (PR-AUC del azar).

| Escenario | Arq. | Pérdida | Pasan | val PR-AUC [IC 95 %] | val F1 | val MCC | val ROC-AUC | test PR-AUC | test F1 | test MCC | test ROC-AUC | Comp. media | Comp. cons. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| native | GCN | none | 0/3 | 0,083 [-0,032; 0,198] | 0,134 ± 0,107 | 0,121 ± 0,096 | 0,677 ± 0,097 | 0,008 ± 0,005 | 0,016 ± 0,025 | 0,009 ± 0,038 | 0,561 ± 0,106 | no | no |
| native | GCN | CW | 0/3 | 0,097 [0,068; 0,126] | 0,108 ± 0,018 | 0,116 ± 0,009 | 0,720 ± 0,024 | 0,008 ± 0,002 | 0,014 ± 0,009 | 0,006 ± 0,015 | 0,635 ± 0,067 | no | no |
| native | GCN | focal | 0/3 | 0,105 [0,040; 0,171] | 0,177 ± 0,033 | 0,164 ± 0,031 | 0,719 ± 0,038 | 0,012 ± 0,012 | 0,019 ± 0,029 | 0,027 ± 0,061 | 0,601 ± 0,132 | no | no |
| native | GraphSAGE | none | 0/3 | 0,177 [0,029; 0,326] | 0,210 ± 0,096 | 0,207 ± 0,085 | 0,843 ± 0,034 | 0,004 ± 0,000 | 0,001 ± 0,002 | -0,013 ± 0,006 | 0,391 ± 0,023 | no | no |
| native | GraphSAGE | CW | 3/3 | 0,462 [0,411; 0,512] | 0,406 ± 0,011 | 0,432 ± 0,008 | 0,923 ± 0,003 | 0,010 ± 0,000 | 0,012 ± 0,005 | 0,006 ± 0,005 | 0,711 ± 0,013 | sí | sí |
| native | GraphSAGE | focal | 3/3 | 0,305 [0,203; 0,407] | 0,401 ± 0,032 | 0,387 ± 0,034 | 0,885 ± 0,004 | 0,006 ± 0,000 | 0,002 ± 0,003 | -0,008 ± 0,003 | 0,551 ± 0,029 | sí | sí |
| native | GAT | none | 2/3 | 0,202 [-0,033; 0,438] | 0,273 ± 0,093 | 0,257 ± 0,096 | 0,807 ± 0,108 | 0,005 ± 0,000 | 0,001 ± 0,001 | -0,010 ± 0,001 | 0,529 ± 0,044 | no | no |
| native | GAT | CW | 0/3 | 0,320 [0,220; 0,421] | 0,201 ± 0,052 | 0,266 ± 0,051 | 0,900 ± 0,015 | 0,008 ± 0,001 | 0,022 ± 0,019 | 0,017 ± 0,022 | 0,611 ± 0,044 | no | no |
| native | GAT | focal | 3/3 | 0,332 [0,258; 0,406] | 0,391 ± 0,014 | 0,380 ± 0,009 | 0,885 ± 0,001 | 0,005 ± 0,001 | 0,009 ± 0,009 | 0,001 ± 0,010 | 0,476 ± 0,085 | sí | sí |
| native | TAGCN | none | 0/3 | 0,097 [-0,027; 0,220] | 0,003 ± 0,001 | 0,011 ± 0,023 | 0,763 ± 0,031 | 0,004 ± 0,000 | 0,000 ± 0,000 | -0,020 ± 0,010 | 0,312 ± 0,041 | no | no |
| native | TAGCN | CW | 2/3 | 0,299 [-0,046; 0,645] | 0,360 ± 0,154 | 0,368 ± 0,132 | 0,869 ± 0,064 | 0,008 ± 0,001 | 0,006 ± 0,001 | -0,001 ± 0,002 | 0,667 ± 0,032 | sí | no |
| native | TAGCN | focal | 2/3 | 0,290 [0,063; 0,518] | 0,357 ± 0,064 | 0,345 ± 0,069 | 0,865 ± 0,037 | 0,008 ± 0,002 | 0,008 ± 0,006 | -0,001 ± 0,009 | 0,624 ± 0,079 | sí | no |
| 1:10 | GCN | none | 0/3 | 0,121 [0,036; 0,207] | 0,209 ± 0,050 | 0,199 ± 0,040 | 0,757 ± 0,039 | 0,013 ± 0,013 | 0,025 ± 0,029 | 0,031 ± 0,055 | 0,581 ± 0,101 | no | no |
| 1:10 | GCN | CW | 0/3 | 0,162 [-0,096; 0,419] | 0,159 ± 0,067 | 0,191 ± 0,094 | 0,790 ± 0,116 | 0,006 ± 0,000 | 0,007 ± 0,003 | -0,003 ± 0,003 | 0,526 ± 0,034 | no | no |
| 1:10 | GCN | focal | 0/3 | 0,146 [0,010; 0,281] | 0,227 ± 0,062 | 0,210 ± 0,067 | 0,751 ± 0,092 | 0,008 ± 0,002 | 0,017 ± 0,024 | 0,013 ± 0,035 | 0,577 ± 0,014 | no | no |
| 1:10 | GraphSAGE | none | 3/3 | 0,373 [0,214; 0,533] | 0,443 ± 0,031 | 0,431 ± 0,032 | 0,897 ± 0,010 | 0,006 ± 0,001 | 0,002 ± 0,003 | -0,005 ± 0,003 | 0,501 ± 0,041 | sí | sí |
| 1:10 | GraphSAGE | CW | 3/3 | 0,455 [0,436; 0,474] | 0,399 ± 0,015 | 0,430 ± 0,011 | 0,926 ± 0,004 | 0,011 ± 0,001 | 0,010 ± 0,009 | 0,004 ± 0,009 | 0,713 ± 0,019 | sí | sí |
| 1:10 | GraphSAGE | focal | 3/3 | 0,426 [0,220; 0,632] | 0,448 ± 0,040 | 0,449 ± 0,038 | 0,908 ± 0,019 | 0,010 ± 0,001 | 0,015 ± 0,005 | 0,010 ± 0,006 | 0,666 ± 0,015 | sí | sí |
| 1:10 | GAT | none | 3/3 | 0,404 [0,340; 0,468] | 0,457 ± 0,030 | 0,446 ± 0,032 | 0,908 ± 0,017 | 0,006 ± 0,001 | 0,004 ± 0,001 | -0,004 ± 0,002 | 0,557 ± 0,093 | sí | sí |
| 1:10 | GAT | CW | 0/3 | 0,323 [0,221; 0,425] | 0,197 ± 0,052 | 0,262 ± 0,052 | 0,900 ± 0,015 | 0,008 ± 0,000 | 0,023 ± 0,021 | 0,019 ± 0,023 | 0,609 ± 0,044 | no | no |
| 1:10 | GAT | focal | 0/3 | 0,304 [0,242; 0,366] | 0,259 ± 0,014 | 0,298 ± 0,016 | 0,883 ± 0,014 | 0,005 ± 0,000 | 0,005 ± 0,006 | -0,004 ± 0,007 | 0,482 ± 0,036 | no | no |
| 1:10 | TAGCN | none | 3/3 | 0,320 [0,187; 0,453] | 0,413 ± 0,021 | 0,400 ± 0,020 | 0,886 ± 0,016 | 0,008 ± 0,002 | 0,008 ± 0,001 | 0,001 ± 0,001 | 0,608 ± 0,081 | sí | sí |
| 1:10 | TAGCN | CW | 2/3 | 0,294 [-0,042; 0,629] | 0,350 ± 0,147 | 0,365 ± 0,131 | 0,874 ± 0,068 | 0,008 ± 0,001 | 0,006 ± 0,002 | -0,002 ± 0,002 | 0,651 ± 0,038 | sí | no |
| 1:10 | TAGCN | focal | 3/3 | 0,408 [0,355; 0,460] | 0,435 ± 0,015 | 0,438 ± 0,007 | 0,879 ± 0,018 | 0,004 ± 0,000 | 0,000 ± 0,000 | -0,006 ± 0,001 | 0,411 ± 0,060 | sí | sí |
| 1:10_os | GCN | none | 0/3 | 0,124 [0,032; 0,215] | 0,212 ± 0,050 | 0,203 ± 0,042 | 0,758 ± 0,038 | 0,013 ± 0,012 | 0,023 ± 0,030 | 0,028 ± 0,056 | 0,590 ± 0,085 | no | no |
| 1:10_os | GCN | CW | 0/3 | 0,162 [-0,096; 0,419] | 0,166 ± 0,073 | 0,195 ± 0,097 | 0,791 ± 0,116 | 0,007 ± 0,003 | 0,007 ± 0,003 | -0,002 ± 0,000 | 0,577 ± 0,099 | no | no |
| 1:10_os | GCN | focal | 0/3 | 0,108 [0,080; 0,136] | 0,181 ± 0,024 | 0,163 ± 0,028 | 0,704 ± 0,008 | 0,008 ± 0,003 | 0,018 ± 0,021 | 0,014 ± 0,034 | 0,587 ± 0,019 | no | no |
| 1:10_os | GraphSAGE | none | 3/3 | 0,347 [0,256; 0,439] | 0,433 ± 0,022 | 0,421 ± 0,025 | 0,891 ± 0,006 | 0,005 ± 0,000 | 0,000 ± 0,000 | -0,007 ± 0,001 | 0,459 ± 0,016 | sí | sí |
| 1:10_os | GraphSAGE | CW | 3/3 | 0,438 [0,400; 0,476] | 0,418 ± 0,021 | 0,433 ± 0,019 | 0,918 ± 0,005 | 0,011 ± 0,001 | 0,013 ± 0,009 | 0,008 ± 0,011 | 0,715 ± 0,017 | sí | sí |
| 1:10_os | GraphSAGE | focal | 3/3 | 0,442 [0,395; 0,489] | 0,463 ± 0,015 | 0,456 ± 0,016 | 0,911 ± 0,006 | 0,010 ± 0,000 | 0,011 ± 0,007 | 0,007 ± 0,006 | 0,676 ± 0,012 | sí | sí |
| 1:10_os | GAT | none | 3/3 | 0,385 [0,302; 0,469] | 0,433 ± 0,020 | 0,421 ± 0,020 | 0,900 ± 0,013 | 0,006 ± 0,001 | 0,003 ± 0,002 | -0,005 ± 0,002 | 0,517 ± 0,090 | sí | sí |
| 1:10_os | GAT | CW | 0/3 | 0,345 [0,280; 0,410] | 0,230 ± 0,043 | 0,295 ± 0,043 | 0,909 ± 0,010 | 0,009 ± 0,001 | 0,014 ± 0,021 | 0,008 ± 0,024 | 0,647 ± 0,046 | no | no |
| 1:10_os | GAT | focal | 2/3 | 0,294 [0,040; 0,549] | 0,337 ± 0,067 | 0,331 ± 0,082 | 0,866 ± 0,042 | 0,006 ± 0,002 | 0,003 ± 0,005 | -0,005 ± 0,005 | 0,554 ± 0,092 | sí | no |
| 1:10_os | TAGCN | none | 3/3 | 0,397 [0,320; 0,474] | 0,455 ± 0,027 | 0,445 ± 0,027 | 0,894 ± 0,006 | 0,010 ± 0,001 | 0,002 ± 0,004 | -0,002 ± 0,003 | 0,683 ± 0,014 | sí | sí |
| 1:10_os | TAGCN | CW | 2/3 | 0,292 [-0,043; 0,627] | 0,358 ± 0,148 | 0,363 ± 0,128 | 0,870 ± 0,067 | 0,008 ± 0,001 | 0,007 ± 0,002 | -0,001 ± 0,003 | 0,671 ± 0,040 | sí | no |
| 1:10_os | TAGCN | focal | 3/3 | 0,402 [0,332; 0,473] | 0,446 ± 0,006 | 0,442 ± 0,007 | 0,884 ± 0,006 | 0,005 ± 0,001 | 0,000 ± 0,000 | -0,006 ± 0,001 | 0,452 ± 0,108 | sí | sí |
| 1:1 | GCN | none | 0/3 | 0,185 [0,152; 0,219] | 0,171 ± 0,013 | 0,227 ± 0,011 | 0,856 ± 0,003 | 0,010 ± 0,002 | 0,013 ± 0,004 | 0,005 ± 0,006 | 0,652 ± 0,050 | no | no |
| 1:1 | GCN | CW | 0/3 | 0,185 [0,152; 0,218] | 0,172 ± 0,014 | 0,228 ± 0,012 | 0,856 ± 0,004 | 0,010 ± 0,002 | 0,013 ± 0,005 | 0,005 ± 0,007 | 0,651 ± 0,049 | no | no |
| 1:1 | GCN | focal | 0/3 | 0,079 [0,007; 0,150] | 0,086 ± 0,012 | 0,096 ± 0,017 | 0,726 ± 0,077 | 0,014 ± 0,008 | 0,028 ± 0,013 | 0,033 ± 0,024 | 0,676 ± 0,063 | no | no |
| 1:1 | GraphSAGE | none | 0/3 | 0,364 [0,326; 0,401] | 0,276 ± 0,001 | 0,344 ± 0,003 | 0,924 ± 0,003 | 0,010 ± 0,001 | 0,012 ± 0,006 | 0,005 ± 0,006 | 0,698 ± 0,029 | no | no |
| 1:1 | GraphSAGE | CW | 0/3 | 0,364 [0,326; 0,402] | 0,275 ± 0,001 | 0,344 ± 0,002 | 0,924 ± 0,003 | 0,010 ± 0,001 | 0,012 ± 0,006 | 0,005 ± 0,006 | 0,698 ± 0,028 | no | no |
| 1:1 | GraphSAGE | focal | 0/3 | 0,273 [0,075; 0,470] | 0,095 ± 0,016 | 0,147 ± 0,025 | 0,877 ± 0,013 | 0,006 ± 0,001 | 0,000 ± 0,000 | -0,012 ± 0,004 | 0,548 ± 0,070 | no | no |
| 1:1 | GAT | none | 0/3 | 0,347 [0,317; 0,376] | 0,187 ± 0,035 | 0,256 ± 0,039 | 0,905 ± 0,017 | 0,009 ± 0,002 | 0,018 ± 0,005 | 0,011 ± 0,006 | 0,614 ± 0,087 | no | no |
| 1:1 | GAT | CW | 0/3 | 0,347 [0,317; 0,376] | 0,187 ± 0,035 | 0,256 ± 0,039 | 0,905 ± 0,017 | 0,009 ± 0,002 | 0,018 ± 0,005 | 0,011 ± 0,006 | 0,614 ± 0,087 | no | no |
| 1:1 | GAT | focal | 0/3 | 0,204 [0,044; 0,364] | 0,169 ± 0,049 | 0,217 ± 0,031 | 0,860 ± 0,008 | 0,007 ± 0,001 | 0,009 ± 0,005 | 0,001 ± 0,005 | 0,575 ± 0,054 | no | no |
| 1:1 | TAGCN | none | 0/3 | 0,200 [0,074; 0,327] | 0,223 ± 0,039 | 0,275 ± 0,054 | 0,871 ± 0,065 | 0,009 ± 0,002 | 0,009 ± 0,003 | -0,001 ± 0,004 | 0,684 ± 0,045 | no | no |
| 1:1 | TAGCN | CW | 0/3 | 0,205 [0,065; 0,346] | 0,223 ± 0,043 | 0,274 ± 0,054 | 0,870 ± 0,064 | 0,009 ± 0,002 | 0,009 ± 0,003 | 0,000 ± 0,003 | 0,693 ± 0,051 | no | no |
| 1:1 | TAGCN | focal | 0/3 | 0,229 [0,045; 0,412] | 0,115 ± 0,065 | 0,166 ± 0,090 | 0,864 ± 0,046 | 0,007 ± 0,001 | 0,003 ± 0,004 | -0,009 ± 0,008 | 0,635 ± 0,053 | no | no |

Lectura rápida:

- **Test colapsa en todas las configuraciones** (test PR-AUC entre 0,004 y 0,014, a lo sumo unas dos veces el azar). El mejor test ROC-AUC de las 48 es el de GraphSAGE + CW (0,70 a 0,71 en los cuatro escenarios). La estabilidad se sigue midiendo sobre verdaderos positivos de validación.
- **GCN no pasa en ninguna configuración** (val PR-AUC máxima 0,185).
- **En 1:1, none y class_weighting son prácticamente idénticos** (val PR-AUC igual en 4 decimales en 7 de los 12 pares semilla a semilla; el resto difiere por el no determinismo de GPU). Esto confirma lo dicho para class weighting (peso 1). **Focal loss no converge a «sin ajuste» en 1:1**: con α = 0,75 sigue pesando la clase ilícita 3 a 1 y γ = 2 cambia el gradiente. Su val PR-AUC es menor que la de none en GCN (0,079 frente a 0,185), GraphSAGE (0,273 frente a 0,364) y GAT (0,204 frente a 0,347); solo en TAGCN es parecida (0,229 frente a 0,200).
- **Ninguna configuración 1:1 pasa la compuerta argmax, pero 8 de 12 pasan con el umbral calibrado.** Entrenado balanceado, el umbral 0,5 sobreestima la clase ilícita frente a la prevalencia de validación (1:40). El fallo es en buena parte de umbral y no de ranking (GraphSAGE 1:1 tiene val ROC-AUC 0,92).
- **Parte del desacuerdo entre semillas es del early stopping.** En 14 de los 144 modelos la mejor época es ≤ 5 (por ejemplo, la semilla 44 de TAGCN + CW en native, 1:10 y 1:10_os): el F1 argmax queda plano al inicio y la paciencia de 20 corta el entrenamiento.

## 2. Compuerta a nivel de configuración

### 2.1 Cuántas pasan con cada criterio

| Criterio | 48 | Sin 1:1 (36) | GCN | GraphSAGE | GAT | TAGCN | native | 1:10 | 1:10_os | 1:1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Por modelo: alguna semilla pasa (criterio implícito del análisis v4) | 21 | 21 | 0 | 8 | 5 | 8 | 6 | 7 | 8 | 0 |
| Por modelo: mayoría (≥ 2 de 3) | 21 | 21 | 0 | 8 | 5 | 8 | 6 | 7 | 8 | 0 |
| Por modelo: las 3 semillas | 15 | 15 | 0 | 8 | 3 | 4 | 3 | 6 | 6 | 0 |
| **Configuración: media de F1 y MCC (principal)** | **20** | **20** | 0 | 8 | 4 | 8 | 5 | 7 | 8 | 0 |
| Configuración: media - sd | 15 | 15 | 0 | 8 | 3 | 4 | 3 | 6 | 6 | 0 |
| Configuración: límite inferior IC 95 % t | 15 | 15 | 0 | 8 | 3 | 4 | 3 | 6 | 6 | 0 |
| Sensibilidad: media con F1/MCC calibrados | 32 | 24 | 0 | 11 | 10 | 11 | 6 | 9 | 9 | 8 |

Con n = 3 el semiancho del IC t es 2,48 sd. Que las dos variantes conservadoras coincidan con «pasan las 3 semillas» es un resultado de estos datos, no una identidad: las configuraciones 3/3 tienen sd de F1 entre 0,006 y 0,040, y las que tienen una semilla mala tienen sd grande.

### 2.2 Desacuerdo entre semillas (todas 2 de 3)

| Configuración | Pasan | val F1 (42 / 43 / 44) | val MCC (42 / 43 / 44) | F1 media ± sd | MCC media ± sd | Comp. media | Comp. cons. |
|---|---|---|---|---|---|---|---|
| native · GAT · none | 42, 43 | 0,336 / 0,316 / 0,167 | 0,322 / 0,304 / 0,146 | 0,273 ± 0,093 | 0,257 ± 0,096 | **no** | no |
| native · TAGCN · CW | 42, 43 | 0,448 / 0,450 / 0,183 | 0,449 / 0,439 / 0,215 | 0,360 ± 0,154 | 0,368 ± 0,132 | sí | no |
| native · TAGCN · focal | 42, 44 | 0,397 / 0,283 / 0,390 | 0,388 / 0,265 / 0,380 | 0,357 ± 0,064 | 0,345 ± 0,069 | sí | no |
| 1:10 · TAGCN · CW | 42, 43 | 0,427 / 0,441 / 0,180 | 0,435 / 0,445 / 0,213 | 0,350 ± 0,147 | 0,365 ± 0,131 | sí | no |
| 1:10_os · GAT · focal | 42, 43 | 0,405 / 0,334 / 0,271 | 0,416 / 0,324 / 0,252 | 0,337 ± 0,067 | 0,331 ± 0,082 | sí | no |
| 1:10_os · TAGCN · CW | 42, 43 | 0,422 / 0,464 / 0,189 | 0,417 / 0,455 / 0,217 | 0,358 ± 0,148 | 0,363 ± 0,128 | sí | no |

La única configuración que sale respecto del análisis v4 es **native · GAT · none** (21 → 20). Lo más importante no es cuántas entran sino **qué valor aportan**: antes, las configuraciones con desacuerdo aportaban la media de las semillas que pasan (la «semilla con suerte»); ahora aportan la media de las 3. Eso es lo que mueve H3.

## 3. Hipótesis con la configuración como unidad

### 3.1 Método

Se reutiliza sin cambios `analyze_elliptic_v4.analyse()`: Friedman con bloques completos, Kruskal-Wallis, Wilcoxon pareado con Holm, TOST pareado ±0,05, regresión con val PR-AUC, soporte mínimo 5 configuraciones por arquitectura, α = 0,05 y la misma regla de tres ramas. Solo cambia la unidad: **y = media de `stability_primary` de las 3 semillas** de la configuración, siempre las 3, y la configuración entra según la compuerta de configuración. Como control, el script reproduce el análisis v4 con sus unidades originales y obtiene exactamente las cifras de `results_v4/analysis_v4_summary.txt`.

Se añadió un **Friedman exacto por permutación**: los rangos se permutan dentro de cada bloque, con enumeración completa hasta 2 × 10⁶ permutaciones y Monte Carlo de 200.000 en otro caso. Con 4 bloques y 3 tratamientos el estadístico solo toma 9 valores: χ² = 6,0 da p asintótico 0,0498 pero exacto 0,069, y el mínimo p exacto posible es 0,0046 (W = 1).

### 3.2 Principal frente al análisis v4 vigente

Principal: compuerta de configuración (media), sin 1:1, n = 20 (GraphSAGE 8, TAGCN 8, GAT 4, GCN 0). **n pequeño:** H1 y H3 tienen 4 bloques completos. En H2, GAT queda bajo el soporte mínimo de 5 y solo se comparan GraphSAGE y TAGCN (8 pares): no hay Friedman (k = 2) y el ómnibus es Kruskal-Wallis.

| Explicador | H | Análisis v4 vigente (21 config.) | Principal (20 config.) | ¿Cambia? |
|---|---|---|---|---|
| GNNExplainer | H1 | Friedman p = 0,449 (exacto 0,522), 5 bloques; TOST ≤ 0,0015 → EQ | p = 0,779 (exacto 0,931), 4 bloques; Δ = -0,005 en ambos, TOST 0,0024 y 0,0013 → **EQ** | No |
| GNNExplainer | H2 | SAGE, GAT, TAGCN; Friedman p = 0,472; 3 TOST < 0,05 → EQ | Solo SAGE vs TAGCN: Kruskal p = 0,401; Δ = 0,019, TOST 0,0058 → **EQ** | No (GAT sale por soporte) |
| GNNExplainer | H3 | χ² = 6,00, **p = 0,0498** (exacto 0,069), W = 0,75 → SIG | χ² = 3,50, **p = 0,174** (exacto 0,273), W = 0,44; TOST 0,066 a 0,099 → **ND** | **Sí** |
| PGExplainer | H1 | p = 0,247 (exacto 0,367) → ND | p = 0,472 (exacto 0,653); Δ = -0,097 y -0,070 → **ND** | No |
| PGExplainer | H2 | Friedman p = 0,174, Kruskal p = 0,022 → ND | Kruskal p = 0,462; Δ = 0,074, TOST 0,62 → **ND** | No |
| PGExplainer | H3 | χ² = 6,00, p = 0,0498 (exacto 0,069) → SIG | igual: p = 0,0498 (**exacto 0,069**); Δ none vs CW = 0,52 → SIG con la regla vigente, no con la exacta | No con la regla vigente |
| ShapleyFeatures | H1 | EQ | p = 0,174 (exacto 0,273); TOST < 10⁻⁵ → **EQ** | No |
| ShapleyFeatures | H2 | EQ | Kruskal p = 0,074; Δ = 0,004, TOST < 10⁻⁹ → **EQ** | No |
| ShapleyFeatures | H3 | χ² = 6,50, p = 0,039 (exacto 0,042) → SIG | igual; los 3 TOST < 10⁻⁴, \|Δ\| ≤ 0,006 → SIG **pero prácticamente nulo** | No |

**Por qué cambia H3 de GNNExplainer.** Los 4 bloques son GraphSAGE y TAGCN en 1:10 y 1:10_os. Antes, TAGCN + CW aportaba la media de las semillas 42 y 43; ahora entra también la 44, cortada por el early stopping en la época 3. En 1:10_os queda por debajo de none, y la concordancia baja de W = 0,75 a 0,44.

| Bloque (GNNExplainer) | none | CW (v4: semillas que pasan) | CW (3 semillas) | focal |
|---|---|---|---|---|
| 1:10 · GraphSAGE | 0,920 | 0,964 | 0,964 | 0,922 |
| 1:10 · TAGCN | 0,928 | 0,952 | 0,936 | 0,883 |
| 1:10_os · GraphSAGE | 0,917 | 0,965 | 0,965 | 0,932 |
| 1:10_os · TAGCN | 0,943 | 0,955 | 0,937 | 0,879 |

**Si GAT entra (soporte mínimo 3).**
- GNNExplainer: H2 sigue en EQ (Friedman p = 0,472, 4 bloques; SAGE vs GAT Δ = -0,010, TOST 0,008; GAT vs TAGCN Δ = 0,018, TOST 0,022).
- ShapleyFeatures: sigue en EQ.
- PGExplainer: sigue en ND (Kruskal p = 0,043, Friedman p = 0,174).
- H1 y H3 no cambian.

### 3.3 Sensibilidades

| Variante | n | GNNExplainer H1/H2/H3 | PGExplainer H1/H2/H3 | ShapleyFeatures H1/H2/H3 |
|---|---|---|---|---|
| **Principal**: configuración, media, sin 1:1 | 20 | EQ / EQ (SAGE vs TAGCN) / ND | ND / ND / SIG (exacto 0,069) | EQ / EQ / SIG (Δ ≤ 0,006) |
| Configuración, media, **con 1:1** | 20 | idéntica (ninguna 1:1 pasa) | idéntica | idéntica |
| Configuración, media - sd, sin 1:1 | 15 | EQ / no evaluable (solo GraphSAGE con soporte) / ND (2 bloques) | ND / no evaluable / ND | EQ / no evaluable / EQ |
| Configuración, límite inferior IC, sin 1:1 | 15 | igual a media - sd | igual | igual |
| F1/MCC calibrados, sin 1:1 | 24 | EQ / EQ (3 arq.) / EQ | ND / SIG / SIG | EQ / EQ / EQ |
| F1/MCC calibrados, con 1:1 | 32 | EQ / SIG / EQ | ND / SIG / SIG | EQ / SIG / EQ |
| Sin compuerta, sin 1:1 | 36 | EQ / SIG / SIG | ND / SIG / SIG | EQ / SIG / EQ |
| Sin compuerta, con 1:1 | 48 | EQ / SIG / SIG | ND / SIG / SIG | EQ / SIG / EQ |

EQ = equivalencia (TOST); SIG = efecto significativo; ND = no se detectó con esta potencia.

- **Con 1:1.** Con la compuerta de configuración es idéntica a la principal. Sin compuerta, 1:1 vs native es equivalente para GNNExplainer (Δ = 0,011, TOST < 10⁻⁵) y ShapleyFeatures (Δ = 0,001), y no lo es para PGExplainer (Δ = -0,168). Ninguna rama de H1 cambia al incluir 1:1.
- **H2 sin compuerta es significativo en los tres explicadores.** En GNNExplainer y ShapleyFeatures todas las diferencias caen dentro del margen (\|Δ\| ≤ 0,019 y ≤ 0,005; TOST < 0,003): detectables con 9 a 12 bloques, sin relevancia práctica, e incluyen modelos que no aprendieron. En PGExplainer sí son grandes (GAT supera a las demás por 0,22 a 0,41).
- **H3 de GNNExplainer solo es significativo sin compuerta** (12 bloques, exacto < 0,001), con diferencias pequeñas (CW supera a none por 0,022 y a focal por 0,031, dentro del margen).
- **H3 de PGExplainer es el único efecto grande y de dirección estable.** class_weighting colapsa su estabilidad en GraphSAGE y TAGCN (0,01 a 0,06 frente a 0,48 a 0,71 con none). Es significativo sin compuerta (exacto 0,002) y casi significativo con la compuerta calibrada (exacto 0,052, 6 bloques).

### 3.4 Conclusión

- **H1 se mantiene en todas las variantes.**
- **H2 se mantiene con la compuerta de configuración**, con dos matices: cubre solo GraphSAGE, TAGCN y (con soporte 3) GAT, porque GCN no entra nunca; y las compuertas conservadoras lo dejan sin evaluar.
- **H3 de GNNExplainer no se sostiene.** Dependía de promediar solo las semillas que pasan, y además su p = 0,0498 era asintótico con 4 bloques (exacto 0,069). Lo defendible es el efecto de class_weighting sobre PGExplainer, con la advertencia de potencia.
- **Potencia, explícitamente:** 20 configuraciones, pero solo 4 bloques completos en H1 y H3 y 8 pares en H2. Con 4 bloques, Friedman solo rechaza al 5 % (exacto) si W ≥ 0,81. «No se detectó» no significa «no hay efecto».

## 4. Selección del candidato

Media e IC 95 % t por escenario (n = 3) con su puesto entre las 12 combinaciones, y media agrupada de los 9 modelos. La concordancia del ranking entre escenarios es alta: **W de Kendall = 0,781** (χ² = 25,8, gl = 11, p = 0,007).

| # | Arquitectura + pérdida | native | 1:10 | 1:10_os | Rango medio | Peor | val PR-AUC agrupada [IC 95 %] | Escen. que pasan | Modelos que pasan | test ROC-AUC | Estab. GNNExplainer |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | GraphSAGE + class_weighting | 0,462 [0,411; 0,512] (1.º) | 0,455 [0,436; 0,474] (1.º) | 0,438 [0,400; 0,476] (2.º) | 1,33 | 2 | 0,452 [0,439; 0,464] | 3/3 | 9/9 | 0,713 | 0,964 |
| 2 | GraphSAGE + focal_loss | 0,305 [0,203; 0,407] (4.º) | 0,426 [0,220; 0,632] (2.º) | 0,442 [0,395; 0,489] (1.º) | 2,33 | 4 | 0,391 [0,329; 0,453] | 3/3 | 9/9 | 0,631 | 0,924 |
| 3 | TAGCN + focal_loss | 0,290 [0,063; 0,518] (6.º) | 0,408 [0,355; 0,460] (3.º) | 0,402 [0,332; 0,473] (3.º) | 4,00 | 6 | 0,367 [0,309; 0,425] | 3/3 | 8/9 | 0,495 | 0,889 |
| 4 | GAT + none | 0,202 [-0,033; 0,438] (7.º) | 0,404 [0,340; 0,468] (4.º) | 0,385 [0,302; 0,469] (5.º) | 5,33 | 7 | 0,331 [0,246; 0,415] | 2/3 | 8/9 | 0,534 | 0,933 |
| 5 | GAT + class_weighting | 0,320 [0,220; 0,421] (3.º) | 0,323 [0,221; 0,425] (6.º) | 0,345 [0,280; 0,410] (7.º) | 5,33 | 7 | 0,329 [0,304; 0,355] | 0/3 | 0/9 | 0,623 | 0,940 |
| 6 | GAT + focal_loss | 0,332 [0,258; 0,406] (2.º) | 0,304 [0,242; 0,366] (8.º) | 0,294 [0,040; 0,549] (8.º) | 6,00 | 8 | 0,310 [0,266; 0,354] | 2/3 | 5/9 | 0,504 | 0,928 |
| 7 | GraphSAGE + none | 0,177 [0,029; 0,326] (8.º) | 0,373 [0,214; 0,533] (5.º) | 0,347 [0,256; 0,439] (6.º) | 6,33 | 8 | 0,299 [0,220; 0,379] | 2/3 | 6/9 | 0,450 | 0,917 |
| 8 | TAGCN + none | 0,097 [-0,027; 0,220] (11.º) | 0,320 [0,187; 0,453] (7.º) | 0,397 [0,320; 0,474] (4.º) | 7,33 | 11 | 0,271 [0,163; 0,379] | 2/3 | 6/9 | 0,534 | 0,923 |
| 9 | TAGCN + class_weighting | 0,299 [-0,046; 0,645] (5.º) | 0,294 [-0,042; 0,629] (9.º) | 0,292 [-0,043; 0,627] (9.º) | 7,67 | 9 | 0,295 [0,204; 0,386] | 3/3 | 6/9 | 0,663 | 0,936 |
| 10 | GCN + class_weighting | 0,097 [0,068; 0,126] (10.º) | 0,162 [-0,096; 0,419] (10.º) | 0,162 [-0,096; 0,419] (10.º) | 10,00 | 10 | 0,140 [0,079; 0,202] | 0/3 | 0/9 | 0,579 | 0,937 |
| 11 | GCN + focal_loss | 0,105 [0,040; 0,171] (9.º) | 0,146 [0,010; 0,281] (11.º) | 0,108 [0,080; 0,136] (12.º) | 10,67 | 12 | 0,120 [0,091; 0,148] | 0/3 | 0/9 | 0,588 | 0,912 |
| 12 | GCN + none | 0,083 [-0,032; 0,198] (12.º) | 0,121 [0,036; 0,207] (12.º) | 0,124 [0,032; 0,215] (11.º) | 11,67 | 12 | 0,109 [0,079; 0,140] | 0/3 | 0/9 | 0,577 | 0,914 |

**Principal: GraphSAGE + class_weighting.**
- Rango medio 1,33 (peor puesto 2).
- Mayor val PR-AUC agrupada y la menor dispersión entre semillas (sd media dentro de la configuración 0,014, frente a 0,031 a 0,136 en las otras 11 combinaciones).
- En native, el límite inferior de su IC (0,411) supera la media de todas las demás (máximo 0,332).
- Pasa la compuerta en sus 3 configuraciones y en los 9 modelos, con la mejor val ROC-AUC (0,922), el mejor test ROC-AUC (0,713) y la mayor estabilidad de GNNExplainer (0,964).

**Suplente: GraphSAGE + focal_loss.**
- Rango medio 2,33, 9 de 9 modelos pasan, primero en 1:10_os; su punto débil es native (0,305, 4.º).
- Cubre el riesgo de que la ventaja sea de la pérdida.
- Si se quiere cubrir el riesgo de arquitectura: **TAGCN + focal_loss** (rango 4,0; 8 de 9 pasan), aunque su IC en native es muy ancho (0,063 a 0,518).

Advertencias:

1. **La búsqueda v4 no fue realmente TPE.** `TPESampler(seed=42)` tiene `n_startup_trials = 10` por defecto y solo corrieron 8 trials. Fueron el prior más 7 muestras aleatorias con semilla fija, las mismas para todas las configuraciones de una arquitectura. Por eso GraphSAGE + CW usa los mismos hiperparámetros en los cuatro escenarios (hidden 64, 2 capas, lr 0,0046). Con 100 trials el ranking puede cambiar: conviene correr también el suplente.
2. **El tope de 150 épocas es activo:** en los 9 modelos del candidato la mejor época está entre 144 y 150. Conviene subir `epochs` y `trial_epochs`, y hacer el early stopping sobre PR-AUC y no sobre F1 argmax.
3. **PGExplainer es muy inestable con GraphSAGE + CW:** 0,015, frente a 0,291 con focal y 0,510 con none.
4. **La selección es optimista:** validación eligió hiperparámetros, umbral, compuerta y ahora el candidato. Test sigue colapsado (PR-AUC 0,011, el doble del azar 0,0057). El test ROC-AUC favorece al principal.

## 5. Costo estimado

Tiempos de `duration_s` (RTX 4060).
- Las semillas 43/44 no corren Optuna; su duración da los segundos por época del entrenamiento final.
- La parte de Optuna de la semilla 42 es su duración menos su entrenamiento final a ese ritmo, dividida entre 8 trials. Incluye el armado del escenario, así que tiende a sobreestimar.
- Cota superior por trial: 50 épocas sin poda.

| Combinación | s/época | s/trial estimado | s/trial máx. | 100 trials × 3 escenarios | Final (3 semillas × 3 escenarios) | Total |
|---|---|---|---|---|---|---|
| GraphSAGE + CW | 0,10 a 0,11 | 4,3 a 5,0 | 5,1 a 5,3 | 22,9 min (máx. 25,8) | 2,3 min | **0,42 h** (máx. 0,47) |
| GraphSAGE + focal | 0,10 a 0,13 | 3,8 a 4,7 | 5,1 a 6,5 | 20,6 min (máx. 27,9) | 1,6 min | **0,37 h** (máx. 0,51) |
| TAGCN + focal | 0,27 a 0,36 | 6,6 a 7,2 | 13,7 a 17,9 | 34,8 min (máx. 81,2) | 2,8 min | **0,63 h** (máx. 1,47) |

- Las 12 combinaciones están en `configs48_costo_proyectado.csv`: de 0,27 h (GCN) a 0,74 h (GAT y TAGCN).
- Explicación (medianas de GraphSAGE): 25 s GNNExplainer, 65 s PGExplainer y 22 s ShapleyFeatures por modelo, unos 17 min para 9 modelos.
- Con 500 épocas finales y 150 por trial: unos 100 × 3 × 15 s ≈ 75 min de Optuna más unos 8 min de entrenamiento final, alrededor de 1,4 h. Principal y suplente juntos caben en una noche.

## 6. Archivos

| Archivo | Contenido |
|---|---|
| `configs48_modelos.csv` | 144 modelos: métricas val/test (argmax y calibradas), ROC-AUC, compuerta, hiperparámetros, duración |
| `configs48_metricas.csv` | 48 configuraciones: media, sd e IC 95 % t de 12 métricas, F1/MCC por semilla, 7 criterios de compuerta |
| `configs48_compuerta_resumen.csv`, `configs48_compuerta_desacuerdo.csv` | Secciones 2.1 y 2.2 |
| `configs48_units_estabilidad.csv` | Unidades del análisis por variante (`filter`) |
| `configs48_tests.csv`, `configs48_regresion.csv` | Todas las pruebas (formato de `analysis_v4_tests.csv`) |
| `configs48_friedman_exacto.csv` | Friedman exacto o Monte Carlo por variante, explicador e hipótesis |
| `configs48_hipotesis.txt` | Salida completa de `analyse()` por variante, ramas y p exactos |
| `configs48_seleccion.csv`, `configs48_kendall.json` | Ranking y concordancia |
| `configs48_costo_por_config.csv`, `configs48_costo_proyectado.csv` | Tiempos y proyección |
| `roc_auc_cpu.csv` | ROC-AUC en CPU y control contra el PR-AUC del meta.json |

Reproducir (CPU): `~/.local/bin/uv run --frozen python scripts/v4/reunion_0410/configs48.py` (con `--roc` recalcula el ROC-AUC desde los checkpoints, unos minutos).
