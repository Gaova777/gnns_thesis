# Corrida sin el periodo del cierre del mercado (THE-44)

Fecha: 11 y 12 de octubre de 2026. 216 modelos, 0 errores. 3 semillas del modelo en todo.

## Diseño

El mercado ilegal cierra en el paso de tiempo 43. Se repitió el estudio completo usando solo
los pasos 1 a 42; los pasos 43 a 49 no se usan.

| Parte | Pasos | Ilícitas | Negativos | Razón |
|---|---|---|---|---|
| Entrenamiento | 1 a 29 | 2.871 | 117.933 | 1:41,1 |
| Validación | 30 a 35 | 773 | 20.195 | 1:26,1 |
| Prueba | 36 a 42 | 732 | 31.581 | 1:43,1 |

Seis escenarios (nativo, 1:10, 1:20, 1:10 con SMOTE, 1:100 y 1:200), 4 arquitecturas, 3
pérdidas, 3 semillas, 5 explicadores. Configuración: `configs/experiment_v4_precierre.yaml`.
Scripts: `scripts/v4/run_precierre.sh`, `scripts/v4/cola_precierre.sh` y
`scripts/v4/reunion_0410/curvas.py --variante precierre`.

Advertencias: la partición la escogieron los autores (reparto del fraude 66 %, 18 % y 17 %);
hay 591 ilícitas menos para entrenar que en la corrida original, así que las cifras de
validación de una y otra no son directamente comparables.

## 1. La prueba deja de caer al azar

Escenario nativo; para cada arquitectura, la pérdida elegida por PR-AUC de validación.

| Arquitectura | PR-AUC con el cierre | PR-AUC sin el cierre | ROC-AUC con el cierre | ROC-AUC sin el cierre |
|---|---|---|---|---|
| GraphSAGE | 0,010 | 0,273 | 0,711 | 0,883 |
| TAGCN | 0,008 | 0,244 | 0,667 | 0,875 |
| GAT | 0,005 | 0,172 | 0,476 | 0,848 |
| GCN | 0,012 | 0,085 | 0,601 | 0,729 |
| Azar | 0,006 | 0,023 | 0,5 | 0,5 |

Mejor valor de toda la matriz en la prueba: PR-AUC 0,306 y ROC-AUC 0,901 (GraphSAGE, pesos por
clase, 1:20).

## 2. La mejor pérdida depende del escenario

PR-AUC de validación (azar 0,037):

| Configuración | 1:10 | 1:20 | Nativo | 1:100 | 1:200 |
|---|---|---|---|---|---|
| GraphSAGE, pesos por clase | 0,405 | 0,462 | 0,326 | 0,447 | 0,420 |
| GraphSAGE, focal loss | 0,325 | 0,457 | 0,478 | 0,189 | 0,054 |
| GraphSAGE, sin ajuste | 0,551 | 0,035 | 0,034 | 0,034 | 0,038 |
| TAGCN, pesos por clase | 0,389 | 0,214 | 0,212 | 0,357 | 0,307 |
| TAGCN, focal loss | 0,337 | 0,512 | 0,529 | 0,282 | 0,054 |
| TAGCN, sin ajuste | 0,594 | 0,307 | 0,149 | 0,046 | 0,046 |

- En la corrida original pesos por clase era la mejor en el nativo (GraphSAGE 0,462 frente a
  0,305 de focal loss). Aquí se invierte. La afirmación «pesos por clase detecta mejor» no es
  general.
- Lo que se repite en las dos corridas: con poco fraude (1:100 y 1:200) solo pesos por clase
  resiste.
- «Sin ajuste» es frágil: pasa de 0,55 a 0,03 entre 1:10 y 1:20 en GraphSAGE.

Compuerta (F1 ≥ 0,30 y MCC ≥ 0,15 en validación, umbral 0,5), modelos que pasan de 18 por
celda: GraphSAGE 10, 11 y 6 (pesos por clase, focal loss, sin ajuste); TAGCN 8, 13 y 7; GAT 0,
8 y 11; GCN 0, 3 y 6.

## 3. La estabilidad de las explicaciones no cambia

Spearman sobre las 165 variables, GraphSAGE, GAT y TAGCN, pesos por clase:

| Explicador | 1:10 | 1:20 | Nativo | 1:100 | 1:200 |
|---|---|---|---|---|---|
| GNNExplainer | 0,935 | 0,929 | 0,917 | 0,939 | 0,932 |
| Shapley | 0,992 | 0,993 | 0,994 | 0,991 | 0,991 |
| Expected Gradients | 0,931 | 0,921 | 0,934 | 0,918 | 0,927 |
| Integrated Gradients | 1 | 1 | 1 | 1 | 1 |

- Friedman sobre los seis escenarios, GNNExplainer con pesos por clase (9 modelos): p = 0,27.
- Con las tres pérdidas juntas (27 modelos) sí hay diferencia (p < 0,001): «sin ajuste» baja de
  0,94 en 1:10 a 0,89 en 1:200. Es un efecto de centésimas y ocurre en modelos que no aprenden.
- Por arquitectura (pesos por clase, escenarios principales, GNNExplainer): GCN 0,937,
  GraphSAGE 0,931, TAGCN 0,929, GAT 0,923. En la corrida original GraphSAGE era la primera:
  no hay un orden estable entre arquitecturas.
- Entre semillas del modelo: GNNExplainer 0,946, Shapley 0,957, Integrated Gradients 0,958,
  Expected Gradients 0,763.

## 4. Qué se mantiene y qué cambia

| | Corrida original | Sin el cierre |
|---|---|---|
| Prueba | Azar en todas | Hasta PR-AUC 0,31 y ROC-AUC 0,90 |
| GCN | La más débil | La más débil |
| Mejor pérdida en el nativo | Pesos por clase | Focal loss (empata o gana) |
| Con poco fraude | Solo resiste pesos por clase | Solo resiste pesos por clase |
| Estabilidad frente al desbalance | Sin cambio | Sin cambio |
| Arquitectura más estable | GraphSAGE con GNNExplainer | Ninguna |

## Pendiente

- No se calcularon aquí las dos medidas que no se inflan (Spearman sobre variables distintas
  de 0 y Jaccard de las 10 principales) ni el control de pesos al azar.
- La partición no se ha acordado con el director.
