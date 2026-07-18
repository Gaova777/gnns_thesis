# Fase 1 ROBUSTA — inferencia estadística sobre la matriz factorial

Fecha: 2026-07-17 · Encargo: ENCARGO_robustez_estadistica.md · Ejecuta: Claude Code (4060) · Audita: sesión de análisis

Esta corrida no busca mejorar ningún número sino soportarlo con réplicas. Se añadieron tres fuentes de
variabilidad al factorial y se cuantificó la incertidumbre resultante. Ningún hiperparámetro se ajustó para
que un resultado "se viera bien"; la única intervención fue la replicación. Los tres hallazgos centrales del
factorial se sometieron a prueba y ninguno cambió de dirección.

## Qué se corrió

- Run principal: matriz completa 4 arq × 5 escen × 3 balanceo × 3 explicadores sobre el grafo g0 (v2),
  con S=3 semillas de modelo {42,43,44} → 540 filas de celda, 14907 nodos.
- Run de grafos (PIEZA 3): matriz reducida (4 arq × {1:1, natural} × none × 3 explicadores) sobre g1 y g2
  → 144 filas, 4320 nodos. Los tres grafos comparten configuración (firma +1.5, simetrizado, distractores)
  y densidad (receptive field k=2 mediana ≈29).
- Agregado con media Y dispersión por celda en `results_robust_agg.csv` (228 celdas).

## Chequeo de reproducción (control de que la extensión es fiel)

El corte del run robusto en g0 con model_seed=42 reproduce el factorial en las métricas que no cambiaron de
definición, lo que confirma que el runner robusto es una extensión y no una reescritura que movió los números.

| métrica | robusto (g0, seed 42) | factorial | coincide |
|---|---|---|---|
| spearman | 0.950 | 0.950 | sí |
| plaus_edge | 0.629 | 0.628 | sí |
| plaus_feat | 0.326 | 0.328 | sí |

La fidelidad no se compara porque la PIEZA 2 la redefinió de forma uniforme (ver abajo). NaN silencioso: 0.

---

## 1. Media ± desviación estándar sobre las 3 semillas de modelo (grafo g0)

Cada número trae ahora su dispersión entre entrenamientos; el intervalo de confianza al 95% se obtiene con la
t de Student sobre las tres semillas. Se muestran arquitectura × explicador para las métricas principales.

Estabilidad (Spearman de features):

| arch | GNNExplainer | GNNShap | PGExplainer |
|---|---|---|---|
| GraphSAGE | 0.889 ± 0.030 | 0.973 ± 0.011 | N/A |
| GAT | 0.960 ± 0.025 | 0.974 ± 0.015 | N/A |
| GCN | 0.965 ± 0.026 | 0.989 ± 0.004 | N/A |
| TAGCN | 0.886 ± 0.037 | 0.971 ± 0.012 | N/A |

Plausibilidad de edges (F1 vs edges de la tipología):

| arch | GNNExplainer | PGExplainer |
|---|---|---|
| GraphSAGE | 0.493 ± 0.025 | 0.924 ± 0.032 |
| GAT | 0.540 ± 0.015 | 0.683 ± 0.181 |
| GCN | 0.523 ± 0.042 | 0.734 ± 0.146 |
| TAGCN | 0.426 ± 0.061 | 0.861 ± 0.032 |

Fidelity+ (caída de probabilidad al quitar los top-k importantes; mayor es más fiel):

| arch | GNNExplainer | PGExplainer | GNNShap |
|---|---|---|---|
| GraphSAGE | 0.633 ± 0.153 | 0.133 ± 0.085 | 0.548 ± 0.120 |
| GAT | 0.522 ± 0.130 | 0.119 ± 0.226 | 0.428 ± 0.179 |
| GCN | 0.536 ± 0.084 | 0.121 ± 0.131 | 0.514 ± 0.203 |
| TAGCN | 0.550 ± 0.141 | 0.067 ± 0.061 | 0.372 ± 0.147 |

## 2. Kruskal-Wallis (primario) y ANOVA con tamaños de efecto

Las métricas están acotadas en [0,1] y sus residuales no pasan la prueba de normalidad de Shapiro, de modo que
la prueba primaria es Kruskal-Wallis y se reporta epsilon² como tamaño de efecto; el ANOVA y eta² se incluyen
como contraste. La lectura por umbrales convencionales es 0.01 pequeño, 0.06 medio y 0.14 grande. El factor
explicador domina todas las métricas, la arquitectura tiene efecto grande solo en estabilidad, y el balanceo
resulta prácticamente irrelevante para la calidad de las explicaciones.

| métrica | factor | KW H | p | epsilon² | ANOVA F | eta² |
|---|---|---|---|---|---|---|
| spearman | explicador | 125.6 | <1e-27 | 0.354 | 198.0 | 0.360 |
| spearman | arquitectura | 91.1 | <1e-19 | 0.252 | 36.5 | 0.238 |
| spearman | balanceo | 9.1 | 0.011 | 0.020 | 1.4 | 0.008 |
| plaus_edge | explicador | 195.8 | <1e-43 | 0.553 | 625.7 | 0.640 |
| plaus_edge | arquitectura | 10.5 | 0.015 | 0.021 | 4.5 | 0.037 |
| plaus_edge | balanceo | 2.4 | 0.308 | 0.001 | 1.5 | 0.008 |
| plaus_feat | explicador | 176.2 | <1e-38 | 0.498 | 282.6 | 0.445 |
| plaus_feat | arquitectura | 11.3 | 0.010 | 0.024 | 6.2 | 0.050 |
| plaus_feat | balanceo | 3.5 | 0.173 | 0.004 | 2.1 | 0.012 |

## 3. Prueba pareada: el hallazgo estrella sobrevive a la varianza de modelo

La afirmación de que PGExplainer señala los edges de la tipología mejor que GNNExplainer se puso a prueba con
una comparación pareada por celda, tomando cada combinación de grafo, arquitectura, escenario, balanceo y
semilla de modelo como una pareja. Sobre 225 parejas la diferencia media es de +0.299 a favor de PGExplainer,
positiva en el 92% de los casos, con una prueba de Wilcoxon de rangos con signo que arroja p = 2.6e-35. El
hallazgo por tanto no es un artefacto de un único entrenamiento sino un efecto sostenido a través de la
variabilidad entre modelos.

| explicador | plaus_edge (media ± sd sobre 225 parejas) |
|---|---|
| PGExplainer | 0.800 ± 0.151 |
| GNNExplainer | 0.502 ± 0.056 |

## 4. Puente estabilidad ↔ plausibilidad con intervalo de confianza bootstrap

El puente se estimó por nodo con intervalos de confianza bootstrap de 2000 remuestreos. La correlación global
entre estabilidad y plausibilidad de features es de −0.014 para GNNExplainer, con un intervalo que contiene al
cero, de modo que el puente es nulo. Para GNNShap la correlación es de +0.034 con un intervalo que excluye al
cero, pero su magnitud es despreciable y el tamaño del efecto explica alrededor del uno por mil de la varianza,
lo que se interpreta como nulo en la práctica y significativo solo por el tamaño de muestra. Descompuesto por
tipología el signo se invierte entre categorías, lo que confirma que no existe una ley universal que ligue
estabilidad con plausibilidad.

| corte | r | IC 95% | lectura |
|---|---|---|---|
| GNNExplainer estab↔features (global) | −0.014 | [−0.038, +0.011] | incluye 0 → nulo |
| GNNShap estab↔features (global) | +0.034 | [+0.013, +0.053] | significativo pero despreciable |
| GNNExplainer estab↔edges STRUCTURING | +0.344 | [+0.301, +0.386] | positivo |
| GNNExplainer estab↔edges FAN_OUT | +0.307 | [+0.247, +0.362] | positivo |
| GNNExplainer estab↔edges LAYERING | −0.102 | [−0.146, −0.055] | negativo |
| GNNExplainer estab↔edges FAN_IN | −0.044 | [−0.105, +0.020] | nulo |

## 5. Robustez a la instancia del grafo (PIEZA 3)

Los hallazgos se replican sobre tres instancias independientes del generador. En las 8 combinaciones comunes a
los tres grafos, la plausibilidad de edges de PGExplainer se mantiene alta y por encima de la de GNNExplainer
en las tres instancias, y la estabilidad de GNNExplainer es prácticamente idéntica entre grafos. La conclusión
del eje sintético no depende por tanto de una realización particular de los datos.

| graph_seed | PGExplainer plaus_edge | GNNExplainer plaus_edge | GNNExplainer estabilidad |
|---|---|---|---|
| g0 | 0.820 ± 0.154 | 0.498 ± 0.057 | 0.919 ± 0.059 |
| g1 | 0.765 ± 0.184 | 0.523 ± 0.030 | 0.931 ± 0.052 |
| g2 | 0.829 ± 0.101 | 0.529 ± 0.032 | 0.933 ± 0.049 |

---

## Hallazgo nuevo que habilita la PIEZA 2: disociación plausibilidad–fidelidad

Con la fidelidad medida de forma uniforme para los tres explicadores aparece una tensión que el factorial no
podía mostrar. PGExplainer es el más plausible en edges pero el menos fiel al modelo, con una caída de
probabilidad al remover sus edges de apenas 0.07 a 0.13 según la arquitectura. GNNExplainer es a la inversa,
menos alineado con la tipología de referencia pero mucho más fiel, con caídas de 0.52 a 0.63. La lectura es que
PGExplainer recupera los edges que definen el patrón de lavado según el ground-truth, mientras que GNNExplainer
recupera los edges que el modelo realmente utiliza para decidir, y ambos conjuntos no coinciden. Esta
disociación entre lo que resulta plausible para un humano y lo que es fiel al mecanismo del modelo es un
resultado que solo el régimen sintético con ground-truth de tipología permite exhibir.

## Definición de la Fidelity± manual (PIEZA 2)

El built-in de PyG solo cubría GNNExplainer, de modo que se implementó una fidelidad manual con una definición
común basada en la probabilidad de la clase predicha. fid+ es la caída de esa probabilidad al quitar el top-k de
elementos importantes, donde más caída indica mayor fidelidad. fid− es el cambio al conservar únicamente el
top-k, donde un valor cercano a cero indica mayor fidelidad. Para GNNExplainer y PGExplainer el top-k se aplica
sobre edges, y para GNNShap sobre features, con baseline cero tras el escalado robusto; el top-k es el 25% de
los elementos. La máscara enmascarada no es idéntica entre edge-explainers y feature-explainers y esa diferencia
se declara de forma explícita, ya que la comparación de fidelidad entre las dos familias debe leerse con esa
salvedad.

## Criterios de aceptación (para el recálculo del auditor)

- Columnas `model_seed` y `graph_seed` pobladas; S=3 semillas de modelo por celda en las 228 celdas.
- Fidelity± con las tres columnas de explicador pobladas (75 de 76 celdas por explicador; la faltante es la
  celda GCN / 1:100 / none con 0 TP en las tres semillas, un SKIP legítimo).
- CSV con media Y dispersión por celda (`results_robust_agg.csv`, columnas `*_sd`); NaN silencioso 0.
- Reporte con IC/sd, Kruskal-Wallis y ANOVA con tamaños de efecto, prueba de robustez de PGExplainer y puente
  con IC.
- Ningún hallazgo cambió de dirección: la reproducción del corte seed=42 es exacta, el contraste PG > GNN
  sobrevive con p = 2.6e-35, y el puente sigue nulo con intervalos que lo confirman.

## Entregables (`phase1/`)

`run_phase1_robust.py`, `analyze_robust.py`, `synthetic_aml_g1.pt`, `synthetic_aml_g2.pt`,
`results_robust.csv` (+ `_pernode`), `results_robust_graph.csv` (+ `_pernode`), `results_robust_agg.csv`.

## Salvedades honestas

- La inferencia se apoya en tres semillas de modelo por celda, suficiente para intervalos y para Kruskal-Wallis
  pero modesto; subir a cinco semillas estrecharía los intervalos si se dispusiera de más tiempo de cómputo.
- La comparación de fidelidad entre edge-explainers y feature-explainers usa máscaras distintas por
  construcción y debe leerse con esa salvedad declarada.
- La correlación de GNNShap en el puente es distinguible de cero solo por el tamaño de muestra y su magnitud es
  despreciable; no debe presentarse como evidencia de un puente.
- El régimen sintético carece de shift temporal, a diferencia de Elliptic, y esa diferencia de régimen se
  mantiene al comparar los dos ejes.
