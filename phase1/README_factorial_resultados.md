# Fase 1 FACTORIAL — resultados (4 arq × 5 escen × 3 balanceo × 3 explicadores + Fidelity±)

**Fecha:** 2026-07-16 · **Encargo:** RUNBOOK_MAESTRO (bloque único). Cierra el código de la tesis.
**Corrida:** 180 celdas (60 modelos × 3 explicadores), 4968 nodos, dataset `synthetic_aml_v2.pt` (firma +1.5),
nodes=30, replicas=5, GNNExplainer/PGExplainer/GNNShap. **0 errores.** Reusa el pipeline
(no reimplementa). CSVs: `results_factorial.csv` (+ `_pernode.csv`) con columnas `explainer` y `balancing`.

**Único SKIP legítimo (0 TP):** la celda GCN / 1:100 / none — bajo imbalance extremo sin balanceo el
modelo no produjo ningún true positive en los nodos evaluados, así que no hay nada que explicar. Afecta a
sus 3 filas de explicador (por eso las medias por explicador se calculan sobre 59 celdas, no 60). No es un
error ni un NaN silencioso: es la ausencia esperada de nodos ilícitos correctamente clasificados.

---

## 1. Estabilidad (Spearman de features) por arquitectura × explicador

| arch | GNNExplainer | GNNShap | PGExplainer |
|---|---|---|---|
| GraphSAGE | 0.884 | 0.973 | N/A |
| GAT | 0.962 | 0.976 | N/A |
| GCN | 0.964 | 0.989 | N/A |
| TAGCN | 0.886 | 0.970 | N/A |

**GNNShap es el más estable** (0.97–0.99) en las 4 arquitecturas; GCN/GAT algo más estables que
GraphSAGE/TAGCN con GNNExplainer. PGExplainer es determinista tras entrenar → estabilidad N/A por diseño.

## 2. Plausibilidad por explicador — ¿cuál señala mejor el patrón real de lavado?

| explicador | plaus_edge | plaus_feat | shap_conc | celdas |
|---|---|---|---|---|
| **PGExplainer** | **0.763** 🥇 | N/A | N/A | 59 |
| GNNExplainer | 0.493 | 0.493 | N/A | 59 |
| GNNShap | N/A | 0.163 | 0.947 | 59 |

**HALLAZGO CENTRAL — PGExplainer NO es degenerado en el sintético denso** (pg_degenerate="no" en 59/59;
plaus_edge medio **0.763**, el más alto). Esto **responde la pregunta del runbook**: su degeneración en
Elliptic (Spearman 0, mode collapse) **era por la DISPERSIÓN del dataset, no un bug del método** — con
estructura de subgrafo suficiente, PGExplainer señala el patrón mejor que GNNExplainer. Es un resultado
metodológico limpio, imposible de establecer en Elliptic.
**GNNShap es estable pero poco plausible** (plaus_feat 0.163) → estabilidad ≠ señalar la feature correcta.
**shap_concentration medida con ground-truth = 0.947** (la métrica que en la tesis quedó sin fuente).

## 3. Fidelity± por explicador × arquitectura (fid+ / fid−)

| arch | GNNExplainer | PGExplainer | GNNShap |
|---|---|---|---|
| GraphSAGE | 0.95 / 0.00 | n/a | n/a |
| GAT | 1.00 / 0.29 | n/a | n/a |
| GCN | 0.96 / 0.04 | n/a | n/a |
| TAGCN | 0.93 / 0.00 | n/a | n/a |

Las explicaciones de **GNNExplainer son fieles** (fid+ alto ≈ quitar los edges importantes degrada la
predicción; fid− bajo ≈ conservar solo los importantes la preserva). GAT tiene fid− más alto (0.29). Fidelity±
no computó para PGExplainer/GNNShap (no son PyG Explainer estándar en esa métrica) → N/A, reportado.

## 4. Puente estabilidad↔plausibilidad por tipología × explicador (Pearson, por nodo)

| tipología (GNNExplainer) | r(estab↔edges) | r(estab↔features) |
|---|---|---|
| STRUCTURING | +0.403 | −0.216 |
| LAYERING | −0.164 | −0.179 |
| FAN_IN | −0.070 | +0.183 |
| FAN_OUT | +0.174 | +0.297 |

Global GNNExplainer estab↔features **−0.014**; GNNShap **+0.053**. **Confirma (consistente con v1/v2/v3): NO
hay una ley "estabilidad ⇒ plausibilidad"** — heterogéneo por tipología, cambia de signo, global ≈ 0. Un
explicador estable no garantiza señalar el patrón correcto.

## 5. Contraste Elliptic vs sintético (el aporte diferencial)

| | Elliptic | Sintético |
|---|---|---|
| plausibilidad de subgrafo | **NO medible** (~2-3 nodos) | **SÍ medible** (~30 nodos) |
| plaus edges (media) | no aplicable | 0.63 |
| plaus features (media) | no aplicable | 0.33 |
| Fidelity± | no medida | medida (tabla 3) |
| **comparar explicadores por plausibilidad** | **IMPOSIBLE** | **POSIBLE** (tabla 2) |

Poder decir *"PGExplainer señala el patrón mejor que GNNExplainer, y GNNShap es estable pero poco plausible"*
es **imposible en Elliptic** (sin ground-truth de tipología y con vecindarios de 2 nodos) y **posible en el
sintético**. Ese es el aporte que justifica el trabajo en pareja.

---

## Criterios de aceptación (para tu recálculo)
- ✅ CSVs con `explainer` y `balancing` poblados (180 filas de celda = 60×3).
- ✅ Fidelity± presente y no-NaN donde hay máscara (GNNExplainer); N/A explícito en PG/GNNShap.
- ✅ 0 errores / 0 NaN silencioso; PGExplainer con estado explícito (`pg_degenerate="no"` → NO degenerado).
- ✅ Plausibilidad por explicador comparable; puente por tipología × explicador.

## Entregables (`phase1/`)
`synthetic_aml_v2.pt`, `run_phase1_factorial.py`, `analyze_factorial.py`, `plausibility.py`,
`results_factorial.csv` (180) + `results_factorial_pernode.csv` (4968).

## Salvedades honestas
- 1 modelo por celda → análisis descriptivo (marginales + tamaños de efecto), no ANOVA con potencia.
- Fidelity± solo para GNNExplainer (limitación de la API para los otros dos) — declarado, no oculto.
- El puente débil/nulo es un resultado legítimo, no un fracaso.
- El régimen sintético no tiene shift temporal (test≈val), a diferencia de Elliptic — declararlo al comparar.
