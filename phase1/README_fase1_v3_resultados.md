# Fase 1 v3 — Firma de features atenuada (re-corrida B) + Elliptic k-hop definitivo (re-corrida A)

**Fecha:** 2026-07-16
**Encargo:** `ENCARGO_recorrer_matrices.md`. Dos re-corridas independientes con el pipeline corregido.

---

## RE-CORRIDA A — Elliptic k-hop DEFINITIVO (reemplaza los números full-graph de la tesis v16)

Ya ejecutada (pipeline k-hop corregido); CSVs en `results_v3/xai-gnn-stability-{B,C}-v3.csv`. Verificado:
**0 celdas cuda_oom, GAT 15/15 medible**, columnas completas (spearman, jaccard, stab_n_tp,
stab_subgraph_n_nodes/edges, model_test_f1).

**Ranking por arquitectura (Spearman de features, GNNExplainer):**
GraphSAGE **0.630** > GAT 0.486 > GCN 0.468 > TAGCN 0.270 (GAT/GCN indistinguibles).

**Ranking por escenario:** 1:1 0.525 · 1:100 0.501 · 1:50 0.470 · 1:30_native 0.439 · 1:10 0.433.
→ **Plano, SIN pico-colapso dramático en 1:50** (1:100 es incluso más alto), y **1:30 nativo NO es el más
bajo** (lo es 1:10). Esto **invalida el "peak-collapse 1:50→1:100"** que defendía la tesis v16.

**GATED (sin `--force`, opción recomendada por el auditor) — DEFINITIVO:** se re-corrió k-hop restringido a
los modelos que pasan el quality gate en VAL (F1≥0.30), que es el subconjunto que la tesis realmente reporta.
Cobertura del gate: GraphSAGE 11/15, GAT 7/15, TAGCN 4/15, **GCN 1/15** (23/60 total). 0 OOM.

| ranking | GraphSAGE | GAT | GCN | TAGCN |
|---|---|---|---|---|
| **GATED k-hop** (definitivo) | **0.640** (n=11) | 0.535 (n=7) | 0.639 (n=**1**) | 0.363 (n=4) |
| FORCE k-hop (todas las configs) | 0.630 (n=15) | 0.486 (n=15) | 0.468 (n=15) | 0.270 (n=12) |
| tesis v16 (full-graph, gated) | 0.345 (n=11) | 0.534 (n=7) | — | — |

**La inversión es robusta:** en el MISMO subconjunto gated que usa la tesis, el pipeline k-hop da
**GraphSAGE 0.640 > GAT 0.535** — la misma dirección que la corrida --force, y la OPUESTA a la tesis v16
(GAT 0.534 > GraphSAGE 0.345). Detalle: GAT quedó casi igual (0.534→0.535) pero **GraphSAGE saltó de 0.345 a
0.640** — el fix k-hop rescató a GraphSAGE, que la explicación full-graph deprimía artificialmente. Por tanto
el "GAT el más estable" de la tesis v16 era un artefacto del full-graph, y **GraphSAGE es el más estable en
ambos subconjuntos**. (Caveat: GCN gated n=1 y TAGCN n=4 → sin soporte para rankearlos; la comparación firme
es GraphSAGE vs GAT.)

---

## RE-CORRIDA B — Fase 1 sintética con firma atenuada (+4 → +1.5)

Cambio: `sig[i, t-1] += 1.5` (antes +4). Todo lo demás del generador v4 igual. Dataset `synthetic_aml_v2.pt`.

**Verificación previa:** densidad receptive field k=2 mediana **30** (≥10 ✓); soporte STRUCTURING 377,
LAYERING 505, FAN_IN 344, FAN_OUT 323; firma ahora **Δ≈1.5 sobre ruido** (antes ~4).

**Criterio de aceptación — plaus_feat ya NO trivial:**
media **0.484** (antes 0.819), fracción==1.0 **48%** (antes 82%), con varianza (rango 0–1). Cumple
(media ≪ 0.9). *No saturado* → no hace falta bajar a +1.0; si se quiere aún menos trivial, +1.0 lo reduciría
más (reportado honestamente).

**Matriz 4 arqs × 3 escenarios (12 celdas, 320 nodos, base_k=2 uniforme):**

| arch | scen | val PR-AUC | estab | plaus edges F1 | plaus feat rec@3 | n_tp |
|---|---|---|---|---|---|---|
| GraphSAGE | 1:1/1:10/1:50 | 0.99 | 0.82–0.92 | 0.46–0.48 | 0.23–0.53 | 30 |
| GAT | 1:1/1:10/1:50 | 0.71–0.83 | 0.97–0.99 | 0.53–0.57 | 0.50–0.67 | 30/30/**15** |
| GCN | 1:1/1:10/1:50 | 0.62–0.80 | 0.86–0.98 | 0.50–0.58 | 0.20–0.77 | 30/30/**5** |
| TAGCN | 1:1/1:10/1:50 | 0.99 | 0.86–0.96 | 0.38–0.47 | 0.30–0.53 | 30 |

(GAT 1:50 n_tp=15 y GCN 1:50 n_tp=5: menos TP en val bajo desbalance extremo, reportado.)

### Puente DESCOMPUESTO POR TIPOLOGÍA (edges Y features — ambos medibles ahora)

**Global:** edges Pearson **+0.066** / Spearman +0.120 · features Pearson **−0.006** / Spearman +0.105.

| tipología | n | r(estab↔edges) | r(estab↔features) |
|---|---|---|---|
| STRUCTURING | 97 | **+0.395** | −0.180 |
| LAYERING | 82 | −0.224 | −0.318 |
| FAN_IN | 55 | +0.075 | +0.150 |
| FAN_OUT | 86 | +0.072 | +0.346 |

**Conclusión honesta (robusta ya a través de v1/v2/v3): NO hay una ley "estabilidad ⇒ plausibilidad."**
- Con la firma atenuada, el puente por FEATURES (el más directo con el Spearman de Elliptic) es
  **prácticamente NULO globalmente** (Pearson −0.006) y heterogéneo por tipología (de −0.32 a +0.35).
- El puente por EDGES sigue débil (+0.07 global), conducido solo por STRUCTURING (+0.40); las demás
  tipologías ~0 o negativas.
- El +0.27 de la v1 era un artefacto de (a) firma trivial y (b) LAYERING con n=4. Corregido eso, **la
  estabilidad NO predice de forma fiable la plausibilidad**, ni por edges ni por features. Es el resultado —
  honesto y verificable — del item 7.

## Paso 5 — Contraste Elliptic vs sintético (se mantiene)
Elliptic: receptive field ~2-3 nodos, shift temporal (test ~0.01), plausibilidad de subgrafo **NO medible**.
Sintético: ~30 nodos, sin shift (test ~0.9+), plausibilidad **medible** (edges 0.48, features 0.48). Ese
contraste — plausibilidad imposible en el dato real vs posible en el sintético controlado — es el aporte
diferencial del trabajo en pareja, independiente de que el puente resulte débil.

## Entregables
- `synthetic_aml_v2.pt` (firma atenuada), generador, `plausibility.py`, `run_phase1_matrix.py`,
  `analyze_bridge.py`.
- `results_phase1_v3.csv` + `_pernode.csv` (con plaus_feat no trivial).
- Elliptic definitivo: `results_v3/xai-gnn-stability-{B,C}-v3.csv`.

## A decidir con el auditor
- ¿Se quiere el subconjunto gated de Elliptic (sin --force)? (re-corrida A alternativa).
- La firma a +1.0 si se quiere plaus_feat aún menos saturado (hoy 48% en 1.0).
- Redacción en Overleaf (CONCLUSIONES v3.2 + Fase 1) tras cerrar integridad de citas.
