# Fase 1 v2 — Resultados (re-corrida con generador v4 + análisis por-tipología). Para el auditor.

**Fecha:** 2026-07-16
**Encargo:** `ENCARGO_claude_code_fase1_v2.md`. Re-ejecución con el generador v4 (LAYERING balanceado +
features-firma) y el runner extendido (plausibilidad de features). Aborda los 3 puntos de tu auditoría.

---

## Criterio de aceptación — estado

| Criterio | Estado |
|---|---|
| 4 tipologías con soporte ≥ ~20 nodos (LAYERING ya no en 4) | ✅ STRUCTURING 111, LAYERING 82, FAN_IN 69, FAN_OUT 75 |
| `plaus_feat` poblado en todas las celdas | ✅ 12/12 |
| Puente reportado POR TIPOLOGÍA (edges y features) | ✅ (abajo) |
| 0 NaN silencioso; n_tp y n_plausible reportados | ✅ (todas las celdas; GCN 1:50 con n_tp=7, ver nota) |
| Honestidad sobre features demasiado fáciles | ✅ **reportado: 82% en 1.0** (abajo) |

## Paso 1 — Dataset v4
9.549 nodos · 31.606 edges · 20 features · 16.2% ilícito. Densidad receptive field k=2: **mediana 30**
(≥10 ✓). Soporte parejo por tipología (arriba). Features-firma en índices fijos 6-9, cada una **+4 sobre
ruido solo en sus nodos** (verificado: propios +4.0 vs resto ~0), `typology_feature_index={1:6,2:7,3:8,4:9}`.

## Paso 3 — Matriz 4 arqs × 3 escenarios (12 celdas, 337 nodos, base_k=2 uniforme)

| arch | scen | val PR-AUC | estab (Spearman) | plaus edges (F1) | plaus feat (rec@3) | n_tp |
|---|---|---|---|---|---|---|
| GraphSAGE | 1:1 | 0.999 | 0.832 | 0.491 | 0.867 | 30 |
| GraphSAGE | 1:10 | 1.0 | 0.884 | 0.464 | 0.800 | 30 |
| GraphSAGE | 1:50 | 0.997 | 0.934 | 0.488 | 0.433 | 30 |
| GAT | 1:1 | 0.798 | 0.978 | 0.517 | 0.667 | 30 |
| GAT | 1:10 | 0.794 | 0.979 | 0.560 | 0.900 | 30 |
| GAT | 1:50 | 0.770 | 0.991 | 0.545 | 0.967 | 30 |
| GCN | 1:1 | 0.852 | 0.979 | 0.454 | 1.000 | 30 |
| GCN | 1:10 | 0.836 | 0.986 | 0.528 | 0.933 | 30 |
| GCN | 1:50 | 0.661 | 0.958 | 0.560 | 0.857 | **7** |
| TAGCN | 1:1 | 0.999 | 0.772 | 0.501 | 0.933 | 30 |
| TAGCN | 1:10 | 1.0 | 0.826 | 0.470 | 0.833 | 30 |
| TAGCN | 1:50 | 0.998 | 0.918 | 0.386 | 0.667 | 30 |

**Caveat TAGCN (base_k):** el receptive field real de TAGCN es num_layers·K=6; en este grafo denso, k=6
captura ~2.944 nodos (casi todo el grafo) y degenera la plausibilidad (plausF1 cae a 0.06). Para
comparabilidad de scope entre arqs se usa **k=2 uniforme** (elección del v4), aceptando que la explicación
de TAGCN no reproduce del todo su predicción. Es una limitación declarada, no oculta.
**Nota GCN 1:50:** solo 7 TP en validación (el modelo a 1:50 acierta pocos ilícitos → val PR-AUC 0.66); su
celda tiene soporte bajo, reportado.

## Paso 4 — Puente DESCOMPUESTO POR TIPOLOGÍA (lo que pediste)

**Global:** edges Pearson **+0.159** / Spearman +0.214 · features Pearson **−0.081** / Spearman +0.089.

| tipología | n | estab | plaus edges F1 | **r(estab↔edges)** | plaus feat rec@3 | **r(estab↔feat)** |
|---|---|---|---|---|---|---|
| STRUCTURING | 111 | 0.905 | 0.534 | **+0.409** | 0.865 | −0.074 |
| LAYERING | 82 | 0.896 | 0.404 | −0.118 | 0.793 | −0.321 |
| FAN_IN | 69 | 0.930 | 0.492 | +0.081 | 0.957 | −0.063 |
| FAN_OUT | 75 | 0.946 | 0.527 | −0.140 | 0.653 | +0.409 |

**Conclusión honesta (matiza el resultado de la v1):** el puente estabilidad→plausibilidad **NO es robusto
ni uniforme.** El global de edges es débil (+0.16) y al descomponer va de **+0.41 (STRUCTURING)** a **−0.14
(FAN_OUT)**. Además **es inestable entre corridas**: tu primera auditoría halló que lo conducía FAN_OUT
(+0.54); ahora lo conduce STRUCTURING (+0.41) y FAN_OUT sale negativo. Por tanto **no hay una ley clara
"estabilidad ⇒ plausibilidad"**; a lo sumo una asociación débil y dependiente de la tipología. El +0.27
global de la v1 era frágil, como sospechabas. Un resultado nulo/débil es un resultado.

## Honestidad sobre la plausibilidad de FEATURES (como pediste)
`plaus_feat` media **0.819**, con **82% de los nodos en exactamente 1.0**. La firma (+4 sobre ruido) es
**demasiado separable** → la plausibilidad de features es casi trivial (el explainer casi siempre señala la
feature-firma). Interpretación: confirma que el explainer detecta una feature discriminante **cuando la señal
es muy clara**; para un test exigente hay que **atenuar la firma (bajar el +4)**. No se esconde: por eso el
puente por features es ~0 (no hay variación que correlacionar).

## Paso 5 — Contraste Elliptic vs sintético
| | Elliptic | Sintético |
|---|---|---|
| receptive field mediana | ~2-3 nodos | ~30 nodos |
| shift temporal | sí (test ~0.01) | no (test ~0.9+) |
| plausibilidad de subgrafo | **NO medible** | **SÍ medible** (edges 0.49, feat 0.82) |

El aporte diferencial se mantiene: la plausibilidad es medible en el sintético e imposible en Elliptic.

## Hallazgos de régimen (matices)
- En el sintético **denso, GAT y GCN son los más estables** (Spearman ~0.98) — invierte Elliptic (disperso,
  GraphSAGE lideraba). Diferencia de régimen denso/disperso.
- Estabilidad ≠ exactitud: GAT es el más estable pero el menos preciso (val PR-AUC ~0.78 vs ~1.0 de otros).

## Entregables (en `phase1/`)
- `synthetic_aml_v1.pt` (v4, con typology_feature_index), generador y `plausibility.py`.
- `run_phase1_matrix.py`, `analyze_bridge.py` (con descomposición por tipología).
- `results_phase1_v2.csv` + `results_phase1_v2_pernode.csv` (con `plaus_feat`).

## Pendiente / a decidir contigo
- Si el puente por features importa, **atenuar la firma** (p. ej. +1.5 en vez de +4) y re-correr → daría una
  plausibilidad de features no trivial y un puente medible con el Spearman.
- Verificar estos números y redactar la sección de Fase 1 en Overleaf (no se redactó aquí).
