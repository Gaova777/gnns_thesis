# Fase 1 — Resultados (dataset sintético + plausibilidad). Para la sesión de análisis.

**Fecha:** 2026-07-16
**Qué es:** ejecución completa del `ENCARGO_claude_code_fase1_completo.md` (Pasos 1-5) sobre el generador
sintético (AMLSim quedó bloqueado: MASON v20 sin distribución + graphviz/sudo — declarado como limitación).
Reusa el pipeline corregido (k-hop, GNNExplainer, estabilidad); NO reimplementa nada. Los números están para
que la sesión de análisis los verifique y redacte la sección de Overleaf.

---

## Paso 1 — Dataset sintético + verificación de densidad
`phase1/synthetic_aml_generator.py` (v3: `symmetrize=True` + `n_distractors=3`) → `phase1/synthetic_aml_v1.pt`.
- 9.299 nodos · 29.646 edges · 16 features · 14.0% ilícito.
- Tipologías con ground-truth **por nodo Y por edge**: STRUCTURING, LAYERING, FAN_IN, FAN_OUT.
- **Densidad (criterio de parada):** receptive field k=2 de nodos de tipología = **mediana 31 nodos / 64
  edges** (min 11). CRITERIO ≥10 ✓ (Elliptic era ~2-3). La simetrización es lo que lo hace medible.

## Paso 2 — Integración + smoke
Escalado RobustScaler (fit en train) + GraphSAGE vía `Trainer` (reusado). **VAL pr_auc=0.987, TEST
pr_auc=0.987** (base rate 0.14). Sin shift temporal → test≈val (contraste con Elliptic, cuyo test ~0.01).

## Paso 3 — Matriz reducida (estabilidad + plausibilidad)
`phase1/run_phase1_matrix.py` → `phase1/results_phase1.csv` (4 celdas) + `_pernode.csv` (120 nodos).
2 arquitecturas × 2 escenarios × GNNExplainer, 30 TP de validación/celda, 5 réplicas.

| Arq | Escen | val PR-AUC | Estabilidad (Spearman) | Plausibilidad edges (F1) | n_tp |
|---|---|---|---|---|---|
| GraphSAGE | 1:1 | 0.975 | 0.808 | 0.509 | 30/30 |
| GraphSAGE | 1:10 | 0.987 | 0.843 | 0.493 | 30/30 |
| GAT | 1:1 | 0.864 | 0.971 | 0.545 | 30/30 |
| GAT | 1:10 | 0.867 | 0.975 | 0.564 | 30/30 |

Nota: `edge_plausibility` usa top_k = nº de edges de tipología presentes (comparación balanceada), por eso
precision=recall=f1 por celda. Todos los 30 nodos por celda fueron evaluables (su subgrafo contiene edges de
tipología), gracias a la densidad.

## Paso 4 — Análisis puente: ¿la estabilidad predice la plausibilidad?
**Sí, positivamente.**
- **Nivel de nodo (n=120):** Pearson **+0.274**, Spearman **+0.252**. Split por mediana de estabilidad:
  plausibilidad de nodos estables **0.553** vs inestables **0.503** (**Cohen d = +0.71**, efecto medio-grande).
- **Nivel de celda (n=4):** Pearson **+0.908** (descriptivo; GAT alto-alto, GraphSAGE bajo-bajo).
- Casos: 2 nodos estables-pero-no-plausibles; 0 plausibles-pero-inestables.
- **Salvedades (iguales a Elliptic):** 1 modelo por celda → descriptivo, no ANOVA con pretensión de potencia;
  el n=4 de celda no está potenciado (la correlación robusta es la de nodo, +0.25/d=0.71).

**Matices interesantes (hallazgos en sí):**
- En el sintético **denso, GAT es el MÁS estable y plausible** — INVIERTE el orden de Elliptic (disperso),
  donde GraphSAGE lideraba. Es una diferencia de régimen (denso vs disperso), no una contradicción.
- El explicador más estable/plausible (GAT) **NO es el modelo más preciso** (val PR-AUC 0.86 vs 0.98 de
  GraphSAGE). Estabilidad/plausibilidad de la explicación ≠ exactitud del modelo.

## Paso 5 — Contraste Elliptic vs sintético (aporte diferencial)

| | Elliptic (real) | Sintético (controlado) |
|---|---|---|
| receptive field (mediana) | ~2-3 nodos | ~31 nodos |
| shift temporal | sí (test colapsa) | no (test≈val) |
| TEST pr_auc | ~0.01 (no discrimina) | ~0.99 (discrimina) |
| plausibilidad de subgrafo | **NO medible** (dispersión) | **SÍ medible** (ground-truth) |
| estabilidad (Spearman features) | medible | medible |
| plausibilidad edges F1 | no aplicable | 0.53 (medible) |

La plausibilidad de subgrafo es **imposible en Elliptic** (vecindarios de ~2 nodos) y **medible en el
sintético** (densidad controlada + ground-truth de tipología). Ese contraste es el argumento del capítulo y
lo que justifica el trabajo en pareja. Diferencia de régimen a declarar: el sintético no tiene shift temporal.

---

## Entregables (en `phase1/`)
1. `synthetic_aml_v1.pt` — dataset + ground-truth (typology_node/typology_edge).
2. `synthetic_aml_generator.py` (v3), `plausibility.py` — generador y métrica (validados por análisis).
3. `run_phase1_matrix.py`, `analyze_bridge.py` — orquestación y análisis (reusan el pipeline).
4. `results_phase1.csv` + `results_phase1_pernode.csv` — estabilidad Y plausibilidad por celda/nodo.

## Pendiente / decisiones para la sesión de análisis
- **Plausibilidad de FEATURES** (secundaria, puente más directo con el Spearman): requiere inyectar features
  específicas por tipología con índice conocido en el generador (NOTA del encargo). No hecho aún; la de edges
  (primaria) ya responde el puente. Decidir si se añade.
- Verificar los números y redactar la sección de Fase 1 para Overleaf (reglas de redacción de la tesis; no se
  redactó aquí, como se acordó).
- AMLSim como confirmación de robustez queda pendiente de que el estudiante corra `sudo apt install graphviz
  graphviz-dev` (y de compilar MASON v20 desde fuente).
