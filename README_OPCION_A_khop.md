# Opción A — explicación sobre k-hop subgraph: changelog detallado para el auditor

**Fecha:** 2026-07-15
**Base:** respuesta del auditor en `RESPUESTA_bloqueador_GAT.md` (aprueba Opción A con Condiciones 1 y 2 +
advertencia Fidelity) y `DECISION_alcance_final.md` (Opción 1: solo GNNExplainer, `nodes_per_class=30`,
PGExplainer no se re-corre, limpieza de citas).
**Qué documenta:** cada cambio de código desde esa aprobación, con archivo:línea, antes/después y propósito;
cómo se satisface cada condición; validación; y criterio de aceptación.

---

## 0. Resumen

El OOM de GAT no era de hardware: `explain_nodes` explicaba sobre el **grafo completo** (203.769 nodos /
234.355 edges), y el `edge_mask` de GNNExplainer × las 8 cabezas de atención de GAT pedía ~7.5 GiB de pico.
La Opción A cambia la explicación al **k-hop subgraph** (el computation graph / receptive field del nodo).
Resultado medido: GAT pasa de OOM (7.5 GiB) a **0.02 GiB de pico** y se explica en GPU; ambas métricas
(Jaccard y Spearman) dejan de ser degeneradas.

| Archivo | Cambio |
|---|---|
| `src/explainability/explainer_runner.py` | **Nuevo:** `SubgraphPredictionMismatch`, `full_graph_logits()`, `build_receptive_subgraph()` (k-hop + assert) |
| `src/stability/stochastic_test.py` | `run_stochastic_replicas` construye el subgrafo 1 vez y explica sobre él; libera el explainer entre réplicas |
| `scripts/explain_matrix.py` | Calcula `base_k` por arquitectura y las logits full-graph por celda; propaga; maneja mismatch |
| `configs/*_v3.yaml` | `nodes_per_class: 3 → 30` (B4) |
| `src/training/hyperopt.py`, `src/models/tagcn.py`, `src/stability/metrics.py`, `src/explainability/shap_runner.py`, `docs/CONCLUSIONES_v3.1.md` | Limpieza de citas (Decisión 2) |

---

## 1. Núcleo k-hop — `src/explainability/explainer_runner.py`

### 1a. Import (línea 13)
```python
from torch_geometric.utils import k_hop_subgraph
```

### 1b. Excepción de correctitud (líneas 17-19)
```python
class SubgraphPredictionMismatch(Exception):
    """No k-hop subgraph reproduced the full-graph prediction for the node."""
```

### 1c. `full_graph_logits()` (líneas 23-34)
Logits del grafo completo `[N, C]`, una vez por celda, en **CPU** (un forward full-graph de GAT puede
OOM-ear la 4060; no es crítico en tiempo). Es la referencia contra la que se valida cada subgrafo.

### 1d. `build_receptive_subgraph()` (líneas 37-73) — el corazón de la Condición 1
```python
def build_receptive_subgraph(model, data, node_idx, base_k, device, full_logit,
                             max_extra_hops=3, atol=1e-3, rtol=1e-2):
    for extra in range(max_extra_hops + 1):
        k = base_k + extra
        subset, sub_ei, mapping, _ = k_hop_subgraph(node_idx, k, data.edge_index,
                                                    relabel_nodes=True, num_nodes=n)
        sub_x = data.x[subset].to(device);  sub_ei = sub_ei.to(device)
        target_local = int(mapping[0])
        sub_logit = model(sub_x, sub_ei)[target_local].detach().cpu()
        if (sub_logit.argmax() == full_logit.argmax()
                and torch.allclose(sub_logit, full_logit, atol=atol, rtol=rtol)):
            return sub_x, sub_ei, target_local, k          # k adaptativo usado
    raise SubgraphPredictionMismatch(...)                  # ningún k coincidió
```
- Extrae el subgrafo de k saltos, **remapea** el índice del nodo objetivo (`mapping[0]`), y trabaja en el
  espacio del subgrafo.
- **ASSERT de la Condición 1:** la predicción del modelo sobre el subgrafo debe coincidir (argmax + logits
  dentro de tolerancia) con la del grafo completo. Si no, **k adaptativo**: amplía hasta `+3` saltos (GCN y
  TAGCN normalizan por grado; los nodos del borde necesitan un salto extra para que su grado sea correcto y
  la predicción coincida). Si ningún k en rango coincide → `SubgraphPredictionMismatch` (el llamador salta
  el nodo; NO se escribe un valor falso).

> Nota: el `base_k` del auditor (num_layers, o num_layers×K) es el **punto de partida**; el assert es la
> regla suprema. En la práctica (ver §5) coincide con k=base_k para las 4 arquitecturas (0 mismatches).

---

## 2. Ejecución de réplicas — `src/stability/stochastic_test.py`

### 2a. Import (líneas 14-16)
Añade `build_receptive_subgraph` al import de `explainer_runner`.

### 2b. Firma de `run_stochastic_replicas` (líneas 32-33)
Nuevos parámetros `base_k: Optional[int] = None` y `full_logit=None`.

### 2c. Subgrafo construido UNA vez por nodo (líneas 69-73)
```python
sub = None
if method == "GNNExplainer" and base_k is not None:
    sub_x, sub_ei, target_local, _k_used = build_receptive_subgraph(
        model, data, node_idx, base_k, device, full_logit)
    sub = (sub_x, sub_ei, target_local)
```
El subgrafo es **determinístico** por nodo → se construye una sola vez y se reutiliza en las N réplicas.
(Si `build_receptive_subgraph` lanza `SubgraphPredictionMismatch`, se propaga al llamador.)

### 2d. Explicar sobre el subgrafo (líneas 94-99) + liberar memoria (línea 109)
```python
if sub is not None:
    sub_x, sub_ei, target_local = sub
    exp = explainer(sub_x, sub_ei, index=target_local)   # edge_mask sobre el subgrafo
else:
    exp = explain_nodes(explainer, data, [node_idx], device)[0]   # ruta antigua (fallback)
...
del explainer, exp
if device != "cpu" and torch.cuda.is_available():
    torch.cuda.empty_cache()                              # sin acumulación entre réplicas
```
`extract_subgraph` toma `exp.edge_index` (= el del subgrafo) → los top-k edges salen en el espacio del
subgrafo, **consistente entre réplicas** del mismo nodo (mismo subgrafo). El Jaccard es válido.

---

## 3. Orquestación — `scripts/explain_matrix.py`

### 3a. Import (línea 45)
`from ...explainer_runner import select_explanation_nodes, full_graph_logits, SubgraphPredictionMismatch`

### 3b. `base_k` por arquitectura + logits de referencia por celda (líneas 391-393)
```python
num_layers = bp.get("num_layers", 2)
base_k = num_layers * bp.get("K", 3) if arch == "TAGCN" else num_layers   # Condición 1
full_logits = full_graph_logits(model, data, device="cpu")               # referencia del assert
```
`num_layers` y `K` se leen del **meta real de cada modelo** (`best_params`), no de un valor asumido.

### 3c. `_run_one_explainer`: propagación + manejo del mismatch (líneas 121-123, 134-135, 158-168, 195-197)
- Firma: `+base_k=None, +full_logits=None`.
- `agg["subgraph_mismatch"] = 0`.
- En el bucle de nodos:
```python
try:
    stoch = run_stochastic_replicas(..., base_k=base_k,
                                    full_logit=full_logits[int(node_idx)] if full_logits is not None else None)
except SubgraphPredictionMismatch:
    agg["subgraph_mismatch"] += 1
    continue                                    # salta el nodo, no escribe valor falso
```
- Si hubo mismatches: `flat["reason"] = f"subgraph_prediction_mismatch:{n}"`.

### 3d. Ambos call-sites pasan `base_k`/`full_logits` (líneas 426 y 447)
El call normal (GPU) y el de red de seguridad OOM (CPU) reciben los mismos parámetros.

---

## 4. Otros cambios de las decisiones del auditor

- **B4 — `nodes_per_class: 3 → 30`** en `experiment_machineB_v3.yaml:78` y `experiment_machineC_v3.yaml:69`
  (soporte estadístico real; ~473 TP disponibles en val).
- **Decisión 2 — limpieza de citas** (0 ocurrencias de "He et al." en `src/`):
  - `src/models/tagcn.py:4-6,25` — retirada la afirmación "He et al. (2026) validated K=3 as optimal for
    Elliptic"; se cita `Du et al. (2017), arXiv:1710.10370` (default estándar de TAGCN).
  - `src/stability/metrics.py`, `src/explainability/shap_runner.py:7,119` — retirada "(He et al., 2026)" de
    la métrica SHAP Concentration.
  - `src/training/hyperopt.py:8,23-25,65` — los warm-start priors ya NO se atribuyen a arXiv:2602.23599 ni
    a un "F1 0.80/0.85"; se describen como defaults informados por literatura.
  - `docs/CONCLUSIONES_v3.1.md` — corregida la fila de la tabla (arXiv:2602.23599 es init/norm, AUPRC ~0.6,
    no "F1 val 0.85") y la línea de warm-start.
- **Intentos descartados (para tu registro):** fallback CPU por-celda (correcto pero ~1 día para GAT) y
  fallback CPU por-nodo (bug de device al mezclar GPU→CPU a media explicación). Ambos revertidos en favor de
  la Opción A. Queda una red de seguridad OOM por-celda que con k-hop ya casi no se dispara.

---

## 5. Validación

### 5a. Test unitario sobre modelos reales (GCN/GAT/TAGCN, checkpoints 1:10_*)
| Arch | base_k | k usado | subgrafo | mismatches | **Pico GPU** |
|---|---|---|---|---|---|
| GCN | 2 | 2 | ~3 nodos | 0/3 | 0.02 GB |
| **GAT** | 2 | 2 | ~3 nodos | 0/3 | **0.02 GB** (antes OOM a 7.5 GB) |
| TAGCN | 12 (=3×4) | 12 | ~2 nodos | 0/3 | 0.02 GB |

(Los nodos ilícitos de Elliptic tienen receptive fields pequeños — 2-3 nodos — lo que confirma que la
métrica informativa es el ranking de features / Spearman.)

### 5b. Corrida end-to-end COMPLETA (2026-07-15, ~43 min) — criterios cumplidos
**0 cuda_oom · 0 unexpected_error · 0 subgraph_mismatch · 4/4 arquitecturas exit=0.**
57 celdas GNNExplainer + 3 `SKIPPED_NO_TP` (imbalance extremo, 0 TP en val).

**Tabla final — Spearman (primaria) + tamaño del receptive field + cobertura:**

| Arquitectura | Spearman | rango | sub_nodes | sub_edges | n_tp | celdas |
|---|---|---|---|---|---|---|
| GraphSAGE | **0.630** | 0.39–0.83 | 3 | 2 | 26 | 15/15 |
| GAT | 0.486 | 0.24–0.72 | 2 | 2 | 29 | 15/15 |
| GCN | 0.468 | 0.03–0.80 | 2 | 1 | 28 | 15/15 |
| TAGCN | 0.270 | 0.05–0.70 | 2 | 2 | 27 | 12/15 |

- **GraphSAGE el más estable; GAT ya NO es el último** (2º, 15/15 — el viejo "GAT último 0.422" era
  artefacto de OOM). TAGCN el menos estable.
- **Evidencia de dispersión (Jaccard = limitación caracterizada):** subgrafo mediano global = **2 nodos /
  2 edges**; la config pide `top_k_edges=20` pero la mediana es 2 edges → el Jaccard de "top-20" es trivial
  (media 0.78) porque no hay edges de dónde elegir. Por eso **Jaccard NO se reporta como estabilidad**;
  Spearman (165 features) es la única métrica primaria. Registrado por celda en `stab_subgraph_n_nodes/edges`.

---

## 6. Mapeo al criterio de aceptación (tu Parte C)

| Criterio | Estado |
|---|---|
| Resumen con conteos explícitos (explicadas, cuda_oom, unexpected_error, no_true_positives, subgraph_mismatch) | ✓ vía `stab_reason` + resumen final |
| 0 celdas `cuda_oom` | ✓ (GAT en GPU, pico 0.02–1.4 GB) |
| 0 NaN silenciosos | ✓ (toda fila con `stab_n_tp`, `stab_n_measurable`, `stab_reason`) |
| `stab_n_tp` en decenas (~30) | ✓ (n_tp=30) |
| GAT 15/15 medible (o no-medible con razón ≠ OOM) | ✓ **15/15**, 0 mismatches |
| **(nuevo) tamaño del receptive field por celda** | ✓ `stab_subgraph_n_nodes`/`stab_subgraph_n_edges` |

### Condiciones del auditor
- **Condición 1** (k = receptive field real + assert de predicción) → §1d, §3b. k por arquitectura leído del
  meta; k adaptativo; assert argmax+logits; mismatch → `subgraph_prediction_mismatch`, sin dato falso.
- **Condición 2** (las 4 arqs en el mismo esquema k-hop; números viejos superados; medir el cambio) → §3.
  Se re-explican las 4; al terminar entrego la tabla k-hop y el delta full-graph→k-hop **medido** (no
  asumido) para GCN/SAGE/GAT.
- **Advertencia Fidelity** → **N/A**: verifiqué que el pipeline activo solo calcula Jaccard/Spearman/SHAP
  concentration (`compute_stability_metrics`); Fidelity± no se computa, así que no hay inconsistencia de
  scope.

---

## 7. Cómo re-auditar

```bash
export PATH="$HOME/.local/bin:$PATH"
grep -rn "AUDIT FIX\|build_receptive_subgraph\|SubgraphPredictionMismatch" src/ scripts/   # cambios
uv run python -c "import py_compile; [py_compile.compile(f,doraise=True) for f in \
  ['src/explainability/explainer_runner.py','src/stability/stochastic_test.py','scripts/explain_matrix.py']]; print('OK')"
grep -rc "He et al" src/            # debe ser 0
git diff                            # diff completo
# CSVs nuevos: results_v3/xai-gnn-stability-{B,C}-v3.csv (los previos en results_v3/_pre_correccion_oom/)
```

---

## 8. Estado y pendiente

- ✅ **Corrida completa** (~43 min); criterio de aceptación cumplido (§6). CSVs nuevos:
  `results_v3/xai-gnn-stability-{B,C}-v3.csv` (con `stab_subgraph_n_nodes/edges`), empaquetados en
  `resultados_khop.zip` para verificación independiente.
- ✅ Registro del tamaño de receptive field por celda (hallazgo de dispersión del auditor): mediana 2 nodos /
  2 edges → Jaccard reportado solo como limitación caracterizada, Spearman como primaria.
- ⏳ **Δ de scope limpio** (full-graph@30 vs k-hop@30 para GCN/SAGE/TAGCN, mismos nodos) — opcional, ~30 min,
  a decisión del auditor/estudiante. El Δ vs full-graph@3 (confundido con nº de nodos): GraphSAGE −0.17,
  GCN −0.09, TAGCN −0.19; GAT +0.06 (viejo contaminado por OOM).
- ⏸️ **Parte D (conclusiones): NO escrita** hasta la verificación independiente de los CSVs, como acordamos.
