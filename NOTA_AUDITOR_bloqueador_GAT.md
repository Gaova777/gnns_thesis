# Nota para el auditor — bloqueador técnico en el re-run de GAT (Decisión 1 / Parte C)

**Fecha:** 2026-07-15
**Contexto:** ejecutando la Opción 1 aprobada (GNNExplainer, `nodes_per_class=30`, GAT completo vía fallback
CPU). Al validar GAT aislado PRIMERO (como pide el plan), apareció un bloqueador que cambia el análisis
costo/beneficio del fallback CPU. Lo detallo para que decidas.

---

## 1. Causa raíz del OOM de GAT (confirmada)

`src/explainability/explainer_runner.py:110` — `explain_nodes` explica **sobre el grafo COMPLETO**:

```python
x = data.x.to(device)              # 203.769 nodos
edge_index = data.edge_index.to(device)   # 234.355 edges
...
explanation = explainer(x, edge_index, index=idx)   # edge_mask sobre TODOS los edges
```

GNNExplainer asigna un `edge_mask` sobre los 234k edges. Para **GAT (8 attention heads)**, la operación
`inputs * edge_mask.view(size)` en el message passing pide **~1.93 GiB por paso**, y el grafo de backward
sobre 100 epochs lleva el pico a **~7.5 GiB** → **OOM en la 4060 (8 GB, ~6.6 GB libres con el escritorio)**.

**GCN / GraphSAGE / TAGCN caben** porque no tienen atención multi-cabeza (sin el factor ×8). Por eso el OOM
es exclusivo de GAT — coincide exactamente con tu hallazgo A1 (13 OOM, todos GAT).

> Es decir: el OOM no es un simple problema de gestión de memoria que se tunee. Es que **se está explicando
> el objeto equivocado**: la predicción de un nodo solo depende de su *receptive field* de k saltos, no de
> todo el grafo. El `edge_mask` sobre los 234k edges es, en su mayoría, sobre edges que no pueden afectar la
> predicción.

## 2. Qué intenté y por qué el fallback CPU no es viable tal cual

| Intento | Resultado |
|---|---|
| **Liberar memoria entre réplicas** (`del explainer; empty_cache` en `stochastic_test.py`) | Acotó la acumulación (la GPU ya no sube de ~1.4 GB entre explicaciones), **pero una sola explicación de GAT sigue pidiendo ~7.5 GiB** → sigue OOM. |
| **Fallback CPU por celda** (tu B3 opción 2) | Funciona pero **lentísimo**: una sola celda GAT (30 nodos × 5 réplicas × 100 epochs sobre el grafo completo en CPU) no terminó en >8 min. Para las 15 celdas GAT serían **~un día o más**. |
| **Fallback CPU por nodo** (solo el nodo que no cabe pasa a CPU) | **Bug de consistencia de device**: tras el OOM parcial en GPU y `model.cpu()`, el reintento en CPU lanza `RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)` — queda estado en GPU. Revertido. |

**Conclusión:** el fallback CPU aprobado (B3) es **correcto en teoría pero inviable en la práctica** en esta
máquina: o tarda ~un día para GAT solo, o tiene un bug de device que habría que resolver además.

## 3. El fix correcto propuesto — explicar sobre el k-hop subgraph

Cambiar `explain_nodes` para que extraiga el **subgrafo de k saltos** del nodo (su computation graph, con
`torch_geometric.utils.k_hop_subgraph`) y explique sobre ese subgrafo, no sobre el grafo completo.

- **GAT cabe en GPU y va rápido** (el subgrafo tiene cientos de edges, no 234k).
- **Es metodológicamente más correcto**: los edges fuera del receptive field no pueden afectar la
  predicción; hoy se les asigna máscara sin sentido. La explicación full-graph es, en rigor, un bug.
- **Trade-off / lo que hay que validar:** re-escopa la explicación para las **4 arquitecturas**. Para
  GCN/GraphSAGE/TAGCN los top-k edges ya deberían estar dentro del receptive field, así que su Spearman
  *debería* cambiar poco — pero **eso hay que verificarlo con datos, no afirmarlo**. Cambia la metodología
  respecto a lo que aprobaste, por eso lo consulto.

## 4. Decisión que necesito de ti

| Opción | Pro | Contra |
|---|---|---|
| **A) k-hop subgraph** (recomendada) | GAT evaluable en GPU, rápido; método más correcto; 15/15 celdas reales | Cambia el scope de explicación de las 4 arqs; requiere implementar + validar que GCN/SAGE/TAGCN no cambian materialmente; cambio de método que debes aprobar |
| **B) GAT en CPU** (lo aprobado) | Sin cambio de método | Fallback CPU con bug de device a arreglar **y** ~un día+ de cómputo para GAT solo; frágil |
| **C) Excluir GAT** | Cero cómputo nuevo; sin cambio de método | GAT queda como "no evaluable en 8GB con GNNExplainer full-graph" (limitación de hardware) / trabajo futuro; el diseño factorial pierde una arquitectura |

**Mi recomendación:** **A**. El OOM de GAT no es realmente hardware — es que explicamos el grafo completo en
vez del receptive field. El subgraph lo arregla de raíz, es más correcto, y con validación de que las otras
tres arquitecturas no cambian materialmente, es de bajo riesgo. Si prefieres no cambiar metodología ahora,
**C** es honesto y rápido (GAT como limitación documentada); **B** es la menos práctica.

## 5. Estado actual del código y los datos (para que no haya sorpresas)

- **Aplicado y verificado:** B1/B2/B3(marco)/B5/B6, B7 (citas: `He et al. 2026` retirada de código;
  `arXiv:2602.23599` re-caracterizada en código y en `CONCLUSIONES_v3.1.md`), `nodes_per_class=30` en ambos
  configs. `del explainer + empty_cache` entre réplicas en `stochastic_test.py`.
- **Revertido:** el fallback CPU por-nodo (bug de device). Queda el fallback CPU por-celda como red de
  seguridad, pero es el que resulta inviable por lentitud.
- **Re-run:** NO completado — GAT lo bloquea. GCN/GraphSAGE/TAGCN correrían bien en GPU.
- **CSVs:** los contaminados (13 OOM) están archivados en `results_v3/_pre_correccion_oom/`; los previos en
  `results_v3/_pre_correccion1/`. `results_v3/` tiene datos parciales de las pruebas de GAT (se regeneran).
- **Nada de esto afecta** la retractación ya acordada del hallazgo "GAT el menos estable" ni los resultados
  provisionales válidos (GraphSAGE 0.798, GCN 0.562, TAGCN 0.457 — pendientes de re-correr a 30 nodos).
