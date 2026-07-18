# Estado del código tras la auditoría — respuesta a `README_claude_code_correcciones.md`

**Fecha:** 2026-07-15
**Autor de esta respuesta:** Claude Code (sobre la máquina del estudiante, Ubuntu nativo, RTX 4060 8 GB, entorno uv reconstruido).
**Para:** el auditor externo que revisó los CSVs del Paso 3.

> **Reconocimiento sin peros.** El auditor tiene razón en todo. El README anterior
> (`README_AUDITORIA_CORRECCIONES.md`) afirmaba **"0 OOM / 0 errores": era FALSO**. La verificación usó
> `grep OutOfMemory` (0 aciertos) cuando el texto real del error es `CUDA out of memory` (13 aciertos), y
> se etiquetó como "colapso de estabilidad" lo que en realidad eran **fallos de memoria de GAT**. El
> hallazgo "GAT es el menos estable" queda **RETRACTADO**. Este documento detalla qué se verificó, qué se
> corrigió en código, y qué queda pendiente.

---

## 0. Resumen ejecutivo

| Bloque | Estado |
|---|---|
| **Parte A** (hallazgos del auditor) | ✅ **Todos confirmados** sobre los CSVs reales (evidencia abajo) |
| **Parte B** (correcciones en código) | ✅ B1, B2, B3, B5, B6 **aplicadas y compilan**; ✅ B7 **verificada**; ⚠️ B4 **pendiente** (decisión de alcance); ⚠️ limpieza de citas B7 **pendiente** |
| **Parte C** (re-ejecución) | ⏸️ **NO ejecutada** — esperando decisión de `nodes_per_class` + explainers (el re-run a 50 nodos × 3 explainers en una 4060 son varios días) |
| **Parte D** (conclusiones honestas) | ⏸️ Pendiente — solo tras la Parte C |

**Los CSVs actuales (`results_v3/*.csv`) siguen contaminados** (13 OOM + 6 NaN silenciosos). Se regeneran en la Parte C. Los CSV pre-corrección-1 están en `results_v3/_pre_correccion1/`.

---

## 1. Parte A — verificación de tus hallazgos (todos confirmados)

Verificado con lectura directa de `results_v3/xai-gnn-stability-B-v3.csv` y `-C-v3.csv`.

| # | Hallazgo | Confirmado | Evidencia medida |
|---|---|---|---|
| **A1** | 13 OOM guardados como texto, reportados como éxito | ✅ **Sí** | `stab_error` contiene 13 "CUDA out of memory", **todos GAT**: GNNExplainer ×7 + PGExplainer ×6. El `except Exception` (antes en `explain_matrix.py:401`) los escribía como fila y continuaba. |
| **A2** | 6 NaN silenciosos en GNNShap | ✅ **Sí** | Las 6 filas GNNShap-GAT con Spearman vacío son las **mismas celdas** que hicieron OOM: `1:100/{cw,focal}`, `1:10/{cw,focal,none}`, `1:1/none`. Sin `stab_error`, sin `stab_reason`, sin `stab_n_tp`. |
| **A3** | No se libera GPU entre celdas | ✅ **Sí** | `grep -c empty_cache scripts/explain_matrix.py` = **0** (antes del fix). |
| **A4** | 34 `except` amplios; 3 `except: pass` | ✅ **Sí** | `add_native_figures.py:39,90,134` eran `except: pass`. |
| **A5** | `nodes_per_class: 3` | ✅ **Sí** | `experiment_machineB_v3.yaml:78` y `experiment_machineC_v3.yaml:69` = 3 (hay ~473 TP en val). |
| **A6** | Citas sin verificar | ✅ Verificadas → ver **§3 (B7)** | Una real pero mal citada; otra **inexistente**. |
| **A7** | El fix de PGExplainer NO funcionó | ✅ **Sí** | PGExplainer da **Spearman = 0 en las 51 celdas medibles** ("99% epochs rolled back due to NaN"). El `edge_size=0.005` no lo resolvió en Elliptic. |

**Impacto científico (confirmado):** las 7 celdas GAT "no medibles" en GNNExplainer NO son colapso de
estabilidad — son OOM. El número "GAT = 0.422" salía de **8/15 celdas**; las otras 7 faltaban por hardware.
La comparación GraphSAGE (15/15) vs GAT (8/15) no es limpia. **Retractado.**

**Lo único que sigue en pie (provisional, sin OOM):** GraphSAGE 0.798 (15/15), GCN 0.562 (15/15),
TAGCN 0.457 (12/12). El ranking completo y cualquier afirmación sobre GAT esperan a la Parte C.

---

## 2. Parte B — correcciones aplicadas (con archivo:línea)

Todas verificadas: los 4 archivos de código compilan (`py_compile`), el esquema CSV nuevo importa, y
`torch.cuda.OutOfMemoryError` existe en torch 2.6.0+cu124.

### B1 — separar OOM de otros errores (no reportar OOM como éxito)
`scripts/explain_matrix.py` (~línea 401→410). El `except Exception` único se reemplazó por:
- `except torch.cuda.OutOfMemoryError:` → libera GPU, reintenta en CPU (B3); si CPU también falla,
  registra `stab_reason="cuda_oom"` con `stab_error` explícito.
- `except Exception:` → registra `stab_reason="unexpected_error"` **con traceback** (ya no un NaN mudo).

### B2 — liberar GPU entre celdas/explainers (causa raíz del OOM)
`scripts/explain_matrix.py:437`. `finally: torch.cuda.empty_cache(); gc.collect()` tras **cada** explainer.
`import gc` añadido (línea 24).

### B3 — fallback automático a CPU para la celda que no cabe (GAT completo)
`scripts/explain_matrix.py:410-436`. Al OOM: `torch.cuda.empty_cache()` y reintento con
`_run_one_explainer(model.cpu(), ..., "cpu")`; si tiene éxito, `stab_reason="ran_on_cpu"`. El modelo se
restaura a GPU en `finally`. (El fallback en CPU es más lento pero deja GAT **evaluado completo**, como
pedías en la opción 2 de B3.)

### B5 — cobertura real en cada fila
- `scripts/explain_matrix.py:386` — `stab_n_tp` se escribe **siempre**, incluso en filas de error
  (`_log()` hace `setdefault("n_tp", n_tp)`).
- `scripts/explain_matrix.py:177` — nueva `stab_n_measurable` = nº de nodos/replica-sets con ≥2 réplicas
  usables (para no leer una media de 1 nodo como si fuera de muchos).
- `src/analysis/tracking.py:26-27` — ambas columnas añadidas al `CSV_SCHEMA_FIELDS` fijo.

### B6 — endurecer los `except: pass`
`scripts/add_native_figures.py:39,91,136` — ahora `except Exception as _e: print(...)` (ya no tragan en silencio).

### B7 — verificación de citas (hecha; **corrección en código pendiente**)
- **`arXiv:2602.23599` → EXISTE pero MAL CITADA.** Es *"Normalisation and Initialisation Strategies for
  GNNs in Blockchain Anomaly Detection"* (Dang Sy Duy, Nguyen Duy Chien, Kapil Dev, Jeff Nijsse; feb 2026).
  Trata de **init/normalización**, no de "warm-start priors", y reporta ganancias en **AUPRC (~0.6)**, no el
  *"F1 val 0.85"* que afirma `CONCLUSIONES_v3.1.md`. Aparece en `src/training/hyperopt.py:8,23,65`.
  → **Acción pendiente:** corregir la caracterización y el número en código y tesis. (Nota: su hallazgo real
  —GraphSAGE mejor con Xavier; GAT necesita GraphNorm— **corrobora** nuestro resultado de GraphSAGE.)
- **`He et al. (2026)` → NO SE ENCONTRÓ (parece fabricada).** Aparece en `src/models/tagcn.py:6,25`
  ("validated K=3 as optimal for Elliptic") y `src/stability/metrics.py:143` ("SHAP Concentration metric,
  He et al. 2026"). K=3 es simplemente el default de TAGCN ([Du et al. 2017](https://arxiv.org/abs/1710.10370)).
  → **Acción pendiente:** retirar la afirmación del código y no citarla en la tesis.

> Las citas se **verificaron** pero **aún no se editaron** en el código (para no mezclar cambios de
> integridad con los de la corrida). Se pueden retirar/corregir en cuanto lo apruebes.

---

## 3. Parte C — re-ejecución: **NO ejecutada** (esperando decisión)

`nodes_per_class` sigue en **3** (no se cambió aún — es B4). Motivo: subirlo a 50 y correr los 3 explainers
en una sola 4060 son **varios días** (16× más trabajo/celda, y PGExplainer degenerado consume la mitad del
cómputo para dar Spearman=0). Se le pasó al estudiante la decisión de alcance:

1. **GNNExplainer, 30 nodos (~1 día)** — métrica primaria con soporte estadístico real, GAT completo vía
   empty_cache + fallback CPU; PGExplainer se reporta como degenerado desde los datos actuales.
2. **GNNExplainer + GNNShap, 20 nodos (~2-3 días)** — añade la métrica secundaria.
3. **Los 3, 50 nodos** — la Parte C literal; inviable en tiempo razonable en una 4060.

**Plan de ejecución (cuando se decida):**
1. Fijar `nodes_per_class` en ambos configs (B4).
2. `smoke_test` con la config nueva.
3. **Validación GAT-only** aislada para confirmar que el fix de OOM (B2+B3) evalúa GAT completo
   (0 `cuda_oom`, o si aparece, con `ran_on_cpu`).
4. Re-explicar la matriz.

**Criterio de aceptación (tu Parte C) que el nuevo resumen debe cumplir:**
- El resumen imprime conteos explícitos: N explicadas, N `cuda_oom`, N `unexpected_error`, N `no_true_positives`.
- **0 celdas `cuda_oom`** (GAT completo en GPU o CPU).
- **0 NaN silenciosos** (toda fila NaN con `stab_reason`).
- `stab_n_tp` en TODAS las filas; del orden de decenas, no 3.
- GAT 15/15 medibles (o las no medibles con razón registrada distinta de OOM).

---

## 4. Parte D — conclusiones: pendiente (tras Parte C)

Se respetarán las reglas de redacción (prosa formal en español, términos técnicos en inglés, sin negrillas
en el cuerpo, sin la palabra "empírico", solo `\cite{}` verificadas). Los 7 puntos que exiges se abordarán
con los datos completos, incluida la retractación de los dos hallazgos viejos (tradeoff -0.20 y peak-collapse
1:50→1:100) y la tabla final con celdas medibles + `stab_n_tp` por celda.

---

## 5. Inventario de archivos (estado actual del working tree)

| Archivo | Estado | Cambio |
|---|---|---|
| `scripts/explain_matrix.py` | modificado | B1/B2/B3/B5 (manejo OOM, empty_cache, fallback CPU, n_tp/n_measurable) + Paso 2 previo (TP filtering) |
| `src/stability/metrics.py` | modificado | `<2` réplicas → NaN (Jaccard `:47`, Spearman `:114`) en vez de `1.0` |
| `src/explainability/explainer_runner.py` | modificado | `select_explanation_nodes(only_correct=…)` + `_predicted_illicit_mask` (forward de predicción en CPU) |
| `src/analysis/tracking.py` | modificado | esquema: `pred_*`→`model_test_*`, `+stab_n_tp`, `+stab_n_measurable` |
| `src/analysis/recommendation.py` | modificado | lee `model_test_f1/mcc` |
| `src/analysis/factorial_robust.py` | **nuevo** | `check_design` (guard anti-degeneración) + ART/MixedLM |
| `scripts/add_native_figures.py` | modificado | B6 (`except: pass` → log) |
| `CLAUDE.md` | modificado | quality gate documentado = real (VAL 0.30/0.15) |
| `src/training/hyperopt.py` | **sin tocar** | cita `arXiv:2602.23599` mal caracterizada (B7 pendiente) |
| `src/models/tagcn.py`, `src/stability/metrics.py` | **sin tocar (cita)** | "He et al. 2026" inexistente (B7 pendiente) |
| `configs/*_v3.yaml` | **sin tocar** | `nodes_per_class: 3` (B4 pendiente) |
| `results_v3/*.csv` | **sin regenerar** | aún contaminados (13 OOM + 6 NaN); backup en `_pre_correccion1/` |

---

## 6. Cómo re-auditar mis cambios

```bash
export PATH="$HOME/.local/bin:$PATH"

# 1. Ver todos los cambios marcados
grep -rn "AUDIT FIX" scripts/ src/

# 2. Confirmar que compila e importa
uv run python -c "import py_compile,glob; [py_compile.compile(f,doraise=True) for f in ['scripts/explain_matrix.py','src/analysis/tracking.py','src/stability/metrics.py']]; print('OK')"

# 3. Ver el esquema CSV nuevo
uv run python -c "from src.analysis.tracking import CSV_SCHEMA_FIELDS; print(CSV_SCHEMA_FIELDS)"

# 4. Re-verificar los OOM en los CSV actuales (contaminados, pre-re-run)
python3 -c "import csv,glob; rows=[r for p in glob.glob('results_v3/*.csv') for r in csv.DictReader(open(p))]; print('OOM en stab_error:', sum('out of memory' in (r.get('stab_error') or '').lower() for r in rows))"

# 5. git diff completo
git diff
```

---

## 7. Decisiones abiertas (te las devuelvo)

1. **Alcance del re-run (B4 + Parte C):** ¿nodes_per_class 30 / 20 / 50? ¿qué explainers? (recomendación:
   GNNExplainer @ 30, GAT vía fallback CPU, PGExplainer reportado como degenerado sin re-correr).
2. **Limpieza de citas (B7):** ¿retiro "He et al. 2026" del código y corrijo la caracterización de
   arXiv:2602.23599? (recomendado — integridad de tesis).
3. **PGExplainer:** confirmado degenerado en Elliptic con PyG 2.7. Propuesta: reportarlo como *finding
   metodológico*, no re-correrlo en la matriz.
