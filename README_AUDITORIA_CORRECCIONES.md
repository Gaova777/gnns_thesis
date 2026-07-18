# Auditoría y correcciones — Fase 0 (estabilidad XAI en GNNs / Elliptic)

**Fecha:** 2026-07-14
**Estado:** entorno restaurado · Paso 1 + Paso 2 aplicados y verificados · listo para Paso 3 (re-explicar)
**Alcance de este documento:** deja registro completo de (a) el diagnóstico corregido, (b) las
modificaciones aplicadas al código, (c) cómo verificar y ejecutar, y (d) lo que queda pendiente.

---

## 0. TL;DR

- Los modelos **sí aprenden en validación** (VAL PR-AUC medio **0.339**), pero **colapsan en test**
  (TEST PR-AUC **0.013** ≈ aleatorio). No es subentrenamiento: es **shift temporal** (el test son los
  timesteps 43–49 de Elliptic, tras el cierre del *dark market* en ts43).
- El **runbook v1** ("cambiar `early_stop_metric` de `f1` a `pr_auc`") estaba **mal diagnosticado**, y su
  criterio de éxito (Paso 4) comparaba TEST-vs-VAL → **éxito falso garantizado**. Retractado por su autor.
- **Encuadre elegido:** estudio de **estabilidad XAI sobre modelos que aprenden en validación**, midiendo
  estabilidad sobre **verdaderos positivos** en `val_mask` (no test).
- Se corrigieron los **bugs de medición en código** (sin reentrenar) y se aplicó la **Corrección 1**
  (filtrado a verdaderos positivos). El entorno (`uv` + Python 3.12 + torch/PyG) fue reconstruido porque
  el upgrade a Ubuntu 26.04 lo había roto.

---

## 1. Contexto

Tesis de maestría sobre **estabilidad de métodos XAI** (GNNExplainer, PGExplainer, GNNShap) aplicados a
GNNs (GCN, GraphSAGE, GAT, TAGCN) para detección de transacciones ilícitas en el dataset **Elliptic**,
bajo desbalance (1:1, 1:10, 1:50, 1:100, y el nativo ~1:30).

El pipeline v3 está dividido en dos etapas:
- `scripts/train_matrix.py` → entrena, calibra umbral, escribe `results_models_v3/{run_id}_meta.json` + `.pt`.
- `scripts/explain_matrix.py` → corre explainers + estabilidad solo sobre modelos que pasan el quality gate.

Esta auditoría partió de una preocupación válida: los resultados de estabilidad tenían filas con
`Jaccard=1.0` y `F1=0.0` a la vez, y una versión previa del diagnóstico afirmaba que "los modelos no
discriminan (VAL PR-AUC ~0.06)".

---

## 2. Diagnóstico corregido (con evidencia empírica)

Todos los números se leyeron de los **60 `results_models_v3/*_meta.json`** ya presentes en disco.

### 2.1. Los modelos aprenden en val y colapsan en test

| Métrica (media de 60 modelos) | Valor | Lectura |
|---|---|---|
| **VAL PR-AUC** (Optuna) | **0.339** (máx 0.559) | discriminan en validación |
| VAL F1 / VAL MCC | 0.264 / 0.265 | consistente |
| **TEST PR-AUC** | **0.013** (máx 0.054) | casi aleatorio (base rate ≈ 0.007) |
| TEST F1 / TEST MCC | 0.012 / 0.006 | colapso total |

Por escenario (VAL Optuna PR-AUC → TEST PR-AUC): 1:1 `0.359→0.016` · 1:10 `0.364→0.017` ·
1:30_native `0.342→0.011` · 1:50 `0.343→0.011` · 1:100 `0.285→0.010`.
Por balancing (VAL→TEST): none `0.311→0.012` · class_weighting `0.374→0.013` · focal_loss `0.331→0.014`.

El colapso es **uniforme** en todos los escenarios (incluido 1:1) y balanceos → apunta a **generalización
(shift temporal)**, no a la técnica de balanceo ni al entrenamiento. El propio `docs/CONCLUSIONES_v3.1.md`
ya lo reconocía (limitación #4: "Temporal shift no resuelto").

### 2.2. La trampa del Paso 4 del runbook v1

El chequeo de éxito del runbook v1 leía:
- `test_metrics.pr_auc` cuando `early_stop_metric == "f1"` (estado actual)
- `best_val_score` cuando `early_stop_metric == "pr_auc"` (tras el `sed`)

Reproducido sobre los datos actuales:

```
AS-IS  (early_stop_metric='f1'):   "VAL PR-AUC proxy" = 0.013   (en realidad lee TEST)
DESPUÉS (early_stop_metric='pr_auc'): "VAL PR-AUC"      = 0.339   (best_val_score = VAL)
```

El "salto" 0.013→0.339 es un **cambio de dataset (test→val), no una mejora**. Habría cantado éxito sin que
nada mejorara en test. **Corrección 0 quedó descartada** como arreglo principal.

### 2.3. El artefacto "Jaccard=1.0 & F1=0.0"

En los CSV de estabilidad: **100% de los Jaccard computables valen exactamente 1.0** (B: 32/47 filas
reales; C: 18/27; `std=0`). Dos mecanismos distintos:

| Explicador | Jaccard | Spearman | Causa | ¿Lo arregla 1a? |
|---|---|---|---|---|
| **PGExplainer** | 1.0 | **0.0 universal** | mode collapse / réplicas que fallan → `<2` → 1.0 hardcodeado | **Sí** (→ NaN) |
| **GNNExplainer** | 1.0 | varía 0–0.79 | top-k determinístico en grafo de 234k edges | **No** (es real) |

Y la "F1=0.0" de esas filas **no era del explicador**: era el **F1 en test del modelo** (`pred_f1`) copiado
en cada fila. Por eso se ve `Jaccard=1.0 & F1=0.0` a la vez en modelos colapsados.

---

## 3. Entorno

### 3.1. Qué pasó
El upgrade a **Ubuntu 26.04 (Python 3.14)** desinstaló `uv` y dejó inservible el `.venv` (construido con
Python 3.12). Ningún `uv run …` corría.

### 3.2. Cómo se reconstruyó (2026-07-14)
1. `uv` reinstalado (installer oficial) → queda en **`~/.local/bin/uv`** (⚠ **no** en PATH).
2. `uv python install 3.12` (intérprete gestionado por uv, independiente del 3.14 del sistema).
3. `.venv` viejo borrado + `uv sync --locked` (desde `uv.lock`).

**Verificado:** `torch 2.6.0+cu124 · CUDA True (RTX 4060 8GB) · PyG 2.7.0`. Dataset ya local en `./dataset/`.

### 3.3. Cómo correr (recordatorio)
```bash
export PATH="$HOME/.local/bin:$PATH"     # uv no está en PATH por defecto
uv run python scripts/smoke_test.py --config configs/experiment_machineB_v3.yaml
```

---

## 4. Correcciones aplicadas

Todas en **código**, sin reentrenar. Verificadas con tests unitarios + smoke 15/15 + trial real.

### Paso 1 — bugs de medición

**1a · `<2` réplicas → NaN (no `1.0` falso).**
`src/stability/metrics.py` — `pairwise_jaccard` y `pairwise_spearman` devolvían `{"mean":1.0,...}`
hardcodeado cuando sobrevivían menos de 2 réplicas (inyectaba estabilidad perfecta artificial). Ahora
devuelven **NaN** + `reason:"insufficient_replicas"`.
`scripts/explain_matrix.py` — la agregación por celda ahora es **nan-aware**: descarta nodos no medibles
antes de promediar (un nodo NaN ya no envenena ni inventa la media; si *todos* son NaN, la columna queda
vacía en el CSV en vez de un valor falso).

**1b · `pred_*` → `model_test_*` + nueva columna `stab_n_tp`.**
`src/analysis/tracking.py` (`CSV_SCHEMA_FIELDS` + `log_run`) y `src/analysis/recommendation.py`. Las
columnas predictivas de cada fila de estabilidad **eran el F1/MCC global del modelo en test**, y se leían
como si fueran del explicador. Renombradas a `model_test_*` para que sean inequívocas. Se añadió
**`stab_n_tp`** = nº de verdaderos positivos que respaldan cada estimación de estabilidad (transparencia
de cobertura).

**1c · Quality gate documentado = real.**
`CLAUDE.md`. El doc decía F1≥0.70 / MCC≥0.40; el gate real (en los configs v3) es **VAL F1≥0.30 /
MCC≥0.15 sobre validación**. Alineado, con nota recomendando —para futuros reentrenamientos— gate sobre
**VAL PR-AUC** en vez de F1/MCC-en-argmax (degenerados bajo imbalance).

### Paso 2 — Corrección 1 (estabilidad solo sobre verdaderos positivos)

**`src/explainability/explainer_runner.py`** — `select_explanation_nodes` reescrita (PARCHE 1):
parámetros nuevos `model`, `threshold`, `only_correct`, `device`; con `only_correct=True` restringe a
**verdaderos positivos** (ilícito real y predicho ilícito) y verdaderos negativos. Retrocompatible.
Devuelve `coverage` (disponibles vs. que pasan el filtro, por clase).

**`scripts/explain_matrix.py`** — punto de llamada corregido respecto al parche original (que apuntaba a
`test_mask` y no saltaba de verdad):
- `mask_name="val_mask"` → explicar donde el modelo **sí** acierta (no test, donde hay ~0 TP).
- `threshold=None` (argmax) → el umbral calibrado para test (~0.87) es demasiado conservador para val.
- **`continue` real** + fila `SKIPPED_NO_TP` (`stab_reason="no_true_positives"`) cuando hay 0 TP: no se
  mide "estabilidad" de predicciones incorrectas.
- Se registra `n_tp` en cada fila de estabilidad.

### Corrección 3 — estadística robusta

**`src/analysis/factorial_robust.py`** (nuevo, PARCHE 3). Lo clave es **`check_design`**: un guard que
devuelve `ok=False` cuando la respuesta es casi constante (var≈0) o hay celdas con `n<2` → **no interpretar
F/p** en esos casos. (Nota: `run_mixed_model` requiere datos desagregados por nodo que hoy el pipeline no
guarda; `run_art_anova` es una implementación no estándar de ART — usar con cautela. `check_design` sí es
sólido y es el guard a usar antes de cualquier ANOVA.)

### Limpieza

Los CSV pre-fix (esquema viejo + datos con el artefacto) se archivaron en
**`results_v3/_pre_correccion1/`** para que el re-explicar escriba limpio con el esquema nuevo (si no, se
corrompían al hacer *append*).

---

## 5. Verificación

- **Smoke test v3: 15/15** checks (incluye el esquema CSV nuevo, node selection, GNNExplainer,
  PGExplainer, GNNShap, calibración de umbral, focal alpha, warm-start, pr_auc, metadata JSON).
- **Tests unitarios** (entorno real): NaN en `<2` réplicas; `1.0` genuino preservado para réplicas
  idénticas; filtrado TP correcto; `check_design(ok=False)` sobre datos degenerados.
- **Trial de una celda** (`GCN 1:10 / GNNExplainer`, `--force`) en el `explain_matrix.py` real:
  - Cobertura TP en val: **class_weighting 473/914**, **focal_loss 365/914** → amplia.
  - Fila producida con el **esquema nuevo**:

    | scenario | arch | balancing | explainer | model_test_f1 | stab_jaccard_mean | stab_spearman_mean | stab_n_tp |
    |---|---|---|---|---|---|---|---|
    | 1:10 | GCN | class_weighting | GNNExplainer | 0.0154 | 1.0 | **0.8055** | 3 |

  - Confirma lo esperado: **Spearman informativo** (0.81), **Jaccard sigue 1.0** (determinístico; 1a no lo
    toca), `n_tp` y `model_test_*` correctos.

---

## 6. Cómo ejecutar el pipeline corregido

```bash
export PATH="$HOME/.local/bin:$PATH"

# Paso 3 — RE-EXPLICAR (no reentrena; usa los 60 checkpoints .pt existentes)
uv run python scripts/explain_matrix.py --config configs/experiment_machineB_v3.yaml --force
uv run python scripts/explain_matrix.py --config configs/experiment_machineC_v3.yaml --force
#   (--force para no depender del quality gate; quitarlo si se prefiere filtrar)
#   Filtros útiles: --arch GCN  --scenario "1:10"  --explainer GNNExplainer

# Paso 4 — ANÁLISIS HONESTO (antes de cualquier ANOVA)
#   - revisar cobertura TP por celda (columna stab_n_tp)
#   - confirmar que aparecen NaN (insufficient_replicas) en vez de 1.0 falsos
#   - correr src/analysis/factorial_robust.py::check_design; si ok=False, no interpretar F/p
```

> Nota: se re-explica escribiendo CSV fresco en `results_v3/` (los previos están en
> `results_v3/_pre_correccion1/`).

---

## 7. Pendiente

- **Paso 3** — re-explicar la matriz completa (B + C). Listo para lanzar.
- **Paso 4** — análisis de estabilidad con cobertura TP + estadística robusta (`check_design`).
- **Decisión de alcance (con el director):**
  - **A)** Tesis de estabilidad XAI sobre modelos que aprenden en validación *(este encuadre; sin cómputo
    nuevo)*. **Recomendado ahora.**
  - **B)** Atacar el shift temporal con un modelo temporal (EvolveGCN/TGN) o evaluando sin el corte
    post-ts43. *Cambio de alcance mayor; trabajo futuro.*

---

## 8. Caveats vigentes (para la tesis)

1. **PGExplainer sigue degenerado** en Elliptic con PyG 2.7 (Jaccard 1.0 + Spearman 0 universal, 99% NaN
   epochs). Reportarlo como **finding metodológico**, no como dato de estabilidad.
2. **El Jaccard de GNNExplainer es determinístico** (top-k sobre 234k edges) → varianza 0; **usar Spearman
   como métrica primaria** (varía 0–0.79).
3. **Se explica sobre validación, no test.** Val se usó en early-stopping y selección de HPs, así que hay
   un **sesgo optimista leve** que debe declararse en la metodología.
4. **Citas por verificar** en el código (posteriores al corte de conocimiento; no confirmables):
   `arXiv:2602.23599` (`src/training/hyperopt.py`) y `He et al. (2026)` (`src/models/tagcn.py`,
   `src/stability/metrics.py`). Una cita inventada en la tesis es un problema serio → verificar en arxiv.org.

---

## 9. Archivos modificados / creados

| Archivo | Cambio |
|---|---|
| `src/stability/metrics.py` | 1a — Jaccard y Spearman: `<2` réplicas → NaN + `reason` |
| `scripts/explain_matrix.py` | 1a agregación nan-aware · Paso 2 callsite (val_mask, threshold=None, skip real, n_tp) |
| `src/explainability/explainer_runner.py` | Corr. 1 — `select_explanation_nodes` con `only_correct`/`coverage` + helper `_predicted_illicit_mask` |
| `src/analysis/tracking.py` | 1b — `pred_*`→`model_test_*` en esquema y `log_run` + `stab_n_tp` |
| `src/analysis/recommendation.py` | 1b — lee `model_test_f1`/`model_test_mcc` |
| `src/analysis/factorial_robust.py` | **Nuevo** (PARCHE 3) — `check_design`, MixedLM, ART-ANOVA |
| `CLAUDE.md` | 1c — quality gate documentado = real (VAL 0.30/0.15) |
| `results_v3/_pre_correccion1/` | **Nuevo** — CSVs pre-fix archivados |

---

## 10. Apéndice — números clave

- **Modelos:** 60 (`results_models_v3/`), todos con `early_stop_metric="f1"`, `optuna_metric="pr_auc"`;
  quality_passed **23/60** (gate VAL F1≥0.30 / MCC≥0.15).
- **Dataset Elliptic:** 203,769 nodos · 234,355 edges · 165 features · 4,545 ilícitos.
  Splits temporales: train ts1–34 (3,462 ilícitos), val ts35–42 (914 ilícitos), test ts43–49 (169 ilícitos).
- **Entorno:** torch 2.6.0+cu124 · CUDA True · PyG 2.7.0 · `uv` en `~/.local/bin/uv`.
- **Artefacto de estabilidad (pre-fix):** 100% de Jaccard computables = 1.0; PGExplainer Spearman = 0
  universal; GNNExplainer Spearman 0–0.79.

---

## 11. Resultados del Paso 3 (re-explicación corregida)

**Fecha:** 2026-07-15 · **Cómputo:** ~9.7 h (4 arquitecturas + resume, 4060).

> 🔴 **CORRECCIÓN / RETRACTACIÓN — auditoría externa 2026-07-15.** La versión previa de esta sección afirmaba
> **"0 OOM / 0 errores". Es FALSO.** La verificación usó `grep OutOfMemory` (0 aciertos) cuando el texto real
> del error es `CUDA out of memory` (13 aciertos). Los CSVs contienen **13 fallos CUDA OOM en la columna
> `stab_error`, TODOS en GAT** (GNNExplainer ×7, PGExplainer ×6), capturados por un `except Exception` y
> escritos como fila en lugar de crashear. Además, **6 celdas GAT de GNNShap quedaron como NaN silencioso**
> (sin `stab_error`, sin `stab_reason`, sin `stab_n_tp`). **Consecuencia:** las "7 celdas GAT no medibles" NO
> son colapso de estabilidad — son fallos de memoria. **El número de GAT (0.422 sobre 8/15) es un artefacto de
> hardware, no un hallazgo**, y la comparación GraphSAGE (15/15) vs GAT (8/15) NO es limpia. **Queda
> RETRACTADO** hasta re-ejecutar con GAT evaluado completo. El fix de CPU del Paso 2 solo cubría el forward de
> predicción de TP, NO los explainers (que siguen en GPU y OOM-ean con GAT).

**Cobertura real:** de 60 celdas, GAT falló por OOM en 13 explicaciones + 6 GNNShap NaN silenciosos. Solo
**GCN, GraphSAGE y TAGCN quedaron completos.** El plan de corrección (separar OOM, liberar GPU, fallback CPU,
subir `nodes_per_class`) está en `README_claude_code_correcciones.md` (Partes B–D) y se aplica antes de un re-run.

### Resultado que SÍ se sostiene (arquitecturas sin OOM) — provisional

| Arquitectura | Spearman medio | celdas medibles | ¿válido? |
|---|---|---|---|
| GraphSAGE | 0.798 | 15/15 | ✔ |
| GCN | 0.562 | 15/15 | ✔ |
| TAGCN | 0.457 | 12/12 | ✔ |
| ~~GAT~~ | ~~0.422~~ | 8/15 | ✗ **contaminado por OOM — pendiente re-run** |

No se puede afirmar el ranking completo (ni "GAT el menos estable") hasta que GAT se evalúe en las 15 celdas.
Además, con `nodes_per_class=3` toda media por celda descansa en solo 3 nodos (habiendo ~473 TP en val) →
soporte estadístico frágil; subir a decenas antes de conclusiones.

**Por balanceo:** focal_loss 0.627 ≈ none 0.613 > class_weighting 0.536.
**Por escenario:** 1:100 0.820 · 1:1 0.601 · 1:10 0.549 · 1:50 0.542 · 1:30_native 0.466.

### Validez estadística (`check_design`, Paso 4c)

| Métrica | `ok` | var | veredicto |
|---|---|---|---|
| **Spearman** | **True** | 0.090 | analizable |
| Jaccard | False | 0.000 | **NO interpretar** (degenerado, todo = 1.0) |

Limitación registrada: **1 seed por celda** → sin réplica intra-celda (50 celdas con `n<2`); para un ANOVA
factorial con potencia se necesitan más seeds. Los **marginales por factor** (arriba) sí son válidos.

### Estado de los explainers (confirmado sobre toda la matriz)

- **PGExplainer:** degenerado universal — Spearman = 0 en 47/47 celdas medibles ("99% epochs rolled back
  due to NaN"). → **finding metodológico** (bug PyG 2.7), no dato de estabilidad.
- **GNNShap:** informativo — 41/57 celdas con Spearman ≠ 0. Métrica secundaria usable.
- **GNNExplainer:** la fuente sólida — Spearman con varianza real, discrimina por arquitectura/balanceo.
- **Jaccard:** degenerado (todo lo medible = 1.0) → **Spearman es la métrica primaria.**

### Salidas
- `results_v3/xai-gnn-stability-B-v3.csv` (GCN, GraphSAGE — 90 filas)
- `results_v3/xai-gnn-stability-C-v3.csv` (GAT, TAGCN — 85 filas)
- CSVs pre-fix en `results_v3/_pre_correccion1/`.
