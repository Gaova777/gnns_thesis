# RUNBOOK — Consolidación final de la tesis en el PC principal (GPU)

> Guía operativa paso a paso para dejar la tesis **consolidada y coherente** corriendo en la
> máquina principal con GPU. Escrito tras la auditoría integral de 2026-07-21.
>
> **Idea central que debes tener clara antes de empezar:** consolidar la tesis **NO** implica
> re-correr todo el proyecto. De todo el trabajo pendiente, **solo una cosa necesita GPU** (recomputar
> la estabilidad del eje Elliptic tras un fix de 3 líneas). El resto son ediciones de documento y
> regeneración de tablas a partir de datos que **ya existen**.

---

## 🖥️ HANDOFF — para la instancia de Claude Code en la máquina con GPU

> **Lee esto primero.** Esta parte del trabajo se hizo en la **máquina de análisis (SIN GPU)** en una
> sesión de Claude Code aparte. Lo de abajo ya está **commiteado** en la rama
> **`consolidacion-auditoria`**. Tú corres en el **dispositivo con GPU** — te toca ejecutar los
> procesos pesados.

### Ya hecho y commiteado (rama `consolidacion-auditoria`) — sin GPU
1. **Fix R1** en `src/stability/metrics.py` (`spearman_rank_agreement`): dimensiona el vector de
   rangos por nº de features, no por `top_k`. Idéntico al comportamiento previo con `top_k=None`
   (⇒ el **eje sintético NO cambia**); correcto con `top_k>0` (⇒ afecta al eje Elliptic).
   **⚠️ Falta validarlo con el smoke test en tu máquina (Paso A.verificación).**
2. **Tabla 5.1 reconciliada** (`tesis_latex/tables/synth_factorial.tex`): la columna Fidelity+ ahora
   es uniforme para los 3 explicadores, generada desde `phase1/results_robust_agg.csv`. Cierra la
   contradicción con el texto del Cap. 5. **No requiere re-correr nada** (datos ya existentes).
3. **Nuevo generador** `phase1/gen_tables.py` (solo stdlib; regenera la tabla anterior).
4. **README raíz** actualizado (banner de estado + Hallazgos clave a la narrativa vigente) y **este
   runbook**.
5. **Guion de defensa R2** (`docs/DEFENSA_R2_evidencia_sintetica.md`): respuestas ensayables para la
   objeción "la evidencia fuerte viene del dataset que ustedes construyeron". Material de estudio para
   los autores, no requiere ejecución.

### Cómo traer estos cambios a la máquina GPU
```bash
git fetch origin
git checkout consolidacion-auditoria   # o merge a tu rama de trabajo
# (si la rama aún no está en el remoto, la máquina de análisis debe hacer push primero)
```

### Lo que TE TOCA correr aquí (procesos de mayor complejidad)
En orden — el detalle completo está en las secciones numeradas más abajo:
- **Paso A · verificación** → `scripts/smoke_test.py` para validar el fix R1 (rápido, valida que no rompió nada).
- **Paso B** → **recomputar la estabilidad del eje Elliptic** con la métrica corregida. Es el proceso
  pesado: etapa *explain* sobre las 4 arquitecturas (reusa checkpoints si existen; si no, entrena
  primero — determinista, `seed=42`). **Este es el único paso que realmente necesita GPU.**
- **Paso C** → unificar los CSV de Elliptic.
- **Paso D** → si el Paso B cambió los números, regenerar las tablas de Elliptic (y verificar si el
  ranking de arquitecturas del Cap. 4 sobrevive; ver §9 Impacto).

> Todo lo demás (README, docs retractados, DOI) es sin GPU y puede hacerlo cualquier instancia.

---

## 0. TL;DR — el mapa de qué se toca y qué no

| Bloque | ¿Necesita GPU? | ¿Re-corre experimentos? | Riesgo |
|---|---|---|---|
| **Eje sintético** (`phase1/`, Capítulo 5 — hallazgos principales) | **NO** | **NO — NO TOCAR** | ⚠️ regenerarlo puede desincronizar el manuscrito |
| **Fix R1 + recomputar estabilidad Elliptic** (Capítulo 4) | **SÍ** (solo etapa *explain*) | Solo *explain*, reusando checkpoints | Medio: puede cambiar el ranking de arquitecturas del Cap. 4 |
| **Tabla 5.1 de fidelidad** (cierra el flanco más peligroso) | NO | NO — regenerar desde CSV existente | Ninguno |
| **README + archivar docs retractados** | NO | NO | Ninguno |
| **Verificar DOI He2026** | NO (web) | NO | Ninguno |

**Regla de oro:** el eje sintético (`phase1/`) ya está limpio, es reproducible y coincide con el
manuscrito. **No lo re-corras** durante la consolidación — solo lo pondrías en riesgo.

---

## 1. Preparación del PC principal

```bash
# 1.1 Traer el código actualizado
git pull origin main

# 1.2 Entorno (si es la primera vez en esta máquina)
uv venv
uv sync

# 1.3 Verificar la GPU
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sin GPU')"
```

### 1.4 Diagnóstico: ¿qué artefactos ya existen en esta máquina?

Esto decide cuánto hay que correr en el **Paso B**. Corre este bloque y anota el resultado:

```bash
uv run python - <<'PY'
import os, glob
def n(p): 
    g = glob.glob(p); return len(g)
print("Dataset Elliptic (data/ o dataset/):", os.path.isdir("data"), os.path.isdir("dataset"))
print("Checkpoints Elliptic  results_models_v3/*_best.pt :", n("results_models_v3/*_best.pt"))
print("Metadatos             results_models_v3/*_meta.json:", n("results_models_v3/*_meta.json"))
print("CSV explain Elliptic  results_v3/*.csv            :", n("results_v3/*.csv"))
print("Grafos sintéticos     phase1/*.pt                 :", n("phase1/*.pt"))
print("CSV sintéticos        phase1/results_*.csv        :", n("phase1/results_*.csv"))
PY
```

Interpretación:
- **`*_best.pt` ≈ 60** → tienes los modelos entrenados. En el Paso B **solo corres *explain*** (barato, ~1.5 h).
- **`*_best.pt` = 0** → hay que **entrenar primero**. El entrenamiento está sembrado (`seed=42`), así que
  reproduce los **mismos modelos** de forma determinista; solo cuesta tiempo (~3 h en la 4060).
- **`phase1/*.pt` = 0 pero `phase1/results_*.csv` presentes** → **es lo esperado y está bien**: los CSV
  sintéticos ya contienen todos los resultados del Cap. 5. No necesitas los `.pt`.

---

## 2. PASO A — Aplicar el fix R1 (código, SIN GPU)

**Qué:** el cálculo del Spearman (métrica de estabilidad primaria) tiene un bug cuando se usa con
`top_k > 0` — dimensiona el vector de rangos al tamaño de `top_k` y descarta toda feature cuyo índice
sea mayor. El eje sintético NO está afectado (usa `top_k=None`); el eje Elliptic SÍ.

**Archivo:** `src/stability/metrics.py`, función `spearman_rank_agreement` (~líneas 69-102).

**Código actual (con bug):**
```python
    n = max(len(ranking_a), len(ranking_b))
    ranks_a = np.zeros(n)
    ranks_b = np.zeros(n)
    for pos, feat in enumerate(ranking_a):
        if feat < n:                # ← BUG: descarta features con índice >= top_k
            ranks_a[feat] = pos
    for pos, feat in enumerate(ranking_b):
        if feat < n:
            ranks_b[feat] = pos
```

**Reemplazar por (dimensiona por nº real de features, no por top_k):**
```python
    # Nº real de features = mayor índice presente + 1 (las rankings son argsort de TODAS las features)
    if len(ranking_a) == 0 or len(ranking_b) == 0:
        return 0.0
    n_features = int(max(ranking_a.max(), ranking_b.max())) + 1
    # posición de rango por feature; las no seleccionadas van al "peor" rango (empate al final)
    ranks_a = np.full(n_features, n_features, dtype=float)
    ranks_b = np.full(n_features, n_features, dtype=float)
    top_a = ranking_a[:top_k] if top_k else ranking_a
    top_b = ranking_b[:top_k] if top_k else ranking_b
    for pos, feat in enumerate(top_a):
        ranks_a[int(feat)] = pos
    for pos, feat in enumerate(top_b):
        ranks_b[int(feat)] = pos
```

**Recomendación adicional (consistencia entre ejes):** para que Elliptic y el sintético usen la
**misma** definición de estabilidad (Spearman sobre el ranking completo, como `phase1/`), pon en el
config `stability.top_k_features: null` en `configs/experiment_machineB_v3.yaml` y
`configs/experiment_machineC_v3.yaml`. Así ambos ejes son "apples-to-apples" y lo puedes defender como
métrica idéntica en los dos casos.

**Verificar el fix (sin GPU):**
```bash
uv run python scripts/smoke_test.py --config configs/experiment_machineB_v3.yaml
```
Debe seguir pasando todos los checks. (Si tocas el config a `top_k_features: null`, confirma que el
smoke test no rompe.)

---

## 3. PASO B — Recomputar la estabilidad del eje Elliptic (GPU)

> Objetivo: regenerar el `stab_spearman_mean` de Elliptic con la métrica corregida. Esto **reemplaza**
> los números del Capítulo 4. Jaccard, plausibilidad y fidelidad NO dependen del bug.

El PC principal debe cubrir las **4 arquitecturas**. Se reparten en dos configs (heredado del setup de
2 máquinas). Gracias al fix del subgrafo k-hop, la 4060 ya corre GAT/TAGCN sin OOM.

> ⚠️ **Gotcha de rutas:** `experiment_machineC_v3.yaml` usa `data.root: ./dataset` y
> `experiment_machineB_v3.yaml` usa `./data`. En un solo PC, edita el config C a `data.root: ./data`
> (o copia/enlaza la carpeta) para que ambos usen el mismo dataset.

### Caso 1 — los checkpoints existen (`*_best.pt` ≈ 60): solo *explain*

```bash
# GCN + GraphSAGE (config B)
uv run python scripts/explain_matrix.py --config configs/experiment_machineB_v3.yaml --resume

# GAT + TAGCN (config C, con data.root ya apuntando a ./data)
uv run python scripts/explain_matrix.py --config configs/experiment_machineC_v3.yaml --resume
```

### Caso 2 — no hay checkpoints: entrenar y luego explicar

```bash
# 1) entrenar la matriz (determinista, seed 42) — repite para el config C
uv run python scripts/train_matrix.py   --config configs/experiment_machineB_v3.yaml --max-hours 9 --resume
uv run python scripts/train_matrix.py   --config configs/experiment_machineC_v3.yaml --max-hours 9 --resume

# 2) explicar solo los modelos que pasaron el quality gate
uv run python scripts/explain_matrix.py --config configs/experiment_machineB_v3.yaml --resume
uv run python scripts/explain_matrix.py --config configs/experiment_machineC_v3.yaml --resume
```

Flags útiles para correr por partes: `--arch GAT`, `--scenario "1:50"`, `--balancing focal_loss`,
`--explainer PGExplainer`.

### Verificación del Paso B
- Se generan/actualizan `results_v3/xai-gnn-stability-B-v3.csv` y `...-C-v3.csv`.
- Monitoreo en vivo: `uv run mlflow ui --backend-store-uri sqlite:///mlruns.db`
- Sanity: el pass-rate del quality gate debe seguir siendo **~23/60** (el training no cambió; solo
  cambió la métrica de Spearman). Si el pass-rate cambia, algo más se movió — revisar.

---

## 4. PASO C — Unificar los CSV de Elliptic (SIN GPU)

> ⚠️ **`scripts/merge_results.py` está OBSOLETO** — espera carpetas `results_machineA/B/C` y 144
> configs del pipeline v1. **NO lo uses.** Une los dos CSV de v3 directamente:

```bash
uv run python - <<'PY'
import pandas as pd, glob
files = glob.glob("results_v3/xai-gnn-stability-*-v3.csv")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.drop_duplicates(subset=["scenario","architecture","balancing","explainer"], keep="last")
df.to_csv("results_v3/elliptic_merged_v3.csv", index=False)
print("Filas unificadas:", len(df), "| archivos:", files)
print(df.groupby(["architecture"])["stab_spearman_mean"].mean())
PY
```

Los promedios impresos son el nuevo ranking de estabilidad de arquitecturas sobre Elliptic — **cómpralos
con los del Capítulo 4** (0.630 / 0.486 / 0.468 / 0.270). Si cambian, el Cap. 4 hay que actualizarlo
(ver Paso D y la sección de impacto).

---

## 5. PASO D — Regenerar tablas del manuscrito (SIN GPU)

> ⚠️ **No existe hoy un generador automático de `.tex`** (los archivos en `tesis_latex/tables/` están
> escritos a mano o desde celdas de notebook). Hay que regenerarlos. Dos sub-tareas:

### 5.1 Tabla 5.1 de fidelidad (cierra R3 — el flanco más peligroso)
La contradicción: `tesis_latex/tables/synth_factorial.tex` muestra Fidelity+ solo para GNNExplainer con
la definición vieja (0.93-1.00); el texto y `synth_robust_full.tex` usan la definición uniforme
(GNNExplainer 0.56 / PGExplainer 0.11 / GNNShap 0.46). **Regenerar `synth_factorial.tex` con la columna
de fidelidad de los 3 explicadores** tomada de `phase1/results_robust_agg.csv` (que ya existe, sin GPU).

### 5.2 Tablas de Elliptic (solo si el Paso B cambió los números)
Regenerar desde `results_v3/elliptic_merged_v3.csv`:
`tesis_latex/tables/elliptic_full.tex` (60 configs), `tab:ranking`, `elliptic_stab_scenario.tex`.

> Estas dos regeneraciones conviene automatizarlas con un helper (`phase1/gen_tables.py`) que lea los
> CSV y escupa el `.tex`. **Aún no existe** — pídelo y se crea.

---

## 6. PASO E — Documentación y citas (SIN GPU)

1. ✅ **HECHO** (commit d00d381) — **`README.md` raíz actualizado**: banner de estado + "Hallazgos
   clave" reescritos a la narrativa vigente, con nota de retractación de la narrativa vieja
   (pico-colapso 0.42→0.59→0.24, paradoja del régimen nativo, TAGCN·1:50=0.789, "GAT lidera
   explicación", trade-off −0.20).
2. ✅ **HECHO** — **Docs retractados archivados**: `literature_review.md` y `CONCLUSIONES_v3.1.md`
   movidos a `docs/_archivo/` (con `git mv`, preservando historia) y con banner de OBSOLETO/RETRACTADO
   en la cabecera. En `docs/` solo queda la fuente vigente (`CONCLUSIONES_v3.2.md`) y el guion R2.
3. ✅ **VERIFICADO — PASA** — **DOI de He2026** (`tesis_latex/bibliografia.bib:54-64`,
   `10.1007/s10489-026-07138-9`): registrado en Crossref, es REAL. Título "An explainable graph neural
   network framework for illicit financial transaction detection", Applied Intelligence vol. 56 (2026),
   autores He X/Huang J/Ma K/He H/Li M (coinciden con el .bib). Los números atribuidos (TAGCN+SHAP+
   Elliptic; acc 98.14% / R 86.22% / P 94.23% / F1 90.05% / MCC 0.8913) son los REALES del paper. La
   alerta previa de auditoría queda descartada (el paper aún no estaba indexado cuando se auditó).
   Residual menor: confirmar en el PDF si el término exacto "SHAP Concentration" es del paper.

---

## 7. Checklist final de consolidación

- [x] Fix R1 aplicado en `src/stability/metrics.py` — **smoke test 15/15 en la máquina GPU** (Paso A) ✅
- [ ] (opcional) `top_k_features: null` en ambos configs para consistencia entre ejes
- [x] Estabilidad de Elliptic recomputada (Paso B) — `results_v3/xai-gnn-stability-B-v3.csv` regenerado: 23 passed × 3 explicadores + 37 no-passed × GNNExplainer. **Cero OOM** vía driver troceado (un proceso por config). ✅
- [x] Pass-rate del gate = **23/60** confirmado (sanity) ✅
- [x] CSV de Elliptic: `explain_matrix` globea los 60 meta.json, así que el CSV B único ya contiene los 4 archs; no hace falta merge separado (Paso C) ✅
- [x] Comparado el nuevo ranking vs Cap. 4 → **CAMBIÓ y se actualizó el manuscrito** (Paso C/D). El fix R1 invierte GraphSAGE>GAT a GAT/GCN líderes y **disuelve la "inversión por densidad"**: Elliptic pasa a concordar con el sintético. Reescritos Cap 4/5/6/7. Ver memoria `r1-fix-inverts-graphsage-gat-ranking`. ✅
- [x] Tabla 5.1 de fidelidad regenerada con los 3 explicadores (Paso D.1) ✅ commit d00d381
- [x] Tablas de Elliptic regeneradas (Paso D.2): `elliptic_full`, `elliptic_jaccard`, `elliptic_stab_scenario`, `tab:ranking`, `elliptic_perf` + figuras `ranking_khop`/`estabilidad_escenario`/`contraste_regimen`, todas con métrica corregida ✅
- [x] **Extra (mejora de métricas):** PR-AUC elevado a métrica primaria + bloque **ROC-AUC / precision@k** (contraste honesto: ROC-AUC engañoso bajo imbalance) en Cap 4. Datos en `results_v3/reeval_metrics.csv` (val ROC 0,884 / PR 0,367 vs test ROC 0,653 / PR 0,017) ✅
- [x] **PDF recompilado con biber: 101 páginas, sin errores ni refs/citas undefined** ✅
- [x] README raíz actualizado ✅ (commit d00d381) + docs retractados archivados en `docs/_archivo/` ✅ (Paso E)
- [x] DOI de He2026 verificado (Paso E) ✅ real, Crossref, números coinciden
- [x] Guion de defensa R2 listo — `docs/DEFENSA_R2_evidencia_sintetica.md` ✅ commit 89235c4
- [x] **Eje sintético `phase1/` NO tocado** ✅

---

## 8. Lo que NO debes hacer

- **NO regeneres los `.pt` sintéticos** (`synthetic_aml_v1/v2.pt`). Los CSV que produjeron ya están
  versionados y coinciden con el manuscrito. Regenerarlos con los defaults actuales del generador (que
  tienen historial de tuning) puede dar números distintos y **desincronizar todo el Capítulo 5**.
- **NO re-entrenes el eje sintético** (`phase1/run_phase1_robust.py`) salvo que aceptes actualizar
  todos los números del Cap. 5 y su estadística.
- **NO uses `scripts/merge_results.py`** (obsoleto, v1).
- **NO cambies `seed`, epochs, hidden_dim ni Optuna trials** de los configs "para mejorar" durante la
  consolidación — eso cambia resultados y te obliga a re-escribir capítulos. Si quieres mejorar el
  modelo, es un proyecto aparte, no consolidación.

---

## 9. Impacto esperado de estas correcciones (positivo/negativo)

**Neto: positivo para la integridad y la coherencia del manuscrito, con un único riesgo acotado.**

| Corrección | Efecto | Dirección |
|---|---|---|
| Tabla 5.1 fidelidad (R3) | Elimina una auto-contradicción; refuerza el hallazgo de disociación | ✅ Solo positivo |
| README + archivar docs | Elimina afirmaciones retractadas que contradicen la tesis | ✅ Solo positivo |
| Verificar DOI | Confirma integridad (o detecta a tiempo un problema) | ✅ Positivo / detección temprana |
| **Fix R1 + recomputar Elliptic** | Métrica correcta; **puede cambiar el ranking de arquitecturas del Cap. 4** | ⚠️ Positivo con riesgo acotado |

Sobre el único riesgo (R1): al corregir la métrica, el ranking de estabilidad por arquitectura sobre
Elliptic (GraphSAGE > GAT > GCN > TAGCN) **podría cambiar o diluirse**. Pero:

- Los **hallazgos principales de la tesis (Capítulo 5, sintético) NO se tocan** — el núcleo está a salvo
  pase lo que pase.
- El Capítulo 4 ya es el eje **débil declarado** (1 semilla, sin inferencia, GCN apoyado en n=1). Su
  exposición es limitada: solo "GraphSAGE > GAT" es una comparación firme.
- **Peor caso realista:** el ranking se cae → se re-encuadra como *"corregida la métrica, el eje de
  datos reales, con su baja potencia ya reconocida, no sostiene un ranking de arquitecturas"*. Eso
  **fortalece** tu credibilidad, no la debilita.
- **No corregir NO es opción:** el bug es descubrible por cualquier jurado técnico que lea
  `src/stability/metrics.py`. Es infinitamente mejor llegar con *"lo encontramos y lo corregimos"*.

---

*Runbook generado en la auditoría de consolidación · 2026-07-21*
