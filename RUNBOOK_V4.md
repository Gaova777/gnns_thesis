# RUNBOOK v4 — Rehacer el eje Elliptic con el diseño corregido

> **Para el Claude de Juan Diego (y para Juan Diego).** Léelo completo antes de ejecutar nada.
> Este documento es autocontenido: explica por qué existe la v4, qué cambió en el código, cómo
> correrla, cómo vigilarla sin perder horas de GPU y qué decidir en cada punto de control.
> Fecha: 28-sep-2026 · Entrega del manuscrito: **jueves 15-oct-2026**.

---

## 0. En una línea

El eje Elliptic se entrenó con las etiquetas invertidas (los 157.205 nodos sin etiqueta como
«lícitos» y los 42.019 lícitos reales fuera). La v4 corrige eso y otros siete problemas de diseño
detectados en la auditoría, y trae un plan de corrida por prioridades con monitoreo automático.
**Nada de la v3 se borra**: la v4 escribe en directorios nuevos (`results_v4/`, `results_models_v4/`,
`runs_v4/`, `results_phase1_v4/`).

## 1. Por qué existe la v4

| # | Problema en v3 | Evidencia | Corrección v4 |
|---|---|---|---|
| 1 | Etiquetas invertidas: PyG codifica `{'unknown': 2, '1': 1, '2': 0}` y el loader asumió 0 = desconocido | `reeval_rocauc.log`: «Licit: 157,205 · Unknown: 42,019» | `src/data/loader.py`: 0 lícita, 1 ilícita, 2 → −1. **Excepción `DataIntegrityError` si los conteos no cuadran** (totales y por partición) |
| 2 | Escenarios mezclaban proporción con tamaño (1:1 entrenaba con 6.924 nodos, nativo con 109.833) y 1:50/1:100 descartaban ilícitas diciendo lo contrario | `src/data/imbalance.py` | Escenarios v4 con lícitas fijas y control de tamaño (§3) |
| 3 | Presupuesto distinto por arquitectura (GCN/SAGE 50 trials × 600 épocas; GAT/TAGCN 8 × 150) | configs B vs C | `configs/experiment_v4.yaml`: **mismo presupuesto para las 4** |
| 4 | Umbral calibrado con la prevalencia del test | `trainer.calibrate_threshold` | Calibración solo con validación |
| 5 | Cada modelo explicaba sus propios aciertos → arquitecturas comparadas sobre nodos distintos | `explain_matrix.py` | **30 ilícitas de validación comunes** a todos los modelos (`results_v4/explain_nodes_v4.json`) |
| 6 | PGExplainer: Spearman = 0,0 era un valor centinela (no produce ranking de variables); entrenado con `train_indices[:50]` sin ilícitos; ~99 % de épocas NaN por nodos sin aristas de entrada | `metrics.py`, `explainer_runner.py` | Estabilidad sobre **aristas**; muestra estratificada sin nodos de grado 0; réplica fallida si todas las épocas son NaN |
| 7 | «GNNShap» del código es Shapley por permutaciones sobre variables, no el GNNShap publicado | `shap_runner.py` | Renombrado a **`ShapleyFeatures`** (alias de lectura para CSV viejos) |
| 8 | Spearman top-20 (145 variables empatadas); Jaccard binario en subgrafos de 0-2 aristas; vacío → 0 | `metrics.py` | `spearman_full` (primaria), `spearman_topk` (sensibilidad), Jaccard solo con ≥ 2 aristas; **NaN con razón**, nunca 0 |
| 9 | La compuerta de calidad no se aplicaba en el análisis a 3 semillas | `tab:ic` | `analyze_elliptic_v4.py` filtra por compuerta (y reporta la versión sin filtro como sensibilidad) |
| 10 | Grafo sintético con atajos: el modelo acertaba sin mirar la tipología | AUC de una variable 0,96 | Generador `shortcut_free` (AUC ≤ 0,65) + **prueba de alineamiento** |

El diseño completo, con las alternativas descartadas, está en el documento «Respuestas a Cristian:
diseño de los experimentos» (PDF en el repo de actas de Alejandro). Si una decisión de este runbook
choca con algo que diga Cristian, **manda lo de Cristian** y se actualiza este archivo.

## 2. Qué NO hacer

- No borrar ni sobrescribir nada de `results_v3/`, `results_seedsweep/`, `results_models_v3/` ni
  los CSV de `phase1/`: el manuscrito actual cita esos archivos hasta que se reescriba.
- No correr `train_matrix.py` con las configs v3 (`experiment_machine*_v3.yaml`): el loader nuevo
  rechaza el mapeo viejo, pero los escenarios v3 no tienen sentido con las etiquetas correctas.
- No «arreglar» un `DataIntegrityError` relajando el assert. Si salta, los datos en `data/raw/` no
  son los oficiales: bájalos de `https://data.pyg.org/datasets/elliptic/` y vuelve a probar.
- No citar cifras de Weber et al. (2019) sin verificarlas en el paper.
- No correr dos orquestaciones sobre el mismo `runs_v4/` (hay un lock; respétalo).

## 3. El diseño que se ejecuta

**Datos (verificados contra el dataset oficial):**

| Partición | Ilícitas | Lícitas | Sin etiqueta |
|---|---|---|---|
| Train (ts 1-34) | 3.462 | 26.432 | 106.371 |
| Val (ts 35-42) | 914 | 9.069 | 27.837 |
| Test (ts 43-49) | 169 | 6.518 | 22.997 |

**Escenarios** (solo cambia la máscara de train; grafo, val y test intactos; submuestreo con
semilla de datos fija 2026, la misma para las 3 semillas de modelo):

| Escenario | Ilícitas | Lícitas | Total |
|---|---|---|---|
| `native` (1:7,6) | 3.462 | 26.432 | 29.894 |
| `1:10` | 2.643 | 26.432 | 29.075 |
| `1:50` | 528 | 26.432 | 26.960 |
| `1:100` | 264 | 26.432 | 26.696 |
| `1:1` | 3.462 | 3.462 | 6.924 |
| `native_size_ctrl` (control de tamaño del 1:1) | 802 | 6.122 | 6.924 |

**Matriz:** 6 escenarios × 4 arquitecturas × 3 balanceos (`none`, `class_weighting`, `focal_loss`)
= **72 configuraciones por semilla**, semillas 42 (con Optuna) y 43/44 (`--reuse-hp`: reusan los HP
de la 42, solo reentrenan).

**Presupuesto común:** 8 trials de Optuna, 150 épocas, paciencia 20, `hidden_dim ∈ {64,128,148}`.

**Compuerta de calidad** (validación, argmax): F1 ≥ 0,30 y MCC ≥ 0,15.

**Explicación:** 30 ilícitas de validación comunes · 5 réplicas por explicador · GNNExplainer
(variables y aristas), PGExplainer (aristas; reentrenado en cada réplica), ShapleyFeatures
(variables). Solo modelos que pasan la compuerta (`--include-gated` como sensibilidad).

**Análisis:** unidad = configuración (media de semillas). H1 desbalance (Friedman + TOST ±0,05 vs
`native` + control `native_size_ctrl` vs `1:1`), H2 arquitectura (Holm sobre los 6 pares +
regresión con `val_pr_auc` como covariable), H3 balanceo (sin `1:1`), control estabilidad ~ calidad.
El resumen dice para cada hipótesis qué rama aplica: *efecto significativo* / *equivalencia (TOST)* /
*no se detectó con esta potencia*.

## 4. Cómo correrlo

Requisitos: `uv` instalado y los tres CSV oficiales en `data/raw/`
(`elliptic_txs_{classes,features,edgelist}.csv`). La primera carga genera `data/processed/`.

### Paso 1 — Chequeo previo (CPU, ~10 s)

```bash
uv run --frozen python scripts/v4/preflight.py
```

Debe imprimir `PREFLIGHT OK`. Verifica conteos, config, presupuesto igual para las 4
arquitecturas, disco ≥ 20 GB, GPU visible y que no haya salidas v3 mezcladas.

### Paso 2 — Prueba de humo en GPU (~15-20 min)

```bash
bash scripts/v4/smoke_v4.sh
```

Corre todo el pipeline en miniatura en directorios aislados (`runs_smoke/`, `results_smoke/`,
`results_models_smoke/`) y termina en `HUMO OK`. **Si no termina en `HUMO OK`, no se lanza la
corrida real.** (Ya pasó en CPU en la máquina de Alejandro el 28-sep; en GPU es la primera vez.)

### Paso 3 — Corrida real

**Modo recomendado: paralelo** (entrena en GPU mientras explica en CPU lo ya entrenado).

Terminal A — entrenamiento, todas las semillas por prioridad:

```bash
bash scripts/v4/run_all_v4.sh --stages train --device cuda
```

Terminal B — explicación en CPU, se repite hasta cubrir todo lo entrenado:

```bash
until [ -f runs_v4/TRAIN_DONE ] && bash scripts/v4/run_all_v4.sh --stages explain --device cpu --resume --skip-preflight --no-checkpoint; do
  bash scripts/v4/run_all_v4.sh --stages explain --device cpu --resume --skip-preflight --no-checkpoint || true
  sleep 1200
done
```

Al terminar la terminal A, crear el marcador: `touch runs_v4/TRAIN_DONE`. `--resume` hace que cada
vuelta salte lo ya explicado; los modelos que aún no existen simplemente no se explican todavía.

**Modo simple** (secuencial, una sola terminal):

```bash
bash scripts/v4/run_all_v4.sh            # P0 → P1 → P2 → P3 con checkpoints
bash scripts/v4/run_all_v4.sh --resume   # retomar tras un corte
bash scripts/v4/run_all_v4.sh --from P2 --resume
```

**Bloques de prioridad** (si falta tiempo, se corta por abajo):

| Bloque | Qué | Por qué en este orden |
|---|---|---|
| P0 | Semilla 42 · GCN + GraphSAGE | Las más rápidas: validan el pipeline real y la compuerta en pocas horas |
| P1 | Semilla 42 · GAT + TAGCN | Completan la semilla 42 para las 4 arquitecturas |
| P2 | Semillas 43/44 · 4 arquitecturas (`--reuse-hp`) | Replicación; sin Optuna, más barata |
| P3 | Estabilidad entre semillas + análisis (CPU) | No usa GPU |

### Paso 4 — Eje sintético (en paralelo, **en la máquina de Alejandro**, solo CPU)

```bash
bash scripts/v4/run_phase1_v4.sh --quick      # 1-2 min: valida el cableado
bash scripts/v4/run_phase1_v4.sh --resume     # corrida completa (~40 h CPU, casi todo TAGCN)
```

Primero corre los atajos y la prueba de alineamiento. **Si una arquitectura no queda alineada**, su
plausibilidad no se interpreta como calidad del explicador (va a anexo). GCN quedó al borde en la
prueba (margen +0,138 con 300 épocas; criterio ≥ 0,10): vigilarlo. Para acortar TAGCN:
`bash scripts/v4/run_phase1_v4.sh --resume -- --nodes 20 --replicas 3` solo para esa arquitectura.

### Paso 5 — Análisis (CPU, minutos)

```bash
uv run --frozen python scripts/v4/cross_seed_stability.py
uv run --frozen python scripts/v4/analyze_elliptic_v4.py
uv run --frozen python phase1/analyze_v4.py
```

Salidas: `results_v4/analysis_v4_summary.txt` (la rama de cada hipótesis), `results_v4/tables/*.tex`
(coma decimal, listas para el manuscrito) y `results_phase1_v4/summary_v4.csv`.

## 5. Plan de monitoreo — para no perder horas de GPU

Todas las etapas escriben eventos en `runs_v4/progress.jsonl` y refrescan `runs_v4/HEARTBEAT_<etapa>`.
El monitor lee ambos:

```bash
uv run --frozen python scripts/v4/monitor_run.py --once          # estado + alertas
uv run --frozen python scripts/v4/monitor_run.py --watch 10      # refresca cada 10 min
uv run --frozen python scripts/v4/monitor_run.py --once --json   # para que lo lea un agente
```

Código de salida: **0** sin alertas · **1** solo avisos · **2** alguna crítica. En modo secuencial,
`run_all_v4.sh` se detiene solo con un 2 en cada checkpoint.

**Claude de Juan Diego: vigila la corrida con un bucle.** Mientras haya corridas activas, ejecuta
cada 30 minutos `monitor_run.py --once --json` (por ejemplo con `/loop 30m`), lee las alertas y
actúa según esta tabla. No esperes a que termine un bloque para mirar.

| Alerta | Nivel | Qué significa | Qué hacer |
|---|---|---|---|
| Latido sin actualizar > 45 min | Crítica | Proceso colgado o muerto | `nvidia-smi` y `ps`; si el proceso murió, relanzar el bloque con `--resume`; si está colgado, matarlo y relanzar |
| ≥ 2 OOM seguidos | Crítica | La siguiente configuración también fallará | Parar ese bloque. Para entrenamiento: relanzar esa arquitectura sola; si persiste, restringir `hidden_dim` a {64,128} **solo para esa arquitectura**, declararlo en el log de decisiones y avisar a Alejandro |
| Tasa de compuerta < 30 % tras 10 configuraciones | Crítica | Casi ningún modelo aprende | **Parar todo.** Revisar `val_pr_auc` y `val_f1` de los meta.json; casi seguro es un bug de datos o de escenario. No seguir gastando GPU |
| Mediana de `val_pr_auc` fuera de [0,20 ; 0,95] | Crítica | < 0,20: no aprende (el azar es ≈ 0,09). > 0,95: sospecha de fuga de información | Parar y revisar; con > 0,95 revisar que val/test no entren al entrenamiento |
| NaN en métrica primaria | Crítica | Métrica rota | Parar la etapa y revisar el primer run_id con NaN |
| Explicador con > 50 % «no_aplica» | Crítica | El explicador no produce lo que se mide | Revisar `reason` en el CSV. En PGExplainer, `spearman_full` NaN es esperado y no cuenta |
| Latido > 15 min | Aviso | Unidad lenta | Solo observar |
| Error aislado no-OOM | Aviso | Una configuración falló | Anotarlo; se reintenta al final con `--resume` |

**Puntos de control con decisión humana** (Juan Diego o Alejandro deciden; el Claude prepara el
resumen con `monitor_run.py --once` y las cifras):

| Punto | Cuándo | Qué se mira | Si sale bien | Si sale mal |
|---|---|---|---|---|
| C0 | Fin del humo | `HUMO OK` | Lanzar P0 | Corregir antes de gastar GPU |
| C1 | Fin de P0 (entrenamiento GCN + GraphSAGE) | Tasa de compuerta, mediana de `val_pr_auc`, tiempo por configuración | Seguir con P1 y recalcular el ETA | Parar y diagnosticar |
| C2 | Fin de la semilla 42 (P0 + P1) | Las 4 arquitecturas con soporte (≥ 5 configuraciones que pasan la compuerta) | Seguir con P2 | Si GAT o TAGCN no tienen soporte, se reporta así; no se «arregla» subiendo presupuesto solo para ellas |
| C3 | 02-oct | ¿Semillas 43/44 terminadas o en curso sin errores? | Análisis completo | Entregar con semilla 42 para las que falten, declarado en límites |

**Log de decisiones:** cualquier desviación de este runbook (reintento, cambio de parámetro, bloque
recortado) se anota en `runs_v4/DECISIONES.md` con fecha, qué se cambió y por qué. El manuscrito se
escribe a partir de ese archivo.

## 6. Cronograma (actualizado 28-sep; no se había corrido nada)

Las horas son estimaciones: el entrenamiento, del runbook anterior en la RTX 4060; la explicación,
de las mediciones en CPU del 28-sep (GNNExplainer ≈ 1,5 s por nodo; ShapleyFeatures ≈ 2 s por nodo;
PGExplainer 3-7 min por modelo). **El C1 da el primer tiempo real y con él se recalcula todo.**

| Fecha | Qué | Quién |
|---|---|---|
| Lun 28-sep | PR con la v4 listo para revisar | Alejandro |
| Mar 29-sep | Revisar y mergear el PR; preflight + humo en GPU; **lanzar P0 esa misma tarde** | Juan Diego (+ su Claude) |
| Mar 29-sep | Lanzar el eje sintético en la máquina de Alejandro | Alejandro |
| Mié 30-sep | C1 por la mañana; P1 corriendo; explicación de P0 en CPU | Juan Diego |
| Jue 01-oct | C2; P2 (semillas 43/44) corriendo | Juan Diego |
| Vie 02-oct | **C3: punto de decisión.** Semillas 43/44 terminando; estabilidad entre semillas y análisis | Los dos |
| Sáb 03-oct | Revisión con Cristian: diseño + primeros resultados | Los tres |
| 04 al 09-oct | Reescribir los caps. 4 a 7 y el resumen con las ramas de discusión ya decididas; figuras desde `analyze_elliptic_v4.py` | Los dos |
| Sáb 10-oct | Versión final a Cristian | Los dos |
| 11 al 14-oct | Correcciones | Los dos |
| **Jue 15-oct** | **Entrega** | — |

**Plan B** si P0 no arranca antes del miércoles 30-sep: semilla 42 para las 4 arquitecturas y
semillas 43/44 solo para GCN y GraphSAGE, declarado en los límites.

## 7. Qué avisar a Alejandro (y cómo)

Un resumen corto al cerrar cada punto de control, con: bloque, configuraciones hechas/total, tasa de
compuerta, mediana de `val_pr_auc`, errores y nuevo ETA. La salida de `monitor_run.py --once` sirve
tal cual. Avisar **de inmediato** ante cualquier alerta crítica que obligue a parar.

## 8. Mapa del código v4

| Archivo | Qué hace |
|---|---|
| `src/data/loader.py` | Etiquetas corregidas + `EXPECTED_COUNTS`, `EXPECTED_SPLIT_COUNTS`, `DataIntegrityError` |
| `src/data/imbalance.py` | `create_v4_scenario`, `V4_SCENARIOS`, reporte y verificación de cada escenario |
| `src/training/trainer.py`, `hyperopt.py` | Umbral solo con validación; callback de tiempo por configuración |
| `scripts/train_matrix.py` | Modo v4, RunLog, sigue ante OOM o timeout, `--reuse-hp`, overrides |
| `src/explainability/v4_explain.py` | Nodos comunes, PGExplainer estratificado, Shapley por lotes |
| `src/stability/metrics.py` | `spearman_full`, `spearman_topk`, `spearman_edges`, `jaccard_edges_topk`, NaN con razón |
| `scripts/explain_matrix.py` | Explicación v4 con `--resume`, rankings `.npz` por nodo y réplica |
| `scripts/v4/cross_seed_stability.py` | Estabilidad entre semillas del modelo |
| `scripts/v4/analyze_elliptic_v4.py` | Pruebas H1/H2/H3, rendimiento, tablas LaTeX, resumen por rama |
| `phase1/synthetic_aml_generator.py` | Modo `shortcut_free` (el legado sigue siendo el default de la función) |
| `phase1/alignment_check.py`, `run_phase1_v4.py`, `analyze_v4.py` | Alineamiento, matriz sintética v4, métricas contra el azar |
| `src/monitoring/progress.py` | `RunLog`: eventos JSONL + latido |
| `scripts/v4/preflight.py`, `monitor_run.py`, `smoke_v4.sh`, `run_all_v4.sh`, `run_phase1_v4.sh` | Chequeo previo, monitor, humo, orquestadores |
| `configs/experiment_v4.yaml` | La config única de la v4 |
| `tests/test_*_v4.py` | Pruebas en CPU |
