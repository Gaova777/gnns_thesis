# Guía del proyecto — XAI Stability in GNNs (AML / Elliptic)

## Qué es este proyecto

Tesis de maestría (UTP, MISC, 2026) de Alejandro Gómez Huertas y Juan Diego Garzón Ovalle, dirigida
por el Ph.D. Cristian Rosero Arias. Estudia la **estabilidad de métodos XAI** (GNNExplainer,
PGExplainer, GNNShap) aplicados a GNNs (GCN, GraphSAGE, GAT, TAGCN) entrenadas para detectar
transacciones ilícitas en el dataset Elliptic, bajo distintos niveles de desbalance de clases
(1:1, 1:10, 1:30 nativo, 1:50, 1:100).

El estudio se articula en **dos ejes**: Elliptic real (validez externa, solo permite medir
estabilidad) y un **grafo sintético propio con ground-truth por arista** (`phase1/`, validez interna,
donde sí se pueden medir plausibilidad y fidelidad).

> **La fuente de verdad de los hallazgos es el manuscrito**, no este archivo ni los CSV sueltos:
> `tesis_latex/main.pdf` (103 páginas, 8 capítulos). El README de la raíz es el resumen vigente.

## Estado actual (2026-07-23)

La consolidación posterior a la auditoría está **cerrada**. La rama `consolidacion-auditoria` se
integró a `main` por fast-forward: **ambas apuntan al mismo commit y `main` es la rama de trabajo.**
La rama de consolidación se conserva solo como registro histórico.

### ⚠️ Narrativa retractada (no reintroducir)

Versiones anteriores del README y del manuscrito afirmaban **"inversión por densidad"** y un
**liderazgo de GraphSAGE** en estabilidad. Ambas se **retractaron**: eran artefactos de medición. Si
encuentras esas afirmaciones en algún documento del repo, está desactualizado. Lo vigente:

- Estabilidad Elliptic (Spearman de GNNExplainer, **3 semillas de modelo, 180 modelos**):
  **GAT 0,782 · GCN 0,758 · GraphSAGE 0,735 · TAGCN 0,676**.
- La afirmación correcta es una **partición en dos grupos**, no un ranking de cuatro: grupo alto
  (GAT, GCN) y grupo bajo (GraphSAGE, TAGCN), con diferencias significativas entre grupos y no dentro.
  Kruskal-Wallis global p = 2,8×10⁻⁵; dentro de cada grupo no se rechaza la igualdad.
- La misma partición aparece en el eje sintético, medido de forma independiente. La correlación de
  rangos entre regímenes sigue siendo **+0,80** (el orden de las 4 arquitecturas no cambió).
- **Nunca decir "GAT es la más estable"**: GAT y GCN se permutan entre semillas.
- El pipeline **no es determinista a nivel de pesos** (scatter con atómicos en GPU). Reproduce
  conclusiones, no decimales: el reentrenamiento dio 25/60 sobre el gate frente a 23/60.

## Los dos artefactos de evaluación (contribución central)

Dos defectos del **protocolo de medición**, cada uno suficiente por sí solo para invertir la
conclusión sobre qué arquitectura es más estable:

| Artefacto | Síntoma | Favorecía a | Fix |
|---|---|---|---|
| Memoria / grafo completo | 13 configs de GAT con OOM silencioso; filas vacías promediadas solo sobre las que terminaban | GAT | calcular sobre el **subgrafo receptivo** (7,5 GiB → 0,02 GiB) |
| **Truncamiento de Spearman (fix R1)** | dimensionaba el vector de rangos por `top_k`, descartando toda feature con índice ≥ `top_k`: con `top_k=20` sobre 166 features sobrevivían ~2-3 | GraphSAGE | dimensionar por nº real de features (`src/stability/metrics.py::spearman_rank_agreement`) |

**No revertir el fix R1.** No tiene test de regresión y el smoke test no lo cubre; el eje sintético
no está afectado porque usa `top_k=None`.

## Reglas duras

- **`phase1/` (eje sintético) NO se re-corre.** Sus CSV ya coinciden con el Capítulo 5; regenerarlos
  solo puede desincronizar el manuscrito.
- **La estabilidad se mide sobre los verdaderos positivos de validación** (`mask_name="val_mask"` en
  `scripts/explain_matrix.py`), no sobre test: el clasificador colapsa en test por el shift temporal
  y explicar predicciones erradas no informa. Está declarado abiertamente en el manuscrito (§4.4).
- **PR-AUC y precision@k son las métricas primarias**, no F1 ni ROC-AUC (el ROC-AUC se ve
  engañosamente alto bajo desbalance: 0,88 en validación contra PR-AUC 0,37).

## Pipeline v3 — dividido en dos scripts

La v2 producía modelos con F1 ~0,02-0,08 (la literatura reporta 0,70-0,85). Se corrigieron tres bugs
centrales y se dividió el pipeline en dos etapas para iterar sobre la calidad del modelo de forma
independiente de los explainers.

### 0. Validar (smoke test, ~10 min)

```bash
uv run python scripts/smoke_test.py --config configs/experiment_machineB_v3.yaml
```

Debe pasar **15/15 checks**: estructura de config, quality gate, dataset, escenario de desbalance,
entrenamiento, selección de nodos, los 3 explicadores sin crash/OOM, schema CSV, semántica del alpha
de FocalLoss, warm-start priors, PR-AUC, schema del metadata JSON y calibración de threshold.

### 1. Entrenar la matriz

```bash
uv run python scripts/train_matrix.py --config configs/experiment_machineB_v3.yaml --max-hours 9
```

- Optuna con warm-start (prior de literatura como trial 0) + TPE.
- Entrena el final con `epochs=600`, `patience=50`, early-stop F1; calibra el threshold sobre
  validación remuestreada a la prevalencia de test.
- **Quality gate real (del YAML): VAL F1 ≥ 0,30 y VAL MCC ≥ 0,15** para marcar `quality_passed=True`,
  evaluado sobre **validación**. Los 0,70/0,40 del código son defaults que el YAML sobrescribe.
  De las 60 configuraciones, **23 pasan**.
- Produce `results_models_v3/{run_id}_best.pt` + `{run_id}_meta.json`. Retomable con `--resume`.

> Nota para futuros reentrenamientos: conviene gate sobre **VAL PR-AUC** (~0,34), no F1/MCC en
> argmax-0.5, que son degenerados bajo desbalance extremo.

### 2. Explicar solo los modelos que aprendieron

```bash
uv run python scripts/explain_matrix.py --config configs/experiment_machineB_v3.yaml
```

- Lee los `*_meta.json`, filtra por `quality_passed=True` (`--force` ignora el gate).
- Corre GNNExplainer + PGExplainer + GNNShap + estabilidad (5 réplicas).
- Escribe `results_v3/xai-gnn-stability-B-v3.csv` + MLflow nested runs.
- Flags: `--arch`, `--scenario`, `--balancing`, `--explainer`, `--force`, `--resume`, `--max-hours`.

> **Reproducibilidad:** para evitar OOM en GPUs de 8 GB, la estabilidad se recomputó lanzando **un
> proceso fresco por configuración** (troceando por `--arch/--scenario/--balancing`). El OOM previo
> venía de correr las 60 en un solo proceso.

### Regenerar tablas y figuras (sin GPU)

```bash
uv run python scripts/consolidacion/finalize_elliptic.py   # tablas + figuras desde el CSV
uv run python scripts/consolidacion/reeval_rocauc.py       # ROC-AUC / precision@k (usa checkpoints)
```

## Máquinas

| Máquina | GPU | Arquitecturas | Config |
|---------|-----|---------------|--------|
| **B** | RTX 4060 8GB | GCN, GraphSAGE | `configs/experiment_machineB_v3.yaml` |
| **C** | RTX 3050 4GB | GAT, TAGCN | `configs/experiment_machineC_v3.yaml` |

Difieren en `hidden_dim` (cap por VRAM), `optuna_trials` (50 vs 8), `epochs` (600 vs 150) y
`num_samples` de GNNShap (50 vs 25). Matriz total = 5 escenarios × 4 arquitecturas × 3 balanceos = 60.

## Estructura del proyecto

```
gnns_thesis/
├── tesis_latex/            ← MANUSCRITO (main.tex + 8 capítulos + tables/ + bibliografia.bib)
│   └── main.pdf              103 páginas, versionado
├── presentacion_latex/     ← defensa en Beamer (.tex + .pdf, 26 páginas)
├── docs/                   ← material de defensa (ver abajo)
├── configs/                ← *_v3.yaml son los vigentes; el resto es legacy
├── scripts/
│   ├── train_matrix.py       Paso 1 (entrena + quality gate)
│   ├── explain_matrix.py     Paso 2 (explainers + estabilidad)
│   ├── smoke_test.py         validación pre-run (15 checks)
│   └── consolidacion/        regeneración determinista de tablas/figuras/métricas
├── src/
│   ├── models/               GCN, GraphSAGE, GAT, TAGCN (interfaz común)
│   ├── data/                 loader Elliptic, split temporal causal, escenarios de desbalance
│   ├── training/             trainer, hyperopt (Optuna)
│   ├── explainability/       GNNExplainer, PGExplainer, GNNShap
│   ├── stability/            réplicas estocásticas, perturbación, métricas (Jaccard/Spearman)
│   ├── balancing/            class weighting, focal loss, GraphSMOTE (no cableado en v3)
│   └── analysis/             tracking (MLflow + CSV), factorial, recommendation
├── phase1/                 ← eje sintético (generador + CSV de resultados). NO RE-CORRER
├── results_v3/             ← CSV de provenance (versionados)
├── results_models_v3/      ← checkpoints (NO en git)
└── data/                   ← Elliptic, auto-descargado por PyG (~300MB, NO en git)
```

## Material de defensa

| Archivo | Qué es |
|---|---|
| `docs/DISCURSO_defensa_dos_voces.md` | Guion hablado, 21 slides a dos voces (Alejandro 1-12, Juan Diego 13-21), con tiempos, mapa slide→página del PDF y respuestas ensayadas |
| `docs/GUION_defensa_por_capitulo.md` | Mapa slide→capítulo/sección + preguntas del jurado |
| `docs/DEFENSA_R2_evidencia_sintetica.md` | Respuesta a la objeción de circularidad del eje sintético |
| `docs/ESQUELETO_presentacion_defensa.md` | Esqueleto slide por slide con las figuras |

El PDF de la presentación tiene **26 páginas** para **21 slides de contenido**: intercala 5
separadores de sección. La numeración del guion no es la del PDF (tabla de equivalencia en el
DISCURSO).

## Bugs corregidos (no revertir sin entender)

### En PGExplainer de PyTorch Geometric 2.7 (reportados upstream)

| Bug | Síntoma | Fix |
|-----|---------|-----|
| `edge_size=0.05` (default) | mode collapse: toda la atribución en una arista | `edge_size=0.005` |
| `temp=[5.0,2.0]` (default) | overflow → ~99% de épocas con loss NaN | `temp=[1.0,1.0]` + gradient clipping |

### En `src/`

| Bug | Archivo | Fix |
|-----|---------|-----|
| Truncamiento de Spearman (fix R1) | `src/stability/metrics.py` | dimensionar rangos por nº de features |
| FocalLoss alpha invertido (sub-pesaba la clase rara) | `src/balancing/losses.py` | `alpha` = peso clase rara; default 0,75 |
| Early stopping en MCC ruidoso bajo imbalance | `src/training/trainer.py` | `early_stop_metric` configurable (default F1) |
| Optuna sin prior de literatura, search space estrecho | `src/training/hyperopt.py` | `get_warm_start_priors()` + rango expandido |
| CSV corruption (error embebido en columna numérica) | `src/analysis/tracking.py` | `CSV_SCHEMA_FIELDS` fijo + escritura atómica |
| PGExplainer crash cuando el training falla | `src/stability/stochastic_test.py:71` | `if not success: continue` |
| `--resume` ignoraba el CSV, solo leía MLflow | `src/analysis/tracking.py` | fallback CSV en `get_completed_runs` |

## Entorno y comandos útiles

Requiere Python ≥ 3.12 y `uv`. El dataset Elliptic se descarga solo en `./data/` la primera vez.

```bash
uv venv && uv sync                                          # entorno
uv run mlflow ui --backend-store-uri sqlite:///mlruns.db    # monitoreo en vivo
```

Compilar el manuscrito (TinyTeX + biber):

```bash
cd tesis_latex && pdflatex -interaction=nonstopmode main.tex && biber main \
  && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```
