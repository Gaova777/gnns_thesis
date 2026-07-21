# Estudio de la Estabilidad de Métodos de Explicabilidad (XAI) en Graph Neural Networks para Detección de Lavado de Dinero bajo Desbalance Extremo de Datos

Repositorio de código y experimentación de la tesis de Maestría en Ingeniería en Sistemas y
Computación. Estudia la **estabilidad de métodos XAI** (GNNExplainer, PGExplainer, GNNShap)
aplicados a Graph Neural Networks (GCN, GraphSAGE, GAT, TAGCN) entrenadas para detectar
transacciones ilícitas en el **Elliptic Bitcoin Dataset**, bajo distintos niveles de
desbalance de clases (1:1, 1:10, 1:30 nativo, 1:50, 1:100).

| | |
|---|---|
| **Autores** | Alejandro Gómez Huertas · Juan Diego Garzón Ovalle |
| **Director** | Ph.D. Cristian Rosero |
| **Institución** | Universidad Tecnológica de Pereira (UTP) — Facultad de Ingenierías — Maestría en Ingeniería en Sistemas y Computación (MISC) |
| **Lugar / Año** | Pereira, Colombia · 2026 |
| **Dataset** | Elliptic Bitcoin Dataset (~203k nodos, ~234k aristas, 166 features, 49 time-steps) |
| **Manuscrito** | `tesis_final_v16.pdf` (105 pp) — ver [§ Manuscrito](#-manuscrito) |

---

> ### ⚠️ Estado (2026-07-21): consolidación en curso
> La **fuente de verdad de los hallazgos es el manuscrito** (`tesis_latex/`), no las cifras
> históricas de este README. Tras la auditoría integral, varias afirmaciones previas fueron
> **retractadas** (ver [§ Hallazgos clave](#hallazgos-clave)). El plan de consolidación, el estado
> de los arreglos aplicados y los **pasos pendientes que requieren la máquina con GPU** están en
> **[RUNBOOK_CONSOLIDACION.md](RUNBOOK_CONSOLIDACION.md)**.

---

## Tabla de contenidos

- [Pregunta de investigación y objetivos](#pregunta-de-investigación-y-objetivos)
- [Hallazgos clave](#hallazgos-clave)
- [Diseño experimental](#diseño-experimental)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Pipeline v3 (workflow)](#pipeline-v3-workflow)
- [Configuración por máquina](#configuración-por-máquina)
- [Detalles técnicos](#detalles-técnicos)
- [Bugs metodológicos corregidos](#bugs-metodológicos-corregidos)
- [Resultados y artefactos](#resultados-y-artefactos)
- [📄 Manuscrito](#-manuscrito)
- [Limitaciones y trabajo futuro](#limitaciones-y-trabajo-futuro)

---

## Pregunta de investigación y objetivos

> **¿Cómo se comportan los métodos XAI aplicados a distintas arquitecturas GNN para detección
> AML cuando el sistema enfrenta desbalance extremo, y qué combinación de arquitectura,
> explicador y estrategia de balanceo ofrece el mejor compromiso entre rendimiento predictivo
> y estabilidad explicativa?**

**Objetivo general:** evaluar la estabilidad de los métodos de explicabilidad en GNNs bajo
desbalance extremo en detección de lavado de dinero sobre el Elliptic Dataset, para entregar
recomendaciones técnicas sobre las combinaciones óptimas de arquitectura, explicador y técnica
de balanceo que garanticen explicaciones robustas, consistentes y auditables en entornos regulados.

**Objetivos específicos:**

1. Evaluar el impacto directo del desbalance de clases en la robustez de las explicaciones
   estructurales, comparando cuantitativamente GNNExplainer, PGExplainer y GNNShap a través de
   escenarios incrementales de desbalance.
2. Comparar la resiliencia topológica de GCN, GraphSAGE, GAT y TAGCN, midiendo la consistencia
   de los subgrafos explicativos bajo estrés de datos.
3. Analizar el impacto colateral de las estrategias de mitigación del desbalance (ponderación de
   clases y focal loss) sobre la estabilidad/fidelidad de la interpretación topológica.
4. Construir y validar una matriz de recomendación técnica que articule rendimiento predictivo y
   estabilidad explicativa (tríada arquitectura–explicador–balanceo).

---

## Hallazgos clave

> El estudio se articula en **dos ejes**: **Elliptic real** (validez externa) y un **grafo sintético
> propio** con *ground-truth* de tipologías (validez interna, donde vive la evidencia estadística
> fuerte). Hallazgos vigentes (manuscrito):

1. **Tres dimensiones independientes:** estabilidad, plausibilidad y fidelidad de las explicaciones
   son **empíricamente independientes** — no covarían entre sí.
2. **Puente estabilidad→plausibilidad NULO** (hipótesis central de la tesis, *refutada*): una
   explicación estable no es más plausible (GNNExplainer r ≈ −0,01; IC bootstrap incluye 0). Un
   no-resultado reportado honestamente como resultado.
3. **Disociación plausibilidad ↔ fidelidad:** **PGExplainer** domina la plausibilidad de aristas
   (0,80 vs 0,50; Wilcoxon p ≈ 2,6×10⁻³⁵) pero **colapsa en fidelidad** (0,11 vs 0,56 de
   GNNExplainer). El explicador más "plausible" no es el más fiel.
4. **El balanceo es prácticamente irrelevante** para las tres dimensiones (η² < 0,02).
5. **La estabilidad por arquitectura depende del régimen del grafo:** en el grafo denso sintético
   GCN/GAT lideran (Spearman ≈ 0,96); el orden difiere respecto al eje Elliptic disperso. *(La
   estabilidad del eje Elliptic se está recomputando tras corregir la métrica de Spearman — ver
   [RUNBOOK_CONSOLIDACION.md](RUNBOOK_CONSOLIDACION.md).)*
6. **Contribución metodológica:** dos bugs en el PGExplainer de PyTorch Geometric 2.7 — ver
   [§ Bugs metodológicos corregidos](#bugs-metodológicos-corregidos).

> ⚠️ **Narrativa retractada:** las afirmaciones previas de este README (pico-y-colapso
> 0,42→0,59→0,24, paradoja del régimen nativo, trade-off −0,20, "GAT lidera explicación", pico
> TAGCN·1:50 = 0,789) fueron **retractadas** en el manuscrito tras la auditoría y **ya no son
> válidas**. Provenían de una métrica de estabilidad con bug (ya corregida) y de una corrida con
> fallos de memoria silenciosos en GAT.

---

## Diseño experimental

Diseño factorial sobre la tríada **arquitectura × explicador × balanceo**, cruzada con escenarios
de desbalance:

| Factor | Niveles |
|---|---|
| **Arquitecturas (4)** | GCN · GraphSAGE · GAT · TAGCN |
| **Explicadores (3)** | GNNExplainer · PGExplainer · GNNShap |
| **Balanceo (3)** | sin balanceo · class weighting · focal loss |
| **Escenarios (5)** | 1:1 · 1:10 · **1:30 nativo** · 1:50 · 1:100 |

- **Desbalance extremo** se define operativamente como razón ilícito:lícito ≥ 1:50 (prevalencia ≤ 2 %).
- El escenario **1:30 nativo** preserva la distribución natural del dataset (baseline Weber 2019),
  sin manipular la máscara.
- Los experimentos se distribuyen en **2 máquinas** (la 4090 quedó sin acceso): cada máquina cubre
  un subconjunto de arquitecturas — ver [§ Configuración por máquina](#configuración-por-máquina).

---

## Estructura del proyecto

```
gnns_thesis/
├── configs/
│   ├── experiment_machineB_v3.yaml     # ← config vigente Máquina B (RTX 4060): GCN, GraphSAGE
│   ├── experiment_machineC_v3.yaml     # ← config vigente Máquina C (RTX 3050): GAT, TAGCN
│   ├── experiment_machineB_v2.yaml     # iteración v2 (histórico)
│   ├── experiment_machineB_pgonly.yaml # solo PGExplainer (debugging)
│   └── experiment_machine{A,B,C}.yaml  # configs v1 (legacy)
├── scripts/
│   ├── train_matrix.py          # ← Pipeline v3 · Paso 1: entrena la matriz + quality gate
│   ├── explain_matrix.py        # ← Pipeline v3 · Paso 2: explainers + estabilidad
│   ├── smoke_test.py            # validación end-to-end (~10 min, 13-15 checks)
│   ├── merge_results.py         # unifica CSVs de ambas máquinas
│   ├── run_full_pipeline.py     # orquestador monolítico v1/v2 (LEGACY — no usar para v3)
│   ├── run_training.py / run_explain.py / run_stability.py   # etapas standalone
│   ├── debug_pgexplainer*.py    # diagnósticos de PGExplainer sobre Cora
│   └── add_native_figures.py    # genera figuras del escenario 1:30 nativo
├── src/
│   ├── models/        # GCN, GraphSAGE, GAT, TAGCN (interfaz común)
│   ├── data/          # carga Elliptic, split temporal, escenarios de desbalance
│   ├── training/      # Trainer (early stopping, calibración) + hyperopt (Optuna)
│   ├── balancing/     # FocalLoss, class weighting, GraphSMOTE (implementado, no cableado en v3)
│   ├── explainability/# runners de GNNExplainer/PGExplainer y GNNShap + extracción
│   ├── stability/     # réplicas estocásticas, perturbación, métricas (Jaccard/Spearman)
│   └── analysis/      # tracking MLflow+CSV, ANOVA factorial, matriz de recomendación
├── data/              # dataset Elliptic (auto-descargado por PyG, ~300MB · NO en git)
├── results_models_v3/ # checkpoints *_best.pt + *_meta.json (NO en git)
├── results_v3/        # CSVs del explain stage + MLflow artifacts (NO en git)
├── pyproject.toml     # dependencias (gestionado con uv)
└── README.md
```

> **Nota:** `data/`, `results_*/`, `checkpoints/`, `mlruns/` y `*.pt` están en `.gitignore` — se
> generan al correr el pipeline y no se versionan.

---

## Instalación

Requiere **Python ≥ 3.12** y el gestor [`uv`](https://github.com/astral-sh/uv). PyTorch se instala
desde el índice CUDA 12.4 (`pytorch-cu124`).

```bash
uv venv          # crea el entorno virtual
uv sync          # instala dependencias desde uv.lock
```

**Dependencias principales:** `torch>=2.6`, `torch-geometric>=2.7`, `optuna>=4.0`, `mlflow>=2.0`,
`scikit-learn>=1.5`, `scipy>=1.14`, `statsmodels>=0.14`, `pandas>=2.2`, `matplotlib`, `seaborn`.

El **dataset Elliptic** (~300 MB) se descarga automáticamente en `./data/` la primera vez que se
corre el pipeline (requiere conexión a internet, desde `data.pyg.org`). Sin internet, copiar
manualmente la carpeta `data/` desde otra máquina.

---

## Pipeline v3 (workflow)

El pipeline está **dividido en dos scripts** para poder iterar sobre la calidad del modelo de forma
independiente de los explicadores. La v2 producía modelos con F1 ~0,02–0,08; la v3 corrigió tres
bugs centrales (focal loss, early stopping, search space de Optuna) y agregó un quality gate.

### 0. Validar (smoke test, ~10 min)

```bash
uv run python scripts/smoke_test.py --config configs/experiment_machineB_v3.yaml
```

Corre una config mínima (1:10 · GCN · class_weighting) end-to-end y valida 13–15 checks (estructura
de config, integridad del dataset, escenario de desbalance, entrenamiento, los 3 explicadores sin
crash/OOM, schema CSV, semántica del alpha de FocalLoss, warm-start priors, métrica PR-AUC,
calibración de threshold, schema del metadata JSON).

### 1. Entrenar la matriz

```bash
uv run python scripts/train_matrix.py --config configs/experiment_machineB_v3.yaml --max-hours 9
```

- Corre **Optuna con warm-start** (prior de literatura como trial 0) + búsqueda bayesiana (TPE).
- Entrena el modelo final, **calibra el threshold** sobre validación (remuestreando val a la
  prevalencia de test, para corregir el covariate shift temporal).
- Aplica el **quality gate** sobre validación y persiste `*_best.pt` + `*_meta.json` en
  `results_models_v3/`. **No corre explainers.**
- Flags: `--resume` (saltea configs con `meta.json` ya guardado), `--quick`, `--arch`, `--scenario`,
  `--balancing`, `--device auto`, `--max-hours`, `--no-warm-start`.

### 2. Explicar solo los modelos que aprendieron

```bash
uv run python scripts/explain_matrix.py --config configs/experiment_machineB_v3.yaml
```

- Lee los `*_meta.json`, filtra por `quality_passed=True` (`--force` ignora el gate).
- Corre **GNNExplainer + PGExplainer + GNNShap** + estabilidad (**5 réplicas estocásticas**) sobre
  nodos ilícitos de test.
- Escribe `results_v3/xai-gnn-stability-B-v3.csv` + MLflow nested runs.
- Flags: `--arch`, `--scenario`, `--balancing`, `--explainer`, `--force`, `--resume`, `--max-hours`.

### 3. Merge final (cuando ambas máquinas terminan)

```bash
uv run python scripts/merge_results.py
```

Unifica los CSVs de B y C, deduplica por `[scenario, architecture, balancing, explainer]` y genera
`results/results_merged.csv`.

### Monitoreo

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlruns.db   # UI de MLflow en vivo
```

---

## Configuración por máquina

Los experimentos se reparten en dos GPUs. Ambas configs comparten la estructura (`data`, `scenarios`,
`models`, `training`, `balancing`, `explainability`, `stability`, `analysis`, `tracking`) y los mismos
5 escenarios y 3 técnicas de balanceo; difieren en arquitecturas y presupuesto de cómputo:

| Campo | **Máquina B** (RTX 4060, 8 GB) | **Máquina C** (RTX 3050, 4 GB) |
|---|---|---|
| Arquitecturas | **GCN, GraphSAGE** | **GAT, TAGCN** |
| `data.root` | `./data` | `./dataset` |
| `hidden_dim` choices | `[64,128,140,211,256]` | `[64,128,148]` (cap por VRAM) |
| `optuna_trials` | **50** | **8** |
| `training.epochs` | **600** | **150** |
| `training.patience` | **50** | **20** |
| GNNShap `num_samples` | **50** | **25** |
| `experiment_name` | `xai-gnn-stability-B-v3` | `xai-gnn-stability-C-v3` |

Matriz por máquina = 5 escenarios × 2 arquitecturas × 3 balanceos = **30 configs de training**
(60 entre ambas) × 3 explicadores = 90 runs de explain.

---

## Detalles técnicos

### Modelos (`src/models/`)

Las cuatro arquitecturas comparten interfaz
`__init__(in_channels, hidden_channels=128, num_layers=2, dropout=0.3, num_classes=2)` y
`forward(x, edge_index)`, con patrón `Conv → activación → Dropout`:

| Modelo | Capa PyG | Activación | Extra |
|---|---|---|---|
| `GCN` | `GCNConv` | ReLU | — |
| `GraphSAGE` | `SAGEConv` | ReLU | — |
| `GAT` | `GATConv` | ELU | `heads=4`; `hidden_channels` es por cabeza |
| `TAGCN` | `TAGConv` | ReLU | `K=3` (orden del filtro polinomial) |

Instanciadas vía `build_model()` (factory en `src/training/trainer.py`).

### Datos (`src/data/`)

- **Carga** (`loader.py`): `EllipticBitcoinDataset` de PyG; labels remapeados a
  `0=lícito, 1=ilícito, -1=unknown`.
- **Split temporal causal** (`preprocessing.py`): train = time-steps 1–34, val = 35–42, test = 43–49
  (solo nodos etiquetados). Features normalizadas con **RobustScaler** ajustado solo en train
  (anti-leakage) y clip a [-10, 10].
- **Escenarios de desbalance** (`imbalance.py`): por *undersampling*, modificando solo `train_mask`
  (val/test intactos). `None` → nativo; ratio ≥ 0,1 → subsamplea lícitas; ratio < 0,1 → subsamplea
  ilícitas.

### Entrenamiento (`src/training/`)

- **`Trainer`**: full-batch transductivo, early stopping configurable (`early_stop_metric` ∈
  `{f1, mcc, pr_auc}`, default **F1**). `evaluate()` devuelve `loss, f1, mcc, pr_auc`. Calibra el
  threshold barriendo 101 puntos y maximizando F1, con opción de remuestrear val a la prevalencia de
  test. Guarda best checkpoint + recovery por epoch (con RNG state → reanudable).
- **`hyperopt.py`** (Optuna): **warm-start priors** de literatura (arXiv:2602.23599) como trial 0;
  search space sobre `hidden_dim, num_layers, dropout, lr, weight_decay` (+ `heads` para GAT, `K` para
  TAGCN); `TPESampler` + `MedianPruner`; métrica objetivo por default **PR-AUC**.

### Balanceo (`src/balancing/`)

- **FocalLoss** `FL = -α_t·(1-p_t)^γ·log(p_t)` con `α=0.75, γ=2.0`. **Semántica v3:** `alpha` es el
  peso de la **clase rara** (internamente `[1-alpha, alpha]`) → la clase ilícita recibe 3× el peso.
- **`get_loss_function()`**: `none` → CrossEntropy; `class_weighting` → CE con pesos inversos a
  frecuencia; `focal_loss` → FocalLoss.
- **GraphSMOTE** (`graphsmote.py`): implementado (Zhao et al. 2021) pero **no cableado** en los
  configs v3 — queda como trabajo futuro.

### Explicabilidad (`src/explainability/`)

- **GNNExplainer / PGExplainer** (`explainer_runner.py`): vía la API nativa `torch_geometric.explain`.
  PGExplainer se entrena con gradient clipping y rollback por epoch ante NaN.
- **GNNShap** (`shap_runner.py`): SHAP por permutación sobre el k-hop subgraph (`num_hops=3`), con
  manejo automático de CUDA OOM (halve num_samples y reintenta).

### Estabilidad (`src/stability/`)

- **Réplicas estocásticas** (`stochastic_test.py`): re-ejecuta cada explicador con seeds
  determinísticas (`42 + replica·17`). **En v3 → `num_replicas=5`.** PGExplainer, por ser un modelo
  global paramétrico, se reentrena una vez por réplica.
- **Perturbación** (`perturbation.py`): ruido gaussiano sobre las features con niveles
  `[0.01, 0.05, 0.10]`.
- **Métricas** (`metrics.py`): **Spearman** (acuerdo de rankings de importancia — *primaria*),
  **Jaccard** (intersección/unión de aristas — secundaria, satura por el top-K), **SHAP concentration**
  (parsimonia).

### Análisis (`src/analysis/`)

- **`tracking.py`**: `ExperimentTracker` sobre MLflow (`sqlite:///mlruns.db`) con fallback CSV de
  schema fijo (`CSV_SCHEMA_FIELDS`, anti-corrupción) y escritura atómica (`tmp → os.replace()`).
- **`factorial.py`**: ANOVA factorial (statsmodels) con interacciones + Tukey HSD.
- **`recommendation.py`**: matriz de recomendación con bootstrap CIs.

### Quality gate

> ⚠️ **Valor efectivo:** el gate que decide `quality_passed` se lee del YAML
> (`analysis.quality_gate`): **F1 ≥ 0,30 y MCC ≥ 0,15** (recalibrado tras la calibración de threshold).
> Los defaults de fallback en el código son 0,70 / 0,40, pero **el YAML los sobrescribe**. El gate se
> evalúa sobre **validación**, no test (por el covariate shift temporal "dark market shutdown" entre
> train y test).

Aparte, `analysis.thresholds` (F1 ≥ 0,80, MCC ≥ 0,70, Jaccard ≥ 0,70) son los **criterios ideales de
éxito** para la matriz de recomendación — ninguna configuración los alcanza (umbral regulatorio ideal).

---

## Bugs metodológicos corregidos

Contribución metodológica de la tesis: **dos defectos en PGExplainer de PyTorch Geometric 2.7** que
contaminan cualquier benchmark que use los defaults, reportados upstream.

| Bug | Síntoma | Fix |
|---|---|---|
| `edge_size=0.05` (default) | *mode collapse* universal: toda la masa de atribución en una sola arista | `edge_size=0.005` |
| `temp=[5.0,2.0]` (default) | *overflow* numérico → ~99 % de épocas con loss NaN en grafos grandes | `temp=[1.0,1.0]` + gradient clipping |

Los scripts `scripts/debug_pgexplainer*.py` reproducen y aíslan estos bugs sobre Cora (grafo
balanceado) para distinguir defecto de software vs hallazgo dataset-específico.

**Otros bugs corregidos en `src/` (no revertir):**

| Bug | Archivo | Fix |
|---|---|---|
| FocalLoss alpha invertido (sub-pesaba la clase rara) | `src/balancing/losses.py` | `alpha` = peso clase rara; default 0,75 |
| Early stopping en MCC ruidoso bajo imbalance extremo | `src/training/trainer.py` | `early_stop_metric` configurable (default F1) |
| Optuna sin prior de literatura, search space estrecho | `src/training/hyperopt.py` | `get_warm_start_priors()` + rango expandido |
| CSV corruption (error embebido en columna numérica) | `src/analysis/tracking.py` | `CSV_SCHEMA_FIELDS` fijo + escritura atómica |

---

## Resultados y artefactos

- **Predictivos:** F1, MCC, PR-AUC. GraphSAGE lidera (F1≈0,53; ~73 % pass-rate en escenarios
  forzados). Pass-rate del gate con pico en 1:10 (67 %) y colapso en 1:100 (8 %).
- **Estabilidad:** Spearman observado 0,24–0,79 (consistente con Agarwal 2022: 0,30–0,80). Pico
  absoluto TAGCN · 1:50 · focal loss = 0,789.
- **Estadística:** Kruskal-Wallis entre escenarios (H=4,31, p=0,23, no significativo por N limitado en
  1:100) + Cohen d entre regímenes extremos (efecto grande respaldando el patrón cualitativo).
- **Costos:** pipeline completo RTX 4060 ≈ 4,65 h (train 3,12 h + XAI 1,53 h) vs RTX 3050 ≈ 29,4 h
  (fase XAI 15,7× más lenta; cuello de botella = memoria). GNNShap es el explicador más costoso.

Salidas generadas (no versionadas): `results_models_v3/*_meta.json` (con `best_params`,
`test_metrics`, `calibrated_threshold`, `quality_passed`), `results_v3/*.csv`, y la base MLflow
`mlruns.db`.

---

## 📄 Manuscrito

El manuscrito, la presentación de sustentación y el material de planeación viven en un repositorio
aparte (fuera de este repo de código):

```
thesis manuscritus/thesis_ppt/
├── tesis_final_v16.pdf              # ← manuscrito final vigente (105 pp, 8 capítulos)
├── tesis_final_v{4..15}.pdf         # cadena de versiones histórica
├── overleaf_thesis_v4.zip           # fuente LaTeX (biblatex/biber, ~52 refs en Mendeley)
├── Sustentacion Tesis v4.0.pptx     # presentación de sustentación (26 slides, ~40 min)
├── Sustentacion Tesis v4.0.pdf      # render de referencia de la sustentación
├── speaker_notes_v4.0.md            # notas del ponente por slide
├── Ultimo anteproyecto.docx         # anteproyecto firmado por el director
├── CHANGELOG.md · CAMBIOS_v*.md     # bitácora de cambios v4–v16
├── 0_research_and_planning/         # EDA inicial + figuras
├── secciones_nuevas/                # markdown de caps 5–8 + figuras de resultados
└── docs_v3.1/                       # conclusiones, literatura, figuras
```

### Estructura del manuscrito (8 capítulos, v16)

| # | Capítulo | Pág |
|---|---|---|
| 1 | Introducción (planteamiento, estado del arte, brecha, objetivos, justificación, alcance) | 1 |
| 2 | Marco Contextual | 13 |
| 3 | Fundamentos de IA para Detección de Fraude Financiero | 24 |
| 4 | Metodología (dataset, escenarios, arquitecturas, balanceo, explicadores, protocolo de estabilidad, métricas, Optuna, infraestructura, análisis estadístico) | 39 |
| 5 | Resultados | 51 |
| 6 | Discusión | 72 |
| 7 | Conclusiones | 82 |
| 8 | Perspectivas Futuras y Anexos | 87 |
| — | Referencias | 92 |

### Evolución reciente

- **v14** (102 pp): feedback del revisor externo (33 issues, 26 aplicados).
- **v15** (104 pp): revisión del director — TAGCN reformulado como filtro espacial (no ChebNet),
  Fidelity+/− separadas, definición operativa de "desbalance extremo".
- **v16** (105 pp, final): índice general reparado para incluir los 8 capítulos; 579/579 hipervínculos
  resueltos.
- **Sustentación v4.0:** deck académico formal UTP, 26 slides, ~40 min, defensa a dos voces.

---

## Limitaciones y trabajo futuro

**Pendientes documentados** (requieren re-correr el pipeline): múltiples semillas en 1:100,
evaluación sobre test set, explainers con 100 nodos (vs 30), baselines tabulares (XGBoost/RF), ANOVA
con celdas vacías.

**Perspectivas futuras** (Cap 8): replicación en Elliptic2 (Bellei 2024), GNNs temporales
(EvolveGCN), estabilización específica de PGExplainer en grafos densos, GraphSMOTE, y generalización a
otros datasets de fraude.

---

> Tesis de Maestría en Ingeniería en Sistemas y Computación · Universidad Tecnológica de Pereira · 2026
