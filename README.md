# Estudio de la Estabilidad de Métodos de Explicabilidad (XAI) en Graph Neural Networks para Detección de Lavado de Dinero bajo Desbalance Extremo de Datos

Repositorio de código, manuscrito y experimentación de la tesis de Maestría en Ingeniería en
Sistemas y Computación. Estudia la **estabilidad de métodos XAI** (GNNExplainer, PGExplainer,
GNNShap) aplicados a Graph Neural Networks (GCN, GraphSAGE, GAT, TAGCN) entrenadas para detectar
transacciones ilícitas en el **Elliptic Bitcoin Dataset**, bajo distintos niveles de desbalance de
clases (1:1, 1:10, 1:30 nativo, 1:50, 1:100), complementado con un **grafo sintético con
*ground-truth*** que permite medir plausibilidad y fidelidad.

| | |
|---|---|
| **Autores** | Alejandro Gómez Huertas · Juan Diego Garzón Ovalle |
| **Director** | Ph.D. Cristian Rosero |
| **Institución** | Universidad Tecnológica de Pereira (UTP) — Facultad de Ingenierías — Maestría en Ingeniería en Sistemas y Computación (MISC) |
| **Lugar / Año** | Pereira, Colombia · 2026 |
| **Dataset** | Elliptic Bitcoin Dataset (~203k nodos, ~234k aristas, 166 features, 49 time-steps) + grafo sintético propio |
| **Manuscrito** | [`tesis_latex/main.pdf`](tesis_latex/main.pdf) — 103 pp, 8 capítulos (fuente LaTeX en `tesis_latex/`) |
| **Defensa** | [`presentacion_latex/beamer_defensa.pdf`](presentacion_latex/beamer_defensa.pdf) (26 slides) |

---

> ### ✅ Estado (2026-07-22): consolidación completada
> La **fuente de verdad de los hallazgos es el manuscrito** ([`tesis_latex/`](tesis_latex/)), que
> compila a 103 páginas y está commiteado en la rama `consolidacion-auditoria`. Tras la auditoría
> integral se corrigieron **dos artefactos de evaluación** que invertían conclusiones sobre
> estabilidad; los números de este README ya reflejan la versión corregida. El registro detallado de
> los cambios está en **[CAMBIOS_CONSOLIDACION_2026-07-22.md](CAMBIOS_CONSOLIDACION_2026-07-22.md)** y
> el plan operativo en **[RUNBOOK_CONSOLIDACION.md](RUNBOOK_CONSOLIDACION.md)**.
>
> **Revisión de estilo (2026-07-23):** se auditó el PDF completo en busca de marcas tipográficas
> y giros de redacción ajenos al estilo de los autores (rayas, flechas, comillas curvas, muletillas).
> El manuscrito quedó sin rayas ni flechas en prosa y se redujo el uso repetido de la construcción
> «no solo… sino también» de 19 a 4 apariciones.

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
- [Manuscrito y defensa](#manuscrito-y-defensa)
- [Limitaciones y trabajo futuro](#limitaciones-y-trabajo-futuro)

---

## Pregunta de investigación y objetivos

> **¿Cómo se comportan los métodos XAI aplicados a distintas arquitecturas GNN para detección
> AML cuando el sistema enfrenta desbalance extremo, y qué combinación de arquitectura,
> explicador y estrategia de balanceo ofrece el mejor compromiso entre rendimiento predictivo
> y estabilidad explicativa?**

**Objetivo general:** evaluar la estabilidad de los métodos de explicabilidad en GNNs bajo
desbalance extremo en detección de lavado de dinero, para entregar recomendaciones técnicas sobre
las combinaciones de arquitectura, explicador y técnica de balanceo que garanticen explicaciones
robustas, consistentes y auditables en entornos regulados.

**Objetivos específicos:**

1. Evaluar el impacto del desbalance de clases sobre la robustez de las explicaciones, comparando
   GNNExplainer, PGExplainer y GNNShap a través de escenarios incrementales de desbalance.
2. Comparar la resiliencia de GCN, GraphSAGE, GAT y TAGCN, midiendo la consistencia de los
   subgrafos y rankings explicativos bajo estrés de datos.
3. Analizar el impacto de las estrategias de mitigación del desbalance (ponderación de clases y
   focal loss) sobre la estabilidad y la fidelidad de la interpretación.
4. Construir una matriz de recomendación técnica que articule rendimiento predictivo y estabilidad
   explicativa (tríada arquitectura–explicador–balanceo).

---

## Hallazgos clave

> El estudio se articula en **dos ejes**: **Elliptic real** (validez externa) y un **grafo sintético
> propio** con *ground-truth* de tipologías (validez interna, donde vive la evidencia estadística
> fuerte). Los hallazgos vigentes, tras corregir la métrica de estabilidad, son:

1. **Tres dimensiones independientes.** La estabilidad, la plausibilidad y la fidelidad de las
   explicaciones son dimensiones distintas que no se implican entre sí. Reportar una sola y llamarla
   "calidad" oculta compromisos decisivos según el propósito de la auditoría.
2. **Puente estabilidad→plausibilidad nulo** (hipótesis central de la tesis, *refutada con
   honestidad*): una explicación estable no es por ello más plausible (GNNExplainer r ≈ −0,01; IC
   bootstrap incluye el cero).
3. **Disociación plausibilidad ↔ fidelidad.** PGExplainer domina la plausibilidad de aristas
   (0,80 vs 0,50 de GNNExplainer; Wilcoxon p ≈ 2,6×10⁻³⁵) pero **colapsa en fidelidad** (0,11 vs 0,56).
   El explicador más plausible no es el más fiel.
4. **El balanceo y el nivel de desbalance son secundarios.** El balanceo tiene tamaño de efecto
   despreciable sobre las tres dimensiones (η² < 0,02), y el perfil de estabilidad por escenario de
   desbalance es esencialmente plano. La palanca dominante es el explicador.
5. **Estabilidad por arquitectura: dos grupos, concordantes entre regímenes.** Con la métrica
   corregida y **replicando el entrenamiento con 3 semillas de modelo** (180 modelos en Elliptic), la
   estabilidad no ordena a las cuatro arquitecturas en cuatro posiciones distinguibles: las separa en
   un **grupo alto (GAT, GCN)** y un **grupo bajo (GraphSAGE, TAGCN)**, con diferencias significativas
   entre grupos y no dentro de ellos. La misma partición emerge de forma independiente en el grafo
   sintético denso. Detalle en [§ Resultados](#resultados-y-artefactos).
6. **Contribución metodológica: dos artefactos de evaluación corregidos.** Un fallo de memoria
   (cálculo sobre el grafo completo con OOM silencioso en GAT) y un truncamiento en la métrica de
   Spearman, cada uno suficiente por sí solo para producir una conclusión comparativa falsa. Además,
   dos bugs en el PGExplainer de PyTorch Geometric — ver
   [§ Bugs metodológicos corregidos](#bugs-metodológicos-corregidos).
7. **Rigor métrico bajo desbalance.** El ROC-AUC es engañosamente alto bajo desbalance extremo
   (validación 0,88 vs test 0,65) mientras que PR-AUC y precision@k revelan la dificultad real
   (validación 0,37 vs test 0,02). Por eso PR-AUC y precision@k son las métricas primarias, no F1 ni
   ROC-AUC.

> ⚠️ **Narrativa retractada.** Versiones anteriores de este README y del manuscrito afirmaron una
> **"inversión por densidad"** (GraphSAGE más estable en Elliptic, orden invertido en el sintético) y
> un **liderazgo de GraphSAGE**. Ambas afirmaciones se **retractaron**: eran artefactos de una métrica
> de Spearman con un bug de truncamiento (ya corregido) y de una corrida con fallos de memoria en GAT.
> Corregidos ambos, el orden de estabilidad **concuerda** entre regímenes en lugar de invertirse. La
> correlación de rangos entre ejes pasó de **−0,20 (con bug) a +0,80 (corregida)**. Retractar estas
> conclusiones no debilitó la tesis: la hizo más sólida.

---

## Diseño experimental

Diseño factorial sobre la tríada **arquitectura × explicador × balanceo**, cruzada con escenarios
de desbalance, sobre **dos ejes** (Elliptic real y grafo sintético con *ground-truth*):

| Factor | Niveles |
|---|---|
| **Arquitecturas (4)** | GCN · GraphSAGE · GAT · TAGCN |
| **Explicadores (3)** | GNNExplainer · PGExplainer · GNNShap |
| **Balanceo (3)** | sin balanceo · class weighting · focal loss |
| **Escenarios (5)** | 1:1 · 1:10 · **1:30 nativo** · 1:50 · 1:100 |

- **Desbalance extremo** se define operativamente como razón ilícito:lícito ≥ 1:50 (prevalencia ≤ 2 %).
- El escenario **1:30 nativo** preserva la distribución natural del dataset (baseline Weber 2019).
- Cada explicación se repite **5 veces con semillas distintas** para medir estabilidad; el eje
  sintético añade robustez con **3 semillas de modelo × 3 grafos independientes** y estadística
  (Kruskal-Wallis, Wilcoxon, intervalos por bootstrap).
- Los experimentos se distribuyen en **2 máquinas** — ver [§ Configuración por máquina](#configuración-por-máquina).

---

## Estructura del proyecto

```
gnns_thesis/
├── tesis_latex/                       # ← MANUSCRITO (fuente LaTeX + main.pdf, 103 pp, 8 caps)
│   ├── main.tex · chapter_1..8/ · tables/ · bibliografia.bib
│   └── main.pdf                        # PDF compilado y versionado
├── presentacion_latex/                # ← DEFENSA en Beamer (beamer_defensa.tex + .pdf, 26 slides)
├── docs/                              # guiones y material de defensa
├── CAMBIOS_CONSOLIDACION_2026-07-22.md # registro detallado de la consolidación (punto de entrada)
├── RUNBOOK_CONSOLIDACION.md           # plan operativo de la consolidación
├── configs/
│   ├── experiment_machineB_v3.yaml     # ← config vigente Máquina B (RTX 4060): GCN, GraphSAGE
│   ├── experiment_machineC_v3.yaml     # ← config vigente Máquina C (RTX 3050): GAT, TAGCN
│   └── experiment_machine*.yaml        # configs v1/v2 (legacy)
├── scripts/
│   ├── train_matrix.py          # ← Pipeline v3 · Paso 1: entrena la matriz + quality gate
│   ├── explain_matrix.py        # ← Pipeline v3 · Paso 2: explainers + estabilidad
│   ├── smoke_test.py            # validación end-to-end (~10 min, 15 checks)
│   ├── consolidacion/           # scripts de la consolidación (regenerar tablas/figuras, ROC-AUC)
│   └── ...                      # etapas standalone + diagnósticos de PGExplainer
├── src/
│   ├── models/        # GCN, GraphSAGE, GAT, TAGCN (interfaz común)
│   ├── data/          # carga Elliptic, split temporal, escenarios de desbalance
│   ├── training/      # Trainer (early stopping, calibración) + hyperopt (Optuna)
│   ├── balancing/     # FocalLoss, class weighting, GraphSMOTE (implementado, no cableado en v3)
│   ├── explainability/# runners de GNNExplainer/PGExplainer y GNNShap + extracción
│   ├── stability/     # réplicas estocásticas, perturbación, métricas (Jaccard/Spearman)
│   └── analysis/      # tracking MLflow+CSV, ANOVA factorial, matriz de recomendación
├── phase1/            # eje sintético (grafos + CSVs de resultados) — NO re-correr
├── results_v3/        # CSVs de estabilidad + reeval_metrics.csv (provenance versionada)
├── results_models_v3/ # checkpoints *_best.pt + *_meta.json (NO en git)
├── data/              # dataset Elliptic (auto-descargado por PyG, ~300MB · NO en git)
├── pyproject.toml     # dependencias (gestionado con uv)
└── README.md
```

> **Nota:** `data/`, `results_models_v3/`, `mlruns/` y `*.pt` están en `.gitignore`. Los CSVs de
> provenance de la consolidación (`results_v3/xai-gnn-stability-B-v3.csv`, `results_v3/reeval_metrics.csv`)
> sí se versionan de forma explícita para respaldar las tablas del manuscrito.

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
corre el pipeline (desde `data.pyg.org`). Sin internet, copiar manualmente la carpeta `data/`.

Para **compilar el manuscrito** (requiere una distribución LaTeX con `biber`, p.ej. TinyTeX):

```bash
cd tesis_latex
pdflatex -interaction=nonstopmode main.tex && biber main \
  && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```

---

## Pipeline v3 (workflow)

El pipeline está **dividido en dos scripts** para iterar sobre la calidad del modelo de forma
independiente de los explicadores. La v2 producía modelos con F1 ~0,02–0,08; la v3 corrigió tres
bugs centrales (focal loss, early stopping, search space de Optuna) y agregó un quality gate.

### 0. Validar (smoke test, ~10 min)

```bash
uv run python scripts/smoke_test.py --config configs/experiment_machineB_v3.yaml
```

Corre una config mínima end-to-end y valida 15 checks (estructura de config, quality gate, dataset,
escenario de desbalance, entrenamiento, selección de nodos, los 3 explicadores sin crash/OOM, schema
CSV, semántica del alpha de FocalLoss, warm-start priors, PR-AUC, schema del metadata JSON y
calibración de threshold).

> El smoke test **no** cubre el fix R1 de `spearman_rank_agreement`: esa corrección no tiene test de
> regresión (ver [§ Bugs metodológicos corregidos](#bugs-metodológicos-corregidos)).

### 1. Entrenar la matriz

```bash
uv run python scripts/train_matrix.py --config configs/experiment_machineB_v3.yaml --max-hours 9
```

- Corre **Optuna con warm-start** (prior de literatura como trial 0) + búsqueda bayesiana (TPE).
- Entrena el modelo final, **calibra el threshold** sobre validación (remuestreando val a la
  prevalencia de test, para corregir el covariate shift temporal).
- Aplica el **quality gate** sobre validación y persiste `*_best.pt` + `*_meta.json`. **No corre explainers.**
- Flags: `--resume`, `--quick`, `--arch`, `--scenario`, `--balancing`, `--device auto`, `--max-hours`.

### 2. Explicar solo los modelos que aprendieron

```bash
uv run python scripts/explain_matrix.py --config configs/experiment_machineB_v3.yaml
```

- Lee los `*_meta.json`, filtra por `quality_passed=True` (`--force` ignora el gate).
- Corre **GNNExplainer + PGExplainer + GNNShap** + estabilidad (**5 réplicas**) sobre los
  **verdaderos positivos de validación** (`mask_name="val_mask"` en `explain_matrix.py`), no sobre
  test: el clasificador colapsa en test por el shift temporal y explicar predicciones erradas no
  informa. Está declarado de forma abierta en el manuscrito (Cap. 4, §4.4).
- Escribe `results_v3/xai-gnn-stability-B-v3.csv` + MLflow nested runs.
- Flags: `--arch`, `--scenario`, `--balancing`, `--explainer`, `--force`, `--resume`, `--max-hours`.

> **Nota de reproducibilidad (consolidación):** para evitar OOM en GPUs de 8 GB, la estabilidad se
> recomputó lanzando **un proceso fresco por configuración** (troceando la matriz por
> `--arch/--scenario/--balancing`), de modo que la memoria se libera entre configs. Las tablas y
> figuras de Elliptic se regeneran de forma determinista con `scripts/consolidacion/finalize_elliptic.py`,
> y las métricas ROC-AUC/precision@k con `scripts/consolidacion/reeval_rocauc.py`.

### Monitoreo

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlruns.db   # UI de MLflow en vivo
```

---

## Configuración por máquina

Los experimentos se reparten en dos GPUs. Ambas configs comparten estructura y los mismos 5
escenarios y 3 técnicas de balanceo; difieren en arquitecturas y presupuesto de cómputo:

| Campo | **Máquina B** (RTX 4060, 8 GB) | **Máquina C** (RTX 3050, 4 GB) |
|---|---|---|
| Arquitecturas | **GCN, GraphSAGE** | **GAT, TAGCN** |
| `data.root` | `./data` | `./data` |
| `hidden_dim` choices | `[64,128,140,211,256]` | `[64,128,148]` (cap por VRAM) |
| `optuna_trials` | **50** | **8** |
| `training.epochs` | **600** | **150** |
| `training.patience` | **50** | **20** |
| GNNShap `num_samples` | **50** | **25** |
| `experiment_name` | `xai-gnn-stability-B-v3` | `xai-gnn-stability-C-v3` |

Matriz total = 5 escenarios × 4 arquitecturas × 3 balanceos = **60 configuraciones de training**.
Como `explain_matrix.py` recorre todos los `*_meta.json`, un solo config de explain cubre las 4
arquitecturas. De las 60, **23 superan el quality gate** y reciben el estudio completo de estabilidad.

---

## Detalles técnicos

### Modelos (`src/models/`)

Interfaz común `__init__(in_channels, hidden_channels=128, num_layers=2, dropout=0.3, num_classes=2)`
y `forward(x, edge_index)`, con patrón `Conv → activación → Dropout`:

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
- **Split temporal causal** (`preprocessing.py`): train = time-steps 1–34, val = 35–42, test = 43–49.
  Features normalizadas con **RobustScaler** ajustado solo en train (anti-leakage) y clip a [-10, 10].
- **Escenarios de desbalance** (`imbalance.py`): por *undersampling*, modificando solo `train_mask`
  (val/test intactos).

### Entrenamiento (`src/training/`)

- **`Trainer`**: full-batch transductivo, early stopping configurable (`early_stop_metric` default
  **F1**). `evaluate()` devuelve `loss, f1, mcc, pr_auc`. Calibra el threshold barriendo 101 puntos,
  con opción de remuestrear val a la prevalencia de test. Best checkpoint reanudable (RNG state).
- **`hyperopt.py`** (Optuna): **warm-start priors** de literatura como trial 0; `TPESampler` +
  `MedianPruner`; métrica objetivo por default **PR-AUC**.

### Balanceo (`src/balancing/`)

- **FocalLoss** con `α=0.75, γ=2.0`. **Semántica v3:** `alpha` es el peso de la **clase rara** → la
  clase ilícita recibe 3× el peso.
- **GraphSMOTE** implementado (Zhao et al. 2021) pero **no cableado** en los configs v3.

### Estabilidad (`src/stability/`)

- **Réplicas estocásticas** (`stochastic_test.py`): re-ejecuta cada explicador con seeds
  determinísticas; en v3 → `num_replicas=5`. PGExplainer, por ser paramétrico global, se reentrena
  una vez por réplica.
- **Métricas** (`metrics.py`): **Spearman** (acuerdo de rankings de importancia — *primaria*),
  **Jaccard** (aristas — secundaria, satura por el top-K). La función `spearman_rank_agreement` se
  corrigió en la consolidación: dimensiona el vector de rangos por el número real de features, no por
  `top_k` (el bug truncaba los rankings y distorsionaba la estabilidad de Elliptic).

### Análisis (`src/analysis/`)

- **`tracking.py`**: `ExperimentTracker` sobre MLflow con fallback CSV de schema fijo y escritura
  atómica. **`factorial.py`**: ANOVA factorial (statsmodels) + Tukey HSD. **`recommendation.py`**:
  matriz de recomendación con bootstrap CIs.

### Quality gate

> ⚠️ **Valor efectivo:** el gate que decide `quality_passed` se lee del YAML: **F1 ≥ 0,30 y MCC ≥ 0,15**
> sobre **validación** (no test, por el covariate shift temporal "dark market shutdown"). Los defaults
> de fallback en el código (0,70 / 0,40) son sobrescritos por el YAML. De las 60 configuraciones,
> **23 pasan** y sobre ese subconjunto se afirman las conclusiones de estabilidad.

---

## Bugs metodológicos corregidos

### Artefactos de evaluación (contribución central de la tesis)

Dos defectos en el **protocolo de medición** (no en los métodos) que, cada uno por separado, bastaban
para invertir la conclusión sobre qué arquitectura es más estable:

| Artefacto | Síntoma | A quién favorecía | Fix |
|---|---|---|---|
| **Memoria / grafo completo** | 13 configs de GAT fallaban por OOM silencioso; sus filas quedaban vacías y el promedio de GAT se calculaba solo sobre las que terminaban | GAT | calcular sobre el **subgrafo receptivo** (7,5 GiB → 0,02 GiB) |
| **Truncamiento de Spearman (R1)** | la métrica descartaba features con índice ≥ `top_k` y mutilaba los rankings, deprimiendo más a las arquitecturas de importancia repartida | GraphSAGE | dimensionar el vector de rangos por nº real de features (`src/stability/metrics.py`) |

Corregidos ambos, el ranking de estabilidad se estabiliza (GAT/GCN líderes) y **concuerda** con el
eje sintético. Conecta con la advertencia de Kosan et al. sobre la sensibilidad de las conclusiones al
protocolo de evaluación.

### Bugs en PGExplainer de PyTorch Geometric 2.7 (reportados upstream)

| Bug | Síntoma | Fix |
|---|---|---|
| `edge_size=0.05` (default) | *mode collapse*: toda la masa de atribución en una sola arista | `edge_size=0.005` |
| `temp=[5.0,2.0]` (default) | *overflow* numérico → ~99 % de épocas con loss NaN en grafos grandes | `temp=[1.0,1.0]` + gradient clipping |

### Otros bugs corregidos en `src/` (no revertir)

| Bug | Archivo | Fix |
|---|---|---|
| FocalLoss alpha invertido (sub-pesaba la clase rara) | `src/balancing/losses.py` | `alpha` = peso clase rara; default 0,75 |
| Early stopping en MCC ruidoso bajo imbalance extremo | `src/training/trainer.py` | `early_stop_metric` configurable (default F1) |
| Optuna sin prior de literatura, search space estrecho | `src/training/hyperopt.py` | `get_warm_start_priors()` + rango expandido |
| CSV corruption (error embebido en columna numérica) | `src/analysis/tracking.py` | `CSV_SCHEMA_FIELDS` fijo + escritura atómica |

---

## Resultados y artefactos

**Estabilidad por arquitectura (Spearman de GNNExplainer, métrica corregida, 3 semillas de modelo):**

| Arquitectura | Elliptic (media ± dt) | IC95 % bootstrap | Sintético (denso) | Grupo |
|---|---|---|---|---|
| GAT | **0,782** ± 0,013 | [0,758; 0,805] | 0,960 | alto |
| GCN | **0,758** ± 0,025 | [0,724; 0,787] | 0,968 | alto |
| GraphSAGE | 0,735 ± 0,022 | [0,710; 0,756] | 0,888 | bajo |
| TAGCN | 0,676 ± 0,077 | [0,632; 0,717] | 0,886 | bajo |

> Los valores del eje sintético provienen de la matriz robusta (`phase1/results_robust.csv`, 3 semillas),
> que es la fuente que usa el manuscrito. Una versión anterior de esta tabla citaba la matriz factorial
> de una sola semilla, donde GraphSAGE y TAGCN aparecen invertidos por 0,003.

**Estructura en dos grupos (Wilcoxon pareado sobre celdas comunes):**

| Comparación | Elliptic | Sintético |
|---|---|---|
| Dentro del grupo alto (GAT vs GCN) | p = 0,165 | p = 0,0013 |
| Dentro del grupo bajo (GraphSAGE vs TAGCN) | p = 0,096 | p = 0,519 |
| Entre grupos | p ≤ 0,047 en las 4 comparaciones | p < 0,0001 |

Kruskal-Wallis global sobre Elliptic: H = 23,79, p = 2,8×10⁻⁵, η² = 0,125. Dentro de cada grupo la
igualdad no se rechaza (p = 0,192 y p = 0,080); entre grupos, Mann-Whitney da p = 1,4×10⁻⁶. La única
diferencia intragrupo que alcanza significación es GCN sobre GAT en el eje sintético, por un margen de
0,006 sin relevancia práctica.

- **Concordancia entre regímenes:** correlación de rangos Elliptic ↔ sintético = **+0,80** (con la
  métrica defectuosa era −0,20). GAT vs GraphSAGE **no** es significativo (Wilcoxon p = 0,375; IC95 %
  solapados). Se afirma que GAT/GCN encabezan y TAGCN queda atrás.
- ⚠️ **Cómo leer la columna filtrada.** El soporte por arquitectura es muy desigual (GCN n=1,
  TAGCN n=4, GAT n=7, GraphSAGE n=11): el 0,833 de GCN descansa sobre **una sola** configuración y no
  admite lectura inferencial. El ranking que se afirma es el de la **corrida completa (60)**, donde
  las cuatro arquitecturas tienen soporte comparable; la columna filtrada se reporta como control de
  robustez, y su valor es que **no invierte** el orden, no que lo cuantifique. Por eso la conclusión
  se enuncia como "GAT y GCN encabezan, TAGCN queda atrás" y nunca como un ordenamiento fino.
- **Explicadores:** PGExplainer domina plausibilidad (0,80 vs 0,50) pero colapsa en fidelidad
  (0,11 vs 0,56 de GNNExplainer); GNNShap es el más estable internamente. Puente
  estabilidad→plausibilidad nulo (r ≈ −0,01).
- **Rendimiento predictivo:** los modelos aprenden en validación (PR-AUC ≈ 0,37) pero colapsan en
  test (PR-AUC ≈ 0,02) por el desplazamiento temporal del dataset. El ROC-AUC se ve engañosamente
  alto bajo desbalance (0,88 en validación), por lo que se usan **PR-AUC y precision@k** como
  primarias. La estabilidad se estudia sobre verdaderos positivos de validación, donde el modelo
  discrimina, y se declara de forma transparente.

**Artefactos versionados:** `results_v3/xai-gnn-stability-B-v3.csv` (estabilidad),
`results_v3/reeval_metrics.csv` (ROC-AUC/PR-AUC/precision@k por checkpoint y partición), las tablas
y figuras del manuscrito en `tesis_latex/`, y los scripts de regeneración en `scripts/consolidacion/`.
Los checkpoints (`results_models_v3/*_best.pt`) y la base MLflow no se versionan.

---

## Manuscrito y defensa

El manuscrito y el material de defensa viven **en este repositorio**:

| Artefacto | Ubicación |
|---|---|
| **Manuscrito (fuente LaTeX)** | [`tesis_latex/`](tesis_latex/) — `main.tex` + 8 capítulos + `tables/` + `bibliografia.bib` |
| **Manuscrito (PDF)** | [`tesis_latex/main.pdf`](tesis_latex/main.pdf) — 103 páginas |
| **Presentación de defensa (Beamer)** | [`presentacion_latex/beamer_defensa.pdf`](presentacion_latex/beamer_defensa.pdf) — 26 slides |
| **Guiones de defensa** | `docs/GUION_defensa_por_capitulo.md`, `docs/ESQUELETO_presentacion_defensa.md`, `docs/DEFENSA_R2_evidencia_sintetica.md` |

### Estructura del manuscrito (8 capítulos)

| # | Capítulo |
|---|---|
| 1 | Introducción (planteamiento, estado del arte, brecha, objetivos, hipótesis) |
| 2 | Marco Contextual (AML, sistemas de monitoreo) |
| 3 | Fundamentos de IA y GNNs para detección de fraude |
| 4 | Diseño experimental y resultados sobre Elliptic (EDA, colapso val→test, dos correcciones) |
| 5 | Eje sintético (plausibilidad, fidelidad, análisis estadístico) |
| 6 | Discusión (lectura conjunta, tres dimensiones, contribuciones, limitaciones) |
| 7 | Conclusiones y perspectivas futuras |
| 8 | Anexos (hiperparámetros, resultados completos, entorno de cómputo) |

---

## Limitaciones y trabajo futuro

**Limitaciones declaradas:** la evidencia inferencial más fuerte proviene del eje sintético (único
donde plausibilidad y fidelidad son medibles); el clasificador colapsa en test por el desplazamiento
temporal; el eje real se apoya en un solo dataset anonimizado; y con 3 semillas la evidencia sostiene
la pertenencia a un grupo, no un ordenamiento fino dentro de él (TAGCN, con dt 0,077, es el caso más
sensible).

**Reproducibilidad, en sentido preciso:** el pipeline **no es determinista a nivel de pesos**, porque
los *scatter* del message passing en GPU acumulan con sumas atómicas de orden indeterminado. Lo que sí
se reproduce son las conclusiones: un reentrenamiento independiente de las 60 configuraciones devolvió
25 sobre el gate frente a 23, y la misma partición de estabilidad.

**Perspectivas futuras (Cap 7–8):** extender el eje real a datasets con atributos no anonimizados
(Elliptic2, Bellei 2024), incorporar desplazamiento temporal al grafo sintético, GNNs temporales
(EvolveGCN), estabilización específica de PGExplainer en grafos densos, y GraphSMOTE.

---

> Tesis de Maestría en Ingeniería en Sistemas y Computación · Universidad Tecnológica de Pereira · 2026
