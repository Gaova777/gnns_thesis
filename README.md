# Estabilidad de Métodos XAI en GNNs para Detección de Lavado de Dinero

Pipeline de investigación para el estudio de la estabilidad de métodos de explicabilidad (XAI)
en Graph Neural Networks (GNNs) aplicados a la detección de lavado de dinero en el
**Elliptic Bitcoin Dataset** bajo condiciones de desbalance de datos.

---

## Estructura del Proyecto

```
gnns_thesis/
├── configs/
│   ├── experiment_reduced.yaml      # Config base (144 configs, GraphSMOTE excluido)
│   ├── experiment_machineA.yaml     # GAT + TAGCN, 4 escenarios  → 72 configs
│   ├── experiment_machineB.yaml     # GCN + SAGE,  1:1 y 1:10   → 36 configs
│   └── experiment_machineC.yaml     # GCN + SAGE,  1:50 y 1:100 → 36 configs
├── scripts/
│   ├── run_full_pipeline.py         # Orquestador principal
│   ├── run_machineA.sh / .bat       # Launcher Máquina A (RTX 4090)
│   ├── run_machineB.sh / .bat       # Launcher Máquina B (RTX 4060)
│   ├── run_machineC.sh / .bat       # Launcher Máquina C (RTX 3050)
│   ├── run_machineA_on_4060.sh/.bat # Configs de A corriendo en 4060
│   ├── setup_windows.ps1            # Setup CUDA interactivo para Windows
│   └── merge_results.py             # Unifica CSVs de las 3 máquinas
├── src/
│   ├── data/          # Carga, preprocesamiento, escenarios de desbalance
│   ├── models/        # GCN, GraphSAGE, GAT, TAGCN
│   ├── balancing/     # Focal Loss, Weighted CE (GraphSMOTE excluido v1)
│   ├── training/      # Entrenamiento con early stopping + checkpoints por época
│   ├── explainability/# GNNExplainer, PGExplainer, GNNShap (con recuperación OOM)
│   ├── stability/     # Tests estocásticos + métricas de estabilidad
│   └── analysis/      # Tracking MLflow/CSV atómico, ANOVA factorial
├── checkpoints/       # Checkpoints de recuperación por época (auto-generado)
├── results_machineA/  # Resultados Máquina A (auto-generado)
├── results_machineB/  # Resultados Máquina B (auto-generado)
├── results_machineC/  # Resultados Máquina C (auto-generado)
├── logs/              # Logs con timestamp + heartbeat del watchdog
├── pyproject.toml
└── README.md
```

---

## Requisitos

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — gestor de paquetes
- **GPU NVIDIA con CUDA** (recomendado; CPU funciona pero tarda ~10× más)
- **Dataset Elliptic** en `data/raw/` (ver sección Dataset)

### Instalar uv

```bash
# Linux / WSL2
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## Instalación

```bash
git clone <URL_DEL_REPO>
cd gnns_thesis
uv sync
```

### Verificar CUDA

```bash
# Linux / WSL2
PYTHONPATH=. .venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# Windows (PowerShell)
$env:PYTHONPATH="."; .venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

---

## Dataset Elliptic

El dataset **no** se descarga automáticamente. Cópialo manualmente:

```bash
# Ejemplo desde OneDrive / disco externo
cp /ruta/al/dataset/raw/*    data/raw/
cp /ruta/al/dataset/processed/* data/processed/
```

Los archivos esperados en `data/raw/`:
- `elliptic_txs_features.csv`
- `elliptic_txs_edgelist.csv`
- `elliptic_txs_classes.csv`

---

## Ejecución Distribuida — 3 Máquinas

El experimento está dividido en 144 configuraciones (4 escenarios × 4 archs × 3 balanceos × 3 explainers):

| Máquina | GPU | Configs | Arquitecturas | Escenarios |
|---------|-----|---------|---------------|------------|
| A | RTX 4090 (24 GB) | 72 | GAT, TAGCN | 1:1, 1:10, 1:50, 1:100 |
| B | RTX 4060 (8 GB)  | 36 | GCN, GraphSAGE | 1:1, 1:10 |
| C | RTX 3050 (4 GB)  | 36 | GCN, GraphSAGE | 1:50, 1:100 |

### En Linux / WSL2

```bash
# Primera vez en cada máquina
git pull origin main
uv sync

# Lanzar (incluye watchdog, lock file y auto-restart)
bash scripts/run_machineA.sh   # ← Máquina A
bash scripts/run_machineB.sh   # ← Máquina B
bash scripts/run_machineC.sh   # ← Máquina C
```

### En Windows (nativo)

```powershell
# 1. Setup inicial (verifica CUDA, pregunta instalación automática/manual)
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1

# 2. Lanzar (detecta WSL2; si hay WSL usa el .sh, si no usa Python nativo)
scripts\run_machineA.bat
scripts\run_machineB.bat
scripts\run_machineC.bat
```

### Caso especial: configs de A en una RTX 4060

```bash
bash scripts/run_machineA_on_4060.sh   # detecta VRAM y ajusta GNNShap
# o en Windows:
scripts\run_machineA_on_4060.bat
```
Resultados guardados en `results_machineA_4060/` para no mezclar con la 4090.

---

## Reanudar después de una interrupción

Todos los scripts pasan `--resume` automáticamente. Para reanudar manualmente:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_full_pipeline.py \
    --config configs/experiment_machineX.yaml --resume
```

El pipeline usa tres niveles de recuperación:

1. **Checkpoint por época** — guardado en `checkpoints/{run_id}/checkpoint_last.pt`
   cada 10 épocas (y en época 1). Se carga automáticamente al reanudar.
2. **`.interrupted_runs`** — configs interrumpidas por señal se marcan como
   "skip" en el siguiente `--resume` (no se reintenta parcialmente).
3. **CSV atómico + backup** — escrituras via `tmp → os.replace()`.
   Backup automático cada 5 filas en `results_*.csv.bak`.

---

## Smoke Test

Para verificar que todo funciona sin lanzar el experimento completo:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_full_pipeline.py \
    --quick --config configs/experiment_machineC.yaml --device cpu
```

`--quick` reduce a: 5 épocas, 2 réplicas, 3 nodos, 1 arch × 1 balanceo × 3 explainers.

---

## Monitoreo en tiempo real (MLflow)

```bash
PYTHONPATH=. .venv/bin/python -m mlflow ui --backend-store-uri sqlite:///mlruns.db
# Abrir http://localhost:5000
```

Qué ver en MLflow:
- **Curvas por época**: train loss, val F1, val MCC (actualizan en vivo)
- **Métricas de test**: F1, MCC, PR-AUC por configuración
- **Métricas de estabilidad**: Jaccard mean/std, Spearman mean, shap_oom_retries
- **Runs anidados**: un run padre por config de entrenamiento, un run hijo por explainer

---

## Unificar resultados (post-experimento)

Una vez terminadas las 3 máquinas, copiar los CSV a una sola máquina y ejecutar:

```bash
python scripts/merge_results.py --results-root . --output results/results_merged.csv
```

El script:
- Prioriza `results_machineA/` sobre `results_machineA_4060/` si hay solapamiento
- Deduplica por `(scenario, architecture, balancing, explainer)`
- Reporta cobertura (espera 144 configs únicas)
- Advierte si hay `shap_oom_retries > 0`

---

## Robustez frente a cortes (48–66 h sin supervisión)

| Feature | Dónde | Qué hace |
|---|---|---|
| Checkpoint por época | `trainer.py` | Guarda cada 10 épocas; reanuda sin perder progreso |
| Escritura atómica CSV | `tracking.py` | `tmp → os.replace()`; backup cada 5 filas |
| Manejo de señales | `run_full_pipeline.py` | SIGTERM/SIGINT → cierre limpio + marca interrupted |
| Auto-restart (×3) | Scripts `.sh`/`.bat` | Reinicia pipeline si sale con error, espera 30 s |
| Lock file | Scripts `.sh` | Bloquea doble ejecución accidental |
| Watchdog GPU | Scripts `.sh` | Mata proceso si GPU al 0% por 10 min consecutivos |

---

## Matriz Experimental

```
4 escenarios de desbalance  ×  1:1, 1:10, 1:50, 1:100
4 arquitecturas GNN         ×  GCN, GraphSAGE, GAT, TAGCN
3 técnicas de balanceo      ×  none, class_weighting, focal_loss
3 métodos XAI                  GNNExplainer, PGExplainer, GNNShap
────────────────────────────────────────────────────────
144 evaluaciones de estabilidad  (GraphSMOTE excluido de v1)
```

Por cada combinación se calculan:
- **Métricas predictivas**: F1-Score, MCC, PR-AUC
- **Métricas de estabilidad**: Índice de Jaccard (subgrafos), Correlación de Spearman (rankings de features), Concentración SHAP

---

## Parámetros reducidos (`experiment_reduced.yaml` vs original)

| Parámetro | Original | Reducido | Justificación |
|---|---|---|---|
| `nodes_per_class` | 100 | 30 | Análisis de potencia: n=35 para ±0.05 CI al 95% |
| `num_replicas` | 30 | 20 | C(20,2)=190 pares, suficiente para Δ=0.1 |
| `GNNShap.num_samples` | 100 | 50 | Ranking converge a ≈50 permutaciones |

---

## Stack Tecnológico

| Librería | Versión mín. | Uso |
|---|---|---|
| PyTorch | 2.6 | Deep learning (CUDA 12.4+) |
| PyTorch Geometric | 2.7 | GNNs y explainers nativos |
| MLflow | 2.0 | Tracking de experimentos |
| scikit-learn | 1.5 | Métricas (F1, MCC, PR-AUC) |
| statsmodels | 0.14 | ANOVA factorial + Tukey HSD |
| Optuna | 4.0 | Búsqueda de hiperparámetros |
| tqdm | — | Barras de progreso con ETA |
