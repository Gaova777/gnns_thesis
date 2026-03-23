# Estabilidad de Metodos XAI en GNNs para Deteccion de Lavado de Dinero

Pipeline de investigacion para el estudio de la estabilidad de metodos de explicabilidad (XAI) en Graph Neural Networks (GNNs) aplicados a la deteccion de lavado de dinero en el **Elliptic Bitcoin Dataset** bajo condiciones de desbalance de datos.

## Estructura del Proyecto

```
gnns_thesis/
|-- configs/
|   +-- experiment.yaml        # Configuracion central (modelos, escenarios, hiperparametros)
|-- scripts/
|   |-- run_full_pipeline.py   # Orquestador completo (192 configs experimentales)
|   |-- run_training.py        # Entrenar un modelo individual
|   |-- run_explain.py         # Ejecutar explicabilidad sobre un modelo entrenado
|   +-- run_stability.py       # Ejecutar pruebas de estabilidad
|-- src/
|   |-- data/                  # Carga, preprocesamiento, escenarios de desbalance
|   |-- models/                # GCN, GraphSAGE, GAT, TAGCN
|   |-- balancing/             # Focal Loss, Weighted CE, GraphSMOTE
|   |-- training/              # Entrenamiento con early stopping + Optuna hyperopt
|   |-- explainability/        # GNNExplainer, PGExplainer, SHAP permutacional
|   |-- stability/             # Tests estocasticos, perturbacion, metricas
|   +-- analysis/              # Tracking MLflow/CSV, ANOVA factorial, recomendacion
|-- pyproject.toml             # Dependencias y configuracion de uv
|-- uv.lock                    # Lockfile de dependencias exactas
+-- README.md
```

## Requisitos Previos

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** (gestor de paquetes y entornos)
- **GPU con CUDA 12.4** (recomendado, funciona sin GPU pero es muy lento)

### Instalacion de uv (si no lo tienes)

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Instalacion

```bash
# 1. Clonar el repositorio
git clone <URL_DEL_REPO>
cd gnns_thesis

# 2. Instalar dependencias (uv crea el entorno automaticamente)
uv sync

# 3. Verificar GPU
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

> **Nota:** El dataset Elliptic (~300MB) se descarga automaticamente la primera vez que corres el pipeline. No necesitas descargarlo manualmente.

## Uso Rapido

Todos los comandos deben ejecutarse desde la raiz del proyecto. En **PowerShell** (Windows), agrega `$env:PYTHONPATH="."` antes de cada comando:

### Smoke Test (verificar que todo funciona)

```powershell
$env:PYTHONPATH="."; uv run python scripts/run_full_pipeline.py --quick --clean
```

### Pipeline Completo

```powershell
# Ejecutar la matriz experimental completa (4 escenarios x 4 archs x 4 balanceos x 3 explainers)
$env:PYTHONPATH="."; uv run python scripts/run_full_pipeline.py --clean
```

### Monitoreo en Tiempo Real (MLflow)

```powershell
# En otra terminal:
uv run mlflow ui --backend-store-uri sqlite:///mlruns.db
# Abrir http://localhost:5000 en el navegador
```

En MLflow podras ver:
- **Curvas de entrenamiento** por epoca (loss, F1, MCC)
- **Metricas de test** por configuracion
- **Metricas de estabilidad** por explainer (Jaccard, Spearman)
- **Comparaciones** entre configuraciones

### Resumir Pipeline Interrumpido

```powershell
# Si el pipeline se detuvo (error, apagon, etc.), usar --resume para continuar:
$env:PYTHONPATH="."; uv run python scripts/run_full_pipeline.py --resume
```

### Entrenar un Modelo Individual

```powershell
$env:PYTHONPATH="."; uv run python scripts/run_training.py --model GAT --scenario "1:10" --balancing focal_loss --epochs 300
```

## Configuracion (`configs/experiment.yaml`)

Toda la configuracion del pipeline se gestiona desde un unico archivo YAML:

| Seccion        | Que controla                                                    |
|----------------|----------------------------------------------------------------|
| `data`         | Ruta de datos y rangos de timesteps para splits temporales      |
| `scenarios`    | Ratios de desbalance a evaluar (1:1, 1:10, 1:50, 1:100)        |
| `models`       | Arquitecturas GNN disponibles y espacio de hiperparametros      |
| `training`     | Epocas, paciencia (early stopping), seeds                       |
| `balancing`    | Tecnicas: none, class_weighting, focal_loss, graphsmote         |
| `explainability`| Metodos XAI y parametros (GNNExplainer, PGExplainer, GNNShap) |
| `stability`    | Replicas estocasticas, niveles de ruido, top-k                  |
| `analysis`     | Umbrales de exito (F1, MCC, Jaccard) y parametros de bootstrap  |
| `tracking`     | Backend (mlflow/csv), nombre del experimento                    |

## Donde se Guardan los Resultados

| Tipo                       | Ubicacion                        |
|----------------------------|----------------------------------|
| Checkpoints de modelos     | `results/models/*.pt`            |
| Metricas (CSV backup)      | `results/xai-gnn-stability.csv`  |
| Metricas (MLflow DB)       | `mlruns.db`                      |
| Dataset (auto-descarga)    | `data/raw/`, `data/processed/`   |

> Todos estos archivos estan en `.gitignore` — no se suben al repositorio.

## Matriz Experimental

El pipeline ejecuta **192 configuraciones** experimentales:

```
4 escenarios de desbalance   x   (1:1, 1:10, 1:50, 1:100)
4 arquitecturas GNN          x   (GCN, GraphSAGE, GAT, TAGCN)
4 tecnicas de balanceo       x   (none, weighted CE, Focal Loss, GraphSMOTE)
3 metodos XAI                    (GNNExplainer, PGExplainer, GNNShap)
= 192 evaluaciones de estabilidad
```

Para cada combinacion se calculan:
- **Metricas predictivas**: F1-Score, MCC, PR-AUC
- **Metricas de estabilidad**: Indice de Jaccard (subgrafos), Correlacion de Spearman (rankings de features), Concentracion SHAP

## Modulos del Codigo

### `src/data/`
- **`loader.py`**: Carga del Elliptic Dataset via PyG con remapeo de etiquetas (licit=0, illicit=1)
- **`preprocessing.py`**: Split temporal causal (train: ts 1-34, val: ts 35-42, test: ts 43-49) + normalizacion StandardScaler
- **`imbalance.py`**: Generador de escenarios de desbalance por undersampling (preserva estructura del grafo)

### `src/models/`
GCN, GraphSAGE, GAT (multi-head), TAGCN (filtros polinomiales K=3)

### `src/balancing/`
- **`losses.py`**: Cross-Entropy estandar, ponderada, y Focal Loss
- **`graphsmote.py`**: GraphSMOTE con encoder GCN + generador de aristas

### `src/training/`
- **`trainer.py`**: Loop de entrenamiento full-batch con early stopping en MCC y logging MLflow por epoca
- **`hyperopt.py`**: Busqueda de hiperparametros con Optuna

### `src/explainability/`
- **`explainer_runner.py`**: GNNExplainer y PGExplainer via API nativa de PyG 2.7
- **`shap_runner.py`**: SHAP permutacional como fallback
- **`extraction.py`**: Extraccion de subgrafos y rankings de features

### `src/stability/`
- **`stochastic_test.py`**: Replicas con diferentes semillas para medir consistencia
- **`perturbation.py`**: Inyeccion de ruido gaussiano para medir robustez
- **`metrics.py`**: Jaccard, Spearman, SHAP Concentration, Fidelity

### `src/analysis/`
- **`tracking.py`**: Tracker unificado MLflow/CSV con parent/nested runs
- **`factorial.py`**: ANOVA factorial multifactorial + Tukey HSD post-hoc
- **`recommendation.py`**: Matriz de recomendacion con intervalos de confianza y export LaTeX

## Stack Tecnologico

| Libreria            | Version   | Uso                                        |
|---------------------|-----------|--------------------------------------------|
| PyTorch             | >= 2.6    | Framework de deep learning (CUDA 12.4)     |
| PyTorch Geometric   | >= 2.7    | GNNs y explainers nativos                  |
| Optuna              | >= 4.0    | Optimizacion de hiperparametros             |
| MLflow              | >= 2.0    | Tracking de experimentos en tiempo real     |
| scikit-learn        | >= 1.5    | Metricas (F1, MCC, PR-AUC)                 |
| statsmodels         | >= 0.14   | ANOVA factorial                             |
| tqdm                | -         | Barras de progreso con ETA                  |