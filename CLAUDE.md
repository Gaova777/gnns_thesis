# Guía del proyecto — XAI Stability in GNNs (AML / Elliptic)

## Qué es este proyecto

Tesis de maestría. Estudiamos la **estabilidad de métodos XAI** (GNNExplainer, PGExplainer, GNNShap)
aplicados a GNNs (GCN, GraphSAGE, GAT, TAGCN) entrenadas para detectar transacciones ilícitas
en el dataset Elliptic, bajo distintos niveles de imbalance de clases (1:1, 1:10, 1:50, 1:100).

## Estado actual (2026-04-11)

Los experimentos corren distribuidos en **2 máquinas**. La 4090 quedó sin acceso.

| Máquina | GPU | Architectures | Escenarios | Configs |
|---------|-----|---------------|------------|---------|
| **B** | RTX 4060 8GB | GCN, GraphSAGE | 1:1, 1:10, 1:50, 1:100 | `configs/experiment_machineB.yaml` |
| **C** | RTX 3050 4GB | GAT, TAGCN | 1:1, 1:10, 1:50, 1:100 | `configs/experiment_machineC.yaml` |

**Machine C (3050) ya fue validada** — smoke test 10/10 PASS el 2026-04-11.

## Lo que hay que hacer en Machine B (RTX 4060)

### 1. Clonar / pullear el repo

```bash
git pull origin main
```

### 2. Crear el entorno (si es la primera vez)

```powershell
uv venv
uv sync
```

### 3. Validar con el smoke test (~3 min)

```powershell
uv run python scripts/smoke_test.py --config configs/experiment_machineB.yaml
```

Debe mostrar **10/10 checks passed**. Si algo falla, reportar el error antes de continuar.

### 4. Lanzar el pipeline completo

```powershell
uv run python scripts/run_full_pipeline.py --config configs/experiment_machineB.yaml --max-hours 9.5
```

Para correr desatendido toda la noche y guardar el log:

```powershell
uv run python scripts/run_full_pipeline.py --config configs/experiment_machineB.yaml --max-hours 9.5 | Tee-Object -FilePath logs/machineB.log
```

### 5. Si se interrumpe, retomar

```powershell
uv run python scripts/run_full_pipeline.py --config configs/experiment_machineB.yaml --resume --max-hours 9.5
```

## Estructura del proyecto

```
gnns_thesis/
├── configs/
│   ├── experiment_machineB.yaml   ← config para esta máquina (4060)
│   └── experiment_machineC.yaml   ← config para la 3050
├── scripts/
│   ├── run_full_pipeline.py       ← pipeline principal (usar este)
│   ├── smoke_test.py              ← validación pre-run (~3 min)
│   └── merge_results.py           ← merge de resultados de ambas máquinas
├── src/
│   ├── models/                    ← GCN, GraphSAGE, GAT, TAGCN
│   ├── data/                      ← loader, preprocessing, imbalance
│   ├── training/                  ← trainer, hyperopt (Optuna)
│   ├── explainability/            ← GNNExplainer, PGExplainer, GNNShap
│   ├── stability/                 ← stochastic replicas, perturbation, metrics
│   ├── balancing/                 ← class_weighting, focal_loss, GraphSMOTE
│   └── analysis/                  ← tracking (MLflow + CSV), factorial, recommendation
├── data/                          ← dataset Elliptic (auto-descargado por PyG, ~300MB)
│                                     ⚠ NO está en git — se descarga al correr el pipeline
└── results_machineB/              ← resultados generados (NO en git)
```

## Dataset

El dataset Elliptic se descarga automáticamente en `./data/` la primera vez que corra el pipeline.
Requiere conexión a internet (~300MB desde data.pyg.org).

Si no hay internet, copiar manualmente la carpeta `data/` desde otra máquina que ya lo tenga.

## Qué hace el pipeline

Para cada combinación de (scenario × architecture × balancing):
1. **Hyperopt** — Optuna busca mejores hiperparámetros (hidden_dim, num_layers, dropout, lr)
2. **Training** — Entrena el GNN con los mejores HPs, guarda checkpoint
3. **Quality gate** — Si F1 < 0.05 y MCC < 0.02, salta los explainers (modelo no aprendió)
4. **Explainability** — GNNExplainer, PGExplainer, GNNShap sobre nodos de test
5. **Stability** — Repite la explicación N veces con distintas seeds, calcula Jaccard y Spearman

Resultados se guardan en `results_machineB/xai-gnn-stability-B.csv` y en MLflow.

## Merge final (cuando ambas máquinas terminen)

Copiar `results_machineC/xai-gnn-stability-C.csv` a esta máquina y correr:

```powershell
uv run python scripts/merge_results.py
```

Genera `results/results_merged.csv` con los 144 configs de ambas máquinas.

## Bugs corregidos (no tocar sin entender)

Estos bugs estaban en el código anterior y ya están fixeados:

| Bug | Archivo | Fix |
|-----|---------|-----|
| CSV corruption — error embebido en columna numérica | `src/analysis/tracking.py` | `CSV_SCHEMA_FIELDS` fijo |
| PGExplainer crash cuando training falla | `src/stability/stochastic_test.py:71` | `if not success: continue` |
| `--resume` ignoraba el CSV, solo leía MLflow | `src/analysis/tracking.py` | CSV fallback en `get_completed_runs` |
| Optuna leía `optuna_trials_fast` en vez de `optuna_trials` | `scripts/run_full_pipeline.py:204` | Lee `optuna_trials` primero |

## Comandos útiles

```powershell
# Ver progreso en MLflow UI (mientras corre)
uv run mlflow ui --backend-store-uri sqlite:///mlruns.db

# Quick test (2 min, parámetros mínimos)
uv run python scripts/run_full_pipeline.py --config configs/experiment_machineB.yaml --quick

# Ver cuántos configs completaron
python -c "import pandas as pd; df = pd.read_csv('results_machineB/xai-gnn-stability-B.csv'); print(df.groupby(['scenario','architecture','balancing','explainer']).size())"
```
