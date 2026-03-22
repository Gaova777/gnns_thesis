# Guía del Usuario: Pipeline de Investigación XAI + GNN 🚀

Esta es la guía oficial de cómo usar, configurar y leer los resultados del pipeline de investigación para el estudio de la estabilidad de métodos XAI en Redes Neuronales de Grafos (GNNs) para el dataset Elliptic.

---

## 1. ⚙️ Archivo de Configuración Principal (`configs/experiment.yaml`)

**TODO** el comportamiento del pipeline, modelos, escenarios y parámetros se define desde el archivo `configs/experiment.yaml`. Nunca deberías tener que tocar el código madre para cambiar algo básico.

Puedes abrir este archivo en tu editor para cambiar:
- **Estrategias de imbalances:** Cambia `ratios` dentro de `scenarios.imbalance_ratios`.
- **Arquitecturas:** Agrega o quita modelos en la lista `models.architectures` (Soporta `GCN`, `GraphSAGE`, `GAT`, `TAGCN`).
- **Hiperparámetros:** Edita los valores de `models.hyperparameter_search` o `training.epochs`, `training.patience`, y `training.learning_rate`.
- **Técnicas de balanceo:** Ajusta factores como `gamma` (Focal Loss) o `up_scale` (GraphSMOTE) en `balancing.techniques`.
- **Pruebas de estabilidad:** Configura los niveles de perturbación (`noise_levels`), el número de grafos (`num_replicas`), y número de nodos a explicar (`nodes_per_class`).
- **Tracking / Guardado:** Cambia el destino de `results_dir`, y escoge entre usar SQLite/MLflow o un simple archivo `.csv`.

---

## 2. 🏃‍♂️ Ejecutando Partes Individuales del Pipeline

El proyecto está diseñado de forma modular. **Antes de correr cualquier comando**, asegúrate de que le estás indicando a Python donde está la carpeta de tu código agregando `$env:PYTHONPATH="."`.

### A. Para Entrenar un Modelo (Pruebas Unitarias o Ajustes Finos)
Prueba entrenar una configuración específica (ej., GAT con Focal Loss y un escenario de 1:50) en la terminal así:
```bash
$env:PYTHONPATH="."; uv run python scripts/run_training.py --model GAT --scenario "1:50" --balancing focal_loss
```
> **Tip:** Usa `--dry-run` para correr apenas 5 épocas y asegurarte que todo funciona antes de iniciar un entrenamiento largo de 300 épocas.

**¿Dónde se guarda?**
Los mejores pesos del modelo entrenado (*checkpoints*) se guardan localmente en:
📂 `results/models/[Modelo]_[Escenario]_[Balanceo]_s[Semilla]_best.pt`

### B. Para Extraer Explicaciones (XAI)
Puedes correr un explicador sobre un modelo pre-entrenado:
```bash
$env:PYTHONPATH="."; uv run python scripts/run_explain.py --model GCN --explainer GNNExplainer --checkpoint results/models/GCN_1:10_none_s42_best.pt
```

**¿Dónde se guarda?**
Crea un archivo local `.json` de fácil lectura en:
📂 `results/explanations/[Modelo]_[Explicador].json`

### C. Para Evaluar la Estabilidad de este Explicador
Permite correr múltiples iteraciones con ruidos o diferentes semillas estocásticas para el cálculo sobre los nodos.
```bash
$env:PYTHONPATH="."; uv run python scripts/run_stability.py --model GCN --explainer GNNExplainer --replicas 30
```

**¿Dónde se guarda?**
Calcula la matriz Jaccard, correlación de Spearman y la Concentración SHAP y las tira en crudo a:
📂 `results/stability/[Modelo]_[Explicador]_stability.json`

---

## 3. 🎯 Ejecutando el Pipeline Completo (La Matriz Experimental)

Una vez que comprobaste que todo funciona y has configurado el `experiment.yaml` acorde a tus deseos, vas a lanzar TODOS los experimentos como lo requiere la tesis usando un único comando:

```bash
$env:PYTHONPATH="."; uv run python scripts/run_full_pipeline.py
```
Este script hace un **grid search profundo**, ejecutando una tras otra de las combinaciones (4 modelos × 4 balances × 4 ratios = 64 variaciones de entrenamiento; multiplicadas por los 3 XAI resultando en 192 evaluaciones de estabilidad con sus réplicas de nodos).

> **Aviso:** Esto demora mucho tiempo. Es recomendado dejarlo correr durante la noche y asegurarse de que el computador no pase a estado de "dormir", o en un servidor dedicado.

---

## 4. 📊 ¿Cómo leer y analizar los Resultados Finales?

Todo el Pipeline está conectado a un trazador automático (`ExperimentTracker`).

### 1. La Base de Datos principal (`results/[nombre_experimento].csv`)
Si configuraste `backend: "csv"` en el yaml (es lo que viene por defecto), todos los 192 experimentos con su `F1-Score`, `MCC`, y `Jaccard Mean / Spearman Mean` se tabularán automáticamente línea por línea en el archivo CSV configurado. 
Podrás abrir este CSV fácilmente en Excel o Python(Pandas) para trazar tus propias curvas o pivot tables de degradación de estabilidad a medida que aumenta el desbalanceo.

### 2. Tablas y Estadísticas (`src/analysis/`)
Dentro de la carpeta `src/analysis/` existen ya las formulas listas de **Tesis**:
- **`factorial.py`**: Es posible correr la función `run_factorial_anova(tu_dataframe)` en los notebooks para extraer los p-values de ANOVA multifactorial buscando las debilidades.
- **`recommendation.py`**: Utiliza `build_recommendation_matrix` que filtra cuáles cumplen los umbrales de MCC>0.70 y Jaccard>0.70 dándote un Intervalo de Confianza (Bootstrapping CIs). Al final, escupe a `results/recommendation_matrix.tex` una **Tabla para LaTeX lista para copiar a tu tesis**. 

### 3. MLFlow (Opcional - Interfaz Web Interactiva)
Si cambias `backend: "mlflow"` dentro de `configs/experiment.yaml`, luego de ejecutar los experimentos puedes simplemente abrir a la consola para ver todas las métricas en visual en vivo:
```bash
uv run mlflow ui
```
¡Se abrirá una pestaña de internet muy limpia donde podrás filtrar métricas como MCC vs Estabilidad en una gráfica y descargar en Excel!