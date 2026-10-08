#!/usr/bin/env bash
# Optimización completa del candidato top (reunión con Cristian del 4-oct; THE-36).
# La matriz v4 (8 trials por configuración) es la fase de selección; de ella sale
# GraphSAGE + class_weighting (1.º en native, 1:10 y 1:20, 2.º en 1:10_os; configs48.py).
# Aquí se le da un Optuna de 150 trials en cada escenario principal y se re-explica:
#   1. semilla 42 · Optuna de 150 trials (mismas épocas por trial y búsqueda que la v4)
#   2. semillas 43/44 · --reuse-hp (hiperparámetros de la 42)
#   3. explicación de los 12 modelos sin filtrar por compuerta
#   4. estabilidad entre semillas
# Todo va a directorios propios (results_models_v4_top/, results_v4_top/) para no mezclarse
# con la matriz. Retomable con --resume. Logs en runs_v4/TOP_*.log.
#
# Uso: bash scripts/v4/run_optuna_top.sh [--device cuda|cpu|auto] [--stages all|train|explain]
#                                         [--arch GraphSAGE] [--balancing class_weighting]
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="configs/experiment_v4.yaml"
DEVICE="auto"
STAGES="all"
ARCH="GraphSAGE"
BAL="class_weighting"
TRIALS=150
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift ;;
    --stages) STAGES="$2"; shift ;;
    --arch) ARCH="$2"; shift ;;
    --balancing) BAL="$2"; shift ;;
    --trials) TRIALS="$2"; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "argumento desconocido: $1" >&2; exit 64 ;;
  esac
  shift
done

PY=(uv run --frozen python)
SCENS=(native 1:10 1:10_os 1:20)
MODELS="results_models_v4_top"
RESULTS="results_v4_top"
LOGS="runs_v4"
DRIVER="$LOGS/TOP_driver.log"
DIRS=(--models-dir "$MODELS" --results-dir "$RESULTS")
mkdir -p "$MODELS" "$RESULTS" "$LOGS"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$DRIVER"; }

say "inicio top $ARCH + $BAL · trials=$TRIALS · device=$DEVICE · stages=$STAGES"

if [[ "$STAGES" != "explain" ]]; then
  for S in "${SCENS[@]}"; do
    "${PY[@]}" scripts/train_matrix.py --config "$CONFIG" --device "$DEVICE" "${DIRS[@]}" \
      --scenario "$S" --arch "$ARCH" --balancing "$BAL" --seed 42 --trials "$TRIALS" \
      --max-minutes-per-config 240 --max-hours 12 --resume \
      >> "$LOGS/TOP_train_${S//:/-}_s42.log" 2>&1
    say "train $S s42 rc=$?"
  done
  for SEED in 43 44; do
    for S in "${SCENS[@]}"; do
      "${PY[@]}" scripts/train_matrix.py --config "$CONFIG" --device "$DEVICE" "${DIRS[@]}" \
        --scenario "$S" --arch "$ARCH" --balancing "$BAL" --seed "$SEED" --reuse-hp --resume \
        >> "$LOGS/TOP_train_${S//:/-}_s${SEED}.log" 2>&1
      say "train $S s$SEED rc=$?"
    done
  done
fi

if [[ "$STAGES" != "train" ]]; then
  for SEED in 42 43 44; do
    "${PY[@]}" scripts/explain_matrix.py --config "$CONFIG" --device "$DEVICE" "${DIRS[@]}" \
      --arch "$ARCH" --balancing "$BAL" --seed "$SEED" --include-gated --resume \
      >> "$LOGS/TOP_explain_s${SEED}.log" 2>&1
    say "explain s$SEED rc=$?"
  done
  "${PY[@]}" scripts/v4/cross_seed_stability.py --results-dir "$RESULTS" \
    >> "$LOGS/TOP_cross_seed.log" 2>&1
  say "cross_seed rc=$?"
fi

say "fin top"
