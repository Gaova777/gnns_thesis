#!/usr/bin/env bash
# THE-36, segunda parte. La optimización completa (run_optuna_top.sh) elige por PR-AUC, pero el
# entrenamiento final se detiene por F1 en argmax (paciencia 20). Con las tasas de aprendizaje
# altas que eligió la búsqueda, 4 de los 12 modelos se detuvieron en la época 21 con el mejor F1
# en la época 1, antes de aprender. Aquí se reentrena con detención temprana por PR-AUC (la
# misma métrica que optimiza Optuna), sin nueva búsqueda, en dos variantes:
#   top_prauc  hiperparámetros de la búsqueda de 150 trials  (results_models_v4_top/)
#   mat_prauc  hiperparámetros de la matriz de 8 trials       (results_models_v4/), control
# El control separa el efecto de los hiperparámetros del efecto de cambiar la detención.
# Cada variante: 4 escenarios × 3 semillas, explicación sin compuerta y estabilidad entre
# semillas, en directorios propios. Retomable con --resume. Logs en runs_v4/TOPPR_*.log.
#
# Uso: bash scripts/v4/run_top_prauc.sh [--device cuda|cpu|auto] [--variants "top_prauc mat_prauc"]
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="configs/experiment_v4.yaml"
DEVICE="auto"
VARIANTS="top_prauc mat_prauc"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift ;;
    --variants) VARIANTS="$2"; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "argumento desconocido: $1" >&2; exit 64 ;;
  esac
  shift
done

PY=(uv run --frozen python)
ARCH="GraphSAGE"; BAL="class_weighting"
SCENS=(native 1:10 1:10_os 1:20)
LOGS="runs_v4"
DRIVER="$LOGS/TOPPR_driver.log"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$DRIVER"; }

for V in $VARIANTS; do
  case "$V" in
    top_prauc) HPFROM="results_models_v4_top" ;;
    mat_prauc) HPFROM="results_models_v4" ;;
    *) echo "variante desconocida: $V" >&2; exit 64 ;;
  esac
  MODELS="results_models_v4_$V"; RESULTS="results_v4_$V"
  DIRS=(--models-dir "$MODELS" --results-dir "$RESULTS")
  mkdir -p "$MODELS" "$RESULTS"
  say "inicio $V · hp de $HPFROM · detención por pr_auc · device=$DEVICE"
  for SEED in 42 43 44; do
    for S in "${SCENS[@]}"; do
      "${PY[@]}" scripts/train_matrix.py --config "$CONFIG" --device "$DEVICE" "${DIRS[@]}" \
        --scenario "$S" --arch "$ARCH" --balancing "$BAL" --seed "$SEED" \
        --hp-from "$HPFROM" --early-stop-metric pr_auc --resume \
        >> "$LOGS/TOPPR_${V}_train_${S//:/-}_s${SEED}.log" 2>&1
      say "$V train $S s$SEED rc=$?"
    done
  done
  for SEED in 42 43 44; do
    "${PY[@]}" scripts/explain_matrix.py --config "$CONFIG" --device "$DEVICE" "${DIRS[@]}" \
      --arch "$ARCH" --balancing "$BAL" --seed "$SEED" --include-gated --resume \
      >> "$LOGS/TOPPR_${V}_explain_s${SEED}.log" 2>&1
    say "$V explain s$SEED rc=$?"
  done
  "${PY[@]}" scripts/v4/cross_seed_stability.py --results-dir "$RESULTS" \
    >> "$LOGS/TOPPR_${V}_cross_seed.log" 2>&1
  say "$V cross_seed rc=$?"
done
say "fin top_prauc"
