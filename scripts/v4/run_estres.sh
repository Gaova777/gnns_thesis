#!/usr/bin/env bash
# Escenarios de estrés 1:100 y 1:200 (reunión con Cristian del 7-oct). En el rango 1:10 a
# 1:38,4 la estabilidad de las explicaciones no se movió, así que se fuerza el desbalance
# más allá del nativo para ver si cae. En modo C el nativo ya usa todos los negativos: para
# llegar a 1:100 y 1:200 sin datos sintéticos se quitan ilícitas (1.328 y 664 de 3.462).
# Mismo protocolo que el resto de la v4 (igual que scripts/v4/run_1to20.sh):
#   1. semilla 42 · 4 arquitecturas · Optuna de 8 trials (presupuesto común)
#   2. semillas 43/44 · --reuse-hp (hiperparámetros de la 42)
#   3. explicación de los 36 modelos por escenario, sin filtrar por la compuerta
#      (--include-gated), un proceso por (arquitectura, semilla)
#   4. estabilidad entre semillas: elliptic_v4_cross_seed_sinfiltro.csv (todos) y
#      elliptic_v4_cross_seed.csv (solo los que pasan la compuerta)
# Retomable: todo va con --resume. Logs en runs_v4/EST_*.log.
#
# Uso: bash scripts/v4/run_estres.sh [--device cuda|cpu|auto] [--stages all|train|explain]
#                                    [--scenarios "1:100_subil 1:200_subil"]
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="configs/experiment_v4.yaml"
DEVICE="auto"
STAGES="all"
SCENS="1:100_subil 1:200_subil"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift ;;
    --stages) STAGES="$2"; shift ;;
    --scenarios) SCENS="$2"; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "argumento desconocido: $1" >&2; exit 64 ;;
  esac
  shift
done

PY=(uv run --frozen python)
ARCHS=(GCN GraphSAGE GAT TAGCN)
LOGS="runs_v4"
DRIVER="$LOGS/EST_driver.log"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$DRIVER"; }

say "inicio estrés · escenarios=[$SCENS] · device=$DEVICE · stages=$STAGES"

for SCEN in $SCENS; do
  TAG="${SCEN//:/-}"
  if [[ "$STAGES" != "explain" ]]; then
    for A in "${ARCHS[@]}"; do
      "${PY[@]}" scripts/train_matrix.py --config "$CONFIG" --device "$DEVICE" \
        --scenario "$SCEN" --arch "$A" --seed 42 --resume >> "$LOGS/EST_${TAG}_train_${A}_s42.log" 2>&1
      say "$SCEN train $A s42 rc=$?"
    done
    for SEED in 43 44; do
      for A in "${ARCHS[@]}"; do
        "${PY[@]}" scripts/train_matrix.py --config "$CONFIG" --device "$DEVICE" \
          --scenario "$SCEN" --arch "$A" --seed "$SEED" --reuse-hp --resume \
          >> "$LOGS/EST_${TAG}_train_${A}_s${SEED}.log" 2>&1
        say "$SCEN train $A s$SEED rc=$?"
      done
    done
  fi

  if [[ "$STAGES" != "train" ]]; then
    for SEED in 42 43 44; do
      for A in "${ARCHS[@]}"; do
        "${PY[@]}" scripts/explain_matrix.py --config "$CONFIG" --device "$DEVICE" \
          --scenario "$SCEN" --arch "$A" --seed "$SEED" --include-gated --resume \
          >> "$LOGS/EST_${TAG}_explain_${A}_s${SEED}.log" 2>&1
        say "$SCEN explain $A s$SEED rc=$?"
      done
    done
  fi
done

if [[ "$STAGES" != "train" ]]; then
  "${PY[@]}" scripts/v4/cross_seed_stability.py >> "$LOGS/EST_cross_seed.log" 2>&1
  say "cross_seed (sin filtro) rc=$?"
  cp results_v4/elliptic_v4_cross_seed.csv results_v4/elliptic_v4_cross_seed_sinfiltro.csv
  "${PY[@]}" scripts/v4/cross_seed_stability.py --gate-only >> "$LOGS/EST_cross_seed.log" 2>&1
  say "cross_seed (compuerta) rc=$?"
fi

say "fin estrés"
