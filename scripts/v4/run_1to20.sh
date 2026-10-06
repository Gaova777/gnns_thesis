#!/usr/bin/env bash
# Escenario 1:20 (reunión con Cristian del 4-oct; THE-35). Reemplaza al 1:1 en el análisis
# principal: con clases balanceadas las tres pérdidas coinciden. Mismo protocolo que el resto
# de la v4, para que sea comparable:
#   1. semilla 42 · 4 arquitecturas · Optuna de 8 trials (presupuesto común)
#   2. semillas 43/44 · --reuse-hp (hiperparámetros de la 42)
#   3. explicación de los 36 modelos, sin filtrar por la compuerta (--include-gated, igual
#      que la pasada de sensibilidad), un proceso por (arquitectura, semilla)
#   4. estabilidad entre semillas (regenera elliptic_v4_cross_seed*.csv con 1:20 incluido)
# Retomable: todo va con --resume. Logs en runs_v4/S20_*.log.
#
# Uso: bash scripts/v4/run_1to20.sh [--device cuda|cpu|auto] [--stages all|train|explain]
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="configs/experiment_v4.yaml"
DEVICE="auto"
STAGES="all"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift ;;
    --stages) STAGES="$2"; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "argumento desconocido: $1" >&2; exit 64 ;;
  esac
  shift
done

PY=(uv run --frozen python)
SCEN="1:20"
ARCHS=(GCN GraphSAGE GAT TAGCN)
LOGS="runs_v4"
DRIVER="$LOGS/S20_driver.log"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$DRIVER"; }

say "inicio 1:20 · device=$DEVICE · stages=$STAGES"

if [[ "$STAGES" != "explain" ]]; then
  for A in "${ARCHS[@]}"; do
    "${PY[@]}" scripts/train_matrix.py --config "$CONFIG" --device "$DEVICE" \
      --scenario "$SCEN" --arch "$A" --seed 42 --resume >> "$LOGS/S20_train_${A}_s42.log" 2>&1
    say "train $A s42 rc=$?"
  done
  for SEED in 43 44; do
    for A in "${ARCHS[@]}"; do
      "${PY[@]}" scripts/train_matrix.py --config "$CONFIG" --device "$DEVICE" \
        --scenario "$SCEN" --arch "$A" --seed "$SEED" --reuse-hp --resume \
        >> "$LOGS/S20_train_${A}_s${SEED}.log" 2>&1
      say "train $A s$SEED rc=$?"
    done
  done
fi

if [[ "$STAGES" != "train" ]]; then
  for SEED in 42 43 44; do
    for A in "${ARCHS[@]}"; do
      "${PY[@]}" scripts/explain_matrix.py --config "$CONFIG" --device "$DEVICE" \
        --scenario "$SCEN" --arch "$A" --seed "$SEED" --include-gated --resume \
        >> "$LOGS/S20_explain_${A}_s${SEED}.log" 2>&1
      say "explain $A s$SEED rc=$?"
    done
  done
  "${PY[@]}" scripts/v4/cross_seed_stability.py >> "$LOGS/S20_cross_seed.log" 2>&1
  say "cross_seed rc=$?"
fi

say "fin 1:20"
