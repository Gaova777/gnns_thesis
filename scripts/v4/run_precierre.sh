#!/usr/bin/env bash
# Partición antes del cierre del mercado (THE-44; reunión con Cristian del 7-oct).
# El test original (t = 43 a 49) cae después del cierre de un mercado ilegal y ningún modelo
# supera al azar. Aquí se repite la matriz principal con todo antes del cierre:
#   entrenamiento t = 1 a 29 · validación t = 30 a 35 · prueba t = 36 a 42
# Mismo protocolo de la v4 (igual que scripts/v4/run_estres.sh), en carpetas separadas:
#   1. semilla 42 · 4 arquitecturas · Optuna de 8 trials
#   2. semillas 43/44 · --reuse-hp
#   3. explicación de los 36 modelos por escenario (--include-gated)
#   4. estabilidad entre semillas (sin filtro y con compuerta)
# Una sola ejecución a la vez. Retomable: todo va con --resume.
# Uso: bash scripts/v4/run_precierre.sh [--device cuda|cpu|auto] [--scenarios "native 1:10 ..."]
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="configs/experiment_v4_precierre.yaml"
DEVICE="auto"
SCENS="native 1:10 1:20 1:10_os"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift ;;
    --scenarios) SCENS="$2"; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "argumento desconocido: $1" >&2; exit 64 ;;
  esac
  shift
done

PY=(uv run --frozen python)
ARCHS=(GCN GraphSAGE GAT TAGCN)
LOGS="runs_v4_precierre"
RES="results_v4_precierre"
mkdir -p "$LOGS" "$RES"
DRIVER="$LOGS/PRE_driver.log"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$DRIVER"; }

say "inicio precierre · escenarios=[$SCENS] · device=$DEVICE"
for SCEN in $SCENS; do
  TAG="${SCEN//:/-}"
  for A in "${ARCHS[@]}"; do
    "${PY[@]}" scripts/train_matrix.py --config "$CONFIG" --device "$DEVICE" \
      --scenario "$SCEN" --arch "$A" --seed 42 --resume >> "$LOGS/PRE_${TAG}_train_${A}_s42.log" 2>&1
    say "$SCEN train $A s42 rc=$?"
  done
  for SEED in 43 44; do
    for A in "${ARCHS[@]}"; do
      "${PY[@]}" scripts/train_matrix.py --config "$CONFIG" --device "$DEVICE" \
        --scenario "$SCEN" --arch "$A" --seed "$SEED" --reuse-hp --resume \
        >> "$LOGS/PRE_${TAG}_train_${A}_s${SEED}.log" 2>&1
      say "$SCEN train $A s$SEED rc=$?"
    done
  done
  for SEED in 42 43 44; do
    for A in "${ARCHS[@]}"; do
      "${PY[@]}" scripts/explain_matrix.py --config "$CONFIG" --device "$DEVICE" \
        --scenario "$SCEN" --arch "$A" --seed "$SEED" --include-gated --resume \
        >> "$LOGS/PRE_${TAG}_explain_${A}_s${SEED}.log" 2>&1
      say "$SCEN explain $A s$SEED rc=$?"
    done
  done
done

"${PY[@]}" scripts/v4/cross_seed_stability.py --results-dir "$RES" >> "$LOGS/PRE_cross_seed.log" 2>&1
say "cross_seed (sin filtro) rc=$?"
cp "$RES/elliptic_v4_cross_seed.csv" "$RES/elliptic_v4_cross_seed_sinfiltro.csv"
"${PY[@]}" scripts/v4/cross_seed_stability.py --results-dir "$RES" --gate-only >> "$LOGS/PRE_cross_seed.log" 2>&1
say "cross_seed (compuerta) rc=$?"
say "fin precierre"
