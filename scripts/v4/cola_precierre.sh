#!/usr/bin/env bash
# Cola posterior a la corrida antes del cierre (THE-44). Una sola ejecución a la vez:
#   0. espera a que termine scripts/v4/run_precierre.sh (por PID)
#   1. escenarios de estrés 1:100 y 1:200 con la misma partición (los que pidió Cristian)
#   2. explicadores por gradiente (Integrated Gradients y Expected Gradients) en los seis
#      escenarios, en results_v4_precierre_grad/
#   3. estabilidad entre semillas de los explicadores por gradiente
# Uso: bash scripts/v4/cola_precierre.sh --driver-pid N
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DRIVER_PID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --driver-pid) DRIVER_PID="$2"; shift ;;
    *) echo "argumento desconocido: $1" >&2; exit 64 ;;
  esac
  shift
done

PY=(uv run --frozen python)
CONFIG="configs/experiment_v4_precierre.yaml"
LOG="runs_v4_precierre/PRE_cola.log"
GRAD="results_v4_precierre_grad"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "cola en espera · driver=$DRIVER_PID"
if [[ -n "$DRIVER_PID" ]]; then
  while kill -0 "$DRIVER_PID" 2>/dev/null; do sleep 30; done
fi
if ! grep -q "fin precierre" runs_v4_precierre/PRE_driver.log; then
  say "ABORTA: la corrida principal terminó sin la línea «fin precierre»"
  exit 1
fi

# 1. escenarios de estrés (el driver escribe otra vez «fin precierre» al terminar)
bash scripts/v4/run_precierre.sh --device cuda --scenarios "1:100_subil 1:200_subil" \
  >> runs_v4_precierre/PRE_nohup.log 2>&1
say "estrés 1:100 y 1:200 rc=$?"

# 2. explicadores por gradiente en los seis escenarios (CPU)
mkdir -p "$GRAD/logs"
for SCEN in native 1:10 1:20 1:10_os 1:100_subil 1:200_subil; do
  TAG="${SCEN//:/-}"
  "${PY[@]}" scripts/explain_matrix.py --config "$CONFIG" --device cpu \
    --results-dir "$GRAD" --logs-dir "$GRAD/logs" --scenario "$SCEN" --include-gated \
    --resume --explainer IntegratedGradients,ExpectedGradients --threads 4 \
    > "$GRAD/logs/run_${TAG}.log" 2>&1
  say "gradientes $SCEN rc=$?"
done

# 3. estabilidad entre semillas (gradientes)
"${PY[@]}" scripts/v4/cross_seed_stability.py --results-dir "$GRAD" > "$GRAD/logs/cross_seed.log" 2>&1
say "cross_seed gradientes rc=$?"
say "fin cola precierre"
