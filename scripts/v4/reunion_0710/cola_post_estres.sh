#!/usr/bin/env bash
# Cola posterior a la corrida de estrés (1:100 y 1:200). Una sola ejecución a la vez: cada
# paso empieza cuando terminó el anterior, y el primero espera a que termine la corrida de
# estrés (scripts/v4/run_estres.sh). Nada corre al tiempo con la GPU.
#
#   0. espera a que termine el driver de estrés (por PID) y comprueba que cerró bien
#   1. control de pesos al azar: lo reanuda si quedó en pausa (kill -CONT) o lo corre completo
#   2. explicadores por gradiente sobre los modelos de 1:100 y 1:200 (CPU)
#   3. estabilidad entre semillas de los explicadores por gradiente
#   4. análisis de la reunión del 7-oct (analisis.py y metrica_robusta.py)
#   5. curvas PR y ROC de los escenarios de estrés (scores, métricas y figuras)
#
# Uso: bash scripts/v4/reunion_0710/cola_post_estres.sh --driver-pid N [--control-pid N]
# Log: runs_v4/EST_cola.log
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

DRIVER_PID=""
CONTROL_PID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --driver-pid) DRIVER_PID="$2"; shift ;;
    --control-pid) CONTROL_PID="$2"; shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "argumento desconocido: $1" >&2; exit 64 ;;
  esac
  shift
done

PY=(uv run --frozen python)
CONFIG="configs/experiment_v4.yaml"
LOG="runs_v4/EST_cola.log"
GLOGS="results_v4_grad/logs"
OUT="results_v4/reunion_0710"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "cola en espera · driver=$DRIVER_PID · control=$CONTROL_PID"

# 0. la corrida de estrés
if [[ -n "$DRIVER_PID" ]]; then
  while kill -0 "$DRIVER_PID" 2>/dev/null; do sleep 30; done
fi
if ! grep -q "fin estrés" runs_v4/EST_driver.log; then
  say "ABORTA: el driver de estrés terminó sin la línea «fin estrés»"
  exit 1
fi
say "corrida de estrés terminada"

# 1. control de pesos al azar: se reanuda si quedó en pausa; si no, se corre completo
if [[ -n "$CONTROL_PID" ]] && kill -0 "$CONTROL_PID" 2>/dev/null; then
  kill -CONT "$CONTROL_PID"
  say "control de pesos al azar reanudado"
  while kill -0 "$CONTROL_PID" 2>/dev/null; do sleep 15; done
else
  "${PY[@]}" scripts/v4/reunion_0710/control_aleatorio.py --seeds 42 43 44 \
    > "$GLOGS/control_aleatorio.log" 2>&1
  say "control de pesos al azar rc=$?"
fi
say "control de pesos al azar terminado ($(($(wc -l < "$OUT/control_aleatorio.csv") - 1)) filas)"

# 2. explicadores por gradiente en los escenarios de estrés
for SCEN in "1:100_subil" "1:200_subil"; do
  TAG="${SCEN//:/-}"
  "${PY[@]}" scripts/explain_matrix.py --config "$CONFIG" --device cpu \
    --results-dir results_v4_grad --logs-dir "$GLOGS" --scenario "$SCEN" --include-gated \
    --resume --explainer IntegratedGradients,ExpectedGradients --threads 4 \
    > "$GLOGS/run_${TAG}.log" 2>&1
  say "gradientes $SCEN rc=$?"
done

# 3. estabilidad entre semillas (gradientes)
"${PY[@]}" scripts/v4/cross_seed_stability.py --results-dir results_v4_grad \
  > "$GLOGS/cross_seed.log" 2>&1
say "cross_seed gradientes rc=$?"

# 4. análisis
"${PY[@]}" scripts/v4/reunion_0710/analisis.py > "$OUT/analisis.log" 2>&1
say "analisis.py rc=$?"
"${PY[@]}" scripts/v4/reunion_0710/metrica_robusta.py > "$OUT/metrica_robusta.log" 2>&1
say "metrica_robusta.py rc=$?"

# 5. curvas PR y ROC con los escenarios de estrés
"${PY[@]}" scripts/v4/reunion_0410/curvas.py --stage scores --device auto \
  > runs_v4/EST_curvas.log 2>&1
say "curvas scores rc=$?"
"${PY[@]}" scripts/v4/reunion_0410/curvas.py --stage metrics >> runs_v4/EST_curvas.log 2>&1
say "curvas metrics rc=$?"
"${PY[@]}" scripts/v4/reunion_0410/curvas.py --stage figures >> runs_v4/EST_curvas.log 2>&1
say "curvas figures rc=$?"
"${PY[@]}" scripts/v4/reunion_0410/curvas.py --stage figures --deck >> runs_v4/EST_curvas.log 2>&1
say "curvas figures (deck) rc=$?"

say "fin cola"
