#!/usr/bin/env bash
# Orquestador v4 (eje Elliptic) por PRIORIDADES, con punto de parada tras cada bloque.
#
#   P0  semilla 42 · GCN + GraphSAGE      train → explain   → CHECKPOINT
#   P1  semilla 42 · GAT + TAGCN          train → explain   → CHECKPOINT
#   P2  semillas 43/44 · 4 arq. (--reuse-hp)  train → explain → CHECKPOINT
#   P3  estabilidad entre semillas + análisis (CPU)
#
# CHECKPOINT = monitor_run.py --once sobre runs_v4/. Exit 2 (alerta crítica) DETIENE la
# orquestación (exit 2) para no quemar GPU sobre un problema; exit 1 (avisos) sigue.
# Cada proceso de explicación es uno por (arquitectura, escenario, semilla): proceso fresco =
# sin acumulación de memoria (la causa del OOM de v3 fue correr todo en un solo proceso).
#
# Uso:
#   bash scripts/v4/run_all_v4.sh                      # todo, P0→P3
#   bash scripts/v4/run_all_v4.sh --resume             # retoma (pasa --resume a train/explain)
#   bash scripts/v4/run_all_v4.sh --only P1            # solo un bloque
#   bash scripts/v4/run_all_v4.sh --from P2 --resume   # desde un bloque en adelante
#   opciones: --device auto|cuda|cpu · --skip-preflight · --no-checkpoint · --dry-run
#             --stages all|train|explain  (paralelo: train en GPU en una terminal y explain en CPU
#             en otra; ver RUNBOOK_V4.md §4)
#             --config configs/experiment_v4.yaml
# Logs: runs_v4/run_all.log (orquestador) y runs_v4/<bloque>_<etapa>_<arq>.log (cada proceso).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="configs/experiment_v4.yaml"
DEVICE="auto"
RESUME=0
ONLY=""
FROM="P0"
PREFLIGHT=1
CHECKPOINT=1
DRY=0
STAGES="all"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME=1 ;;
    --only) ONLY="$2"; shift ;;
    --from) FROM="$2"; shift ;;
    --device) DEVICE="$2"; shift ;;
    --config) CONFIG="$2"; shift ;;
    --skip-preflight) PREFLIGHT=0 ;;
    --no-checkpoint) CHECKPOINT=0 ;;
    --dry-run) DRY=1 ;;
    --stages) STAGES="$2"; shift ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "argumento desconocido: $1" >&2; exit 64 ;;
  esac
  shift
done

PY=(uv run --frozen python)
LOGS="$(${PY[@]} -c "import yaml,sys;print(yaml.safe_load(open('$CONFIG'))['tracking'].get('logs_dir','./runs_v4'))")"
RESULTS="$(${PY[@]} -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['tracking'].get('results_dir','./results_v4'))")"
mapfile -t SCENARIOS < <(${PY[@]} -c "import yaml;[print(s) for s in yaml.safe_load(open('$CONFIG'))['scenarios']['names']]")
mkdir -p "$LOGS"
exec > >(tee -a "$LOGS/run_all.log") 2>&1

LOCK="$LOGS/.run_all.lock"
if [[ -f "$LOCK" ]] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
  echo "ERROR: ya hay una orquestación activa (PID $(cat "$LOCK")). Si no, borra $LOCK"; exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

ts() { date "+%Y-%m-%d %H:%M:%S"; }
say() { echo "[$(ts)] $*"; }
run() {  # run <logfile> <cmd...>
  local lf="$1"; shift
  say "  \$ $* > $lf"
  if [[ "$DRY" -eq 1 ]]; then return 0; fi
  "$@" 2>&1 | tee -a "$lf"
  return "${PIPESTATUS[0]}"
}

RES_FLAG=(); [[ "$RESUME" -eq 1 ]] && RES_FLAG=(--resume)

train_block() {  # train_block <tag> <seed> <reuse_hp 0|1> <arch...>
  [[ "$STAGES" == "explain" ]] && return 0
  local tag="$1" seed="$2" reuse="$3"; shift 3
  local extra=(); [[ "$reuse" -eq 1 ]] && extra=(--reuse-hp)
  for A in "$@"; do
    run "$LOGS/${tag}_train_${A}_s${seed}.log" "${PY[@]}" scripts/train_matrix.py --config "$CONFIG" \
      --device "$DEVICE" --arch "$A" --seed "$seed" "${extra[@]}" "${RES_FLAG[@]}"
  done
}

explain_block() {  # explain_block <tag> <seed> <arch...>
  [[ "$STAGES" == "train" ]] && return 0
  local tag="$1" seed="$2"; shift 2
  for A in "$@"; do
    for S in "${SCENARIOS[@]}"; do
      run "$LOGS/${tag}_explain_${A}_s${seed}.log" "${PY[@]}" scripts/explain_matrix.py --config "$CONFIG" \
        --device "$DEVICE" --arch "$A" --scenario "$S" --seed "$seed" "${RES_FLAG[@]}"
    done
  done
}

checkpoint() {  # checkpoint <tag>
  [[ "$CHECKPOINT" -eq 0 || "$DRY" -eq 1 ]] && { say "CHECKPOINT $1 omitido"; return 0; }
  say "CHECKPOINT $1 — monitor --once"
  set +e
  "${PY[@]}" scripts/v4/monitor_run.py --once --log-dir "$LOGS"
  local rc=$?
  set -e
  if [[ "$rc" -ge 2 ]]; then
    say "ALERTA CRÍTICA en el checkpoint $1: me detengo. Revisa arriba, corrige y relanza con --from <bloque> --resume."
    exit 2
  fi
  say "CHECKPOINT $1 superado (rc=$rc)"
}

ORDER=(P0 P1 P2 P3)
want() {  # ¿se corre el bloque $1?
  if [[ -n "$ONLY" ]]; then [[ "$1" == "$ONLY" ]]; return; fi
  local b started=0
  for b in "${ORDER[@]}"; do
    [[ "$b" == "$FROM" ]] && started=1
    [[ "$b" == "$1" ]] && { [[ "$started" -eq 1 ]]; return; }
  done
  return 1
}

say "run_all_v4 · stages=$STAGES · config=$CONFIG · device=$DEVICE · resume=$RESUME · only=${ONLY:-todo} · from=$FROM · escenarios=${SCENARIOS[*]}"
if [[ "$PREFLIGHT" -eq 1 && "$DRY" -eq 0 ]]; then
  PF=(); [[ "$DEVICE" == "cpu" ]] && PF=(--allow-cpu)
  "${PY[@]}" scripts/v4/preflight.py --config "$CONFIG" "${PF[@]}" || { say "PREFLIGHT falló"; exit 1; }
fi

if want P0; then
  say "=== P0: semilla 42 · GCN + GraphSAGE ==="
  train_block P0 42 0 GCN GraphSAGE
  explain_block P0 42 GCN GraphSAGE
  checkpoint P0
fi
if want P1; then
  say "=== P1: semilla 42 · GAT + TAGCN ==="
  train_block P1 42 0 GAT TAGCN
  explain_block P1 42 GAT TAGCN
  checkpoint P1
fi
if want P2; then
  say "=== P2: semillas 43/44 · 4 arquitecturas · --reuse-hp ==="
  for SEED in 43 44; do
    train_block P2 "$SEED" 1 GCN GraphSAGE GAT TAGCN
    explain_block P2 "$SEED" GCN GraphSAGE GAT TAGCN
  done
  checkpoint P2
fi
if want P3 && [[ "$STAGES" != "train" ]]; then
  say "=== P3: estabilidad entre semillas + análisis ==="
  run "$LOGS/P3_cross_seed.log" "${PY[@]}" scripts/v4/cross_seed_stability.py --results-dir "$RESULTS"
  run "$LOGS/P3_analyze.log" "${PY[@]}" scripts/v4/analyze_elliptic_v4.py \
    --csv "$RESULTS/elliptic_v4_stability.csv" --out-dir "$RESULTS"
fi
say "run_all_v4 terminado"
