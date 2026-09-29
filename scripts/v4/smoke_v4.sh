#!/usr/bin/env bash
# Corrida GPU de HUMO del pipeline v4 (~15-20 min objetivo en la RTX 4060 / 3050).
#   modo de etiquetas C · native × 4 arquitecturas + 1:10_os (SMOTE) × GCN
#   × 1 balanceo (class_weighting) × 1 semilla (42)
#   0 trials de Optuna (HP por defecto) · 3 épocas · 3 nodos · 2 réplicas · 3 explicadores
#   → analyze --selftest → monitor --once
# Directorios AISLADOS (runs_smoke/, results_smoke/, results_models_smoke/): no toca runs_v4/.
# Uso:  bash scripts/v4/smoke_v4.sh [--cpu] [--keep]
#   --cpu   fuerza device=cpu (para probar el cableado sin GPU; más lento)
#   --keep  no borra los directorios de humo previos
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="configs/experiment_v4.yaml"
RUNS="runs_smoke"
RES="results_smoke"
MODELS="results_models_smoke"
DEVICE="auto"
KEEP=0
for arg in "$@"; do
  case "$arg" in
    --cpu)  DEVICE="cpu" ;;
    --keep) KEEP=1 ;;
    *) echo "argumento desconocido: $arg" >&2; exit 64 ;;
  esac
done

PY=(uv run --frozen python)
ts() { date "+%Y-%m-%d %H:%M:%S"; }
T0=$(date +%s)
step() { echo; echo "[$(ts)] ── $* (t=$(( $(date +%s) - T0 ))s)"; }

if [[ "$KEEP" -eq 0 ]]; then
  rm -rf "$RUNS" "$RES" "$MODELS"
fi
mkdir -p "$RUNS" "$RES" "$MODELS"
LOG="$RUNS/smoke.log"
exec > >(tee -a "$LOG") 2>&1

ARCHS=(GCN GraphSAGE GAT TAGCN)

step "0/4 preflight (sin exigir GPU si --cpu)"
PF_ARGS=(--extra-output-dirs "$RUNS" "$RES" "$MODELS")
[[ "$DEVICE" == "cpu" ]] && PF_ARGS+=(--allow-cpu)
"${PY[@]}" scripts/v4/preflight.py "${PF_ARGS[@]}" || { echo "PREFLIGHT falló: no se lanza el humo"; exit 1; }

step "1/4 entrenar (native, class_weighting, seed 42, HP por defecto, 3 épocas)"
for A in "${ARCHS[@]}"; do
  "${PY[@]}" scripts/train_matrix.py --config "$CONFIG" --device "$DEVICE" \
      --arch "$A" --scenario native --balancing class_weighting --seed 42 \
      --trials 0 --epochs 3 --max-minutes-per-config 10 --no-mlflow \
      --models-dir "$MODELS" --results-dir "$RES" --logs-dir "$RUNS"
done
"${PY[@]}" scripts/train_matrix.py --config "$CONFIG" --device "$DEVICE" \
    --arch GCN --scenario 1:10_os --balancing class_weighting --seed 42 \
    --trials 0 --epochs 3 --max-minutes-per-config 10 --no-mlflow \
    --models-dir "$MODELS" --results-dir "$RES" --logs-dir "$RUNS"

step "2/4 explicar (3 explicadores, 3 nodos, 2 réplicas; --include-gated: 3 épocas no pasan la compuerta)"
for A in "${ARCHS[@]}"; do
  "${PY[@]}" scripts/explain_matrix.py --config "$CONFIG" --device "$DEVICE" \
      --arch "$A" --scenario native --balancing class_weighting --seed 42 \
      --include-gated --n-nodes 3 --replicas 2 \
      --models-dir "$MODELS" --results-dir "$RES" --logs-dir "$RUNS"
done
"${PY[@]}" scripts/explain_matrix.py --config "$CONFIG" --device "$DEVICE" \
    --arch GCN --scenario 1:10_os --balancing class_weighting --seed 42 \
    --include-gated --n-nodes 3 --replicas 2 \
    --models-dir "$MODELS" --results-dir "$RES" --logs-dir "$RUNS"
# TODO(otro agente): explain_matrix no expone --epochs para GNNExplainer/PGExplainer; el humo
# usa las épocas del YAML (100). Si se añade un override, bajarlo aquí a ~10.

step "3/4 análisis --selftest + análisis real sobre el CSV de humo (no debe romper)"
"${PY[@]}" scripts/v4/analyze_elliptic_v4.py --selftest
"${PY[@]}" scripts/v4/analyze_elliptic_v4.py --csv "$RES/elliptic_v4_stability.csv" --out-dir "$RES" \
  || echo "AVISO: analyze sobre el humo falló (esperable con 1 semilla/1 escenario si exige soporte mínimo)"

step "4/4 monitor --once (umbrales de PR-AUC/compuerta relajados: 3 épocas no aprenden)"
set +e
"${PY[@]}" scripts/v4/monitor_run.py --once --log-dir "$RUNS" --prauc-lo 0 --gate-min-n 999
RC=$?

# Chequeos estructurales del humo: 4 modelos, 12 unidades de explicación con status ok.
"${PY[@]}" - "$MODELS" "$RES" <<'PYEOF'
import sys, glob, csv
models, res = sys.argv[1], sys.argv[2]
metas = glob.glob(f"{models}/*_meta.json")
rows = list(csv.DictReader(open(f"{res}/elliptic_v4_stability.csv")))
ok = [r for r in rows if r.get("status") == "ok"]
print(f"modelos={len(metas)} (esperado 5) · filas explain={len(rows)} ok={len(ok)} (esperado 15)")
bad = [(r["run_id"], r["explainer"], r.get("status"), r.get("reason")) for r in rows if r.get("status") != "ok"]
for b in bad:
    print("  NO-OK:", b)
sys.exit(0 if len(metas) == 5 and len(ok) == 15 else 3)
PYEOF
SRC=$?
set -e

echo
echo "[$(ts)] humo terminado en $(( $(date +%s) - T0 )) s · monitor rc=$RC · estructura rc=$SRC · log: $LOG"
if [[ "$RC" -ge 2 || "$SRC" -ne 0 ]]; then
  echo "HUMO FALLÓ"; exit 1
fi
echo "HUMO OK"
