#!/usr/bin/env bash
# Eje SINTÉTICO v4 (CPU basta: grafo de ~11k nodos). No toca los CSV legados de phase1/.
#   1. atajos: AUC de una sola variable, grafo v4 y legado  → results_phase1_v4/shortcuts_*.txt
#   2. alineamiento modelo↔tipología (4 arq.), v4 y legado   → results_phase1_v4/alignment*.json
#   3. matriz v4 (RunLog stage "synthetic" en runs_v4/)       → results_phase1_v4/results_v4.csv
#   4. resumen real vs azar                                   → results_phase1_v4/summary_v4.csv
# Uso: bash scripts/v4/run_phase1_v4.sh [--resume] [--quick] [--skip-alignment] [-- <args extra de run_phase1_v4.py>]
#   --quick  1 arquitectura (GCN), 1 escenario (natural), 1 balanceo, 1 semilla, pocas épocas
#            (≈1-2 min en CPU; valida el cableado)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT="results_phase1_v4"
LOGS="runs_v4"
RESUME=()
QUICK=0
ALIGN=1
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME=(--resume) ;;
    --quick) QUICK=1 ;;
    --skip-alignment) ALIGN=0 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "argumento desconocido: $1" >&2; exit 64 ;;
  esac
  shift
done
PY=(uv run --frozen python)
mkdir -p "$OUT" "$LOGS"
exec > >(tee -a "$LOGS/synthetic.log") 2>&1
ts() { date "+%Y-%m-%d %H:%M:%S"; }
T0=$(date +%s)

if [[ "$QUICK" -eq 1 ]]; then
  OUT="$OUT/quick"; mkdir -p "$OUT"
  MATRIX=(--archs GCN --scenarios natural --balancings class_weighting --model-seeds 42 --epochs 60
          --nodes 5 --replicas 2 --ex-epochs 30 --pg-epochs 5 --pg-train-nodes 20 --shap-samples 10
          --random-reps 5 --log-dir "$OUT/runs")
  ALIGN_ARGS=(--archs GCN --epochs 100)
else
  MATRIX=(--log-dir "$LOGS")
  ALIGN_ARGS=()
fi

echo "[$(ts)] 1/4 atajos (AUC de una variable; objetivo v4 ≤ 0,65)"
"${PY[@]}" phase1/synthetic_aml_generator.py --v4 --report --out "$OUT/synthetic_v4_g42.pt" | tee "$OUT/shortcuts_v4.txt"
"${PY[@]}" phase1/synthetic_aml_generator.py --report --out "$OUT/synthetic_legacy_g42.pt" | tee "$OUT/shortcuts_legacy.txt"

if [[ "$ALIGN" -eq 1 ]]; then
  echo "[$(ts)] 2/4 alineamiento (criterio: caída_tipología − caída_azar ≥ 0,10)"
  "${PY[@]}" phase1/alignment_check.py "${ALIGN_ARGS[@]}" --out "$OUT/alignment.json"
  "${PY[@]}" phase1/alignment_check.py --legacy "${ALIGN_ARGS[@]}" --out "$OUT/alignment_legacy.json"
fi

echo "[$(ts)] 3/4 matriz v4"
"${PY[@]}" phase1/run_phase1_v4.py "${MATRIX[@]}" "${RESUME[@]}" \
  --alignment-json "$OUT/alignment.json" --out "$OUT/results_v4.csv" "${EXTRA[@]}"

echo "[$(ts)] 4/4 resumen real vs azar"
"${PY[@]}" phase1/analyze_v4.py --csv "$OUT/results_v4.csv" --out "$OUT/summary_v4.csv"
echo "[$(ts)] eje sintético v4 terminado en $(( $(date +%s) - T0 )) s"
