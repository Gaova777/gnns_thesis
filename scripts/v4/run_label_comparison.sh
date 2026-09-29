#!/usr/bin/env bash
# PASO 0 del plan v4 (reunión con Cristian, 28-sep-2026): ¿se pueden tratar los sin etiqueta
# como lícitos? Se entrenan los tres modos de etiqueta SOLO en el escenario nativo, semilla 42,
# 4 arquitecturas × 3 balanceos, con el mismo presupuesto, y se comparan en varas comunes.
#
#   A  unknown        negativos = sin etiqueta          (lo que corrió v3 por error)
#   B  licit          negativos = lícitas revisadas
#   C  licit_unknown  negativos = lícitas + sin etiqueta (PRINCIPAL)
#
# C se entrena en los directorios PRINCIPALES (results_models_v4/): son exactamente los
# modelos nativo/semilla 42 de P0-P1 de run_all_v4.sh, que luego se saltan con --resume.
# A y B van a results_models_v4_labels/<modo>/. Sin explicadores: solo entrenamiento.
#
# Uso:  bash scripts/v4/run_label_comparison.sh [--resume] [--device cuda|cpu|auto]
#                                               [--modes "unknown licit licit_unknown"]
#                                               [--trials N --epochs N]   (humo)
# Al final: scripts/v4/compare_label_modes.py → results_v4_labels/label_comparison.{csv,md}
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="configs/experiment_v4.yaml"
DEVICE="auto"
MODES="licit_unknown licit unknown"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) EXTRA+=(--resume) ;;
    --device) DEVICE="$2"; shift ;;
    --modes) MODES="$2"; shift ;;
    --trials) EXTRA+=(--trials "$2"); shift ;;
    --epochs) EXTRA+=(--epochs "$2"); shift ;;
    --config) CONFIG="$2"; shift ;;
    --out-root) OUT_ROOT="$2"; shift ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "argumento desconocido: $1" >&2; exit 64 ;;
  esac
  shift
done
OUT_ROOT="${OUT_ROOT:-}"
PY=(uv run --frozen python)
LOGS="${OUT_ROOT:+$OUT_ROOT/}runs_v4_labels"
mkdir -p "$LOGS"
exec > >(tee -a "$LOGS/label_comparison.log") 2>&1
ts() { date "+%Y-%m-%d %H:%M:%S"; }

DIRS=()
for M in $MODES; do
  if [[ "$M" == "licit_unknown" && -z "$OUT_ROOT" ]]; then
    MD="$(${PY[@]} -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['tracking']['models_dir'])")"
    RD="$(${PY[@]} -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['tracking']['results_dir'])")"
    LD="$(${PY[@]} -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['tracking']['logs_dir'])")"
  else
    MD="${OUT_ROOT:+$OUT_ROOT/}results_models_v4_labels/$M"
    RD="${OUT_ROOT:+$OUT_ROOT/}results_v4_labels/$M"
    LD="$LOGS/$M"
  fi
  DIRS+=("$M=$MD")
  echo "[$(ts)] === modo $M → $MD ==="
  for A in GCN GraphSAGE GAT TAGCN; do
    "${PY[@]}" scripts/train_matrix.py --config "$CONFIG" --device "$DEVICE" \
      --label-mode "$M" --scenario native --seed 42 --arch "$A" \
      --models-dir "$MD" --results-dir "$RD" --logs-dir "$LD" "${EXTRA[@]}" \
      2>&1 | tee -a "$LOGS/train_${M}_${A}.log"
  done
done

echo "[$(ts)] === comparación ==="
"${PY[@]}" scripts/v4/compare_label_modes.py --out-dir "${OUT_ROOT:+$OUT_ROOT/}results_v4_labels" \
  $(for d in "${DIRS[@]}"; do printf -- "--models %s " "$d"; done)
echo "[$(ts)] listo"
