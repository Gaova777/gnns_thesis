#!/usr/bin/env bash
# Machine B launcher — RTX 4060 (8GB)
# Configs: GCN + GraphSAGE, scenarios 1:1 and 1:10 = 36 configs
set -euo pipefail

MACHINE="B"
CONFIG="configs/experiment_machineB.yaml"
RESULTS_DIR="./results_machineB"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/machineB.log"
EXPECTED_CONFIGS=36

# ── Setup ──────────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

ts() { date "+%Y-%m-%d %H:%M:%S"; }
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  XAI-GNN Pipeline — Machine ${MACHINE}                        "
echo "  $(ts)                                                        "
echo "════════════════════════════════════════════════════════════════"

# ── Dependency check ───────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "[$(ts)] .venv not found — running uv sync..."
    uv sync
    echo "[$(ts)] Dependencies installed."
fi

PYTHON=".venv/bin/python"

# ── GPU / VRAM info ────────────────────────────────────────────────────────────
GPU_INFO=$(PYTHONPATH=. "$PYTHON" - <<'PYEOF'
import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    vram_gb = p.total_memory / 1024**3
    print(f"GPU detected:   {p.name}")
    print(f"VRAM total:     {vram_gb:.1f} GB")
    print(f"CUDA version:   {torch.version.cuda}")
    print(f"SHAP mode:      normal (num_samples=50)")
else:
    print("WARNING: No CUDA GPU detected — running on CPU (will be slow)")
    print("GPU detected:   CPU")
    print("VRAM total:     N/A")
    print("SHAP mode:      normal (num_samples=50)")
PYEOF
)
echo "$GPU_INFO"
echo "Configs to run: ${EXPECTED_CONFIGS} (GCN+SAGE × scenarios 1:1,1:10 × 3 balancing × 3 explainers)"
echo ""

# ── Confirmation ───────────────────────────────────────────────────────────────
read -rp "Launch pipeline? [Y/n] " confirm
confirm="${confirm:-Y}"
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# ── Run ────────────────────────────────────────────────────────────────────────
START_TIME=$(date +%s)
echo "[$(ts)] Starting pipeline..."

PYTHONPATH=. "$PYTHON" scripts/run_full_pipeline.py \
    --config "$CONFIG" \
    --resume

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$(( ELAPSED % 60 ))

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  PIPELINE COMPLETE — Machine ${MACHINE}"
echo "  $(ts)"
echo "  Time elapsed:   ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "  Results dir:    ${RESULTS_DIR}"
echo "  Log file:       ${LOG_FILE}"
if [ "$EXIT_CODE" -ne 0 ]; then
    echo "  WARNING: Pipeline exited with code ${EXIT_CODE}"
fi

CSV=$(find "$RESULTS_DIR" -name "*.csv" 2>/dev/null | head -1)
if [ -n "$CSV" ]; then
    COMPLETED=$(tail -n +2 "$CSV" 2>/dev/null | wc -l || echo "?")
    echo "  Configs logged: ${COMPLETED} rows in CSV"
fi
echo "════════════════════════════════════════════════════════════════"

exit "$EXIT_CODE"
