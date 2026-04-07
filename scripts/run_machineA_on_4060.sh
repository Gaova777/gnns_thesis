#!/usr/bin/env bash
# Machine A configs on an RTX 4060 (8GB) — same YAML as run_machineA.sh
# Runs GAT + TAGCN, all 4 scenarios, all 3 balancing = 72 configs
# Results saved separately so machineA (4090) results are not overwritten.
set -euo pipefail

BASE_CONFIG="configs/experiment_machineA.yaml"
OVERRIDE_CONFIG="/tmp/experiment_machineA_4060_override.yaml"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/machineA_4060.log"
EXPECTED_CONFIGS=72

# ── Setup ──────────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

ts() { date "+%Y-%m-%d %H:%M:%S"; }
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  XAI-GNN Pipeline — Machine A configs on 4060               "
echo "  Base config: ${BASE_CONFIG}                                 "
echo "  $(ts)                                                        "
echo "════════════════════════════════════════════════════════════════"

# ── Dependency check ───────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "[$(ts)] .venv not found — running uv sync..."
    uv sync
    echo "[$(ts)] Dependencies installed."
fi

PYTHON=".venv/bin/python"

# ── VRAM detection and SHAP mode selection ─────────────────────────────────────
VRAM_CONFIG=$(PYTHONPATH=. "$PYTHON" - <<'PYEOF'
import sys
import torch

if not torch.cuda.is_available():
    print("NO_CUDA")
    sys.exit(0)

p = torch.cuda.get_device_properties(0)
vram_gb = p.total_memory / 1024**3
print(f"GPU detected:   {p.name}")
print(f"VRAM total:     {vram_gb:.1f} GB")
print(f"CUDA version:   {torch.version.cuda}")

if vram_gb >= 20:
    print(f"SHAP mode:      normal (num_samples=50) — {vram_gb:.0f}GB >= 20GB threshold")
    print(f"SHAP_SAMPLES=50")
elif vram_gb >= 6:
    print(f"SHAP mode:      4060-safe (num_samples=25) — {vram_gb:.1f}GB in [6,20) GB range")
    print(f"SHAP_SAMPLES=25")
else:
    print(f"ERROR: VRAM {vram_gb:.1f}GB < 6GB — GAT/TAGCN will OOM. Use Machine C config instead.")
    print(f"SHAP_SAMPLES=ERROR")
PYEOF
)

echo "$VRAM_CONFIG"

# Extract SHAP_SAMPLES line
SHAP_LINE=$(echo "$VRAM_CONFIG" | grep "^SHAP_SAMPLES=")
SHAP_SAMPLES="${SHAP_LINE#SHAP_SAMPLES=}"

if [ "$SHAP_SAMPLES" = "ERROR" ]; then
    echo ""
    echo "FATAL: Insufficient VRAM for GAT/TAGCN on this machine."
    echo "       GAT with hidden_channels=128 requires ~6GB+ VRAM."
    echo "       This script is intended for RTX 4060 (8GB) or better."
    exit 1
fi

if echo "$VRAM_CONFIG" | grep -q "^NO_CUDA"; then
    echo "WARNING: No CUDA detected — running on CPU. This will be extremely slow."
    SHAP_SAMPLES=25
fi

echo ""
echo "Configs to run: ${EXPECTED_CONFIGS} (GAT+TAGCN × 4 scenarios × 3 balancing × 3 explainers)"
echo "GNNShap samples: ${SHAP_SAMPLES} (auto-selected based on VRAM)"
echo "Results dir:    ./results_machineA_4060"
echo "Experiment:     xai-gnn-stability-A-4060"
echo ""

# ── Build override YAML ────────────────────────────────────────────────────────
# We copy experiment_machineA.yaml and patch three fields:
#   - GNNShap.num_samples  → SHAP_SAMPLES
#   - tracking.results_dir → ./results_machineA_4060
#   - tracking.experiment_name → xai-gnn-stability-A-4060
PYTHONPATH=. "$PYTHON" - <<PYEOF
import yaml, copy, sys

with open("${BASE_CONFIG}") as f:
    cfg = yaml.safe_load(f)

# Patch GNNShap num_samples
for m in cfg["explainability"]["methods"]:
    if m["name"] == "GNNShap":
        m["num_samples"] = ${SHAP_SAMPLES}

# Separate results so machineA (4090) results are never overwritten
cfg["tracking"]["results_dir"] = "./results_machineA_4060"
cfg["tracking"]["experiment_name"] = "xai-gnn-stability-A-4060"

with open("${OVERRIDE_CONFIG}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"[override] Written to ${OVERRIDE_CONFIG}")
print(f"[override]   GNNShap.num_samples = ${SHAP_SAMPLES}")
print(f"[override]   results_dir         = ./results_machineA_4060")
print(f"[override]   experiment_name     = xai-gnn-stability-A-4060")
PYEOF

echo ""

# ── Confirmation ───────────────────────────────────────────────────────────────
read -rp "Launch pipeline? [Y/n] " confirm
confirm="${confirm:-Y}"
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# ── Run with inter-config pause ────────────────────────────────────────────────
# The pipeline handles the full loop internally.
# We add a wrapper that pauses 5 s after each config via a trap + subshell trick.
# Simplest portable approach: pass --pause-between-configs to the pipeline if
# supported, otherwise just run normally (the pause is nice-to-have).
START_TIME=$(date +%s)
echo "[$(ts)] Starting pipeline with config: ${OVERRIDE_CONFIG}"
echo "[$(ts)] 5-second inter-config GC pauses are handled inside the pipeline."

PYTHONPATH=. "$PYTHON" scripts/run_full_pipeline.py \
    --config "$OVERRIDE_CONFIG" \
    --resume \
    --inter-config-pause 5

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$(( ELAPSED % 60 ))

# Cleanup temp file
rm -f "$OVERRIDE_CONFIG"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  PIPELINE COMPLETE — Machine A on 4060"
echo "  $(ts)"
echo "  Time elapsed:   ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "  Results dir:    ./results_machineA_4060"
echo "  Log file:       ${LOG_FILE}"
echo "  GNNShap used:   ${SHAP_SAMPLES} samples"
if [ "$EXIT_CODE" -ne 0 ]; then
    echo "  WARNING: Pipeline exited with code ${EXIT_CODE}"
fi

CSV=$(find "./results_machineA_4060" -name "*.csv" 2>/dev/null | head -1)
if [ -n "$CSV" ]; then
    COMPLETED=$(tail -n +2 "$CSV" 2>/dev/null | wc -l || echo "?")
    echo "  Configs logged: ${COMPLETED} rows in CSV"
fi
echo "════════════════════════════════════════════════════════════════"

exit "$EXIT_CODE"
