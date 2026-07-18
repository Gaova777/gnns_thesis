#!/usr/bin/env bash
# Machine A configs on an RTX 4060 (8GB) — same YAML as run_machineA.sh
# Runs GAT + TAGCN, all 4 scenarios, all 3 balancing = 72 configs
# Results saved separately so machineA (4090) results are not overwritten.
set -euo pipefail

BASE_CONFIG="configs/experiment_machineA.yaml"
OVERRIDE_CONFIG="/tmp/experiment_machineA_4060_override.yaml"
RESULTS_DIR="./results_machineA_4060"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/machineA_4060.log"
HEARTBEAT_LOG="${LOG_DIR}/machineA_4060_heartbeat.log"
EXPECTED_CONFIGS=72
LOCK="/tmp/gnns_thesis_machineA4060.lock"
MAX_RETRIES=3

# ── Setup ──────────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

ts() { date "+%Y-%m-%d %H:%M:%S"; }

# ── MEJORA 5: Lock file anti-doble-ejecución ───────────────────────────────────
if [ -f "$LOCK" ]; then
    OLD_PID=$(cat "$LOCK")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "ERROR: Ya hay una ejecución activa (PID $OLD_PID, lock: $LOCK)"
        echo "       Si el proceso ya terminó, elimina el lock: rm $LOCK"
        exit 1
    else
        echo "[$(ts)] Lock con PID muerto ($OLD_PID) encontrado — limpiando"
        rm -f "$LOCK"
    fi
fi
echo $$ > "$LOCK"
# Clean lock AND temp YAML on exit
trap 'rm -f "$LOCK" "$OVERRIDE_CONFIG"' EXIT

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  XAI-GNN Pipeline — Machine A on 4060  (PID $$)             "
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

SHAP_LINE=$(echo "$VRAM_CONFIG" | grep "^SHAP_SAMPLES=")
SHAP_SAMPLES="${SHAP_LINE#SHAP_SAMPLES=}"

if [ "$SHAP_SAMPLES" = "ERROR" ]; then
    echo ""
    echo "FATAL: Insufficient VRAM for GAT/TAGCN on this machine."
    echo "       GAT with hidden_channels=128 requires ~6GB+ VRAM."
    exit 1
fi

if echo "$VRAM_CONFIG" | grep -q "^NO_CUDA"; then
    echo "WARNING: No CUDA detected — running on CPU. This will be extremely slow."
    SHAP_SAMPLES=25
fi

echo ""
echo "Configs to run: ${EXPECTED_CONFIGS} (GAT+TAGCN × 4 scenarios × 3 balancing × 3 explainers)"
echo "GNNShap samples: ${SHAP_SAMPLES} (auto-selected based on VRAM)"
echo "Results dir:    ${RESULTS_DIR}"
echo ""

# ── Build override YAML ────────────────────────────────────────────────────────
PYTHONPATH=. "$PYTHON" - <<PYEOF
import yaml

with open("${BASE_CONFIG}") as f:
    cfg = yaml.safe_load(f)

for m in cfg["explainability"]["methods"]:
    if m["name"] == "GNNShap":
        m["num_samples"] = ${SHAP_SAMPLES}

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

# ── MEJORA 6: GPU hang watchdog ────────────────────────────────────────────────
gpu_watchdog() {
    local main_pid=$1
    local hb_log=$2
    local consecutive_zero=0

    while kill -0 "$main_pid" 2>/dev/null; do
        sleep 300

        if ! kill -0 "$main_pid" 2>/dev/null; then
            break
        fi

        local ts_now
        ts_now=$(date "+%Y-%m-%d %H:%M:%S")

        if command -v nvidia-smi &>/dev/null; then
            local util
            util=$(nvidia-smi --query-gpu=utilization.gpu \
                   --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
            echo "[$ts_now] heartbeat PID=$main_pid GPU_util=${util}%" >> "$hb_log"

            if [ "${util:-1}" -eq 0 ]; then
                consecutive_zero=$((consecutive_zero + 1))
                if [ "$consecutive_zero" -ge 2 ]; then
                    echo "[$ts_now] WATCHDOG: GPU at 0% for 2 consecutive checks — killing PID $main_pid" \
                         >> "$hb_log"
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WATCHDOG: killing hung process PID $main_pid"
                    kill "$main_pid" 2>/dev/null || true
                    break
                fi
            else
                consecutive_zero=0
            fi
        else
            echo "[$ts_now] heartbeat PID=$main_pid (nvidia-smi not available)" >> "$hb_log"
        fi
    done
}

# ── MEJORA 4: Auto-restart loop ────────────────────────────────────────────────
START_TIME=$(date +%s)
RETRY=0

while [ "$RETRY" -lt "$MAX_RETRIES" ]; do
    echo "[$(ts)] Starting pipeline (attempt $((RETRY + 1))/$MAX_RETRIES)..."

    PYTHONPATH=. "$PYTHON" scripts/run_full_pipeline.py \
        --config "$OVERRIDE_CONFIG" \
        --resume \
        --inter-config-pause 5 &
    PIPELINE_PID=$!

    gpu_watchdog "$PIPELINE_PID" "$HEARTBEAT_LOG" &
    WATCHDOG_PID=$!

    wait "$PIPELINE_PID"
    EXIT_CODE=$?
    kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true

    if [ "$EXIT_CODE" -eq 0 ]; then
        break
    fi

    RETRY=$((RETRY + 1))
    if [ "$RETRY" -lt "$MAX_RETRIES" ]; then
        echo "[$(ts)] Exit code $EXIT_CODE — reintentando en 30s (intento $RETRY/$MAX_RETRIES)..."
        sleep 30
    else
        echo "[$(ts)] Max retries ($MAX_RETRIES) alcanzados. Pipeline no completado."
    fi
done

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$(( ELAPSED % 60 ))

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  PIPELINE COMPLETE — Machine A on 4060"
echo "  $(ts)"
echo "  Time elapsed:   ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "  Results dir:    ${RESULTS_DIR}"
echo "  Log file:       ${LOG_FILE}"
echo "  GNNShap used:   ${SHAP_SAMPLES} samples"
echo "  Retries used:   ${RETRY}/${MAX_RETRIES}"
if [ "$EXIT_CODE" -ne 0 ]; then
    echo "  WARNING: Final exit code was ${EXIT_CODE}"
fi

CSV=$(find "${RESULTS_DIR}" -name "*.csv" 2>/dev/null | head -1)
if [ -n "$CSV" ]; then
    COMPLETED=$(tail -n +2 "$CSV" 2>/dev/null | wc -l || echo "?")
    echo "  Configs logged: ${COMPLETED} rows in CSV"
fi
echo "════════════════════════════════════════════════════════════════"

exit "$EXIT_CODE"
