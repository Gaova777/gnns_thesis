#!/usr/bin/env bash
# Machine B launcher — RTX 4060 (8GB)
# Configs: GCN + GraphSAGE, scenarios 1:1 and 1:10 = 36 configs
set -euo pipefail

MACHINE="B"
CONFIG="configs/experiment_machineB.yaml"
RESULTS_DIR="./results_machineB"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/machineB.log"
HEARTBEAT_LOG="${LOG_DIR}/machineB_heartbeat.log"
EXPECTED_CONFIGS=36
LOCK="/tmp/gnns_thesis_machineB.lock"
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
trap 'rm -f "$LOCK"' EXIT

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  XAI-GNN Pipeline — Machine ${MACHINE}  (PID $$)             "
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
PYTHONPATH=. "$PYTHON" - <<'PYEOF'
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
    print("GPU detected:   CPU / VRAM total: N/A / SHAP mode: normal")
PYEOF
echo "Configs to run: ${EXPECTED_CONFIGS} (GCN+SAGE × scenarios 1:1,1:10 × 3 balancing × 3 explainers)"
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
        --config "$CONFIG" \
        --resume &
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
echo "  PIPELINE COMPLETE — Machine ${MACHINE}"
echo "  $(ts)"
echo "  Time elapsed:   ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "  Results dir:    ${RESULTS_DIR}"
echo "  Log file:       ${LOG_FILE}"
echo "  Retries used:   ${RETRY}/${MAX_RETRIES}"
if [ "$EXIT_CODE" -ne 0 ]; then
    echo "  WARNING: Final exit code was ${EXIT_CODE}"
fi

CSV=$(find "$RESULTS_DIR" -name "*.csv" 2>/dev/null | head -1)
if [ -n "$CSV" ]; then
    COMPLETED=$(tail -n +2 "$CSV" 2>/dev/null | wc -l || echo "?")
    echo "  Configs logged: ${COMPLETED} rows in CSV"
fi
echo "════════════════════════════════════════════════════════════════"

exit "$EXIT_CODE"
