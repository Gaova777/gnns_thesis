#!/usr/bin/env python3
"""Fase 2 del seed sweep de GNNShap: GAT/TAGCN, en la 4060 (sin acceso a la 3050).

Encadenado DESPUES del driver de config B: espera a que el log de B contenga
'[driver] DONE' para no competir por la GPU, y luego:
  1. Entrena GAT/TAGCN seeds 43 y 44 con --reuse-hp (reutiliza HP de seed 42).
  2. Explica con GNNShap, troceado por (arch, scenario, balancing) en procesos frescos.

GAT + GNNShap es el peor caso de OOM en 8 GB; se mitiga con PYTORCH_CUDA_ALLOC_CONF
expandable_segments, el troceado por config (memoria liberada entre chunks) y el
halving automatico de num_samples de GNNShap ante OOM. Idempotente via --resume.
"""
import subprocess, os, time

ROOT = "/home/juan/Escritorio/gnn_thesis/gnns_thesis"
CONFIG = f"{ROOT}/configs/experiment_seedsweep_C.yaml"
B_LOG = f"{ROOT}/gnnshap_seedsweep_B.log"
UV = os.path.expanduser("~/.local/bin/uv")

env = dict(os.environ)
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
env["PATH"] = os.path.expanduser("~/.local/bin") + ":" + env.get("PATH", "")

def run(cmd, label):
    print(f"\n[driver-C] === {label} ===", flush=True)
    t0 = time.time()
    r = subprocess.run([UV, "run", "python"] + cmd, cwd=ROOT, env=env)
    dt = time.time() - t0
    status = "OK" if r.returncode == 0 else f"FAIL rc={r.returncode}"
    print(f"[driver-C] {label} -> {status} en {dt:.0f}s", flush=True)

# ---------- FASE 0: esperar a que termine el driver de config B ----------
print("[driver-C] esperando a que termine config B (log: DONE)...", flush=True)
waited = 0
while True:
    try:
        if "[driver] DONE" in open(B_LOG, encoding="utf-8", errors="ignore").read():
            break
    except FileNotFoundError:
        pass
    time.sleep(120)
    waited += 120
    if waited % 1800 == 0:
        print(f"[driver-C] aun esperando a B ({waited//60} min)...", flush=True)
print("[driver-C] config B terminó. Arrancando GAT/TAGCN.", flush=True)

# ---------- FASE 1: entrenar GAT/TAGCN seeds 43 y 44 (reuse-hp) ----------
print("[driver-C] FASE 1: entrenar seeds 43 y 44 (GAT/TAGCN, reuse-hp)", flush=True)
for seed in [43, 44]:
    run(["scripts/train_matrix.py", "--config", CONFIG, "--seed", str(seed),
         "--reuse-hp", "--resume"], f"train seed {seed}")

# ---------- FASE 2: GNNShap troceado por config ----------
print("\n[driver-C] FASE 2: GNNShap troceado (GAT/TAGCN, 3 seeds c/u)", flush=True)
ARCHS = ["GAT", "TAGCN"]
SCENARIOS = ["1:1", "1:10", "1:30_native", "1:50", "1:100"]
BALS = ["none", "class_weighting", "focal_loss"]
i, total = 0, len(ARCHS) * len(SCENARIOS) * len(BALS)
for arch in ARCHS:
    for scen in SCENARIOS:
        for bal in BALS:
            i += 1
            run(["scripts/explain_matrix.py", "--config", CONFIG,
                 "--explainer", "GNNShap", "--arch", arch, "--scenario", scen,
                 "--balancing", bal, "--resume"],
                f"GNNShap {i}/{total}: {arch} | {scen} | {bal}")

print("\n[driver-C] DONE", flush=True)
