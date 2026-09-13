#!/usr/bin/env python3
"""Fase 1 del seed sweep de GNNShap para config B (GCN/GraphSAGE).

Objetivo: extender la replicacion de 3 semillas (42/43/44) de GNNExplainer a GNNShap,
solo para las dos arquitecturas de config B. GAT/TAGCN quedan para la maquina C.

Dos fases:
  1. Entrenar los modelos de seed 43 y 44 con --reuse-hp (reutiliza los hiperparametros
     de la seed canonica 42; salta Optuna). Produce checkpoints _s43/_s44 en results_models_v3.
  2. Explicar con GNNShap, troceado por (arch, scenario, balancing) en procesos frescos
     (a prueba de OOM), escribiendo a results_seedsweep/ junto a las filas de GNNExplainer.

Desacoplado (nohup setsid) + log; sobrevive al cierre de terminal/sesion. Idempotente
via --resume, asi que se puede relanzar sin repetir lo hecho.
"""
import subprocess, os, time

ROOT = "/home/juan/Escritorio/gnn_thesis/gnns_thesis"
CONFIG = f"{ROOT}/configs/experiment_seedsweep.yaml"
UV = os.path.expanduser("~/.local/bin/uv")

env = dict(os.environ)
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
env["PATH"] = os.path.expanduser("~/.local/bin") + ":" + env.get("PATH", "")

def run(cmd, label):
    print(f"\n[driver] === {label} ===", flush=True)
    t0 = time.time()
    r = subprocess.run([UV, "run", "python"] + cmd, cwd=ROOT, env=env)
    dt = time.time() - t0
    status = "OK" if r.returncode == 0 else f"FAIL rc={r.returncode}"
    print(f"[driver] {label} -> {status} en {dt:.0f}s", flush=True)
    return r.returncode

# ---------- FASE 1: entrenar seeds 43 y 44 (reuse-hp) ----------
print("[driver] FASE 1: entrenar seeds 43 y 44 (GCN/GraphSAGE, reuse-hp)", flush=True)
for seed in [43, 44]:
    run(["scripts/train_matrix.py", "--config", CONFIG, "--seed", str(seed),
         "--reuse-hp", "--resume"], f"train seed {seed}")

# ---------- FASE 2: GNNShap troceado por config (3 seeds por chunk) ----------
print("\n[driver] FASE 2: GNNShap troceado (GCN/GraphSAGE, 3 seeds c/u)", flush=True)
ARCHS = ["GCN", "GraphSAGE"]
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

print("\n[driver] DONE", flush=True)
