#!/usr/bin/env python3
"""Fase D de la cadena de 3 semillas: PGExplainer + reeval de metricas de rendimiento.

Encadenado DESPUES del driver de config C (GAT/TAGCN): espera a que el log de C
contenga '[driver-C] DONE' para no competir por la GPU. Luego:

  FASE 1  PGExplainer troceado por (arch, scenario, balancing) en procesos frescos.
          GCN/GraphSAGE usan experiment_seedsweep.yaml; GAT/TAGCN usan
          experiment_seedsweep_C.yaml. Cada chunk explica las semillas 42/43/44 que
          ya tengan checkpoint. Cierra la simetria de los tres explicadores en el
          sweep (PGExplainer es de aristas y degenera en Elliptic; el valor es la
          completitud y confirmar que la degeneracion es estable entre semillas).
  FASE 2  Reeval inference-only sobre TODOS los checkpoints (seed 42/43/44), en CPU
          para no arriesgar OOM, con --dump-curves. Produce results_v3/reeval_metrics.csv
          (con columna seed) y results_v3/reeval_curves.csv, o sea la version a 3 semillas
          de ROC-AUC / PR-AUC / precision@k y los puntos para graficar las curvas PR/ROC
          (la unica maquina con checkpoints puede generarlas; ver docs/PENDIENTE_curvas_pr_roc.md).
  FASE 3  Escribe el centinela CHAIN_DONE para que la tarea programada de cierre
          (finalize) sepa que toda la cadena B->C->D termino.

GAT + explicador es el peor caso de OOM en 8 GB; se mitiga con
PYTORCH_CUDA_ALLOC_CONF=expandable_segments y el troceado por config (memoria
liberada entre chunks). Idempotente via --resume: relanzar no repite lo hecho.
"""
import subprocess, os, time

ROOT = "/home/juan/Escritorio/gnn_thesis/gnns_thesis"
CFG_B = f"{ROOT}/configs/experiment_seedsweep.yaml"      # GCN/GraphSAGE
CFG_C = f"{ROOT}/configs/experiment_seedsweep_C.yaml"    # GAT/TAGCN
C_LOG = f"{ROOT}/gnnshap_seedsweep_C.log"
SENTINEL = f"{ROOT}/CHAIN_DONE"
UV = os.path.expanduser("~/.local/bin/uv")

env = dict(os.environ)
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
env["PATH"] = os.path.expanduser("~/.local/bin") + ":" + env.get("PATH", "")


def run(cmd, label):
    print(f"\n[driver-D] === {label} ===", flush=True)
    t0 = time.time()
    r = subprocess.run([UV, "run", "python"] + cmd, cwd=ROOT, env=env)
    dt = time.time() - t0
    status = "OK" if r.returncode == 0 else f"FAIL rc={r.returncode}"
    print(f"[driver-D] {label} -> {status} en {dt:.0f}s", flush=True)
    return r.returncode


# ---------- FASE 0: esperar a que termine el driver de config C ----------
print("[driver-D] esperando a que termine config C (log: [driver-C] DONE)...", flush=True)
waited = 0
while True:
    try:
        if "[driver-C] DONE" in open(C_LOG, encoding="utf-8", errors="ignore").read():
            break
    except FileNotFoundError:
        pass
    time.sleep(120)
    waited += 120
    if waited % 1800 == 0:
        print(f"[driver-D] aun esperando a C ({waited // 60} min)...", flush=True)
print("[driver-D] config C termino. Arrancando PGExplainer 3 semillas.", flush=True)

# ---------- FASE 1: PGExplainer troceado por config ----------
print("\n[driver-D] FASE 1: PGExplainer troceado (4 arquitecturas, 3 seeds c/u)", flush=True)
SCENARIOS = ["1:1", "1:10", "1:30_native", "1:50", "1:100"]
BALS = ["none", "class_weighting", "focal_loss"]
ARCH_CFG = [("GCN", CFG_B), ("GraphSAGE", CFG_B), ("GAT", CFG_C), ("TAGCN", CFG_C)]
i, total = 0, len(ARCH_CFG) * len(SCENARIOS) * len(BALS)
for arch, cfg in ARCH_CFG:
    for scen in SCENARIOS:
        for bal in BALS:
            i += 1
            run(["scripts/explain_matrix.py", "--config", cfg,
                 "--explainer", "PGExplainer", "--arch", arch, "--scenario", scen,
                 "--balancing", bal, "--resume"],
                f"PGExplainer {i}/{total}: {arch} | {scen} | {bal}")

# ---------- FASE 2: reeval ROC-AUC / PR-AUC / precision@k (CPU, 3 seeds) ----------
print("\n[driver-D] FASE 2: reeval metricas + puntos de curvas (CPU, checkpoints 42/43/44)", flush=True)
run(["scripts/consolidacion/reeval_rocauc.py", "--device", "cpu", "--dump-curves"],
    "reeval ROC-AUC/PR-AUC/p@k + puntos de curvas PR/ROC (3 semillas)")

# ---------- FASE 3: centinela de cadena completa ----------
with open(SENTINEL, "w", encoding="utf-8") as fh:
    fh.write("cadena B->C->D completa\n")
print(f"\n[driver-D] escrito centinela {SENTINEL}", flush=True)
print("\n[driver-D] DONE", flush=True)
