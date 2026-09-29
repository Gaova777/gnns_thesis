#!/usr/bin/env python
"""Preflight v4 — comprobaciones baratas (CPU, < 5 min) ANTES de gastar GPU.

Checks (cada uno imprime OK / FALLO / AVISO):
  1. Elliptic en data/raw: conteos oficiales contados DIRECTO de los CSV
       global  4.545 ilícitos / 42.019 lícitos / 157.205 desconocidos
       train (t 1-34) 3.462/26.432 · val (35-42) 914/9.069 · test (43-49) 169/6.518
     y, además, por el loader del pipeline (src.data.loader.load_elliptic + split temporal +
     validate_split_counts). Si el loader no se puede importar se avisa y vale el conteo directo.
     --skip-loader salta la segunda vía (la primera nunca se salta). Con el loader se validan
     también los tres modos de etiqueta (C lícitas+sin etiqueta, B lícitas, A sin etiqueta).
  2. configs/experiment_v4.yaml existe, lista las 4 arquitecturas y el presupuesto es el MISMO
     para las 4 (ninguna entrada de arquitectura sobrescribe epochs/trials/patience/hidden...).
  3. Espacio libre en disco ≥ --min-disk-gb (default 20 GB) en el repo.
  4. GPU visible (torch.cuda) y memoria libre ≥ --min-gpu-free-gb (default 3 GB).
     --allow-cpu lo convierte en aviso (para probar el preflight en una máquina sin GPU).
  5. Directorios de salida v4 (results_dir, models_dir, logs_dir del YAML) distintos de los v3
     y sin resultados v3 mezclados (archivos *v3*, machineB/C, meta.json sin
     pipeline_version=v4). Un progress.jsonl previo es AVISO (¿quedó de un smoke?).
Salida: «PREFLIGHT OK» y exit 0, o la lista de fallos y exit 1.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EXPECTED_GLOBAL = {"illicit": 4545, "licit": 42019, "unknown": 157205}
SPLIT = {"train": (1, 34), "val": (35, 42), "test": (43, 49)}
EXPECTED_SPLIT = {"train": {"illicit": 3462, "licit": 26432},
                  "val": {"illicit": 914, "licit": 9069},
                  "test": {"illicit": 169, "licit": 6518}}
ARCHS = {"GCN", "GraphSAGE", "GAT", "TAGCN"}
BUDGET_KEYS = {"epochs", "patience", "optuna_trials", "trials", "trial_epochs", "trial_patience",
               "hidden_dim", "hidden_channels", "num_layers", "learning_rate", "lr",
               "hyperparameter_search", "training"}
V3_DIRS = {"results_v3", "results_models_v3", "results_seedsweep", "results_machineB", "results_machineC"}

fails: list[str] = []
warns: list[str] = []


def ok(msg):
    print(f"  OK     {msg}")


def fail(msg):
    fails.append(msg); print(f"  FALLO  {msg}")


def warn(msg):
    warns.append(msg); print(f"  AVISO  {msg}")


# ---------------------------------------------------------------------------------------
def count_csv(raw: Path):
    cls = {}
    with open(raw / "elliptic_txs_classes.csv", newline="") as f:
        for r in csv.DictReader(f):
            cls[r["txId"]] = r["class"]
    ts = {}
    with open(raw / "elliptic_txs_features.csv") as f:
        for line in f:
            a, b, _ = line.split(",", 2)
            ts[a] = int(float(b))
    glob = {"illicit": 0, "licit": 0, "unknown": 0}
    split = {k: {"illicit": 0, "licit": 0} for k in SPLIT}
    name = {"1": "illicit", "2": "licit", "unknown": "unknown"}
    for tx, c in cls.items():
        lab = name[c]; glob[lab] += 1
        if lab == "unknown":
            continue
        t = ts[tx]
        for k, (lo, hi) in SPLIT.items():
            if lo <= t <= hi:
                split[k][lab] += 1
    return glob, split


def check_elliptic(data_root: Path, skip_loader: bool):
    print("[1] Elliptic (data/raw)")
    raw = data_root / "raw"
    need = ["elliptic_txs_classes.csv", "elliptic_txs_features.csv", "elliptic_txs_edgelist.csv"]
    missing = [n for n in need if not (raw / n).exists()]
    if missing:
        fail(f"faltan en {raw}: {missing}"); return
    glob, split = count_csv(raw)
    if glob == EXPECTED_GLOBAL:
        ok(f"conteo directo global {glob}")
    else:
        fail(f"conteo directo global {glob} != {EXPECTED_GLOBAL}")
    if split == EXPECTED_SPLIT:
        ok(f"conteo directo por partición {split}")
    else:
        fail(f"conteo directo por partición {split} != {EXPECTED_SPLIT}")
    if skip_loader:
        warn("--skip-loader: no se verificó el loader del pipeline"); return
    try:
        from src.data.loader import load_elliptic, validate_split_counts
        from src.data.preprocessing import create_time_split_masks
    except Exception as exc:  # noqa: BLE001
        warn(f"loader del pipeline no importable ({type(exc).__name__}: {exc}); vale el conteo directo")
        return
    try:
        d = load_elliptic(root=str(data_root), validate=True)
        create_time_split_masks(d, SPLIT["train"], SPLIT["val"], SPLIT["test"])
        got = validate_split_counts(d, SPLIT)
        ok(f"loader del pipeline: etiquetas y particiones validadas {got}")
        # Los tres modos de etiqueta (C principal, B y A para la comparación) cuadran.
        from copy import deepcopy
        from src.data.loader import apply_label_mode
        for mode in ("licit_unknown", "licit", "unknown"):
            dm = deepcopy(d)
            apply_label_mode(dm, mode)
            create_time_split_masks(dm, SPLIT["train"], SPLIT["val"], SPLIT["test"])
            gm = validate_split_counts(dm, SPLIT)
            ok(f"modo de etiquetas {mode}: train {gm['train']} · test {gm['test']}")
    except Exception as exc:  # noqa: BLE001
        fail(f"loader del pipeline: {type(exc).__name__}: {str(exc)[:300]}")


def check_config(cfg_path: Path):
    print(f"[2] Config {cfg_path}")
    if not cfg_path.exists():
        fail(f"no existe {cfg_path}"); return None
    import yaml
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    archs = cfg.get("models", {}).get("architectures", [])
    names = {a.get("name") if isinstance(a, dict) else str(a) for a in archs}
    if names != ARCHS:
        fail(f"arquitecturas {sorted(names)} != {sorted(ARCHS)}")
    else:
        ok(f"4 arquitecturas: {sorted(names)}")
    over = {a.get("name"): sorted(set(a) & BUDGET_KEYS) for a in archs if isinstance(a, dict) and set(a) & BUDGET_KEYS}
    if over:
        fail(f"presupuesto distinto por arquitectura (claves que sobrescriben): {over}")
    else:
        ok("ninguna arquitectura sobrescribe el presupuesto común")
    # por si el YAML trae bloques por arquitectura en otro sitio (p. ej. training.per_arch)
    blob = json.dumps(cfg)
    for k in ("per_arch", "per_architecture", "arch_overrides"):
        if k in blob:
            fail(f"el YAML contiene '{k}': revisar que el presupuesto sea común")
    hs, tr = cfg.get("models", {}).get("hyperparameter_search", {}), cfg.get("training", {})
    missing = [k for k, src in (("optuna_trials", hs), ("epochs", tr), ("patience", tr)) if k not in src]
    if missing:
        fail(f"faltan {missing} en el presupuesto común")
    else:
        ok(f"presupuesto común: trials={hs.get('optuna_trials')} epochs={tr.get('epochs')} "
           f"patience={tr.get('patience')} seeds={tr.get('seeds')}")
    return cfg


def check_disk(min_gb: float):
    print("[3] Disco")
    free = shutil.disk_usage(ROOT).free / 2**30
    (ok if free >= min_gb else fail)(f"libre {free:.1f} GB (mínimo {min_gb} GB)")


def check_gpu(min_free_gb: float, allow_cpu: bool):
    print("[4] GPU")
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        fail(f"torch no importa: {exc}"); return
    if not torch.cuda.is_available():
        (warn if allow_cpu else fail)("torch.cuda no ve ninguna GPU"); return
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        name = torch.cuda.get_device_name(i)
        msg = f"cuda:{i} {name}: libre {free/2**30:.1f} / {total/2**30:.1f} GB (mínimo {min_free_gb} GB)"
        (ok if free / 2**30 >= min_free_gb else fail)(msg)


def _v3_markers(d: Path) -> list[str]:
    bad = []
    for p in d.rglob("*"):
        n = p.name
        if p.is_file() and ("_v3" in n or "-v3" in n or "machineB" in n or "machineC" in n
                            or n.startswith("xai-gnn-stability-B")):
            bad.append(str(p.relative_to(ROOT)))
        elif p.is_file() and n.endswith("_meta.json"):
            try:
                if json.load(open(p)).get("pipeline_version") != "v4":
                    bad.append(str(p.relative_to(ROOT)) + " (sin pipeline_version=v4)")
            except Exception:  # noqa: BLE001
                bad.append(str(p.relative_to(ROOT)) + " (meta ilegible)")
        if len(bad) >= 10:
            break
    return bad


def check_outputs(cfg: dict | None, extra_dirs: list[str]):
    print("[5] Directorios de salida")
    tr = (cfg or {}).get("tracking", {})
    dirs = [tr.get("results_dir", "./results_v4"), tr.get("models_dir", "./results_models_v4"),
            tr.get("logs_dir", "./runs_v4")] + extra_dirs
    for ds in dirs:
        d = (ROOT / ds).resolve()
        if d.name in V3_DIRS:
            fail(f"{ds} es un directorio v3"); continue
        if not d.exists():
            ok(f"{ds} no existe todavía (se creará)"); continue
        bad = _v3_markers(d)
        if bad:
            fail(f"{ds} contiene resultados v3: {bad[:5]}")
        else:
            ok(f"{ds} sin resultados v3")
        pj = d / "progress.jsonl"
        if pj.exists():
            n = sum(1 for _ in open(pj, encoding="utf-8"))
            warn(f"{ds}/progress.jsonl ya tiene {n} eventos (¿de un smoke? el monitor los mezclará; "
                 "muévelo si es una corrida nueva)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiment_v4.yaml")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--min-disk-gb", type=float, default=20.0)
    ap.add_argument("--min-gpu-free-gb", type=float, default=3.0)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--skip-loader", action="store_true")
    ap.add_argument("--extra-output-dirs", nargs="*", default=["results_phase1_v4"])
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    print(f"PREFLIGHT v4 — {ROOT}")
    check_elliptic(ROOT / a.data_root, a.skip_loader)
    cfg = check_config(ROOT / a.config)
    check_disk(a.min_disk_gb)
    check_gpu(a.min_gpu_free_gb, a.allow_cpu)
    check_outputs(cfg, a.extra_output_dirs)
    print()
    if fails:
        print(f"PREFLIGHT FALLÓ ({len(fails)}):")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(f"PREFLIGHT OK{f'  ({len(warns)} avisos)' if warns else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
