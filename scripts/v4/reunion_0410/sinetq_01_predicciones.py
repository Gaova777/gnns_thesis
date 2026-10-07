#!/usr/bin/env python
"""Paso 1: probabilidades y marcas de todos los modelos nativos A, B y C sobre el grafo completo.

Reproduce las tasas de marcado de scripts/v4/compare_label_modes.py (que lee
meta["cross_label_eval"]["<split>"]["flag_rate"]) y las extiende a train y a conteos
absolutos. Verifica nodo a nodo contra el meta.json la tasa de val/test.

Salidas (results_v4/reunion_0410/sin_etiqueta/):
  probs.npz              matriz modelos x nodos (float32) + umbrales + metadatos
  nodos.csv.gz           índice, timestep, partición, grupo (verdad de 3 clases)
  tasas_marcado.csv      una fila por modelo x partición x grupo (n, marcados, tasa)
  verificacion_meta.csv  diferencia contra cross_label_eval del meta.json

Uso: ~/.local/bin/uv run --frozen python scripts/v4/reunion_0410/sinetq_01_predicciones.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sinetq_common import (GROUP_NAMES, OUT, ensure_dirs, build_from_meta, list_metas,
                           load_data, pick_device, split_of)


def main() -> None:
    ensure_dirs()
    data, _ = load_data()
    y3 = data.y3.numpy()
    ts = data.timestep.numpy()
    split = split_of(ts)
    group = np.vectorize(GROUP_NAMES.get)(y3)
    pd.DataFrame({"node": np.arange(len(y3)), "timestep": ts, "split": split,
                  "group": group}).to_csv(OUT / "nodos.csv.gz", index=False)

    metas = (list_metas("B") + list_metas("A") + list_metas("C", seeds=(42, 43, 44)))
    device = pick_device()
    print(f"Dispositivo: {device} · modelos: {len(metas)}")
    x = data.x.to(device)
    ei = data.edge_index.to(device)

    probs, rows, ver = [], [], []
    for m in metas:
        t0 = time.time()
        model = build_from_meta(m, data.num_node_features, device)
        with torch.no_grad():
            p = F.softmax(model(x, ei), dim=-1)[:, 1].float().cpu().numpy()
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        thr = float(m["calibrated_threshold"])
        flag = p >= thr
        key = f"{m['_mode']}|{m['architecture']}|{m['balancing']}|{m['seed']}"
        probs.append(p.astype(np.float32))
        for sp in ("train", "val", "test"):
            for g in ("licita", "ilicita", "sin_etiqueta"):
                msk = (split == sp) & (group == g)
                rows.append({"key": key, "mode": m["_mode"], "arch": m["architecture"],
                             "balancing": m["balancing"], "seed": m["seed"], "split": sp,
                             "group": g, "n": int(msk.sum()), "flagged": int(flag[msk].sum()),
                             "rate": float(flag[msk].mean()),
                             "mean_prob": float(p[msk].mean())})
        ce = m["cross_label_eval"]
        for sp in ("val", "test"):
            for g, gname in (("licit", "licita"), ("illicit", "ilicita"),
                             ("unknown", "sin_etiqueta")):
                msk = (split == sp) & (group == gname)
                ver.append({"key": key, "split": sp, "group": g,
                            "rate_meta": ce[sp]["flag_rate"][g],
                            "rate_recomputed": float(flag[msk].mean()),
                            "diff_nodes": int(round(abs(ce[sp]["flag_rate"][g]
                                                        - flag[msk].mean()) * msk.sum()))})
        print(f"  {key:45s} thr={thr:.3f} val sin etiqueta marcados="
              f"{flag[(split == 'val') & (group == 'sin_etiqueta')].mean():.4f} "
              f"({time.time() - t0:.1f}s)")

    meta_tab = pd.DataFrame([{"key": f"{m['_mode']}|{m['architecture']}|{m['balancing']}|"
                                     f"{m['seed']}",
                              "mode": m["_mode"], "arch": m["architecture"],
                              "balancing": m["balancing"], "seed": m["seed"],
                              "threshold": float(m["calibrated_threshold"]),
                              "val_pr_auc": float(m["val_metrics"]["pr_auc"]),
                              "gate": bool(m["quality_passed"])} for m in metas])
    np.savez_compressed(OUT / "probs.npz", probs=np.stack(probs), keys=meta_tab["key"].values,
                        thresholds=meta_tab["threshold"].values)
    meta_tab.to_csv(OUT / "modelos.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "tasas_marcado.csv", index=False)
    v = pd.DataFrame(ver)
    v.to_csv(OUT / "verificacion_meta.csv", index=False)
    print(f"Verificación contra meta.json: máx. |diferencia| = "
          f"{(v.rate_meta - v.rate_recomputed).abs().max():.6f} "
          f"(máx. {v.diff_nodes.max()} nodos)")


if __name__ == "__main__":
    main()
