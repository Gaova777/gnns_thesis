"""Experimento corto (reunión 04-10, pregunta B / H2): ¿por qué GCN queda atrás en Elliptic?

No toca nada existente; escribe en results_v4/reunion_0410/pgexpl_gcn/.

1. Estadísticas del grafo en modo C: grado de entrada, homofilia de aristas por clase
   (fracción de vecinos de entrada de una ilícita que también son ilícitas), nodos aislados.
2. Mismo bucle de entrenamiento y MISMOS hiperparámetros para todos los modelos (para aislar
   la arquitectura del presupuesto de Optuna):
     MLP        sin grafo, solo las variables del nodo
     GCN        src.models.gcn.GCN (Kipf y Welling: D^-1/2 (A+I) D^-1/2, el nodo propio
                comparte peso con los vecinos)
     GCN_root   GCNConv + Linear del propio nodo en cada capa (separación ego/vecinos)
     GraphSAGE  src.models.sage.GraphSAGE (peso raíz separado, agregación media)
   Escenario nativo, modo C, balanceos none y class_weighting, semillas 42 y 43,
   300 épocas, se reporta la mejor PR-AUC de validación y la final, y la PR-AUC de test en
   la época de mejor validación.

Uso:
  ~/.local/bin/uv run --frozen python scripts/v4/reunion_0410/gcn_vs_baselines.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import auc, precision_recall_curve
from torch_geometric.nn import GCNConv

from src.balancing.losses import compute_class_weights
from src.data.loader import apply_label_mode, load_elliptic
from src.data.preprocessing import preprocess
from src.models.gcn import GCN
from src.models.sage import GraphSAGE

OUT = ROOT / "results_v4/reunion_0410/pgexpl_gcn"


class MLP(nn.Module):
    def __init__(self, cin, hid, num_layers=2, dropout=0.3):
        super().__init__()
        dims = [cin] + [hid] * (num_layers - 1) + [2]
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        for lin in self.lins[:-1]:
            x = F.dropout(F.relu(lin(x)), self.dropout, self.training)
        return self.lins[-1](x)


class GCNRoot(nn.Module):
    """GCNConv(x) + W_root x in every layer: GCN with a separate weight for the node itself."""

    def __init__(self, cin, hid, num_layers=2, dropout=0.3):
        super().__init__()
        dims = [cin] + [hid] * (num_layers - 1) + [2]
        self.convs = nn.ModuleList(GCNConv(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.roots = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, (c, r) in enumerate(zip(self.convs, self.roots)):
            x = c(x, edge_index) + r(x)
            if i < len(self.convs) - 1:
                x = F.dropout(F.relu(x), self.dropout, self.training)
        return x


def pr_auc(y, p):
    pr, rc, _ = precision_recall_curve(y, p, pos_label=1)
    return float(auc(rc, pr))


def graph_stats(data):
    ei = data.edge_index
    y = data.y
    lab = data.y3 if hasattr(data, "y3") else y
    indeg = torch.bincount(ei[1], minlength=data.num_nodes)
    outdeg = torch.bincount(ei[0], minlength=data.num_nodes)
    st = {}
    for name, m in (("train", data.train_mask), ("val", data.val_mask)):
        il = m & (y == 1)
        st[f"{name}_indeg_median_illicit"] = float(indeg[il].float().median())
        st[f"{name}_indeg_median_neg"] = float(indeg[m & (y == 0)].float().median())
        st[f"{name}_frac_indeg0_illicit"] = float((indeg[il] == 0).float().mean())
        st[f"{name}_frac_isolated_illicit"] = float(((indeg[il] + outdeg[il]) == 0).float().mean())
        st[f"{name}_frac_deg_le2_illicit"] = float(((indeg[il] + outdeg[il]) <= 2).float().mean())
    # edge homophily per class (undirected view: both endpoints), using the 3-class truth
    src, dst = ei
    both = (lab[src] >= 0) & (lab[dst] >= 0)
    for c, nm in ((1, "illicit"), (0, "licit")):
        e = both & ((lab[src] == c) | (lab[dst] == c))
        same = (lab[src] == lab[dst]) & e
        st[f"homophily_labelled_edges_{nm}"] = float(same.sum() / e.sum())
    # mode C view: fraction of an illicit node's graph neighbours (any direction) that are illicit
    e_il = (y[src] == 1) | (y[dst] == 1)
    st["modeC_frac_neighbours_illicit_given_illicit"] = float(((y[src] == 1) & (y[dst] == 1)).sum() / e_il.sum())
    st["modeC_prevalence_illicit"] = float((y == 1).sum() / (y >= 0).sum())
    return st


def run(model_name, bal, seed, data, cfg, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cin = data.num_node_features
    hid, nl, dp = cfg["hidden"], cfg["layers"], cfg["dropout"]
    model = {"MLP": lambda: MLP(cin, hid, nl, dp),
             "GCN": lambda: GCN(cin, hid, nl, dp),
             "GCN_root": lambda: GCNRoot(cin, hid, nl, dp),
             "GraphSAGE": lambda: GraphSAGE(cin, hid, nl, dp)}[model_name]().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    x, ei, y = data.x.to(device), data.edge_index.to(device), data.y.to(device)
    tm, vm, te = data.train_mask.to(device), data.val_mask.to(device), data.test_mask.to(device)
    w = compute_class_weights(data.y, data.train_mask).to(device) if bal == "class_weighting" else None
    yv, yt = y[vm].cpu().numpy(), y[te].cpu().numpy()
    best, best_ep, best_test, hist = -1, -1, None, []
    t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train(); opt.zero_grad()
        out = model(x, ei)
        loss = F.cross_entropy(out[tm], y[tm], weight=w)
        loss.backward(); opt.step()
        if ep % 5 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                p = torch.softmax(model(x, ei), -1)[:, 1]
            v = pr_auc(yv, p[vm].cpu().numpy())
            hist.append((ep, v))
            if v > best:
                best, best_ep = v, ep
                best_test = pr_auc(yt, p[te].cpu().numpy())
    return {"model": model_name, "balancing": bal, "seed": seed, "best_val_pr_auc": best,
            "best_epoch": best_ep, "final_val_pr_auc": hist[-1][1], "test_pr_auc_at_best": best_test,
            "val_pr_auc_ep50": dict(hist).get(50), "val_pr_auc_ep150": dict(hist).get(150),
            "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--models", nargs="+", default=["MLP", "GCN", "GCN_root", "GraphSAGE"])
    ap.add_argument("--hp", default="common", choices=["common", "gcn_optuna"])
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    with open(ROOT / "configs/experiment_v4.yaml", encoding="utf-8") as f:
        d = yaml.safe_load(f)["data"]
    data = load_elliptic(root=d.get("root", "./data"))
    apply_label_mode(data, "licit_unknown")
    preprocess(data, train_range=tuple(d.get("train_timesteps", (1, 34))),
               val_range=tuple(d.get("val_timesteps", (35, 42))),
               test_range=tuple(d.get("test_timesteps", (43, 49))))
    OUT.mkdir(parents=True, exist_ok=True)
    st = graph_stats(data)
    st["num_node_features"] = int(data.num_node_features)
    (OUT / "gcn_graph_stats.json").write_text(json.dumps(st, indent=2), encoding="utf-8")
    print(json.dumps(st, indent=2))
    # common: the GraphSAGE/GCN warm-start prior chosen by Optuna in several cells
    # (hidden 64, dropout 0.488, lr 0.0046, wd 2.7e-5); gcn_optuna: GCN native's own choice.
    cfg = ({"hidden": 64, "layers": 2, "dropout": 0.4879639408647978, "lr": 0.004622589001020831,
            "wd": 2.6587543983272695e-05} if a.hp == "common" else
           {"hidden": 148, "layers": 2, "dropout": 0.12602063719411183, "lr": 0.007902619549708232,
            "wd": 0.0008536189862866829})
    cfg["epochs"] = a.epochs
    rows = []
    for seed in a.seeds:
        for bal in ("none", "class_weighting"):
            for mn in a.models:
                r = run(mn, bal, seed, data, cfg, a.device)
                r["hp"] = a.hp
                rows.append(r)
                print(r, flush=True)
                pd.DataFrame(rows).to_csv(OUT / f"gcn_vs_baselines_{a.hp}.csv", index=False)


if __name__ == "__main__":
    main()
