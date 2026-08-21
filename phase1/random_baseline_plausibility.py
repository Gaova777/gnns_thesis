"""Baseline aleatorio de plausibilidad sobre el grafo sintetico principal (g0).

Ancla el 0,80 de PGExplainer y el 0,50 de GNNExplainer: cuanto obtiene un
explicador que ordena las aristas AL AZAR, bajo EXACTAMENTE el mismo protocolo
(subgrafo receptivo de 2 saltos, top-k balanceado = numero de aristas de patron,
misma funcion plausibility.edge_plausibility).

No requiere modelo ni GPU: el baseline aleatorio depende solo de la topologia,
por lo que se evalua sobre todos los nodos de patron del grafo natural.

Uso:  uv run python phase1/random_baseline_plausibility.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_geometric.utils import k_hop_subgraph  # noqa: E402

from phase1.synthetic_aml_generator import generate_aml_graph  # noqa: E402
from phase1 import plausibility  # noqa: E402

BASE_K = 2        # mismos 2 saltos que run_phase1_robust.py
T = 300           # sorteos Monte Carlo por nodo
rng = np.random.default_rng(0)

d = generate_aml_graph()                     # defaults = grafo principal v2 (seed=42)
N = int(d.num_nodes)
E = int(d.edge_index.size(1))
patt = torch.where(d.typology_node > 0)[0].tolist()
feat_index = d.typology_feature_index
Fdim = int(d.x.size(1))
print(f"grafo g0: N={N}  E={E}  nodos de patron={len(patt)}  F={Fdim}")

edge_exp, edge_mc, feat_mc = [], [], []
n_eval = 0
for nid in patt:
    subset, sub_ei, mapping, ebool = k_hop_subgraph(
        int(nid), BASE_K, d.edge_index, relabel_nodes=True, num_nodes=N)
    sub_edge_typ = d.typology_edge[ebool].numpy()
    node_typ = int(d.typology_node[nid])
    true_edges = np.where(sub_edge_typ == node_typ)[0]
    if len(true_edges) == 0:
        continue                              # no evaluable, igual que en el pipeline real
    n_eval += 1
    E_sub = len(sub_edge_typ)
    edge_exp.append(len(true_edges) / E_sub)  # expectativa exacta con k balanceado

    f1s = []
    for _ in range(T):
        mask = rng.random(E_sub)              # importancia de arista AL AZAR
        pl = plausibility.edge_plausibility(mask, sub_ei, sub_edge_typ, node_typ)
        f1s.append(pl["f1"])
    edge_mc.append(float(np.mean(f1s)))

    if node_typ in feat_index:
        ff = []
        for _ in range(T):
            fp = plausibility.feature_plausibility(
                rng.random(Fdim), [feat_index[node_typ]], top_k=3)
            ff.append(fp["f1"])
        feat_mc.append(float(np.mean(ff)))

edge_mc = np.array(edge_mc)
edge_exp = np.array(edge_exp)
feat_mc = np.array(feat_mc)

print(f"\nnodos evaluables (con aristas de patron en su subgrafo 2-hop): {n_eval}")
print("\n================ PLAUSIBILIDAD DE ARISTAS (metrica primaria) ================")
print(f"  baseline aleatorio  (Monte Carlo, T={T}) : {edge_mc.mean():.4f}  +/- {edge_mc.std():.4f}")
print(f"  baseline aleatorio  (expectativa |patron|/|subgrafo|): {edge_exp.mean():.4f}")
print(f"  referencia manuscrito: PGExplainer 0,80 | GNNExplainer 0,50")
print(f"  PGExplainer  / azar = {0.80 / edge_mc.mean():.1f}x")
print(f"  GNNExplainer / azar = {0.50 / edge_mc.mean():.1f}x")
if len(feat_mc):
    print("\n================ PLAUSIBILIDAD DE FEATURES (secundaria) ================")
    print(f"  baseline aleatorio  (Monte Carlo, T={T}) : {feat_mc.mean():.4f}  +/- {feat_mc.std():.4f}")
    print(f"  (top-3 de {Fdim} features; una sola feature-firma es la correcta)")
