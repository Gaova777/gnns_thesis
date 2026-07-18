"""
Fase 1 — matriz reducida sobre el dataset sintético: estabilidad Y plausibilidad juntas.
Reusa el pipeline corregido (build_model, Trainer, create_explainer, k_hop_subgraph,
pairwise_spearman) + phase1/plausibility.py. NO reimplementa modelos/explainers/estabilidad.

Para cada celda (arquitectura × escenario de desbalance) entrena 1 modelo y, sobre cada
verdadero positivo de VALIDACIÓN (mismo criterio que Elliptic), mide:
  - estabilidad: Spearman del ranking de features entre réplicas (idéntico a Elliptic).
  - plausibilidad de edges (primaria): edge_plausibility del edge_mask (consenso entre
    réplicas) vs los edges de la tipología del nodo (ground-truth).
Escribe CSV por-nodo y resumen por-celda.
"""
import sys, csv, argparse, statistics as st
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, ".")
sys.path.insert(0, "phase1")
from sklearn.preprocessing import RobustScaler
from torch_geometric.utils import k_hop_subgraph
from src.training.trainer import build_model, Trainer
from src.explainability.explainer_runner import create_explainer
from src.explainability.extraction import extract_feature_ranking
from src.stability.metrics import pairwise_spearman
from src.data.imbalance import create_imbalance_scenario
import plausibility

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_scaled(path="phase1/synthetic_aml_v1.pt"):
    d = torch.load(path, weights_only=False)
    sc = RobustScaler().fit(d.x[d.train_mask].numpy())
    d.x = torch.clamp(torch.tensor(sc.transform(d.x.numpy()), dtype=torch.float), -10, 10)
    return d


def train_model(d, arch, epochs=120):
    torch.manual_seed(42)
    kw = {"heads": 4} if arch == "GAT" else ({"K": 3} if arch == "TAGCN" else {})
    model = build_model(arch, in_channels=d.x.size(1), hidden_channels=64,
                        num_layers=2, dropout=0.3, **kw)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    tr = Trainer(model, nn.CrossEntropyLoss(), opt, device=DEVICE, patience=25,
                 disable_checkpointing=True, early_stop_metric="pr_auc")
    tr.train(d, epochs=epochs, verbose=False, run_name=f"synth_{arch}")
    model.eval()
    return model, tr.evaluate(d, "val_mask")


@torch.no_grad()
def tp_val_nodes(model, d, n):
    out = model(d.x.to(DEVICE), d.edge_index.to(DEVICE)).argmax(1).cpu()
    illicit_val = torch.where(d.val_mask & (d.y == 1))[0]
    tp = illicit_val[out[illicit_val] == 1]
    rng = np.random.RandomState(42)
    return rng.choice(tp.numpy(), size=min(n, len(tp)), replace=False).tolist() if len(tp) else []


def explain_node(model, d, nid, base_k, num_replicas, ex_epochs):
    """Devuelve (rankings[list], mean_edge_mask, sub_edge_typ, node_typ, n_nodes, n_edges)."""
    subset, sub_ei, mapping, edge_bool = k_hop_subgraph(
        int(nid), base_k, d.edge_index, relabel_nodes=True, num_nodes=d.num_nodes)
    sub_x = d.x[subset].to(DEVICE); sub_ei_d = sub_ei.to(DEVICE); tl = int(mapping[0])
    sub_edge_typ = d.typology_edge[edge_bool].numpy()
    node_typ = int(d.typology_node[nid])
    rankings, edge_masks, feat_masks = [], [], []
    for r in range(num_replicas):
        torch.manual_seed(42 + r * 17)
        ex = create_explainer(model, "GNNExplainer", epochs=ex_epochs, lr=0.01)
        exp = ex(sub_x, sub_ei_d, index=tl)
        rankings.append(extract_feature_ranking(explanation=exp))
        em = (exp.edge_mask.detach().cpu().numpy() if exp.edge_mask is not None
              else np.zeros(sub_ei.size(1)))
        edge_masks.append(em)
        # node_mask: importancia por feature (para plausibilidad de features)
        nm = getattr(exp, "node_mask", None)
        if nm is not None:
            fm = nm.detach().cpu().numpy()
            fm = np.abs(fm).mean(axis=0) if fm.ndim == 2 else np.abs(fm)
            feat_masks.append(fm)
        del ex, exp
        if DEVICE != "cpu": torch.cuda.empty_cache()
    mean_em = np.mean(np.stack(edge_masks), axis=0)
    mean_fm = np.mean(np.stack(feat_masks), axis=0) if feat_masks else None
    return rankings, mean_em, mean_fm, sub_ei, sub_edge_typ, node_typ, subset.numel(), sub_ei.size(1)


def run_cell(d0, arch, target_ratio, scen_name, nodes_per_class, num_replicas, ex_epochs,
             per_node_rows):
    d = create_imbalance_scenario(d0, target_ratio, seed=42) if target_ratio is not None else d0
    model, valm = train_model(d, arch)
    # base_k=2 UNIFORME para las 4 arqs → scope de subgrafo comparable (elección del auditor).
    # CAVEAT TAGCN: su receptive field real es num_layers*K=6; en este grafo denso k=6 captura
    # ~2944 nodos (casi todo el grafo) y degenera la plausibilidad. Se usa k=2 (scope local,
    # comparable) aceptando que la explicación de TAGCN no reproduce del todo su predicción.
    base_k = 2
    nodes = tp_val_nodes(model, d, nodes_per_class)
    feat_index = getattr(d, "typology_feature_index", None)
    spearmans, plaus_f1, plaus_p, plaus_r, subn, featpl = [], [], [], [], [], []
    for nid in nodes:
        rankings, mean_em, mean_fm, sub_ei, sub_edge_typ, node_typ, nn_, ne_ = explain_node(
            model, d, nid, base_k, num_replicas, ex_epochs)
        # estabilidad (Spearman de rankings de features entre réplicas)
        sp = pairwise_spearman([np.asarray(r) for r in rankings], top_k=None)["mean"]
        # plausibilidad de edges (consenso) vs ground-truth de la tipología
        pl = plausibility.edge_plausibility(mean_em, sub_ei, sub_edge_typ, node_typ, top_k=None)
        # plausibilidad de features: ¿el explainer señala la feature-firma de la tipología?
        fp = None
        if mean_fm is not None and feat_index is not None and node_typ in feat_index:
            fpd = plausibility.feature_plausibility(mean_fm, [feat_index[node_typ]], top_k=3)
            fp = fpd["recall"] if fpd is not None else None  # recall@3 de la feature-firma
        row = dict(architecture=arch, scenario=scen_name, node=int(nid), node_typ=node_typ,
                   spearman=None if sp != sp else float(sp),
                   plaus_f1=None, plaus_prec=None, plaus_rec=None, plaus_feat=fp,
                   sub_n_nodes=nn_, sub_n_edges=ne_)
        if pl is not None:
            row.update(plaus_f1=pl["f1"], plaus_prec=pl["precision"], plaus_rec=pl["recall"])
            plaus_f1.append(pl["f1"]); plaus_p.append(pl["precision"]); plaus_r.append(pl["recall"])
        if fp is not None:
            featpl.append(fp)
        if sp == sp:
            spearmans.append(float(sp))
        subn.append(nn_)
        per_node_rows.append(row)
    def m(x): return float(st.mean(x)) if x else float("nan")
    return dict(architecture=arch, scenario=scen_name, val_pr_auc=round(valm["pr_auc"], 3),
                n_tp=len(nodes), n_plausible=len(plaus_f1),
                spearman_mean=round(m(spearmans), 3),
                plaus_f1_mean=round(m(plaus_f1), 3), plaus_prec_mean=round(m(plaus_p), 3),
                plaus_rec_mean=round(m(plaus_r), 3), plaus_feat_mean=round(m(featpl), 3),
                subgraph_n_nodes=int(st.median(subn)) if subn else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=["GraphSAGE", "GAT"])
    ap.add_argument("--scenarios", nargs="+", default=["1:1", "1:10"])
    ap.add_argument("--nodes", type=int, default=30)
    ap.add_argument("--replicas", type=int, default=5)
    ap.add_argument("--ex-epochs", type=int, default=100)
    ap.add_argument("--out", default="phase1/results_phase1.csv")
    ap.add_argument("--data", default="phase1/synthetic_aml_v1.pt")
    args = ap.parse_args()
    RATIO = {"1:1": 1.0, "1:10": 0.1, "1:50": 0.02, "natural": None}

    d0 = load_scaled(args.data)
    print(f"device={DEVICE} | archs={args.archs} scenarios={args.scenarios} "
          f"nodes={args.nodes} replicas={args.replicas}")
    per_node_rows, cell_rows = [], []
    for arch in args.archs:
        for scen in args.scenarios:
            cell = run_cell(d0, arch, RATIO[scen], scen, args.nodes, args.replicas,
                            args.ex_epochs, per_node_rows)
            cell_rows.append(cell)
            print(f"  {arch:10} {scen:5} | valPRAUC={cell['val_pr_auc']} "
                  f"Spearman={cell['spearman_mean']} plausF1={cell['plaus_f1_mean']} "
                  f"n_tp={cell['n_tp']} n_plaus={cell['n_plausible']} sub_nodes={cell['subgraph_n_nodes']}")
    # escribir CSVs
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cell_rows[0].keys())); w.writeheader(); w.writerows(cell_rows)
    pn = args.out.replace(".csv", "_pernode.csv")
    with open(pn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_node_rows[0].keys())); w.writeheader(); w.writerows(per_node_rows)
    print(f"\nescrito: {args.out} ({len(cell_rows)} celdas) + {pn} ({len(per_node_rows)} nodos)")


if __name__ == "__main__":
    main()
