"""
Fase 1 FACTORIAL — 4 arq × 5 escenarios × 3 balanceos × 3 explicadores + Fidelity±.
Reusa el pipeline (build_model, Trainer, create_explainer, train_pgexplainer,
explain_node_shap, compute_shap_concentration, pairwise_spearman, get_loss_function,
plausibility). NO reimplementa modelos/explainers/estabilidad.

Cada explicador aporta métricas distintas (se escribe NaN donde no aplica):
  - GNNExplainer: estabilidad (Spearman features), plaus_edge, plaus_feat, fidelity±.
  - GNNShap:      estabilidad (Spearman features), plaus_feat, shap_concentration.
  - PGExplainer:  plaus_edge, fidelity±, degeneracy flag (entrena 1 modelo paramétrico
                  → determinista; su estabilidad se reporta como N/A, es un resultado).
"""
import sys, csv, argparse, statistics as st, warnings
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, "."); sys.path.insert(0, "phase1")
warnings.filterwarnings("ignore")
from sklearn.preprocessing import RobustScaler
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.explain.metric import fidelity
from src.training.trainer import build_model, Trainer
from src.explainability.explainer_runner import create_explainer, train_pgexplainer
from src.explainability.shap_runner import explain_node_shap, compute_shap_concentration
from src.explainability.extraction import extract_feature_ranking
from src.stability.metrics import pairwise_spearman
from src.data.imbalance import create_imbalance_scenario
from src.balancing.losses import get_loss_function
import plausibility

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RATIO = {"1:1": 1.0, "1:10": 0.1, "1:50": 0.02, "1:100": 0.01, "natural": None}


def load_scaled(path):
    d = torch.load(path, weights_only=False)
    sc = RobustScaler().fit(d.x[d.train_mask].numpy())
    d.x = torch.clamp(torch.tensor(sc.transform(d.x.numpy()), dtype=torch.float), -10, 10)
    return d


def train_model(d, arch, balancing, epochs=120):
    torch.manual_seed(42)
    kw = {"heads": 4} if arch == "GAT" else ({"K": 3} if arch == "TAGCN" else {})
    model = build_model(arch, in_channels=d.x.size(1), hidden_channels=64, num_layers=2, dropout=0.3, **kw)
    loss = get_loss_function(balancing, labels=d.y, mask=d.train_mask, device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    tr = Trainer(model, loss, opt, device=DEVICE, patience=25, disable_checkpointing=True,
                 early_stop_metric="pr_auc")
    tr.train(d, epochs=epochs, verbose=False, run_name=f"f_{arch}_{balancing}")
    model.eval()
    return model, tr.evaluate(d, "val_mask")


@torch.no_grad()
def tp_val_nodes(model, d, n):
    out = model(d.x.to(DEVICE), d.edge_index.to(DEVICE)).argmax(1).cpu()
    iv = torch.where(d.val_mask & (d.y == 1))[0]
    tp = iv[out[iv] == 1]
    rng = np.random.RandomState(42)
    return rng.choice(tp.numpy(), size=min(n, len(tp)), replace=False).tolist() if len(tp) else []


def _fid(explainer, exp):
    try:
        fp, fm = fidelity(explainer, exp); return float(fp), float(fm)
    except Exception:
        return None, None


def explain_cell(model, d, nodes, explainer_name, base_k, num_replicas, ex_epochs, shap_samples,
                 feat_index, top_k_edges, per_node_rows, meta):
    # PGExplainer se corre en CPU: en GPU sus fallos son device-side asserts que corrompen
    # el contexto CUDA; en CPU son excepciones atrapables. Es determinista tras entrenar,
    # así que su estabilidad se reporta N/A.
    pg, pg_ok, ex_dev = None, True, DEVICE
    if explainer_name == "PGExplainer":
        ex_dev = "cpu"; model = model.to("cpu")
        pg = create_explainer(model, "PGExplainer", epochs=ex_epochs, lr=0.003)
        try:
            pg_ok = train_pgexplainer(pg, d, device="cpu")
        except Exception:
            pg_ok = False
    agg = {"sp": [], "ple": [], "plf": [], "conc": [], "fp": [], "fm": [], "subn": [], "emstd": []}
    for nid in nodes:
        subset, sub_ei, mapping, ebool = k_hop_subgraph(int(nid), base_k, d.edge_index,
                                                        relabel_nodes=True, num_nodes=d.num_nodes)
        tl = int(mapping[0])
        sub_x = d.x[subset].to(ex_dev); sub_ei_d = sub_ei.to(ex_dev)
        sub_edge_typ = d.typology_edge[ebool].numpy(); node_typ = int(d.typology_node[nid])
        rankings, edge_masks, feat_masks, conc, fps, fms = [], [], [], [], [], []
        reps = 1 if explainer_name == "PGExplainer" else num_replicas   # PG determinista tras entrenar
        for r in range(reps):
            torch.manual_seed(42 + r * 17)
            try:
                if explainer_name == "GNNExplainer":
                    ex = create_explainer(model, "GNNExplainer", epochs=ex_epochs, lr=0.01)
                    exp = ex(sub_x, sub_ei_d, index=tl)
                    if exp.edge_mask is not None: edge_masks.append(exp.edge_mask.detach().cpu().numpy())
                    nm = getattr(exp, "node_mask", None)
                    if nm is not None:
                        rankings.append(np.asarray(extract_feature_ranking(explanation=exp)))
                        fm = nm.detach().cpu().numpy(); feat_masks.append(np.abs(fm).mean(0) if fm.ndim == 2 else np.abs(fm))
                    fp, fmn = _fid(ex, exp)
                    if fp is not None: fps.append(fp); fms.append(fmn)
                    del ex, exp
                elif explainer_name == "PGExplainer":
                    if not pg_ok: break
                    exp = pg(sub_x, sub_ei_d, index=tl, target=d.y[int(nid)].to("cpu"))
                    if exp.edge_mask is not None: edge_masks.append(exp.edge_mask.detach().cpu().numpy())
                    fp, fmn = _fid(pg, exp)
                    if fp is not None: fps.append(fp); fms.append(fmn)
                    del exp
                elif explainer_name == "GNNShap":
                    res = explain_node_shap(model, d, int(nid), num_samples=shap_samples, device=DEVICE, seed=42 + r * 17)
                    rankings.append(np.asarray(res["feature_ranking"]))
                    feat_masks.append(np.abs(np.asarray(res["shap_values"])))
                    conc.append(res.get("concentration") or compute_shap_concentration(np.asarray(res["shap_values"])))
            except Exception:
                if explainer_name == "PGExplainer": pg_ok = False; break
                continue
            if ex_dev != "cpu": torch.cuda.empty_cache()
        # ---- métricas por nodo ----
        row = dict(architecture=meta[0], scenario=meta[1], balancing=meta[2], explainer=explainer_name,
                   node=int(nid), node_typ=node_typ, spearman=None, plaus_edge=None, plaus_feat=None,
                   shap_conc=None, fid_plus=None, fid_minus=None, sub_n_nodes=int(subset.numel()))
        if len(rankings) >= 2:
            s = pairwise_spearman([np.asarray(x) for x in rankings])["mean"]
            if s == s: row["spearman"] = float(s); agg["sp"].append(float(s))
        if edge_masks:
            mem = np.mean(np.stack(edge_masks), axis=0)
            agg["emstd"].append(float(np.std(mem)))
            pl = plausibility.edge_plausibility(mem, sub_ei, sub_edge_typ, node_typ)
            if pl: row["plaus_edge"] = pl["f1"]; agg["ple"].append(pl["f1"])
        if feat_masks and feat_index and node_typ in feat_index:
            mfm = np.mean(np.stack(feat_masks), axis=0)
            fpd = plausibility.feature_plausibility(mfm, [feat_index[node_typ]], top_k=3)
            if fpd: row["plaus_feat"] = fpd["recall"]; agg["plf"].append(fpd["recall"])
        if conc: row["shap_conc"] = float(np.mean(conc)); agg["conc"].append(float(np.mean(conc)))
        if fps: row["fid_plus"] = float(np.mean(fps)); row["fid_minus"] = float(np.mean(fms))
        if fps: agg["fp"].append(float(np.mean(fps))); agg["fm"].append(float(np.mean(fms)))
        agg["subn"].append(int(subset.numel()))
        per_node_rows.append(row)
    if explainer_name == "PGExplainer": model.to(DEVICE)   # restaurar a GPU para el siguiente explicador
    def m(x): return round(float(st.mean(x)), 3) if x else float("nan")
    degenerate = (explainer_name == "PGExplainer" and (not pg_ok or (agg["emstd"] and np.mean(agg["emstd"]) < 1e-3)))
    return dict(architecture=meta[0], scenario=meta[1], balancing=meta[2], explainer=explainer_name,
                val_pr_auc=meta[3], n_tp=len(nodes), spearman_mean=m(agg["sp"]),
                plaus_edge_mean=m(agg["ple"]), plaus_feat_mean=m(agg["plf"]), shap_conc_mean=m(agg["conc"]),
                fid_plus_mean=m(agg["fp"]), fid_minus_mean=m(agg["fm"]),
                subgraph_n_nodes=int(st.median(agg["subn"])) if agg["subn"] else 0,
                pg_degenerate=("yes" if degenerate else ("no" if explainer_name == "PGExplainer" else "")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="phase1/synthetic_aml_v2.pt")
    ap.add_argument("--archs", nargs="+", default=["GraphSAGE", "GAT", "GCN", "TAGCN"])
    ap.add_argument("--scenarios", nargs="+", default=["1:1", "1:10", "1:50", "1:100", "natural"])
    ap.add_argument("--balancings", nargs="+", default=["none", "class_weighting", "focal_loss"])
    ap.add_argument("--explainers", nargs="+", default=["GNNExplainer", "PGExplainer", "GNNShap"])
    ap.add_argument("--nodes", type=int, default=30)
    ap.add_argument("--replicas", type=int, default=5)
    ap.add_argument("--ex-epochs", type=int, default=100)
    ap.add_argument("--shap-samples", type=int, default=40)
    ap.add_argument("--out", default="phase1/results_factorial.csv")
    a = ap.parse_args()
    d0 = load_scaled(a.data)
    feat_index = getattr(d0, "typology_feature_index", None)
    print(f"device={DEVICE} | {len(a.archs)}arch × {len(a.scenarios)}scen × {len(a.balancings)}bal "
          f"× {len(a.explainers)}expl | nodes={a.nodes} replicas={a.replicas}")
    cell_rows, per_node_rows = [], []
    for arch in a.archs:
        for scen in a.scenarios:
            for bal in a.balancings:
                d = create_imbalance_scenario(d0, RATIO[scen], seed=42) if RATIO[scen] is not None else d0
                model, valm = train_model(d, arch, bal)
                base_k = 2
                nodes = tp_val_nodes(model, d, a.nodes)
                meta = (arch, scen, bal, round(valm["pr_auc"], 3))
                if not nodes:
                    for ex in a.explainers:
                        cell_rows.append(dict(architecture=arch, scenario=scen, balancing=bal, explainer=ex,
                                              val_pr_auc=meta[3], n_tp=0, spearman_mean=float("nan"),
                                              plaus_edge_mean=float("nan"), plaus_feat_mean=float("nan"),
                                              shap_conc_mean=float("nan"), fid_plus_mean=float("nan"),
                                              fid_minus_mean=float("nan"), subgraph_n_nodes=0, pg_degenerate=""))
                    print(f"  {arch:10}{scen:8}{bal:16} SKIP (0 TP)")
                    continue
                for ex in a.explainers:
                    c = explain_cell(model, d, nodes, ex, base_k, a.replicas, a.ex_epochs, a.shap_samples,
                                     feat_index, 20, per_node_rows, meta)
                    cell_rows.append(c)
                    print(f"  {arch:10}{scen:8}{bal:16}{ex:13} spear={c['spearman_mean']} "
                          f"plE={c['plaus_edge_mean']} plF={c['plaus_feat_mean']} conc={c['shap_conc_mean']} "
                          f"fid+={c['fid_plus_mean']} n_tp={c['n_tp']} {('PGdeg='+c['pg_degenerate']) if ex=='PGExplainer' else ''}")
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cell_rows[0].keys())); w.writeheader(); w.writerows(cell_rows)
    pn = a.out.replace(".csv", "_pernode.csv")
    with open(pn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_node_rows[0].keys())); w.writeheader(); w.writerows(per_node_rows)
    print(f"\nescrito: {a.out} ({len(cell_rows)} celdas) + {pn} ({len(per_node_rows)} nodos)")


if __name__ == "__main__":
    main()
