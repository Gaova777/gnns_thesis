"""
Fase 1 ROBUSTO — extiende run_phase1_factorial.py con replicación estadística.
Encargo ENCARGO_robustez_estadistica.md. NO cambia hallazgos; solo cuantifica incertidumbre.

Tres piezas añadidas sobre el factorial:
  PIEZA 1 — semillas de MODELO por celda (nueva fuente de varianza → habilita inferencia).
            train_model(seed) variable; cada celda entrena S modelos independientes. La columna
            model_seed distingue esta varianza de la del EXPLICADOR (r*17, estabilidad) y de la
            del GRAFO (graph_seed). El submuestreo de imbalance queda FIJO (seed=42) para que la
            única fuente que varía entre model_seeds sea el entrenamiento, no los datos.
  PIEZA 2 — Fidelity± MANUAL y uniforme para los 3 explicadores (el built-in de PyG solo cubría
            GNNExplainer). Definición común basada en probabilidad de la clase predicha:
              fid+ = p_orig - p(quitar top-k importantes)   (más caída = más fiel)
              fid- = p_orig - p(conservar SOLO top-k)        (más cerca de 0 = más fiel)
            Se enmascaran EDGES para GNNExplainer/PGExplainer y FEATURES para GNNShap (no es la
            misma máscara; se declara). top-k = fid_frac del nº de edges/features (default 25%).
  PIEZA 3 — semillas de GRAFO: --data acepta varios .pt; la columna graph_seed permite probar que
            los hallazgos no dependen de una instancia del generador.

Salida: results_robust*.csv (celda) + _pernode.csv, con columnas graph_seed y model_seed. Escribe
de forma incremental (reescribe el CSV tras cada super-celda) y admite --resume: tolera cortes en
corridas largas sin perder lo hecho.
"""
import os, sys, csv, argparse, statistics as st, warnings
import numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "phase1")
warnings.filterwarnings("ignore")
from sklearn.preprocessing import RobustScaler
from torch_geometric.utils import k_hop_subgraph
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

CELL_FIELDS = ["graph_seed", "architecture", "scenario", "balancing", "model_seed", "explainer",
               "val_pr_auc", "n_tp", "spearman_mean", "plaus_edge_mean", "plaus_feat_mean",
               "shap_conc_mean", "fid_plus_mean", "fid_minus_mean", "subgraph_n_nodes", "pg_degenerate"]
NODE_FIELDS = ["graph_seed", "architecture", "scenario", "balancing", "model_seed", "explainer",
               "node", "node_typ", "spearman", "plaus_edge", "plaus_feat", "shap_conc",
               "fid_plus", "fid_minus", "sub_n_nodes"]


def load_scaled(path):
    d = torch.load(path, weights_only=False)
    sc = RobustScaler().fit(d.x[d.train_mask].numpy())
    d.x = torch.clamp(torch.tensor(sc.transform(d.x.numpy()), dtype=torch.float), -10, 10)
    return d


def train_model(d, arch, balancing, epochs=120, seed=42):
    torch.manual_seed(seed)                     # PIEZA 1: semilla variable (antes fija en 42)
    kw = {"heads": 4} if arch == "GAT" else ({"K": 3} if arch == "TAGCN" else {})
    model = build_model(arch, in_channels=d.x.size(1), hidden_channels=64, num_layers=2, dropout=0.3, **kw)
    loss = get_loss_function(balancing, labels=d.y, mask=d.train_mask, device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    tr = Trainer(model, loss, opt, device=DEVICE, patience=25, disable_checkpointing=True,
                 early_stop_metric="pr_auc")
    tr.train(d, epochs=epochs, verbose=False, run_name=f"r_{arch}_{balancing}_{seed}")
    model.eval()
    return model, tr.evaluate(d, "val_mask")


@torch.no_grad()
def tp_val_nodes(model, d, n):
    out = model(d.x.to(DEVICE), d.edge_index.to(DEVICE)).argmax(1).cpu()
    iv = torch.where(d.val_mask & (d.y == 1))[0]
    tp = iv[out[iv] == 1]
    rng = np.random.RandomState(42)
    return rng.choice(tp.numpy(), size=min(n, len(tp)), replace=False).tolist() if len(tp) else []


# ---------- PIEZA 2: Fidelity± manual, misma definición para los 3 explicadores ----------
@torch.no_grad()
def _proba(model, x, ei, node):
    return torch.softmax(model(x, ei), dim=1)[node]


@torch.no_grad()
def fidelity_edges(model, x, ei, node, edge_mask, frac):
    """Enmascara EDGES (GNNExplainer, PGExplainer). fid+ quita top-k, fid- conserva solo top-k."""
    E = ei.size(1); m = np.asarray(edge_mask, dtype=float)
    if E == 0 or m.size != E:
        return None, None
    k = max(1, int(round(frac * E)))
    top = torch.as_tensor(np.argsort(-m)[:k].copy(), dtype=torch.long, device=ei.device)
    keep = torch.ones(E, dtype=torch.bool, device=ei.device); keep[top] = False
    p0 = _proba(model, x, ei, node); c = int(p0.argmax())
    fp = float(p0[c] - _proba(model, x, ei[:, keep], node)[c])    # quitar top-k → dejar complemento
    fm = float(p0[c] - _proba(model, x, ei[:, ~keep], node)[c])   # conservar SOLO top-k
    return fp, fm


@torch.no_grad()
def fidelity_features(model, x, ei, node, feat_mask, frac):
    """Enmascara FEATURES (GNNShap). Baseline = 0 (mediana aprox. tras RobustScaler)."""
    F = x.size(1); m = np.asarray(feat_mask, dtype=float)
    if m.size != F:
        return None, None
    k = max(1, int(round(frac * F)))
    top = torch.as_tensor(np.argsort(-m)[:k].copy(), dtype=torch.long, device=x.device)
    p0 = _proba(model, x, ei, node); c = int(p0.argmax())
    x_rem = x.clone(); x_rem[:, top] = 0.0                        # quitar top-k features
    x_keep = torch.zeros_like(x); x_keep[:, top] = x[:, top]      # conservar SOLO top-k
    return float(p0[c] - _proba(model, x_rem, ei, node)[c]), float(p0[c] - _proba(model, x_keep, ei, node)[c])


def explain_cell(model, d, nodes, explainer_name, base_k, num_replicas, ex_epochs, shap_samples,
                 feat_index, fid_frac, per_node_rows, meta, tag):
    """meta=(arch,scen,bal,val_pr_auc); tag=(graph_seed,model_seed)."""
    gseed, mseed = tag
    pg, pg_ok, ex_dev = None, True, DEVICE
    if explainer_name == "PGExplainer":
        ex_dev = "cpu"; model = model.to("cpu")           # en GPU sus fallos son device-side asserts
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
        rankings, edge_masks, feat_masks, conc = [], [], [], []
        reps = 1 if explainer_name == "PGExplainer" else num_replicas   # PG determinista tras entrenar
        for r in range(reps):
            torch.manual_seed(42 + r * 17)                # semilla del EXPLICADOR (estabilidad), NO del modelo
            try:
                if explainer_name == "GNNExplainer":
                    ex = create_explainer(model, "GNNExplainer", epochs=ex_epochs, lr=0.01)
                    exp = ex(sub_x, sub_ei_d, index=tl)
                    if exp.edge_mask is not None: edge_masks.append(exp.edge_mask.detach().cpu().numpy())
                    nm = getattr(exp, "node_mask", None)
                    if nm is not None:
                        rankings.append(np.asarray(extract_feature_ranking(explanation=exp)))
                        fm = nm.detach().cpu().numpy(); feat_masks.append(np.abs(fm).mean(0) if fm.ndim == 2 else np.abs(fm))
                    del ex, exp
                elif explainer_name == "PGExplainer":
                    if not pg_ok: break
                    exp = pg(sub_x, sub_ei_d, index=tl, target=d.y[int(nid)].to("cpu"))
                    if exp.edge_mask is not None: edge_masks.append(exp.edge_mask.detach().cpu().numpy())
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
        row = dict(graph_seed=gseed, architecture=meta[0], scenario=meta[1], balancing=meta[2],
                   model_seed=mseed, explainer=explainer_name, node=int(nid), node_typ=node_typ,
                   spearman=None, plaus_edge=None, plaus_feat=None, shap_conc=None,
                   fid_plus=None, fid_minus=None, sub_n_nodes=int(subset.numel()))
        if len(rankings) >= 2:
            s = pairwise_spearman([np.asarray(x) for x in rankings])["mean"]
            if s == s: row["spearman"] = float(s); agg["sp"].append(float(s))
        mean_edge = np.mean(np.stack(edge_masks), axis=0) if edge_masks else None
        mean_feat = np.mean(np.stack(feat_masks), axis=0) if feat_masks else None
        if mean_edge is not None:
            agg["emstd"].append(float(np.std(mean_edge)))
            pl = plausibility.edge_plausibility(mean_edge, sub_ei, sub_edge_typ, node_typ)
            if pl: row["plaus_edge"] = pl["f1"]; agg["ple"].append(pl["f1"])
        if mean_feat is not None and feat_index and node_typ in feat_index:
            fpd = plausibility.feature_plausibility(mean_feat, [feat_index[node_typ]], top_k=3)
            if fpd: row["plaus_feat"] = fpd["recall"]; agg["plf"].append(fpd["recall"])
        if conc: row["shap_conc"] = float(np.mean(conc)); agg["conc"].append(float(np.mean(conc)))
        # ---- PIEZA 2: fidelidad manual sobre la máscara media (edges para GNN/PG, features para GNNShap) ----
        fp = fm = None
        if explainer_name in ("GNNExplainer", "PGExplainer") and mean_edge is not None:
            fp, fm = fidelity_edges(model, sub_x, sub_ei_d, tl, mean_edge, fid_frac)
        elif explainer_name == "GNNShap" and mean_feat is not None:
            fp, fm = fidelity_features(model, sub_x, sub_ei_d, tl, mean_feat, fid_frac)
        if fp is not None:
            row["fid_plus"] = fp; row["fid_minus"] = fm; agg["fp"].append(fp); agg["fm"].append(fm)
        agg["subn"].append(int(subset.numel()))
        per_node_rows.append(row)
    if explainer_name == "PGExplainer": model.to(DEVICE)
    def m(x): return round(float(st.mean(x)), 3) if x else float("nan")
    degenerate = (explainer_name == "PGExplainer" and (not pg_ok or (agg["emstd"] and np.mean(agg["emstd"]) < 1e-3)))
    return dict(graph_seed=gseed, architecture=meta[0], scenario=meta[1], balancing=meta[2],
                model_seed=mseed, explainer=explainer_name, val_pr_auc=meta[3], n_tp=len(nodes),
                spearman_mean=m(agg["sp"]), plaus_edge_mean=m(agg["ple"]), plaus_feat_mean=m(agg["plf"]),
                shap_conc_mean=m(agg["conc"]), fid_plus_mean=m(agg["fp"]), fid_minus_mean=m(agg["fm"]),
                subgraph_n_nodes=int(st.median(agg["subn"])) if agg["subn"] else 0,
                pg_degenerate=("yes" if degenerate else ("no" if explainer_name == "PGExplainer" else "")))


def flush(out, cell_rows, node_rows):
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CELL_FIELDS); w.writeheader(); w.writerows(cell_rows)
    pn = out.replace(".csv", "_pernode.csv")
    with open(pn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=NODE_FIELDS); w.writeheader(); w.writerows(node_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["phase1/synthetic_aml_v2.pt"],
                    help="uno o varios grafos .pt (PIEZA 3).")
    ap.add_argument("--graph-ids", nargs="+", type=int, default=None,
                    help="etiqueta graph_seed por cada --data (default: índice 0..G-1). Permite "
                         "correr grafos extra (g1,g2) en otra invocación sin colisionar con g0.")
    ap.add_argument("--archs", nargs="+", default=["GraphSAGE", "GAT", "GCN", "TAGCN"])
    ap.add_argument("--scenarios", nargs="+", default=["1:1", "1:10", "1:50", "1:100", "natural"])
    ap.add_argument("--balancings", nargs="+", default=["none", "class_weighting", "focal_loss"])
    ap.add_argument("--explainers", nargs="+", default=["GNNExplainer", "PGExplainer", "GNNShap"])
    ap.add_argument("--model-seeds", nargs="+", type=int, default=[42, 43, 44])   # PIEZA 1
    ap.add_argument("--nodes", type=int, default=30)
    ap.add_argument("--replicas", type=int, default=5)
    ap.add_argument("--ex-epochs", type=int, default=100)
    ap.add_argument("--shap-samples", type=int, default=40)
    ap.add_argument("--fid-frac", type=float, default=0.25)   # top-k = 25% de edges/features
    ap.add_argument("--out", default="phase1/results_robust.csv")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    cell_rows, per_node_rows, done = [], [], set()
    if a.resume and os.path.exists(a.out):
        cell_rows = list(csv.DictReader(open(a.out)))
        pn = a.out.replace(".csv", "_pernode.csv")
        if os.path.exists(pn): per_node_rows = list(csv.DictReader(open(pn)))
        by_key = {}
        for r in cell_rows:
            k = (r["graph_seed"], r["architecture"], r["scenario"], r["balancing"], r["model_seed"])
            by_key.setdefault(k, set()).add(r["explainer"])
        done = {k for k, exs in by_key.items() if set(a.explainers).issubset(exs)}
        print(f"resume: {len(done)} super-celdas ya hechas, {len(cell_rows)} filas de celda cargadas")

    total = len(a.data) * len(a.archs) * len(a.scenarios) * len(a.balancings) * len(a.model_seeds)
    print(f"device={DEVICE} | {len(a.data)}graph × {len(a.archs)}arch × {len(a.scenarios)}scen × "
          f"{len(a.balancings)}bal × {len(a.model_seeds)}mseed = {total} super-celdas × {len(a.explainers)}expl "
          f"| nodes={a.nodes} replicas={a.replicas} fid_frac={a.fid_frac}")
    gids = a.graph_ids if a.graph_ids is not None else list(range(len(a.data)))
    assert len(gids) == len(a.data), "--graph-ids debe tener un valor por cada --data"
    done_n = 0
    for gseed, gpath in zip(gids, a.data):
        d0 = load_scaled(gpath)
        feat_index = getattr(d0, "typology_feature_index", None)
        print(f"[graph_seed={gseed}] {gpath}  N={d0.num_nodes} E={d0.edge_index.size(1)}")
        for arch in a.archs:
            for scen in a.scenarios:
                d = create_imbalance_scenario(d0, RATIO[scen], seed=42) if RATIO[scen] is not None else d0
                for bal in a.balancings:
                    for mseed in a.model_seeds:
                        done_n += 1
                        key = (str(gseed), arch, scen, bal, str(mseed))
                        if key in done:
                            print(f"  [{done_n}/{total}] {arch:10}{scen:8}{bal:16}ms={mseed} SKIP (resume)")
                            continue
                        model, valm = train_model(d, arch, bal, seed=mseed)
                        base_k = 2
                        nodes = tp_val_nodes(model, d, a.nodes)
                        meta = (arch, scen, bal, round(valm["pr_auc"], 3))
                        if not nodes:
                            for ex in a.explainers:
                                cell_rows.append(dict(graph_seed=gseed, architecture=arch, scenario=scen,
                                    balancing=bal, model_seed=mseed, explainer=ex, val_pr_auc=meta[3], n_tp=0,
                                    spearman_mean=float("nan"), plaus_edge_mean=float("nan"),
                                    plaus_feat_mean=float("nan"), shap_conc_mean=float("nan"),
                                    fid_plus_mean=float("nan"), fid_minus_mean=float("nan"),
                                    subgraph_n_nodes=0, pg_degenerate=""))
                            print(f"  [{done_n}/{total}] {arch:10}{scen:8}{bal:16}ms={mseed} SKIP (0 TP)")
                            flush(a.out, cell_rows, per_node_rows); continue
                        for ex in a.explainers:
                            c = explain_cell(model, d, nodes, ex, base_k, a.replicas, a.ex_epochs,
                                             a.shap_samples, feat_index, a.fid_frac, per_node_rows, meta,
                                             (gseed, mseed))
                            cell_rows.append(c)
                            print(f"  [{done_n}/{total}] g{gseed} {arch:10}{scen:8}{bal:14}ms={mseed} {ex:13} "
                                  f"spear={c['spearman_mean']} plE={c['plaus_edge_mean']} plF={c['plaus_feat_mean']} "
                                  f"fid+={c['fid_plus_mean']} fid-={c['fid_minus_mean']} n_tp={c['n_tp']} "
                                  f"{('PGdeg='+c['pg_degenerate']) if ex=='PGExplainer' else ''}")
                        flush(a.out, cell_rows, per_node_rows)
    print(f"\nescrito: {a.out} ({len(cell_rows)} celdas) + pernode ({len(per_node_rows)} nodos)")


if __name__ == "__main__":
    main()
