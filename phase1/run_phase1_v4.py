"""Fase 1 v4 — eje sintético sin atajos, con líneas base y campo receptivo correcto.

NO reemplaza a run_phase1_robust.py (sus CSV sostienen el Capítulo 5 y no se re-corren).
Escribe en results_phase1_v4/ y registra progreso con RunLog (stage "synthetic", ./runs_v4).

Cambios respecto de run_phase1_robust.py (cada uno corrige un problema verificado):
  1. Grafo v4 (synthetic_aml_generator.V4_DEFAULTS): sin atajos de grado ni de firma.
     --legacy-shortcuts reproduce el generador viejo.
  2. PGExplainer: se entrenaba con train_indices[:50] (todos fondo lícito, ids < 8000) y modo
     phenomenon con target = etiqueta. Ahora: muestra ESTRATIFICADA de train (≥ la mitad
     ilícitos, --pg-train-nodes) y modo «model»: el target es la clase PREDICHA por el modelo,
     en entrenamiento y en inferencia (PyG 2.7 solo admite explanation_type="phenomenon" para
     PGExplainer; phenomenon con target = predicción es exactamente explicar el modelo).
     Se entrena sobre el subgrafo receptivo de cada nodo (igual que se explica).
  3. Fidelidad: fid+ Y fid− reportadas (antes fid− se calculaba y se omitía del análisis), y
     LÍNEA BASE ALEATORIA con la misma cantidad de aristas/variables elegidas al azar para fid±
     y para la plausibilidad (--random-reps sorteos). Columnas *_rand y *_gain = real − azar.
  4. Campo receptivo por arquitectura: TAGCN agrega K saltos por capa → num_layers × K = 6
     (antes se explicaba con base_k=2); GCN/GraphSAGE/GAT = num_layers = 2.
  5. Ground truth por INSTANCIA de patrón (pattern_instance_edge): con 6 saltos el subgrafo
     contiene otras estructuras de la misma tipología; contarlas como acierto inflaría la
     plausibilidad de TAGCN. --gt typology reproduce la definición vieja.
  6. «Verdaderos positivos» con el umbral max-F1 de validación (no argmax 0,5, que en v4 deja
     celdas sin positivos predichos); el mismo umbral define el target de PGExplainer.
  7. Escenarios con niveles REALES: el 1:10 legado colapsaba al nativo (≈1:5 legado, ≈1:6,2 v4)
     porque src/data/imbalance solo submuestreaba lícitos para r ≥ 0,1 y no hay suficientes.
     v4_common.make_scenario recorta la clase que sobra (así 1:10 sí se alcanza), pero 1:10
     queda a 1,6× del nativo; default v4 = 1:1, natural (≈1:6,2), 1:50, 1:100: 4 niveles
     separados ≥ 5×. Cualquier escenario a < 2× del nativo se descarta con aviso
     (--allow-collapsed lo fuerza).
Plausibilidad de features: recall@3 de la feature-firma (azar exacto = 3/F).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics as st
import sys
import time
import warnings

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "phase1"))
warnings.filterwarnings("ignore")
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer  # noqa: E402
from torch_geometric.utils import k_hop_subgraph  # noqa: E402

import plausibility  # noqa: E402
import v4_common as C  # noqa: E402
from src.monitoring.progress import RunLog  # noqa: E402

DEFAULT_SCENARIOS = ["1:1", "natural", "1:50", "1:100"]
EXPLAINERS = ["GNNExplainer", "PGExplainer", "ShapleyFeatures"]  # "ShapleyFeatures" = Shapley por permutaciones sobre variables (antes etiquetado "GNNShap")
MODEL_CFG = dict(mode="multiclass_classification", task_level="node", return_type="raw")

CELL_FIELDS = ["graph_seed", "generator", "architecture", "scenario", "train_ratio", "balancing",
               "model_seed", "explainer", "receptive_hops", "aligned", "val_pr_auc", "test_pr_auc",
               "n_tp", "status",
               "spearman_mean", "plaus_edge_mean", "plaus_edge_rand", "plaus_edge_gain",
               "plaus_feat_mean", "plaus_feat_rand", "plaus_feat_gain",
               "fid_plus_mean", "fid_plus_rand", "fid_plus_gain",
               "fid_minus_mean", "fid_minus_rand", "fid_minus_gain",
               "subgraph_n_nodes", "subgraph_n_edges", "pg_degenerate", "pg_train_illicit_frac",
               "seconds"]
NODE_FIELDS = ["graph_seed", "architecture", "scenario", "balancing", "model_seed", "explainer",
               "node", "node_typ", "spearman", "plaus_edge", "plaus_edge_rand", "plaus_feat",
               "fid_plus", "fid_plus_rand", "fid_minus", "fid_minus_rand", "sub_n_nodes", "sub_n_edges"]


# ----------------------------------------------------------------------------------------
# Fidelidad (misma definición que run_phase1_robust.py) + línea base aleatoria
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def _p(model, x, ei, node):
    return torch.softmax(model(x, ei), dim=1)[node]


@torch.no_grad()
def fid_edges(model, x, ei, node, order, k):
    """fid+ = p0 − p(sin las k aristas de ``order``); fid− = p0 − p(solo esas k)."""
    E = ei.size(1)
    top = torch.as_tensor(np.asarray(order[:k]).copy(), dtype=torch.long)
    keep = torch.ones(E, dtype=torch.bool); keep[top] = False
    p0 = _p(model, x, ei, node); c = int(p0.argmax())
    return float(p0[c] - _p(model, x, ei[:, keep], node)[c]), float(p0[c] - _p(model, x, ei[:, ~keep], node)[c])


@torch.no_grad()
def fid_feats(model, x, ei, node, order, k):
    top = torch.as_tensor(np.asarray(order[:k]).copy(), dtype=torch.long)
    p0 = _p(model, x, ei, node); c = int(p0.argmax())
    xr = x.clone(); xr[:, top] = 0.0
    xk = torch.zeros_like(x); xk[:, top] = x[:, top]
    return float(p0[c] - _p(model, xr, ei, node)[c]), float(p0[c] - _p(model, xk, ei, node)[c])


# ----------------------------------------------------------------------------------------
# PGExplainer en modo model con muestra estratificada
# ----------------------------------------------------------------------------------------
def stratified_train_nodes(d, n, seed):
    """≥ la mitad ilícitos (todos los disponibles hasta n/2); si hay menos, se recortan los
    lícitos para que la proporción ilícita siga siendo ≥ 0,5."""
    rng = np.random.RandomState(seed)
    ill = torch.where(d.train_mask & (d.y == 1))[0].numpy()
    lic = torch.where(d.train_mask & (d.y == 0))[0].numpy()
    n_ill = min(len(ill), int(math.ceil(n / 2)))
    n_lic = min(len(lic), n - n_ill, n_ill)
    pick = np.concatenate([rng.choice(ill, n_ill, replace=False), rng.choice(lic, n_lic, replace=False)])
    rng.shuffle(pick)
    return pick.tolist(), (n_ill / max(1, len(pick)))


def train_pg(model, d, pred, hops, n_nodes, epochs, seed):
    torch.manual_seed(seed)
    algo = PGExplainer(epochs=epochs, lr=0.003, edge_size=0.005, edge_ent=1.0, temp=[1.0, 1.0])
    pg = Explainer(model=model, algorithm=algo, explanation_type="phenomenon",
                   edge_mask_type="object", model_config=MODEL_CFG)
    nodes, ill_frac = stratified_train_nodes(d, n_nodes, seed)
    subs = []
    for nid in nodes:
        subset, sei, mapping, _ = k_hop_subgraph(int(nid), hops, d.edge_index, relabel_nodes=True,
                                                  num_nodes=d.num_nodes)
        subs.append((d.x[subset], sei, int(mapping[0]), pred[subset]))
    opt = algo.optimizer
    orig_step = opt.step

    def step_clip(*a, **k):
        torch.nn.utils.clip_grad_norm_(algo.parameters(), max_norm=1.0)
        return orig_step(*a, **k)
    opt.step = step_clip
    nan_epochs = 0
    for ep in range(epochs):
        try:
            snap = {k: v.clone() for k, v in algo.state_dict().items()}
        except Exception:
            snap = None
        bad = False
        for sx, sei, tl, tgt in subs:
            loss = algo.train(ep, model, sx, sei, target=tgt, index=tl)
            loss = float(loss) if torch.is_tensor(loss) else loss
            if loss != loss:
                bad = True; break
        if bad:
            nan_epochs += 1
            if snap is not None:
                algo.load_state_dict(snap)
    ok = not any(torch.isnan(p).any() for p in algo.parameters()) and nan_epochs < epochs
    return pg, ok, ill_frac


# ----------------------------------------------------------------------------------------
# Una celda (modelo × explicador)
# ----------------------------------------------------------------------------------------
def gt_edges(d, ebool, nid, gt):
    if gt == "instance" and hasattr(d, "pattern_instance_edge"):
        return (d.pattern_instance_edge[ebool] == int(d.pattern_instance_node[nid])).numpy().astype(int), 1
    return d.typology_edge[ebool].numpy(), int(d.typology_node[nid])


def explain_cell(model, d, nodes, ex_name, hops, a, pred, rng, feat_index, node_rows, key):
    t0 = time.time()
    pg, pg_ok, ill_frac = None, True, float("nan")
    if ex_name == "PGExplainer":
        pg, pg_ok, ill_frac = train_pg(model, d, pred, hops, a.pg_train_nodes, a.pg_epochs, key["model_seed"])
    F = d.x.size(1)
    agg = {k: [] for k in ["sp", "ple", "pler", "plf", "fp", "fpr", "fm", "fmr", "subn", "sube", "emstd"]}
    for nid in nodes:
        subset, sei, mapping, ebool = k_hop_subgraph(int(nid), hops, d.edge_index, relabel_nodes=True,
                                                      num_nodes=d.num_nodes)
        tl = int(mapping[0]); sx = d.x[subset]
        etyp, target_id = gt_edges(d, ebool, nid, a.gt)
        node_typ = int(d.typology_node[nid])
        rankings, emasks, fmasks = [], [], []
        reps = 1 if ex_name == "PGExplainer" else a.replicas
        for r in range(reps):
            torch.manual_seed(42 + r * 17)                 # semilla del EXPLICADOR (estabilidad)
            if ex_name == "GNNExplainer":
                ex = Explainer(model=model, algorithm=GNNExplainer(epochs=a.ex_epochs, lr=0.01),
                               explanation_type="model", node_mask_type="attributes",
                               edge_mask_type="object", model_config=MODEL_CFG)
                e = ex(sx, sei, index=tl)
                emasks.append(e.edge_mask.detach().numpy())
                fm = np.abs(e.node_mask.detach().numpy()).mean(0)
                fmasks.append(fm); rankings.append(np.argsort(-fm))
            elif ex_name == "PGExplainer":
                if not pg_ok:
                    break
                e = pg(sx, sei, index=tl, target=pred[subset])
                emasks.append(e.edge_mask.detach().numpy())
            else:
                from src.explainability.shap_runner import explain_node_shap
                res = explain_node_shap(model, d, int(nid), num_samples=a.shap_samples, device="cpu",
                                        seed=42 + r * 17)
                sv = np.abs(np.asarray(res["shap_values"]))
                fmasks.append(sv); rankings.append(np.argsort(-sv))
        row = dict(key, explainer=ex_name, node=int(nid), node_typ=node_typ, spearman=None,
                   plaus_edge=None, plaus_edge_rand=None, plaus_feat=None, fid_plus=None,
                   fid_plus_rand=None, fid_minus=None, fid_minus_rand=None,
                   sub_n_nodes=int(subset.numel()), sub_n_edges=int(sei.size(1)))
        row.pop("generator", None); row.pop("train_ratio", None)
        if len(rankings) >= 2:
            from src.stability.metrics import pairwise_spearman
            s = pairwise_spearman(rankings)["mean"]
            if s == s:
                row["spearman"] = float(s); agg["sp"].append(float(s))
        E = sei.size(1)
        if emasks and E > 0:
            me = np.mean(np.stack(emasks), 0); agg["emstd"].append(float(np.std(me)))
            pl = plausibility.edge_plausibility(me, sei, etyp, target_id)
            if pl:
                rnd = [plausibility.edge_plausibility(rng.random(E), sei, etyp, target_id)["f1"]
                       for _ in range(a.random_reps)]
                row["plaus_edge"], row["plaus_edge_rand"] = pl["f1"], float(np.mean(rnd))
                agg["ple"].append(pl["f1"]); agg["pler"].append(row["plaus_edge_rand"])
            k = max(1, int(round(a.fid_frac * E)))
            fp, fm = fid_edges(model, sx, sei, tl, np.argsort(-me), k)
            rr = [fid_edges(model, sx, sei, tl, rng.permutation(E), k) for _ in range(a.random_reps)]
            row.update(fid_plus=fp, fid_minus=fm, fid_plus_rand=float(np.mean([x[0] for x in rr])),
                       fid_minus_rand=float(np.mean([x[1] for x in rr])))
        elif fmasks and ex_name == "ShapleyFeatures":
            mf = np.mean(np.stack(fmasks), 0)
            k = max(1, int(round(a.fid_frac * F)))
            fp, fm = fid_feats(model, sx, sei, tl, np.argsort(-mf), k)
            rr = [fid_feats(model, sx, sei, tl, rng.permutation(F), k) for _ in range(a.random_reps)]
            row.update(fid_plus=fp, fid_minus=fm, fid_plus_rand=float(np.mean([x[0] for x in rr])),
                       fid_minus_rand=float(np.mean([x[1] for x in rr])))
        if fmasks and feat_index and node_typ in feat_index:
            mf = np.mean(np.stack(fmasks), 0)
            fpd = plausibility.feature_plausibility(mf, [feat_index[node_typ]], top_k=3)
            row["plaus_feat"] = fpd["recall"]; agg["plf"].append(fpd["recall"])
        if row["fid_plus"] is not None:
            agg["fp"].append(row["fid_plus"]); agg["fpr"].append(row["fid_plus_rand"])
            agg["fm"].append(row["fid_minus"]); agg["fmr"].append(row["fid_minus_rand"])
        agg["subn"].append(row["sub_n_nodes"]); agg["sube"].append(row["sub_n_edges"])
        node_rows.append(row)

    def m(x):
        return round(float(st.mean(x)), 4) if x else float("nan")

    def gain(x, y):
        return round(m(x) - m(y), 4) if x and y else float("nan")
    deg = ex_name == "PGExplainer" and (not pg_ok or (agg["emstd"] and np.mean(agg["emstd"]) < 1e-3))
    plf_rand = round(3 / F, 4) if ex_name != "PGExplainer" else float("nan")
    status = "ok"
    if ex_name == "PGExplainer" and not pg_ok:
        status = "pg_failed"
    elif not agg["subn"]:
        status = "no_aplica"
    return dict(key, explainer=ex_name, receptive_hops=hops, n_tp=len(nodes), status=status,
                spearman_mean=m(agg["sp"]),
                plaus_edge_mean=m(agg["ple"]), plaus_edge_rand=m(agg["pler"]), plaus_edge_gain=gain(agg["ple"], agg["pler"]),
                plaus_feat_mean=m(agg["plf"]), plaus_feat_rand=plf_rand,
                plaus_feat_gain=round(m(agg["plf"]) - plf_rand, 4) if agg["plf"] and plf_rand == plf_rand else float("nan"),
                fid_plus_mean=m(agg["fp"]), fid_plus_rand=m(agg["fpr"]), fid_plus_gain=gain(agg["fp"], agg["fpr"]),
                fid_minus_mean=m(agg["fm"]), fid_minus_rand=m(agg["fmr"]), fid_minus_gain=gain(agg["fm"], agg["fmr"]),
                subgraph_n_nodes=int(st.median(agg["subn"])) if agg["subn"] else 0,
                subgraph_n_edges=int(st.median(agg["sube"])) if agg["sube"] else 0,
                pg_degenerate=("yes" if deg else "no") if ex_name == "PGExplainer" else "",
                pg_train_illicit_frac=round(ill_frac, 3) if ill_frac == ill_frac else "",
                seconds=round(time.time() - t0, 1))


def flush(out, cells, nodes):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for path, fields, rows in ((out, CELL_FIELDS, cells), (out.replace(".csv", "_pernode.csv"), NODE_FIELDS, nodes)):
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
        os.replace(tmp, path)


def load_alignment(path):
    import json
    if not path or not os.path.exists(path):
        return {}
    res = json.load(open(path)).get("results", {})
    return {k: v.get("aligned_strict", v.get("aligned")) for k, v in res.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", default=None, help=".pt ya generados (si no, --graph-seeds)")
    ap.add_argument("--graph-seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--legacy-shortcuts", action="store_true", help="generador legado (con atajos)")
    ap.add_argument("--signature-shift", type=float, default=None)
    ap.add_argument("--archs", nargs="+", default=C.ARCHS)
    ap.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    ap.add_argument("--allow-collapsed", action="store_true")
    ap.add_argument("--balancings", nargs="+", default=["none", "class_weighting", "focal_loss"])
    ap.add_argument("--explainers", nargs="+", default=EXPLAINERS)
    ap.add_argument("--model-seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--nodes", type=int, default=30)
    ap.add_argument("--replicas", type=int, default=5)
    ap.add_argument("--ex-epochs", type=int, default=100)
    ap.add_argument("--pg-epochs", type=int, default=30)
    ap.add_argument("--pg-train-nodes", type=int, default=60)
    ap.add_argument("--shap-samples", type=int, default=40)
    ap.add_argument("--fid-frac", type=float, default=0.25)
    ap.add_argument("--random-reps", type=int, default=10)
    ap.add_argument("--gt", choices=["instance", "typology"], default="instance")
    ap.add_argument("--alignment-json", default="results_phase1_v4/alignment.json")
    ap.add_argument("--out", default="results_phase1_v4/results_v4.csv")
    ap.add_argument("--log-dir", default="./runs_v4")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    log = RunLog(a.log_dir, "synthetic")
    cells, node_rows, done = [], [], set()
    if a.resume and os.path.exists(a.out):
        cells = list(csv.DictReader(open(a.out)))
        pn = a.out.replace(".csv", "_pernode.csv")
        if os.path.exists(pn):
            node_rows = list(csv.DictReader(open(pn)))
        by = {}
        for r in cells:
            by.setdefault((r["graph_seed"], r["architecture"], r["scenario"], r["balancing"], r["model_seed"]),
                          set()).add(r["explainer"])
        done = {k for k, v in by.items() if set(a.explainers) <= v}
    aligned = load_alignment(a.alignment_json)

    graphs = ([(i, C.load_graph(p)) for i, p in enumerate(a.data)] if a.data else
              [(s, C.make_graph(seed=s, legacy=a.legacy_shortcuts, signature_shift=a.signature_shift))
               for s in a.graph_seeds])
    scen = list(a.scenarios)
    collapsed = C.collapsing_scenarios(graphs[0][1], scen)
    if collapsed and not a.allow_collapsed:
        print(f"AVISO: descarto escenarios que colapsan al nativo (1:{1/C.native_ratio(graphs[0][1]):.1f}): {collapsed}")
        scen = [s for s in scen if s not in collapsed]
    units = [(g, ar, sc, b, ms) for g, _ in graphs for ar in a.archs for sc in scen
             for b in a.balancings for ms in a.model_seeds]
    log.plan(len(units), items=["|".join(map(str, u)) for u in units])
    print(f"{len(units)} super-celdas × {len(a.explainers)} explicadores | escenarios={scen} | gt={a.gt}")
    gdict = dict(graphs)
    for i, (gs, arch, sc, bal, ms) in enumerate(units, 1):
        run_id = f"syn_g{gs}_{arch}_{sc}_{bal}_s{ms}"
        if (str(gs), arch, sc, bal, str(ms)) in done:
            log.skip(run_id, "resume"); continue
        d0 = gdict[gs]
        d = C.make_scenario(d0, sc, seed=42)
        gen = "legacy" if not getattr(d0, "generator_params", {}).get("shortcut_free", False) else "v4"
        tr_ratio = f"1:{(d.y[d.train_mask] == 0).sum().item() / max(1, (d.y[d.train_mask] == 1).sum().item()):.1f}"
        with log.step(run_id) as rec:
            model, mm = C.train_model(d, arch, bal, epochs=a.epochs, patience=a.patience, seed=ms)
            thr = C.val_threshold(model, d)           # umbral max-F1 en val (ver v4_common)
            pred = C.predict(model, d, thr)
            hops = C.receptive_hops(arch)
            nodes = C.tp_val_nodes(model, d, a.nodes, threshold=thr)
            key = dict(graph_seed=gs, generator=gen, architecture=arch, scenario=sc, train_ratio=tr_ratio,
                       balancing=bal, model_seed=ms, aligned=aligned.get(arch, ""),
                       val_pr_auc=mm["val_pr_auc"], test_pr_auc=mm["test_pr_auc"])
            rng = np.random.default_rng(ms)
            metrics = {"val_pr_auc": mm["val_pr_auc"], "test_pr_auc": mm["test_pr_auc"], "n_tp": len(nodes),
                       "threshold": round(thr, 4)}
            for ex in a.explainers:
                if not nodes:
                    c = dict(key, explainer=ex, receptive_hops=hops, n_tp=0, status="no_aplica")
                else:
                    c = explain_cell(model, d, nodes, ex, hops, a, pred, rng, getattr(d0, "typology_feature_index", None),
                                     node_rows, key)
                cells.append(c)
                metrics[f"{ex}_status"] = c["status"]
                for mname in ("plaus_edge_gain", "fid_plus_gain", "spearman_mean"):
                    if mname in c:
                        metrics[f"{ex}_{mname}"] = c[mname]
                print(f"[{i}/{len(units)}] {run_id} {ex:12} {c['status']} spear={c.get('spearman_mean')} "
                      f"plE={c.get('plaus_edge_mean')}(azar {c.get('plaus_edge_rand')}) "
                      f"plF={c.get('plaus_feat_mean')} fid+={c.get('fid_plus_mean')}(azar {c.get('fid_plus_rand')}) "
                      f"fid-={c.get('fid_minus_mean')}(azar {c.get('fid_minus_rand')}) {c.get('seconds', '')}s")
            rec["metrics"] = metrics
            flush(a.out, cells, node_rows)
    flush(a.out, cells, node_rows)
    print(f"escrito {a.out} ({len(cells)} celdas)")


if __name__ == "__main__":
    main()
