"""Diagnóstico de PGExplainer por balanceo (reunión 04-10, pregunta A / H3).

No modifica nada existente: carga checkpoints de results_models_v4/, reusa las funciones del
pipeline v4 (mismos nodos, mismos subgrafos, mismo create_explainer/train_pgexplainer) y
escribe solo en results_v4/reunion_0410/pgexpl_gcn/.

Para cada modelo mide:
  M1  Logits del modelo (la entrada del MLP de PGExplainer, porque PyG usa la salida de la
      última capa MessagePassing, que aquí es la capa de 2 clases): escala, saturación,
      fracción de nodos de entrenamiento de PGExplainer predichos ilícitos.
  M2  Dependencia del modelo respecto a las aristas (oclusión): para los nodos de
      entrenamiento de PGExplainer, cambio del margen logit(1)-logit(0) al quitar TODAS las
      aristas; para los nodos explicados con >= 2 aristas, cambio al quitar cada arista sola
      y la diferencia entre aristas (¿hay un orden «verdadero» que recuperar?).
  M3  Entrenamiento de PGExplainer instrumentado: por época, término de entropía cruzada,
      de tamaño y de entropía de la máscara; y cuánto cambia la entropía cruzada entre
      máscara todo-1 y todo-0 (si es ~0, la pérdida no informa sobre aristas).
  M4  Reproducción de las 5 réplicas: logits pre-sigmoide del MLP por arista, Spearman de
      aristas (misma métrica que el pipeline) y acuerdo del orden con el de la oclusión.

Uso:
  ~/.local/bin/uv run --frozen python scripts/v4/reunion_0410/diag_pgexplainer.py \
      --runs 1-10_GraphSAGE_none 1-10_GraphSAGE_class_weighting ... --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks

from explain_matrix import _build_model, load_settings  # noqa: E402
from src.data.loader import apply_label_mode, load_elliptic  # noqa: E402
from src.data.preprocessing import preprocess  # noqa: E402
from src.explainability.explainer_runner import (  # noqa: E402
    create_explainer, full_graph_logits, train_pgexplainer)
from src.explainability.v4_explain import (  # noqa: E402
    ILLICIT, SubgraphPredictionMismatch, exact_subgraph, num_mp_layers, receptive_hops,
    replica_seed, sample_pg_train_nodes, select_common_explain_nodes)
from src.stability.metrics import spearman_edges  # noqa: E402

OUT = ROOT / "results_v4/reunion_0410/pgexpl_gcn"


def margin(logits):
    return (logits[..., 1] - logits[..., 0])


@torch.no_grad()
def model_logits(model, x, ei, edge_weight_mask=None):
    if edge_weight_mask is None:
        return model(x, ei)
    keep = edge_weight_mask.bool()
    return model(x, ei[:, keep])


def hard_edges(model, ei, index, num_nodes, algo):
    _, hard = algo._get_hard_masks(model, index, ei, num_nodes=num_nodes)
    return hard


def instrumented_train(explainer, model, x, ei, target, train_idx, epochs, log_rows, tag):
    """Same as train_pgexplainer (clipping + rollback) but records loss terms per epoch."""
    algo = explainer.algorithm
    rec = {}
    orig_base = algo._calculate_base_loss
    orig_reg = algo._apply_homo_regularization

    def base(y_hat, y):
        v = orig_base(y_hat, y)
        rec["ce"] = float(v)
        return v

    def reg(loss, edge_mask):
        out = orig_reg(loss, edge_mask)
        m = edge_mask.sigmoid()
        rec["size"] = float(m.sum() * algo.coeffs["edge_size"])
        rec["total"] = float(out)
        rec["mask_mean"] = float(m.mean()) if m.numel() else float("nan")
        rec["n_edges"] = int(m.numel())
        return out

    algo._calculate_base_loss = base
    algo._apply_homo_regularization = reg
    # Reuse the pipeline's training routine (clip + rollback); we only add a per-step hook
    # by wrapping algo.train.
    orig_train = algo.train
    per_epoch = {}

    def train_hook(epoch, *a, **k):
        loss = orig_train(epoch, *a, **k)
        d = per_epoch.setdefault(epoch, {"ce": [], "size": [], "ent": [], "mask_mean": []})
        d["ce"].append(rec.get("ce", np.nan))
        d["size"].append(rec.get("size", np.nan))
        d["ent"].append(rec.get("total", np.nan) - rec.get("ce", np.nan) - rec.get("size", np.nan))
        d["mask_mean"].append(rec.get("mask_mean", np.nan))
        return loss

    algo.train = train_hook
    ok = train_pgexplainer(explainer, None, device=str(x.device), x=x, edge_index=ei,
                           target=target, train_nodes=train_idx, require_clean_epoch=True)
    algo.train = orig_train
    algo._calculate_base_loss = orig_base
    algo._apply_homo_regularization = orig_reg
    for ep, d in sorted(per_epoch.items()):
        log_rows.append({**tag, "epoch": ep, **{k: float(np.nanmean(v)) for k, v in d.items()}})
    return ok


@torch.no_grad()
def ce_mask_sensitivity(model, x, ei, target, train_idx, algo):
    """CE of the model prediction-target with all hard edges kept (mask=1) vs all removed
    (mask=0), per training node. If the gap is ~0 the CE term carries no edge signal."""
    ce1, ce0, dm = [], [], []
    for i in train_idx:
        hard = hard_edges(model, ei, int(i), x.size(0), algo)
        lg1 = model(x, ei)[int(i)]
        lg0 = model(x, ei[:, ~hard])[int(i)]
        t = target[int(i)]
        ce1.append(float(torch.nn.functional.cross_entropy(lg1[None], t[None])))
        ce0.append(float(torch.nn.functional.cross_entropy(lg0[None], t[None])))
        dm.append(float(margin(lg1) - margin(lg0)))
    return np.array(ce1), np.array(ce0), np.array(dm)


def run_one(run_id, data, metas, s, nodes, device, R, epochs_rows, node_rows):
    m = metas[run_id]
    model = _build_model(m, data.num_node_features, s["models_dir"], device)
    full = full_graph_logits(model, data, device="cpu")
    pred = full.argmax(-1)
    base_k = receptive_hops(model, m.get("best_params", {}) or {}, m["architecture"])
    H = max(num_mp_layers(model), int(base_k))
    Rr = 2 * H + 1
    tag = {"run_id": run_id, "arch": m["architecture"], "scenario": m["scenario"],
           "balancing": m["balancing"]}
    summ = dict(tag)

    # ── M1: logits (MLP input) ──────────────────────────────────────────────
    mg = margin(full)
    tm = data.train_mask
    summ["train_frac_pred_illicit"] = float((pred[tm] == 1).float().mean())
    summ["train_margin_std"] = float(mg[tm].std())
    summ["train_abs_margin_median"] = float(mg[tm].abs().median())
    p = torch.softmax(full, -1)[:, 1]
    summ["train_frac_p_sat"] = float(((p[tm] > 0.99) | (p[tm] < 0.01)).float().mean())
    summ["explained_frac_pred_illicit"] = float((pred[nodes] == 1).float().mean())

    # ── PG training contexts per replica (same sampler/seed as the pipeline) ─────────
    rep_masks = {n: [] for n in nodes}
    rep_logits = {n: [] for n in nodes}
    ce_gaps, dms, tr_pred_ill = [], [], []
    ctxs = {}
    for n in nodes:
        try:
            ctxs[n] = exact_subgraph(model, data, [n], Rr, full, device)
        except SubgraphPredictionMismatch:
            ctxs[n] = None
    for r in range(R):
        seed = replica_seed(r)
        torch.manual_seed(seed)
        tr = sample_pg_train_nodes(data, s["pg_train_nodes"], s["pg_train_min_illicit"], seed=seed)
        tctx = exact_subgraph(model, data, tr, Rr, full, device)
        tx, tei = tctx["x"], tctx["edge_index"]
        tidx = [int(v) for v in tctx["local"]]
        ttarget = pred[tctx["subset"]].to(device)
        tr_pred_ill.append(float((pred[tr] == 1).float().mean()))
        explainer = create_explainer(model, "PGExplainer", epochs=s["pg_epochs"],
                                     lr=s["pg_lr"], explanation_type="model")
        # M3 (before training, mask-independent): CE with all edges vs none
        algo = explainer.algorithm
        ce1, ce0, dm = ce_mask_sensitivity(model, tx, tei, ttarget, tidx, algo)
        ce_gaps.append(np.mean(np.abs(ce1 - ce0)))
        dms.append(np.median(np.abs(dm)))
        ok = instrumented_train(explainer, model, tx, tei, ttarget, tidx, s["pg_epochs"],
                                epochs_rows, {**tag, "replica": r})
        if not ok:
            continue
        for n in nodes:
            ctx = ctxs[n]
            if ctx is None:
                continue
            x, ei, tgt = ctx["x"], ctx["edge_index"], int(ctx["local"][0])
            with torch.no_grad():
                emb = algo._get_embeddings(model, x, ei)
                hard = hard_edges(model, ei, tgt, x.size(0), algo)
                inp = algo._get_inputs(emb, ei, tgt)
                lg = algo.mlp(inp).view(-1)
            rep_logits[n].append(lg[hard].cpu().numpy())
            rep_masks[n].append(torch.sigmoid(lg[hard]).cpu().numpy())
        del explainer
    summ["pg_train_frac_pred_illicit"] = float(np.mean(tr_pred_ill))
    summ["ce_gap_all_vs_none_mean"] = float(np.mean(ce_gaps))
    summ["train_abs_dmargin_noedges_median"] = float(np.mean(dms))

    # ── M2 + M4 per explained node ───────────────────────────────────────────
    sp_mask, sp_logit, agree_occ, occ_gap, in_sep = [], [], [], [], []
    for n in nodes:
        ctx = ctxs[n]
        if ctx is None or not rep_masks[n]:
            continue
        x, ei, tgt = ctx["x"], ctx["edge_index"], int(ctx["local"][0])
        algo_tmp = create_explainer(model, "PGExplainer", epochs=1, lr=1e-3,
                                    explanation_type="model").algorithm
        hard = hard_edges(model, ei, tgt, x.size(0), algo_tmp)
        hid = torch.where(hard)[0]
        E = len(hid)
        row = {**tag, "node": int(n), "n_edges": E}
        if E >= 2:
            with torch.no_grad():
                base = float(margin(model(x, ei)[tgt]))
                occ = []
                for e in hid.tolist():
                    keep = torch.ones(ei.size(1), dtype=torch.bool, device=ei.device)
                    keep[e] = False
                    occ.append(base - float(margin(model(x, ei[:, keep])[tgt])))
                emb = algo_tmp._get_embeddings(model, x, ei)
                inp = algo_tmp._get_inputs(emb, ei, tgt)[hard].cpu().numpy()
            occ = np.array(occ)
            M = np.stack(rep_masks[n]); Lg = np.stack(rep_logits[n])
            row["sp_mask"] = spearman_edges(M)["mean"]
            row["sp_mlp_logit"] = spearman_edges(Lg)["mean"]
            row["occ_abs_max"] = float(np.abs(occ).max())
            row["occ_gap"] = float(np.ptp(occ))
            row["mask_within_gap_median"] = float(np.median(np.ptp(M, axis=1)))
            row["logit_within_gap_median"] = float(np.median(np.ptp(Lg, axis=1)))
            row["mask_rep_mean_median"] = float(np.median(M.mean(axis=1)))
            row["input_sep"] = float(np.linalg.norm(inp[:, None] - inp[None], axis=-1).max())
            if np.ptp(occ) > 0:
                rhos = [stats.spearmanr(Lg[k], occ).statistic for k in range(len(Lg))
                        if np.ptp(Lg[k]) > 0]
                row["rho_pg_vs_occlusion"] = float(np.nanmean(rhos)) if rhos else np.nan
            sp_mask.append(row["sp_mask"]); sp_logit.append(row["sp_mlp_logit"])
            occ_gap.append(row["occ_gap"]); in_sep.append(row["input_sep"])
            agree_occ.append(row.get("rho_pg_vs_occlusion", np.nan))
        node_rows.append(row)
    summ["n_nodes_2plus"] = len(sp_mask)
    summ["sp_edges_mask_repro"] = float(np.nanmean(sp_mask)) if sp_mask else np.nan
    summ["sp_edges_mlp_logit"] = float(np.nanmean(sp_logit)) if sp_logit else np.nan
    summ["occ_gap_median"] = float(np.nanmedian(occ_gap)) if occ_gap else np.nan
    summ["input_sep_median"] = float(np.nanmedian(in_sep)) if in_sep else np.nan
    summ["rho_pg_vs_occlusion_mean"] = float(np.nanmean(agree_occ)) if agree_occ else np.nan
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/experiment_v4.yaml"))
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--replicas", type=int, default=5)
    ap.add_argument("--tag", default="diag")
    a = ap.parse_args()
    torch.set_num_threads(4)
    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    s = load_settings(cfg, argparse.Namespace(results_dir=None, models_dir=None,
                                              logs_dir=None, n_nodes=None, replicas=None))
    data = load_elliptic(root=s["data"].get("root", "./data"))
    apply_label_mode(data, "licit_unknown")
    d = s["data"]
    preprocess(data, train_range=tuple(d.get("train_timesteps", (1, 34))),
               val_range=tuple(d.get("val_timesteps", (35, 42))),
               test_range=tuple(d.get("test_timesteps", (43, 49))))
    nodes = select_common_explain_nodes(data, s["n_nodes"], s["nodes_seed"],
                                        ROOT / "results_v4/explain_nodes_v4.json")["nodes"]
    metas = {}
    for p in sorted(s["models_dir"].glob("*_meta.json")):
        mm = json.loads(p.read_text(encoding="utf-8"))
        metas[mm["run_id"]] = mm
    OUT.mkdir(parents=True, exist_ok=True)
    summ_rows, ep_rows, node_rows = [], [], []
    for rid in a.runs:
        t0 = time.time()
        sm = run_one(rid, data, metas, s, nodes, a.device, a.replicas, ep_rows, node_rows)
        sm["seconds"] = round(time.time() - t0, 1)
        summ_rows.append(sm)
        print(json.dumps(sm, default=float), flush=True)
        pd.DataFrame(summ_rows).to_csv(OUT / f"{a.tag}_summary.csv", index=False)
        pd.DataFrame(ep_rows).to_csv(OUT / f"{a.tag}_epochs.csv", index=False)
        pd.DataFrame(node_rows).to_csv(OUT / f"{a.tag}_nodes.csv", index=False)


if __name__ == "__main__":
    main()
