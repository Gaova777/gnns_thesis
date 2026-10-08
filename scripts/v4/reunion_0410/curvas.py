"""
Curvas ROC y PR de la corrida v4 (modo de etiqueta C) para la reunion con Cristian del 04-oct.

Pedido del director: PR-AUC junto con ROC-AUC, F1 junto con KS, y las CURVAS completas con la
incertidumbre de las 3 semillas (media + banda), marcando el umbral calibrado del meta.json.

Dos etapas:

  --stage scores   (GPU si hay memoria, si no CPU) reconstruye los 180 modelos de
                   results_models_v4/, hace UNA pasada hacia adelante por modelo sobre el grafo
                   del escenario (create_v4_scenario con la semilla de datos fija del config) y
                   guarda las probabilidades de la clase ilicita de validacion y test en
                   scores/{run_id}.npz (+ scores/_labels.npz con las etiquetas). Sin entrenar.
  --stage metrics  (CPU) desde los npz: metricas por modelo y agregadas por configuracion.
  --stage figures  (CPU) desde los npz: figuras por escenario, figura resumen y anexo 1:1.
  --stage all      las tres (por defecto).

La reconstruccion copia la de scripts/explain_matrix.py (_build_model) y la de
scripts/train_matrix.py (load_elliptic -> apply_label_mode -> preprocess -> create_v4_scenario).
El PR-AUC principal es el del pipeline (trapecio sobre la curva PR, src/training/trainer.py);
se guarda ademas la average precision. Se verifica contra val_metrics.pr_auc y
test_metrics.pr_auc del meta.json (diferencia absoluta en el CSV).

KS = max_t (TPR(t) - FPR(t)), que es el estadistico de Kolmogorov-Smirnov de dos muestras entre
las distribuciones de score de positivos y negativos.

Uso:
  ~/.local/bin/uv run --frozen python scripts/v4/reunion_0410/curvas.py
  ~/.local/bin/uv run --frozen python scripts/v4/reunion_0410/curvas.py --stage figures
Salidas en results_v4/reunion_0410/curvas/.
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

OUT = REPO / "results_v4" / "reunion_0410" / "curvas"
SCORES = OUT / "scores"
FIGS = OUT / "figuras"
MODELS = REPO / "results_models_v4"
CONFIG = REPO / "configs" / "experiment_v4.yaml"

SCENARIOS = ["native", "1:10", "1:10_os", "1:20", "1:1"]
MAIN_SCENARIOS = ["native", "1:10", "1:10_os", "1:20"]  # 1:20 reemplaza al 1:1 (7-oct)
ARCHS = ["GCN", "GraphSAGE", "GAT", "TAGCN"]
LOSSES = ["none", "class_weighting", "focal_loss"]
SEEDS = [42, 43, 44]
SPLITS = ["val", "test"]

SCEN_LABEL = {"native": "nativo (1:38,4)", "1:10": "1:10 (submuestreo)",
              "1:10_os": "1:10 con SMOTE", "1:20": "1:20 (submuestreo)",
              "1:1": "1:1 (submuestreo)"}
SPLIT_LABEL = {"val": "validación", "test": "test"}
LOSS_LABEL = {"none": "sin balanceo", "class_weighting": "pesos por clase",
              "focal_loss": "focal loss"}
# Paleta categorica de referencia (skill dataviz, slots 1-4) + estilo de linea como
# codificacion secundaria.
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
LOSS_COLOR = dict(zip(LOSSES, PAL))
LOSS_LS = {"none": "-", "class_weighting": "--", "focal_loss": "-."}
ARCH_COLOR = dict(zip(ARCHS, PAL))
ARCH_LS = {"GCN": "-", "GraphSAGE": "--", "GAT": "-.", "TAGCN": ":"}
# --deck: versión para láminas (sin título interno, que va en el pie de la lámina, y letra
# más grande). Se guarda en figuras/deck/.
DECK = False

GRID = np.linspace(0.0, 1.0, 501)


def run_id(scen, arch, loss, seed):
    rid = f"{scen}_{arch}_{loss}".replace(":", "-")
    return rid if seed == 42 else f"{rid}_s{seed}"


# ─────────────────────────────────────────────────────────────────────────────
# Etapa 1: scores
# ─────────────────────────────────────────────────────────────────────────────

def stage_scores(device_arg: str, force: bool):
    import torch
    import torch.nn.functional as F
    import yaml
    from src.data.imbalance import create_v4_scenario
    from src.data.loader import apply_label_mode, load_elliptic
    from src.data.preprocessing import preprocess
    from src.training.trainer import build_model

    SCORES.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    d = cfg["data"]
    sub_seed = int(cfg.get("scenarios", {}).get("subsample_seed", 2026))

    device = device_arg
    if device == "auto":
        device = "cpu"
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            device = "cuda" if free > 2.5 * 2**30 else "cpu"
            print(f"GPU libre: {free / 2**30:.2f} GiB")
    print(f"Dispositivo: {device}")

    data = load_elliptic(root=str(REPO / d.get("root", "./data")))
    mode = apply_label_mode(data, d.get("label_mode", "licit_unknown"))
    preprocess(data, train_range=tuple(d["train_timesteps"]),
               val_range=tuple(d["val_timesteps"]), test_range=tuple(d["test_timesteps"]))
    print(f"Modo de etiqueta: {mode}")
    n_real = data.num_nodes
    val_idx = torch.where(data.val_mask)[0].numpy()
    test_idx = torch.where(data.test_mask)[0].numpy()
    y = data.y.cpu().numpy()
    np.savez_compressed(SCORES / "_labels.npz", val_idx=val_idx, test_idx=test_idx,
                        y_val=y[val_idx].astype(np.int8), y_test=y[test_idx].astype(np.int8),
                        label_mode=mode)
    print(f"val: {len(val_idx)} nodos ({int(y[val_idx].sum())} ilicitos) | "
          f"test: {len(test_idx)} nodos ({int(y[test_idx].sum())} ilicitos)")

    missing = []
    for scen in SCENARIOS:
        sdata, _ = create_v4_scenario(data, scen, subsample_seed=sub_seed, verbose=False)
        # Los nodos reales conservan su indice; SMOTE solo agrega al final.
        assert torch.equal(sdata.val_mask[:n_real], data.val_mask)
        assert torch.equal(sdata.test_mask[:n_real], data.test_mask)
        assert not sdata.val_mask[n_real:].any() and not sdata.test_mask[n_real:].any()
        for arch in ARCHS:
            for loss in LOSSES:
                for seed in SEEDS:
                    rid = run_id(scen, arch, loss, seed)
                    out_p = SCORES / f"{rid}.npz"
                    if out_p.exists() and not force:
                        continue
                    mp = MODELS / f"{rid}_meta.json"
                    if not mp.exists():
                        missing.append(rid)
                        print(f"  FALTA meta: {rid}")
                        continue
                    meta = json.loads(mp.read_text(encoding="utf-8"))
                    ck = MODELS / meta.get("checkpoint", f"{rid}_best.pt")
                    if not ck.exists():
                        missing.append(rid)
                        print(f"  FALTA checkpoint: {ck.name}")
                        continue
                    bp = meta.get("best_params", {}) or {}
                    kw = {}
                    if arch == "GAT" and "heads" in bp:
                        kw["heads"] = bp["heads"]
                    if arch == "TAGCN" and "K" in bp:
                        kw["K"] = bp["K"]
                    model = build_model(arch, in_channels=sdata.num_node_features,
                                        hidden_channels=bp.get("hidden_dim", 128),
                                        num_layers=bp.get("num_layers", 2),
                                        dropout=bp.get("dropout", 0.3), **kw)
                    model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
                    t0 = time.time()
                    dev = device
                    for attempt in (0, 1):
                        try:
                            model = model.to(dev).eval()
                            with torch.no_grad():
                                logits = model(sdata.x.to(dev), sdata.edge_index.to(dev))
                                p = F.softmax(logits, dim=-1)[:, 1].float().cpu().numpy()
                            break
                        except RuntimeError as exc:  # OOM -> CPU
                            if attempt == 0 and "out of memory" in str(exc).lower():
                                print(f"  OOM en {rid}; reintento en CPU")
                                torch.cuda.empty_cache()
                                dev = "cpu"
                                continue
                            raise
                    np.savez_compressed(
                        out_p, p_val=p[val_idx].astype(np.float32),
                        p_test=p[test_idx].astype(np.float32),
                        threshold=float(meta["calibrated_threshold"]),
                        meta_val_pr_auc=float(meta["val_metrics"]["pr_auc"]),
                        meta_test_pr_auc=float(meta["test_metrics"]["pr_auc"]),
                        meta_val_f1=float(meta["val_metrics_calibrated"]["f1"]),
                        meta_test_f1=float(meta["test_metrics"]["f1"]),
                        quality_passed=bool(meta.get("quality_passed", False)),
                        device=dev)
                    print(f"  {rid:42s} {dev} {time.time() - t0:5.1f}s")
                    del model, logits
                    if dev == "cuda":
                        torch.cuda.empty_cache()
        del sdata
    (OUT / "faltantes.txt").write_text("\n".join(missing) + ("\n" if missing else ""),
                                       encoding="utf-8")
    print(f"Modelos faltantes: {len(missing)}")


# ─────────────────────────────────────────────────────────────────────────────
# Metricas y curvas
# ─────────────────────────────────────────────────────────────────────────────

def load_labels():
    z = np.load(SCORES / "_labels.npz")
    return {"val": z["y_val"].astype(int), "test": z["y_test"].astype(int)}


def load_scores(rid):
    p = SCORES / f"{rid}.npz"
    return np.load(p) if p.exists() else None


def metrics_one(y, p, thr):
    from sklearn.metrics import (auc, average_precision_score, f1_score, matthews_corrcoef,
                                 precision_recall_curve, precision_score, recall_score,
                                 roc_auc_score, roc_curve)
    from scipy.stats import ks_2samp
    prec, rec, _ = precision_recall_curve(y, p, pos_label=1)
    fpr, tpr, thr_roc = roc_curve(y, p)
    j = int(np.argmax(tpr - fpr))
    ks_p = ks_2samp(p[y == 1], p[y == 0]).pvalue
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return {
        "roc_auc": roc_auc_score(y, p), "pr_auc": auc(rec, prec),
        "avg_precision": average_precision_score(y, p),
        "ks": float(tpr[j] - fpr[j]), "ks_threshold": float(min(thr_roc[j], 1.0)),
        "ks_pvalue": float(ks_p),
        "threshold": thr, "f1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred), "fpr_at_thr": fp / max(int((y == 0).sum()), 1),
        "n_flagged": int(pred.sum()), "tp": tp, "fp": fp,
        "prevalence": float(y.mean()), "n_pos": int(y.sum()), "n": int(len(y)),
    }


def curves(y, p):
    """ROC (TPR en la rejilla de FPR) y PR (precision en la rejilla de recall), sin suavizar.

    PR: para cada recall r de la rejilla se toma el punto de mayor umbral con recall >= r
    (sin la envolvente "precision interpolada", que esconderia los picos)."""
    from sklearn.metrics import precision_recall_curve, roc_curve
    fpr, tpr, _ = roc_curve(y, p)
    roc = np.interp(GRID, fpr, tpr)
    prec, rec, _ = precision_recall_curve(y, p, pos_label=1)
    # Se descarta el punto final (recall 0, precision 1) que sklearn agrega por convencion:
    # no corresponde a ningun umbral y dibujaria un pico artificial en recall 0.
    rec_a, prec_a = rec[:-1][::-1], prec[:-1][::-1]  # recall ascendente, umbral descendente
    idx = np.clip(np.searchsorted(rec_a, GRID, side="left"), 0, len(rec_a) - 1)
    pr = prec_a[idx]
    return roc, pr


def point_at(y, p, thr):
    pred = p >= thr
    tp = (pred & (y == 1)).sum()
    fp = (pred & (y == 0)).sum()
    tpr = tp / max((y == 1).sum(), 1)
    fpr = fp / max((y == 0).sum(), 1)
    prec = tp / pred.sum() if pred.sum() else np.nan
    return fpr, tpr, tpr, prec   # (fpr, tpr) en ROC; (recall, precision) en PR


def all_curves(labels):
    """dict[(scen, arch, loss, split)] -> {"roc": (3,G), "pr": (3,G), "pts": [...], "seeds"}"""
    out = {}
    for scen in SCENARIOS:
        for arch in ARCHS:
            for loss in LOSSES:
                for split in SPLITS:
                    rocs, prs, pts, seeds = [], [], [], []
                    for seed in SEEDS:
                        z = load_scores(run_id(scen, arch, loss, seed))
                        if z is None:
                            continue
                        y, p = labels[split], z[f"p_{split}"].astype(np.float64)
                        r, pr = curves(y, p)
                        rocs.append(r); prs.append(pr); seeds.append(seed)
                        pts.append(point_at(y, p, float(z["threshold"])))
                    if rocs:
                        out[(scen, arch, loss, split)] = {
                            "roc": np.array(rocs), "pr": np.array(prs),
                            "pts": np.array(pts, dtype=float), "seeds": seeds}
    return out


def roughness(pr_curve):
    """Variacion total de la curva PR que no se explica por su cambio neto: 0 = monotona."""
    v = pr_curve[(GRID >= 0.02) & (GRID <= 0.98)]
    return float(np.abs(np.diff(v)).sum() - abs(v[-1] - v[0]))


def stage_metrics():
    labels = load_labels()
    rows = []
    for scen in SCENARIOS:
        for arch in ARCHS:
            for loss in LOSSES:
                for seed in SEEDS:
                    rid = run_id(scen, arch, loss, seed)
                    z = load_scores(rid)
                    if z is None:
                        print(f"  sin scores: {rid}")
                        continue
                    for split in SPLITS:
                        y, p = labels[split], z[f"p_{split}"].astype(np.float64)
                        m = metrics_one(y, p, float(z["threshold"]))
                        _, pr = curves(y, p)
                        m["pr_roughness"] = roughness(pr)
                        m["pr_auc_meta"] = float(z[f"meta_{split}_pr_auc"])
                        m["pr_auc_absdiff_meta"] = abs(m["pr_auc"] - m["pr_auc_meta"])
                        m["f1_meta"] = float(z[f"meta_{split}_f1"])
                        rows.append({"run_id": rid, "scenario": scen, "arch": arch,
                                     "balancing": loss, "seed": seed, "split": split,
                                     "quality_passed": bool(z["quality_passed"]), **m})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "curvas_metricas_por_modelo.csv", index=False)
    keys = ["scenario", "arch", "balancing", "split"]
    cols = ["roc_auc", "pr_auc", "avg_precision", "ks", "f1", "precision", "recall", "mcc",
            "threshold", "fpr_at_thr", "pr_roughness", "prevalence"]
    g = df.groupby(keys, sort=False)
    agg = g[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_seeds"] = g.size()
    agg["n_gate_passed"] = g["quality_passed"].sum()
    agg = agg.reset_index()
    agg.to_csv(OUT / "curvas_metricas_por_config.csv", index=False)
    mx = df["pr_auc_absdiff_meta"].max()
    mf = (df["f1"] - df["f1_meta"]).abs().max()
    print(f"{len(df) // 2} modelos | max |PR-AUC - meta| = {mx:.2e} | max |F1 - meta| = {mf:.2e}")

    # Dominancia y cruces entre perdidas (curvas medias), por escenario x arquitectura x split.
    C = all_curves(labels)
    dom = []
    sel = (GRID >= 0.02) & (GRID <= 0.98)
    for scen in SCENARIOS:
        for arch in ARCHS:
            for split in SPLITS:
                for kind in ("roc", "pr"):
                    means = {l: C[(scen, arch, l, split)][kind].mean(0)[sel]
                             for l in LOSSES if (scen, arch, l, split) in C}
                    if len(means) < 2:
                        continue
                    # Curva superior en cada punto de la rejilla
                    M = np.vstack([means[l] for l in means])
                    top = np.array(list(means))[M.argmax(0)]
                    share = {l: float((top == l).mean()) for l in means}
                    for a_i, a in enumerate(LOSSES):
                        for b in LOSSES[a_i + 1:]:
                            if a not in means or b not in means:
                                continue
                            diff = means[a] - means[b]
                            # Tolerancia: 0,005 absoluto en ROC; 5 % relativo en PR (en test
                            # las precisiones rondan 0,006 y un umbral absoluto lo ocultaria todo).
                            tol = (np.full_like(diff, 0.005) if kind == "roc" else
                                   0.05 * np.maximum(means[a], means[b]))
                            s = np.sign(diff[np.abs(diff) > tol])
                            crosses = int((np.diff(s) != 0).sum()) if len(s) > 1 else 0
                            dom.append({"scenario": scen, "arch": arch, "split": split,
                                        "curve": kind, "loss_a": a, "loss_b": b,
                                        "frac_a_above": float((diff > 0).mean()),
                                        "n_crossings": crosses,
                                        "dominance": (a if (diff >= -tol).all() else
                                                      b if (diff <= tol).all() else "cruce")})
                    dom.append({"scenario": scen, "arch": arch, "split": split, "curve": kind,
                                "loss_a": "TOP", "loss_b": "",
                                "frac_a_above": np.nan, "n_crossings": np.nan,
                                "dominance": ";".join(f"{l}={share[l]:.2f}" for l in share)})
    pd.DataFrame(dom).to_csv(OUT / "curvas_dominancia.csv", index=False)
    return df, agg


# ─────────────────────────────────────────────────────────────────────────────
# Figuras
# ─────────────────────────────────────────────────────────────────────────────

def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = (15, 17, 15, 14) if DECK else (11, 12, 11, 10)
    plt.rcParams.update({
        "font.size": fs[0], "axes.titlesize": fs[1], "axes.labelsize": fs[2],
        "legend.fontsize": fs[3],
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
        "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
        "axes.formatter.use_locale": False, "figure.facecolor": "white",
    })
    return plt


def _comma(ax):
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: (f"{v:g}").replace(".", ","))
    ax.xaxis.set_major_formatter(f)
    ax.yaxis.set_major_formatter(f)


def _draw(ax, kind, arr, pts, color, ls, label, logy=False):
    m, lo, hi = arr.mean(0), arr.min(0), arr.max(0)
    if logy:
        floor = 1e-4
        m, lo, hi = np.maximum(m, floor), np.maximum(lo, floor), np.maximum(hi, floor)
    ax.fill_between(GRID, lo, hi, color=color, alpha=0.18, lw=0, zorder=2)
    ax.plot(GRID, m, color=color, ls=ls, lw=2, label=label, zorder=3)
    if kind == "roc":
        xs, ys = pts[:, 0], pts[:, 1]
    else:
        xs, ys = pts[:, 2], pts[:, 3]
    ok = np.isfinite(ys)
    if logy:
        ys = np.maximum(ys, 1e-4)
    ax.scatter(xs[ok], ys[ok], s=46, color=color, edgecolor="white", linewidth=1.2,
               zorder=5, marker="o")


def _axes_setup(ax, kind, split, prevalence, logy):
    if kind == "roc":
        ax.plot([0, 1], [0, 1], color="0.55", ls=":", lw=1.2, zorder=1)
        ax.set_xlabel("Tasa de falsos positivos (FPR)")
        ax.set_ylabel("Tasa de verdaderos positivos (TPR)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    else:
        ax.axhline(prevalence, color="0.55", ls=":", lw=1.2, zorder=1)
        ax.set_xlabel("Exhaustividad (recall)")
        ax.set_ylabel("Precisión" + (" (escala log)" if logy else ""))
        ax.set_xlim(0, 1)
        if logy:
            ax.set_yscale("log"); ax.set_ylim(1e-3, 1.05)
        else:
            ax.set_ylim(0, 1.01)
    ax.grid(ls=":", color="0.85", zorder=0)
    if not (kind == "pr" and logy):
        _comma(ax)
    else:
        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))


def _save(fig, name):
    out = FIGS / "deck" if DECK else FIGS
    out.mkdir(parents=True, exist_ok=True)
    fmts = (("png", {"dpi": 200}),) if DECK else (("png", {"dpi": 200}), ("pdf", {}))
    for ext, kw in fmts:
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", **kw)


def _legend_handles(plt, items):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Line2D([0], [0], color=c, ls=ls, lw=2, label=l) for l, c, ls in items]
    h.append(Patch(color="0.5", alpha=0.25, label="banda mín.-máx. de 3 semillas"))
    h.append(Line2D([0], [0], marker="o", color="0.4", ls="", markeredgecolor="white",
                    markersize=8, label="umbral calibrado (una marca por semilla)"))
    h.append(Line2D([0], [0], color="0.55", ls=":", lw=1.2, label="azar (diagonal / prevalencia)"))
    return h


def stage_figures():
    plt = _style()
    labels = load_labels()
    prev = {s: float(labels[s].mean()) for s in SPLITS}
    C = all_curves(labels)
    agg = pd.read_csv(OUT / "curvas_metricas_por_config.csv")

    def auc_txt(scen, arch, loss, split, kind):
        r = agg[(agg.scenario == scen) & (agg.arch == arch) & (agg.balancing == loss)
                & (agg.split == split)]
        if r.empty:
            return ""
        col = "roc_auc" if kind == "roc" else "pr_auc"
        return f"{r[col + '_mean'].iloc[0]:.3f}".replace(".", ",")

    # a) por escenario: 2 filas (ROC, PR) x 4 columnas (arquitecturas), color = perdida
    for scen in SCENARIOS:
        for split in SPLITS:
            logy = split == "test"
            fig, axes = plt.subplots(2, 4, figsize=(18, 9.2))
            for j, arch in enumerate(ARCHS):
                for i, kind in enumerate(("roc", "pr")):
                    ax = axes[i, j]
                    txt = []
                    for loss in LOSSES:
                        c = C.get((scen, arch, loss, split))
                        if c is None:
                            continue
                        _draw(ax, kind, c[kind], c["pts"], LOSS_COLOR[loss], LOSS_LS[loss],
                              LOSS_LABEL[loss], logy=(kind == "pr" and logy))
                        txt.append((auc_txt(scen, arch, loss, split, kind), LOSS_COLOR[loss]))
                    _axes_setup(ax, kind, split, prev[split], kind == "pr" and logy)
                    if i == 0:
                        ax.set_title(arch, fontweight="bold")
                    if j > 0:
                        ax.set_ylabel("")
                    # AUC medias por perdida, en texto neutro con marca de color
                    lab = "ROC-AUC" if kind == "roc" else "PR-AUC"
                    y0 = 0.05 if kind == "roc" else 0.95
                    va = "bottom" if kind == "roc" else "top"
                    for k, (t, col) in enumerate(txt):
                        yy = (y0 + (len(txt) - 1 - k) * 0.075) if kind == "roc" \
                            else y0 - k * 0.075
                        ax.text(0.97, yy, f"{lab} {t}", transform=ax.transAxes, ha="right",
                                va=va, fontsize=13 if DECK else 9.5, color="#0b0b0b",
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, lw=1.5))
            fig.legend(handles=_legend_handles(
                plt, [(LOSS_LABEL[l], LOSS_COLOR[l], LOSS_LS[l]) for l in LOSSES]),
                loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.035))
            extra = " (anexo)" if scen == "1:1" else ""
            if not DECK:
                fig.suptitle(f"Curvas ROC y PR en {SPLIT_LABEL[split]} · escenario "
                             f"{SCEN_LABEL[scen]}{extra} · media de 3 semillas · prevalencia "
                             f"{prev[split] * 100:.2f} %".replace(".", ","), fontsize=13.5)
            fig.tight_layout(rect=(0, 0.03 if DECK else 0.02, 1, 1 if DECK else 0.97))
            pref = "anexo_" if scen == "1:1" else ""
            _save(fig, f"{pref}curvas_{scen.replace(':', '-')}_{split}")
            plt.close(fig)

    # b) resumen: mejor perdida de cada arquitectura en native (elegida por PR-AUC de VALIDACION)
    nat = agg[(agg.scenario == "native") & (agg.split == "val")]
    best = {a: nat[nat.arch == a].sort_values("pr_auc_mean", ascending=False).balancing.iloc[0]
            for a in ARCHS}
    (OUT / "mejor_perdida_native.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11))
    for j, split in enumerate(SPLITS):
        for i, kind in enumerate(("roc", "pr")):
            ax = axes[i, j]
            logy = kind == "pr" and split == "test"
            for arch in ARCHS:
                c = C[("native", arch, best[arch], split)]
                t = auc_txt("native", arch, best[arch], split, kind)
                _draw(ax, kind, c[kind], c["pts"], ARCH_COLOR[arch], ARCH_LS[arch],
                      f"{arch} · {LOSS_LABEL[best[arch]]} ({'ROC' if kind == 'roc' else 'PR'}"
                      f"-AUC {t})", logy=logy)
            _axes_setup(ax, kind, split, prev[split], logy)
            ax.set_title(f"{'ROC' if kind == 'roc' else 'PR'} · {SPLIT_LABEL[split]} "
                         f"(prevalencia {prev[split] * 100:.2f} %)".replace(".", ","))
            ax.legend(loc="lower right" if kind == "roc" else "upper right",
                      fontsize=12 if DECK else 9.5,
                      frameon=True, framealpha=0.9)
    if not DECK:
        fig.suptitle("Escenario nativo: mejor pérdida de cada arquitectura (elegida por PR-AUC "
                     "de validación)\nmedia de 3 semillas, banda mín.-máx., puntos = umbral "
                     "calibrado de cada semilla", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 1 if DECK else 0.95))
    _save(fig, "resumen_native_mejor_por_arquitectura")
    plt.close(fig)
    print(f"Figuras en {FIGS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "scores", "metrics", "figures"])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--force", action="store_true", help="recalcular scores ya guardados")
    ap.add_argument("--deck", action="store_true",
                    help="figuras para láminas: sin título interno y letra grande (figuras/deck/)")
    a = ap.parse_args()
    global DECK
    DECK = a.deck
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage in ("all", "scores"):
        stage_scores(a.device, a.force)
    if a.stage in ("all", "metrics"):
        stage_metrics()
    if a.stage in ("all", "figures"):
        stage_figures()


if __name__ == "__main__":
    main()
