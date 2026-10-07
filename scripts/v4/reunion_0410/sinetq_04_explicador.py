#!/usr/bin/env python
"""Paso 6: explicador sencillo de la marca de B.

(a) Surrogates interpretables que predicen la marca del mejor B sobre los sin etiqueta de
    validación con features crudas + variables de grafo: árbol de decisión (profundidad 3 y 4)
    y regresión logística L1. Fidelidad medida fuera de muestra en los sin etiqueta de test.
    El mismo surrogate se ajusta a la marca de B sobre las etiquetadas de validación (lo que B
    usa para marcar ilícitas) para comparar qué features usa en cada caso.
(b) Atribuciones gradiente x entrada (baseline = 0, que en el espacio RobustScaler es la
    mediana de train) del logit «ilícita menos lícita» respecto de las features, para una
    muestra de sin etiqueta marcados, ilícitas TP, lícitas FP y sin etiqueta no marcados de
    validación. Se separa la atribución sobre la fila del propio nodo y sobre los vecinos.
    Se repite para el mejor B de cada arquitectura.

Uso: ~/.local/bin/uv run --frozen python scripts/v4/reunion_0410/sinetq_04_explicador.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402

from sinetq_common import (BEST_B, FEATURE_NAMES, FIG, OUT, build_from_meta,  # noqa: E402
                           ensure_dirs, list_metas, load_data, load_preds, pick_device)

GRAPH_COLS = ["in_deg", "out_deg", "deg_undirected", "comp_size", "n_vec_ilicitos",
              "n_vec_licitos", "n_vec_sinetq", "n_ilicitas_2saltos", "tam_vecindario_2saltos",
              "clustering"]
N_SAMPLE = 300
RNG = np.random.default_rng(2026)


def topk_jaccard(a: np.ndarray, b: np.ndarray, k: int = 15) -> float:
    sa, sb = set(np.argsort(-a)[:k]), set(np.argsort(-b)[:k])
    return len(sa & sb) / len(sa | sb)


def surrogates(xr, gf, nodos, flag, res):
    split, grp = nodos["split"].values, nodos["group"].values
    X = np.hstack([xr, gf[GRAPH_COLS].values])
    names = FEATURE_NAMES + GRAPH_COLS
    tr_m = (split == "val") & (grp == "sin_etiqueta")
    te_m = (split == "test") & (grp == "sin_etiqueta")
    lab_m = (split == "val") & (grp != "sin_etiqueta")
    out = {}
    for depth in (3, 4):
        t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=100, random_state=0,
                                   class_weight="balanced").fit(X[tr_m], flag[tr_m])
        p_tr, p_te = t.predict_proba(X[tr_m])[:, 1], t.predict_proba(X[te_m])[:, 1]
        out[f"arbol_d{depth}"] = {
            "auc_val": float(roc_auc_score(flag[tr_m], p_tr)),
            "auc_test": float(roc_auc_score(flag[te_m], p_te)),
            "bacc_test": float(balanced_accuracy_score(flag[te_m], p_te >= .5)),
            "reglas": export_text(t, feature_names=names, decimals=3),
            "importancias_top": sorted(
                [(names[j], float(v)) for j, v in enumerate(t.feature_importances_) if v > 0],
                key=lambda z: -z[1]),
        }
    # árbol para la marca de B sobre las etiquetadas (lícita/ilícita) de validación
    t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=0,
                               class_weight="balanced").fit(X[lab_m], flag[lab_m])
    out["arbol_d3_etiquetadas"] = {
        "auc_val": float(roc_auc_score(flag[lab_m], t.predict_proba(X[lab_m])[:, 1])),
        "reglas": export_text(t, feature_names=names, decimals=3),
        "importancias_top": sorted(
            [(names[j], float(v)) for j, v in enumerate(t.feature_importances_) if v > 0],
            key=lambda z: -z[1])}
    # logística L1 (features estandarizadas con los sin etiqueta de val)
    sc = StandardScaler().fit(X[tr_m])
    coefs = {}
    for name, m in (("sin_etiqueta", tr_m), ("etiquetadas", lab_m)):
        lr = LogisticRegression(penalty="l1", C=0.05, solver="liblinear",
                                class_weight="balanced", max_iter=2000)
        Z = np.clip(sc.transform(X[m]), -10, 10)
        lr.fit(Z, flag[m])
        coefs[name] = lr.coef_[0]
        auc_in = roc_auc_score(flag[m], lr.decision_function(Z))
        r = {"auc_in": float(auc_in), "n_coef_no_nulos": int((lr.coef_[0] != 0).sum()),
             "top15": [(names[j], float(lr.coef_[0][j]))
                       for j in np.argsort(-np.abs(lr.coef_[0]))[:15]]}
        if name == "sin_etiqueta":
            Zt = np.clip(sc.transform(X[te_m]), -10, 10)
            r["auc_test"] = float(roc_auc_score(flag[te_m], lr.decision_function(Zt)))
        out[f"logistica_l1_{name}"] = r
    a, b = np.abs(coefs["sin_etiqueta"]), np.abs(coefs["etiquetadas"])
    out["logistica_comparacion"] = {
        "spearman_abs_coef": float(spearmanr(a, b).statistic),
        "jaccard_top15": topk_jaccard(a, b),
        "corr_signo_coef": float(np.corrcoef(coefs["sin_etiqueta"], coefs["etiquetadas"])[0, 1])}
    res["surrogates"] = out


def grad_x_input(model, x, ei, nodes, device):
    """Para cada nodo: atribución gradiente x entrada sobre su propia fila (165) y suma
    sobre las filas de los demás nodos (165), del logit z1 - z0."""
    self_a = np.zeros((len(nodes), x.shape[1]), dtype=np.float32)
    neigh_a = np.zeros((len(nodes), x.shape[1]), dtype=np.float32)
    xg = x.clone().to(device).requires_grad_(True)
    for k, i in enumerate(nodes):
        if xg.grad is not None:
            xg.grad = None
        out = model(xg, ei)
        (out[i, 1] - out[i, 0]).backward()
        a = (xg.grad * xg).detach()
        row = a[i].cpu().numpy()
        tot = a.abs().sum(0).cpu().numpy()
        self_a[k] = row
        neigh_a[k] = tot - np.abs(row)
    return self_a, neigh_a


def main() -> None:
    ensure_dirs()
    data, xr = load_data()
    nodos, modelos, probs, flags = load_preds()
    gf = pd.read_csv(OUT / "features_grafo.csv.gz")
    keys = list(modelos["key"])
    res: dict = {"modelo": BEST_B}
    surrogates(xr, gf, nodos, flags[keys.index(BEST_B)], res)

    # Mejor B de cada arquitectura por PR-AUC de validación
    mb = modelos[modelos["mode"] == "B"].sort_values("val_pr_auc", ascending=False)
    best_per_arch = mb.groupby("arch").head(1)["key"].tolist()
    metas = {f"{m['_mode']}|{m['architecture']}|{m['balancing']}|{m['seed']}": m
             for m in list_metas("B")}
    device = pick_device()
    x = data.x
    ei = data.edge_index.to(device)
    split, grp = nodos["split"].values, nodos["group"].values
    val = split == "val"
    attr_rows = []
    res["atribuciones"] = {}
    for key in best_per_arch:
        f = flags[keys.index(key)]
        sets = {
            "sin_etiqueta_marcado": np.where(val & (grp == "sin_etiqueta") & f)[0],
            "ilicita_TP": np.where(val & (grp == "ilicita") & f)[0],
            "licita_FP": np.where(val & (grp == "licita") & f)[0],
            "sin_etiqueta_no_marcado": np.where(val & (grp == "sin_etiqueta") & ~f)[0],
        }
        model = build_from_meta(metas[key], data.num_node_features, device)
        for p in model.parameters():
            p.requires_grad_(False)
        prof = {}
        info = {}
        for name, idx in sets.items():
            idx = RNG.choice(idx, min(len(idx), N_SAMPLE), replace=False)
            sa, na = grad_x_input(model, x, ei, idx, device)
            prof[name] = {"abs_self": np.abs(sa).mean(0), "signed_self": sa.mean(0),
                          "neigh": na.mean(0)}
            share = np.abs(sa).sum(1) / (np.abs(sa).sum(1) + na.sum(1) + 1e-12)
            info[name] = {"n": int(len(idx)), "frac_atrib_propia_mediana": float(
                np.median(share))}
            for j, fn in enumerate(FEATURE_NAMES):
                attr_rows.append({"key": key, "grupo": name, "feature": fn,
                                  "abs_self": float(prof[name]["abs_self"][j]),
                                  "signed_self": float(prof[name]["signed_self"][j]),
                                  "neigh": float(prof[name]["neigh"][j])})
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        comp = {}
        for a, b in (("sin_etiqueta_marcado", "ilicita_TP"), ("sin_etiqueta_marcado", "licita_FP"),
                     ("ilicita_TP", "licita_FP"),
                     ("sin_etiqueta_marcado", "sin_etiqueta_no_marcado")):
            pa, pb = prof[a]["abs_self"], prof[b]["abs_self"]
            comp[f"{a}__vs__{b}"] = {
                "spearman_abs": float(spearmanr(pa, pb).statistic),
                "jaccard_top15": topk_jaccard(pa, pb),
                "corr_signed": float(np.corrcoef(prof[a]["signed_self"],
                                                 prof[b]["signed_self"])[0, 1])}
        top = {g: [FEATURE_NAMES[j] for j in np.argsort(-prof[g]["abs_self"])[:15]]
               for g in prof}
        res["atribuciones"][key] = {"grupos": info, "comparaciones": comp, "top15": top}
        print(key, json.dumps(comp, indent=0))

    pd.DataFrame(attr_rows).to_csv(OUT / "atribuciones_grad_x_input.csv.gz", index=False)

    # Figura: perfil de |grad x input| propio, mejor B, marcados vs ilícitas TP
    a = pd.DataFrame(attr_rows)
    a = a[a.key == BEST_B].pivot(index="feature", columns="grupo", values="abs_self")
    a = a.loc[FEATURE_NAMES]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    for g, c in (("ilicita_TP", "#d1495b"), ("sin_etiqueta_marcado", "#edae49"),
                 ("licita_FP", "#2a7ab9")):
        ax.plot(range(len(a)), a[g] / a[g].sum(), lw=1, c=c, label=g.replace("_", " "))
    ax.axvline(92.5, c="grey", ls=":", lw=1)
    ax.set_xlabel("Feature (0-92 locales, 93-164 agregadas)")
    ax.set_ylabel("|grad x input| propio, normalizado")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.scatter(a["ilicita_TP"] / a["ilicita_TP"].sum(),
               a["sin_etiqueta_marcado"] / a["sin_etiqueta_marcado"].sum(), s=10, c="#edae49")
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k--", lw=.7)
    for fn in a.index:
        if (a.loc[fn, "ilicita_TP"] / a["ilicita_TP"].sum() > .02 or
                a.loc[fn, "sin_etiqueta_marcado"] / a["sin_etiqueta_marcado"].sum() > .02):
            ax.annotate(fn, (a.loc[fn, "ilicita_TP"] / a["ilicita_TP"].sum(),
                             a.loc[fn, "sin_etiqueta_marcado"] /
                             a["sin_etiqueta_marcado"].sum()), fontsize=7)
    c = res["atribuciones"][BEST_B]["comparaciones"]["sin_etiqueta_marcado__vs__ilicita_TP"]
    ax.set_title(f"Spearman = {c['spearman_abs']:.2f}".replace(".", ","), fontsize=9)
    ax.set_xlabel("Ilícitas TP")
    ax.set_ylabel("Sin etiqueta marcados")
    fig.suptitle(f"Atribución gradiente x entrada · {BEST_B} · validación", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "atribuciones_perfil.png", dpi=150)
    plt.close(fig)

    (OUT / "explicador.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    s = res["surrogates"]
    print(s["arbol_d3"]["reglas"])
    print({k: {kk: vv for kk, vv in v.items() if kk != "reglas"} for k, v in s.items()})
    for k, v in res["atribuciones"].items():
        print(k, v["grupos"])


if __name__ == "__main__":
    main()
