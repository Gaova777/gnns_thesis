#!/usr/bin/env python
"""Pasos 4 y 5: estructura del grafo y tiempo.

Grafo (dirigido, tal como lo usan los modelos): grado de entrada, salida y total; tamaño de la
componente débilmente conexa; vecinos (1 salto, sin dirección) ilícitos, lícitos y sin
etiqueta; ilícitas a 2 saltos (el campo receptivo de un GNN de 2 capas); coeficiente de
clustering del grafo no dirigido. Se comparan los grupos en validación y se mide la tasa de
marcado de los sin etiqueta por bins de grado y según tengan vecinos ilícitos.

Tiempo: tasa de marcado por timestep (1-49) para lícitas, ilícitas y sin etiqueta, con el
mejor B, la banda de los 9 B y el mejor C. Más un indicador de drift por timestep: mediana
del PSI por feature de cada grupo en el timestep t frente a las lícitas de train.

Uso: ~/.local/bin/uv run --frozen python scripts/v4/reunion_0410/sinetq_03_grafo_tiempo.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from sinetq_common import (BEST_B, FIG, GROUP4, GROUP4_COLOR, GROUP4_LABEL, OUT,  # noqa: E402
                           ensure_dirs, group4, load_data, load_preds, psi)

BEST_C = "C|GraphSAGE|class_weighting|42"   # mejor C (semilla 42) por PR-AUC de validación


def graph_features(edge_index: np.ndarray, n: int, grp: np.ndarray) -> pd.DataFrame:
    src, dst = edge_index
    indeg = np.bincount(dst, minlength=n)
    outdeg = np.bincount(src, minlength=n)
    A = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)).tocsr()
    U = ((A + A.T) > 0).astype(np.float32).tocsr()
    U.setdiag(0)
    U.eliminate_zeros()
    deg_u = np.asarray(U.sum(1)).ravel()
    ncomp, lab = connected_components(U, directed=False)
    comp_size = np.bincount(lab)[lab]
    is_il = (grp == "ilicita").astype(np.float32)
    is_li = (grp == "licita").astype(np.float32)
    is_un = (grp == "sin_etiqueta").astype(np.float32)
    n_il = U @ is_il
    n_li = U @ is_li
    n_un = U @ is_un
    U2 = (U @ U)
    U2 = ((U2 + U) > 0).astype(np.float32).tocsr()
    U2.setdiag(0)
    U2.eliminate_zeros()
    n_il2 = U2 @ is_il
    n_li2 = U2 @ is_li
    size2 = np.asarray(U2.sum(1)).ravel()
    G = nx.from_scipy_sparse_array(U)
    clus = nx.clustering(G)
    clus = np.array([clus[i] for i in range(n)])
    # Componentes de la misma componente: proporción de ilícitas etiquetadas
    il_per_comp = np.bincount(lab, weights=is_il)[lab]
    return pd.DataFrame({"in_deg": indeg, "out_deg": outdeg, "deg_total": indeg + outdeg,
                         "deg_undirected": deg_u, "comp_size": comp_size,
                         "comp_n_ilicitas": il_per_comp,
                         "n_vec_ilicitos": n_il, "n_vec_licitos": n_li, "n_vec_sinetq": n_un,
                         "frac_vec_sinetq": np.divide(n_un, deg_u, out=np.full(n, np.nan),
                                                      where=deg_u > 0),
                         "n_ilicitas_2saltos": n_il2, "n_licitas_2saltos": n_li2,
                         "tam_vecindario_2saltos": size2, "clustering": clus})


def main() -> None:
    ensure_dirs()
    data, xr = load_data()
    nodos, modelos, probs, flags = load_preds()
    n = len(nodos)
    grp = nodos["group"].values
    split = nodos["split"].values
    ts = nodos["timestep"].values
    keys = list(modelos["key"])
    kb, kc = keys.index(BEST_B), keys.index(BEST_C)
    g4 = group4(nodos, flags[kb])
    gf = graph_features(data.edge_index.numpy(), n, grp)
    gf.to_csv(OUT / "features_grafo.csv.gz", index=False)
    res: dict = {"modelo": BEST_B}

    # ── comparación por grupo en validación ──────────────────────────────────
    val = split == "val"
    cols = ["in_deg", "out_deg", "deg_total", "comp_size", "n_vec_ilicitos", "n_vec_licitos",
            "n_vec_sinetq", "frac_vec_sinetq", "n_ilicitas_2saltos", "tam_vecindario_2saltos",
            "clustering"]
    rows = []
    for g in GROUP4:
        m = val & (g4 == g)
        r = {"grupo": g, "n": int(m.sum())}
        for c in cols:
            r[f"{c}_media"] = float(np.nanmean(gf.loc[m, c]))
            r[f"{c}_mediana"] = float(np.nanmedian(gf.loc[m, c]))
        r["frac_con_vecino_ilicito"] = float((gf.loc[m, "n_vec_ilicitos"] > 0).mean())
        r["frac_con_ilicita_2saltos"] = float((gf.loc[m, "n_ilicitas_2saltos"] > 0).mean())
        r["frac_sin_vecinos_etiquetados"] = float(
            ((gf.loc[m, "n_vec_ilicitos"] + gf.loc[m, "n_vec_licitos"]) == 0).mean())
        r["frac_clustering_pos"] = float((gf.loc[m, "clustering"] > 0).mean())
        rows.append(r)
    comp_tab = pd.DataFrame(rows)
    comp_tab.to_csv(OUT / "grafo_por_grupo_val.csv", index=False)
    res["grafo_por_grupo_val"] = comp_tab.round(4).to_dict(orient="records")

    # Tasa de marcado de sin etiqueta (val) según grado y vecinos ilícitos
    unk = val & (grp == "sin_etiqueta")
    fl = flags[kb][unk]
    d = gf.loc[unk].copy()
    d["marcado"] = fl
    bins = [0, 1, 2, 3, 5, 10, 1e9]
    d["bin_grado"] = pd.cut(d["deg_undirected"], bins=bins, right=False,
                            labels=["0", "1", "2", "3-4", "5-9", ">=10"])
    t_deg = d.groupby("bin_grado", observed=True)["marcado"].agg(["size", "mean"])
    t_il1 = d.groupby(d["n_vec_ilicitos"] > 0)["marcado"].agg(["size", "mean"])
    t_il2 = d.groupby(d["n_ilicitas_2saltos"] > 0)["marcado"].agg(["size", "mean"])
    t_lab = d.groupby((d["n_vec_ilicitos"] + d["n_vec_licitos"]) == 0)["marcado"].agg(
        ["size", "mean"])
    res["marcado_por_grado"] = t_deg.reset_index().astype({"bin_grado": str}).values.tolist()
    res["marcado_por_vecino_ilicito_1salto"] = t_il1.reset_index().values.tolist()
    res["marcado_por_ilicita_2saltos"] = t_il2.reset_index().values.tolist()
    res["marcado_sin_vecinos_etiquetados"] = t_lab.reset_index().values.tolist()
    res["frac_marcados_con_ilicita_2saltos"] = float((d.loc[d.marcado,
                                                            "n_ilicitas_2saltos"] > 0).mean())
    # AUC de cada variable estructural para predecir la marca (dirección libre)
    aucs = {}
    for c in cols:
        v = d[c].fillna(-1).values
        a = roc_auc_score(d["marcado"], v)
        aucs[c] = float(max(a, 1 - a))
    res["auc_estructura_predice_marca"] = aucs

    # Lo mismo para los 9 B: tasa con y sin ilícita a 2 saltos, y AUC del grado
    rows = []
    for i, k in enumerate(keys):
        if not k.startswith("B|"):
            continue
        f = flags[i][unk]
        has2 = d["n_ilicitas_2saltos"].values > 0
        a = roc_auc_score(f, d["deg_undirected"].values)
        rows.append({"key": k, "tasa_marcado": float(f.mean()),
                     "tasa_con_ilicita_2saltos": float(f[has2].mean()),
                     "tasa_sin_ilicita_2saltos": float(f[~has2].mean()),
                     "frac_marcados_con_ilicita_2saltos": float(has2[f].mean()),
                     "auc_grado": float(a)})
    pd.DataFrame(rows).to_csv(OUT / "grafo_todos_B.csv", index=False)
    res["frac_sinetq_val_con_ilicita_2saltos"] = float((d["n_ilicitas_2saltos"] > 0).mean())

    # Composición del grafo: aristas entre grupos (todo el grafo)
    src, dst = data.edge_index.numpy()
    et = pd.crosstab(pd.Series(grp[src], name="origen"), pd.Series(grp[dst], name="destino"))
    et.to_csv(OUT / "aristas_entre_grupos.csv")
    res["aristas_entre_grupos"] = et.to_dict()

    # Figura estructura
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax = axes[0]
    for g in GROUP4:
        m = val & (g4 == g)
        v = gf.loc[m, "deg_undirected"].values
        xs = np.arange(0, 16)
        ax.plot(xs, [(v == x).mean() if x < 15 else (v >= 15).mean() for x in xs], "-o", ms=3,
                c=GROUP4_COLOR[g], label=GROUP4_LABEL[g])
    ax.set_xlabel("Grado no dirigido (15 = 15 o más)")
    ax.set_ylabel("Fracción de nodos")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.bar(range(len(t_deg)), t_deg["mean"], color="#edae49")
    ax.set_xticks(range(len(t_deg)))
    ax.set_xticklabels([f"{b}\n(n={s})" for b, s in zip(t_deg.index, t_deg["size"])],
                       fontsize=8)
    ax.set_xlabel("Grado no dirigido del sin etiqueta")
    ax.set_ylabel("Tasa de marcado de B")
    ax = axes[2]
    labels = ["sin ilícita\na 2 saltos", "con ilícita\na 2 saltos"]
    ax.bar([0, 1], t_il2["mean"].values, color=["#8d99ae", "#d1495b"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{lab}\n(n={s})" for lab, s in zip(labels, t_il2["size"])],
                       fontsize=8)
    ax.set_ylabel("Tasa de marcado de B")
    fig.suptitle(f"Validación · {BEST_B}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "estructura_grafo.png", dpi=150)
    plt.close(fig)

    # ── tiempo ─────────────────────────────────────────────────────────────────
    rows = []
    bkeys = [i for i, k in enumerate(keys) if k.startswith("B|")]
    for t in range(1, 50):
        mt = ts == t
        for g in ("licita", "ilicita", "sin_etiqueta"):
            m = mt & (grp == g)
            r = {"timestep": t, "grupo": g, "n": int(m.sum()),
                 "tasa_bestB": float(flags[kb][m].mean()) if m.any() else np.nan,
                 "tasa_bestC": float(flags[kc][m].mean()) if m.any() else np.nan}
            rates = [flags[i][m].mean() for i in bkeys] if m.any() else [np.nan]
            r["tasa_B_min"], r["tasa_B_mediana"], r["tasa_B_max"] = (
                float(np.min(rates)), float(np.median(rates)), float(np.max(rates)))
            rows.append(r)
    tt = pd.DataFrame(rows)
    # drift: mediana del PSI por feature frente a las lícitas de train
    ref = (split == "train") & (grp == "licita")
    drift = []
    for t in range(1, 50):
        for g in ("licita", "sin_etiqueta"):
            m = (ts == t) & (grp == g)
            if m.sum() < 50:
                drift.append({"timestep": t, "grupo": g, "psi_mediana_vs_licitas_train": np.nan})
                continue
            v = [psi(xr[ref, j], xr[m, j]) for j in range(xr.shape[1])]
            drift.append({"timestep": t, "grupo": g,
                          "psi_mediana_vs_licitas_train": float(np.median(v))})
    tt = tt.merge(pd.DataFrame(drift), on=["timestep", "grupo"], how="left")
    tt.to_csv(OUT / "marcado_por_timestep.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax = axes[0]
    col = {"licita": "#2a7ab9", "ilicita": "#d1495b", "sin_etiqueta": "#edae49"}
    lab = {"licita": "Lícitas", "ilicita": "Ilícitas", "sin_etiqueta": "Sin etiqueta"}
    for g in ("licita", "ilicita", "sin_etiqueta"):
        s = tt[tt.grupo == g]
        ax.plot(s.timestep, s.tasa_bestB, "-o", ms=3, c=col[g], label=f"{lab[g]} (mejor B)")
        ax.fill_between(s.timestep, s.tasa_B_min, s.tasa_B_max, color=col[g], alpha=.15)
        if g == "sin_etiqueta":
            ax.plot(s.timestep, s.tasa_bestC, "--", c="k", lw=1.2,
                    label="Sin etiqueta (mejor C)")
    for x, txt in ((34.5, "train | val"), (42.5, "val | test")):
        ax.axvline(x, c="grey", ls=":", lw=1)
        ax.text(x + .2, .95, txt, fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_ylabel("Tasa de marcado")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Tasa de marcado por timestep (banda: mínimo y máximo de los 9 B)",
                 fontsize=10)
    ax = axes[1]
    for g in ("licita", "sin_etiqueta"):
        s = tt[tt.grupo == g]
        ax.plot(s.timestep, s.psi_mediana_vs_licitas_train, "-o", ms=3, c=col[g],
                label=f"{lab[g]} en t vs lícitas de train")
    ax.axvline(34.5, c="grey", ls=":", lw=1)
    ax.axvline(42.5, c="grey", ls=":", lw=1)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mediana del PSI por feature")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "marcado_por_timestep.png", dpi=150)
    plt.close(fig)

    def agg(rng, g, col_):
        s = tt[(tt.grupo == g) & tt.timestep.between(*rng)]
        return float((s[col_] * s.n).sum() / s.n.sum())
    res["tiempo"] = {
        f"{g}_{name}": agg(r, g, "tasa_bestB")
        for g in ("licita", "ilicita", "sin_etiqueta")
        for name, r in (("train", (1, 34)), ("val", (35, 42)), ("test_43_49", (43, 49)))}
    s = tt[tt.grupo == "sin_etiqueta"].set_index("timestep")
    res["tiempo"]["corr_tasa_sinetq_vs_drift"] = float(
        s[["tasa_bestB", "psi_mediana_vs_licitas_train"]].corr(method="spearman").iloc[0, 1])
    (OUT / "grafo_tiempo.json").write_text(json.dumps(res, indent=2, ensure_ascii=False,
                                                      default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str)[:9000])


if __name__ == "__main__":
    main()
