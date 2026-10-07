#!/usr/bin/env python
"""Paso 2 y 3: población (PSI, PCA, t-SNE, centroides) y cobertura (vecino más cercano,
clasificador de dominio lícita vs sin etiqueta).

Modelo principal: el mejor B por PR-AUC de validación (sinetq_common.BEST_B). Las cifras que
dependen de la marca (marcado / no marcado) se repiten para los 9 modelos B.

PSI: se calcula sobre las variables crudas (antes de la normalización), con 10 bins en los
deciles de la referencia. Umbrales de lectura: < 0,1 estable, 0,1 a 0,25 cambio moderado,
> 0,25 cambio fuerte (convención de riesgo crediticio; Siddiqi, 2006).

Uso: ~/.local/bin/uv run --frozen python scripts/v4/reunion_0410/sinetq_02_poblacion.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from sinetq_common import (BEST_B, FEATURE_NAMES, FIG, GROUP4, GROUP4_COLOR, GROUP4_LABEL,  # noqa: E402
                           OUT, ensure_dirs, group4, load_data, load_preds, psi)

RNG = np.random.default_rng(2026)


def psi_table(xr: np.ndarray, ref_mask, act_mask) -> np.ndarray:
    return np.array([psi(xr[ref_mask, j], xr[act_mask, j]) for j in range(xr.shape[1])])


def summarize_psi(vals: np.ndarray) -> dict:
    return {"n_gt_0.10": int((vals > 0.10).sum()), "n_gt_0.25": int((vals > 0.25).sum()),
            "mediana": float(np.median(vals)), "max": float(vals.max())}


def domain_auc(Xa: np.ndarray, Xb: np.ndarray, seed: int = 0, max_n: int = 20000) -> float:
    """AUC (5-fold, out-of-fold) de un clasificador que distingue a de b."""
    rng = np.random.default_rng(seed)
    if len(Xa) > max_n:
        Xa = Xa[rng.choice(len(Xa), max_n, replace=False)]
    if len(Xb) > max_n:
        Xb = Xb[rng.choice(len(Xb), max_n, replace=False)]
    X = np.vstack([Xa, Xb])
    y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                          method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def main() -> None:
    ensure_dirs()
    data, xr = load_data()
    xn = data.x.numpy()
    nodos, modelos, probs, flags = load_preds()
    split = nodos["split"].values
    grp = nodos["group"].values
    kbest = list(modelos["key"]).index(BEST_B)
    g4 = group4(nodos, flags[kbest])
    val = split == "val"
    tr = split == "train"
    res: dict = {"modelo": BEST_B}

    # ── PSI ────────────────────────────────────────────────────────────────────
    comps = {
        "sinetq_vs_licitas": (val & (grp == "licita"), val & (grp == "sin_etiqueta")),
        "sinetq_vs_ilicitas": (val & (grp == "ilicita"), val & (grp == "sin_etiqueta")),
        "marcados_vs_no_marcados": (val & (g4 == "sin_etiqueta_no_marcado"),
                                    val & (g4 == "sin_etiqueta_marcado")),
        "marcados_vs_ilicitas": (val & (grp == "ilicita"), val & (g4 == "sin_etiqueta_marcado")),
        "marcados_vs_licitas": (val & (grp == "licita"), val & (g4 == "sin_etiqueta_marcado")),
        # referencias de escala
        "control_licitas_val_vs_train": (tr & (grp == "licita"), val & (grp == "licita")),
        "control_ilicitas_vs_licitas_val": (val & (grp == "licita"), val & (grp == "ilicita")),
        "train_sinetq_vs_licitas": (tr & (grp == "licita"), tr & (grp == "sin_etiqueta")),
    }
    psi_df = pd.DataFrame({"feature": FEATURE_NAMES})
    psi_sum = {}
    for name, (ref, act) in comps.items():
        v = psi_table(xr, ref, act)
        psi_df[name] = v
        psi_sum[name] = summarize_psi(v)
        psi_sum[name]["n_ref"] = int(ref.sum())
        psi_sum[name]["n_act"] = int(act.sum())
    psi_df.to_csv(OUT / "psi_por_feature.csv", index=False)
    res["psi_resumen"] = psi_sum
    res["psi_top15"] = {name: psi_df.nlargest(15, name)[["feature", name]]
                        .round(3).values.tolist() for name in comps}

    # Mismo resumen de PSI marcados vs no marcados para los 9 B
    rows = []
    for i, k in enumerate(modelos["key"]):
        if not k.startswith("B|"):
            continue
        gi = group4(nodos, flags[i])
        uf, un = val & (gi == "sin_etiqueta_marcado"), val & (gi == "sin_etiqueta_no_marcado")
        il, li = val & (grp == "ilicita"), val & (grp == "licita")
        a = psi_table(xr, un, uf)
        b = psi_table(xr, il, uf)
        c = psi_table(xr, li, uf)
        rows.append({"key": k, "n_marcados_val": int(uf.sum()),
                     "psi_marc_vs_nomarc_gt025": int((a > .25).sum()),
                     "psi_marc_vs_ilic_gt025": int((b > .25).sum()),
                     "psi_marc_vs_lic_gt025": int((c > .25).sum()),
                     "psi_marc_vs_nomarc_mediana": float(np.median(a)),
                     "psi_marc_vs_ilic_mediana": float(np.median(b)),
                     "psi_marc_vs_lic_mediana": float(np.median(c))})
    pd.DataFrame(rows).to_csv(OUT / "psi_todos_B.csv", index=False)

    # Figura PSI: distribución por comparación
    fig, ax = plt.subplots(figsize=(8, 4.2))
    order = ["sinetq_vs_licitas", "sinetq_vs_ilicitas", "marcados_vs_no_marcados",
             "marcados_vs_ilicitas", "marcados_vs_licitas", "control_licitas_val_vs_train"]
    ax.boxplot([psi_df[c].values for c in order], vert=False, showfliers=True,
               flierprops={"markersize": 2})
    ax.set_yticklabels(["sin etiqueta vs lícitas", "sin etiqueta vs ilícitas",
                        "marcados vs no marcados", "marcados vs ilícitas",
                        "marcados vs lícitas", "control: lícitas val vs train"])
    ax.axvline(0.1, ls="--", c="grey", lw=.8)
    ax.axvline(0.25, ls="--", c="k", lw=.8)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("PSI por feature (165 features, validación)")
    fig.tight_layout()
    fig.savefig(FIG / "psi_boxplot.png", dpi=150)
    plt.close(fig)

    # ── PCA y centroides ───────────────────────────────────────────────────────
    Xv = xn[val]
    gv = g4[val]
    pca = PCA(n_components=20, random_state=0).fit(Xv)
    Zv = pca.transform(Xv)
    res["pca_varianza_explicada_2"] = pca.explained_variance_ratio_[:2].tolist()
    cents_full = {g: Xv[gv == g].mean(0) for g in GROUP4}
    cents_pca = {g: Zv[gv == g, :2].mean(0) for g in GROUP4}
    dist = []
    for i, a in enumerate(GROUP4):
        for b in GROUP4[i + 1:]:
            dist.append({"a": a, "b": b,
                         "dist_165d": float(np.linalg.norm(cents_full[a] - cents_full[b])),
                         "dist_pca2": float(np.linalg.norm(cents_pca[a] - cents_pca[b]))})
    pd.DataFrame(dist).to_csv(OUT / "distancias_centroides.csv", index=False)
    res["distancias_centroides"] = dist

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, (Z, title) in zip(axes, [(Zv[:, :2], "PCA (validación)"), (None, "t-SNE")]):
        if Z is None:
            continue
        for g in ["sin_etiqueta_no_marcado", "licita", "sin_etiqueta_marcado", "ilicita"]:
            idx = np.where(gv == g)[0]
            idx = RNG.choice(idx, min(len(idx), 3000), replace=False)
            ax.scatter(Z[idx, 0], Z[idx, 1], s=3, alpha=.45, c=GROUP4_COLOR[g],
                       label=f"{GROUP4_LABEL[g]} (n={int((gv == g).sum())})", rasterized=True)
            c = Z[gv == g].mean(0)
            ax.scatter(*c, s=160, marker="X", c=GROUP4_COLOR[g], edgecolors="k", lw=1.2)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        lo, hi = np.percentile(Z, [1, 99], axis=0)
        pad = (hi - lo) * 0.15
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.legend(markerscale=4, fontsize=8, loc="best")

    # t-SNE sobre submuestra estratificada (todas las ilícitas, 1.500 por grupo restante)
    sub = []
    for g in GROUP4:
        idx = np.where(gv == g)[0]
        sub.append(RNG.choice(idx, min(len(idx), 1500), replace=False))
    sub = np.concatenate(sub)
    T = TSNE(n_components=2, perplexity=40, init="pca", random_state=0,
             max_iter=1000).fit_transform(Zv[sub])
    ax = axes[1]
    for g in ["sin_etiqueta_no_marcado", "licita", "sin_etiqueta_marcado", "ilicita"]:
        m = gv[sub] == g
        ax.scatter(T[m, 0], T[m, 1], s=3, alpha=.6, c=GROUP4_COLOR[g], label=GROUP4_LABEL[g],
                   rasterized=True)
    ax.set_title("t-SNE (submuestra, 20 componentes PCA)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=4, fontsize=8)
    fig.suptitle(f"Validación (ts 35-42) · marca del modelo {BEST_B}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "pca_tsne_grupos.png", dpi=150)
    plt.close(fig)

    # vecindad en t-SNE: de los vecinos (k=10) de cada marcado, qué fracción es ilícita/lícita
    nn_t = NearestNeighbors(n_neighbors=11).fit(T)
    _, ind = nn_t.kneighbors(T)
    gs = gv[sub]
    comp = {}
    for g in GROUP4:
        m = gs == g
        neigh = gs[ind[m, 1:]]
        comp[g] = {h: float((neigh == h).mean()) for h in GROUP4}
    res["tsne_vecindad_k10"] = comp

    # ── Cobertura: vecino más cercano en el espacio normalizado ──────────────
    ref_l = xn[tr & (grp == "licita")]
    ref_i = xn[tr & (grp == "ilicita")]
    nn_l = NearestNeighbors(n_neighbors=1).fit(ref_l)
    nn_i = NearestNeighbors(n_neighbors=1).fit(ref_i)
    dl = nn_l.kneighbors(Xv)[0][:, 0]
    di = nn_i.kneighbors(Xv)[0][:, 0]
    thr95 = float(np.percentile(dl[gv == "licita"], 95))
    cov = {}
    for g in GROUP4:
        m = gv == g
        cov[g] = {"n": int(m.sum()),
                  "mediana_dist_licita_train": float(np.median(dl[m])),
                  "mediana_dist_ilicita_train": float(np.median(di[m])),
                  "frac_mas_cerca_de_ilicita": float((di[m] < dl[m]).mean()),
                  "frac_fuera_soporte_licito": float((dl[m] > thr95).mean())}
    res["cobertura_nn"] = {"umbral_p95_licitas_val": thr95, "grupos": cov}
    unk = (gv == "sin_etiqueta_marcado") | (gv == "sin_etiqueta_no_marcado")
    fuera = dl > thr95
    res["cobertura_nn"]["tasa_marcado_sinetq_fuera_soporte"] = float(
        (gv[unk & fuera] == "sin_etiqueta_marcado").mean())
    res["cobertura_nn"]["tasa_marcado_sinetq_dentro_soporte"] = float(
        (gv[unk & ~fuera] == "sin_etiqueta_marcado").mean())
    res["cobertura_nn"]["frac_sinetq_fuera_soporte"] = float(fuera[unk].mean())
    # 1-NN y 5-NN sobre las etiquetadas de train (clasificador sin grafo)
    X_lab = np.vstack([ref_l, ref_i])
    y_lab = np.r_[np.zeros(len(ref_l)), np.ones(len(ref_i))]
    nn5 = NearestNeighbors(n_neighbors=5).fit(X_lab)
    vote = y_lab[nn5.kneighbors(Xv)[1]].mean(1)
    res["knn5_mayoria_ilicita"] = {g: float((vote[gv == g] >= .6).mean()) for g in GROUP4}

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(np.log10(max(dl.min(), 1e-2)), np.log10(dl.max()), 60)
    for g in GROUP4:
        ax.hist(dl[gv == g], bins=bins, histtype="step", density=True, lw=1.4,
                color=GROUP4_COLOR[g], label=GROUP4_LABEL[g])
    ax.axvline(thr95, c="k", ls="--", lw=.8, label="p95 de lícitas val")
    ax.set_xscale("log")
    ax.set_xlabel("Distancia a la lícita de train más cercana (espacio normalizado)")
    ax.set_ylabel("Densidad")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "cobertura_distancia_licita.png", dpi=150)
    plt.close(fig)

    # ── Cobertura: clasificador de dominio (crudo, sin grafo) ────────────────
    dom = {
        "train_licitas_vs_sinetq": domain_auc(xr[tr & (grp == "licita")],
                                              xr[tr & (grp == "sin_etiqueta")]),
        "val_licitas_vs_sinetq": domain_auc(xr[val & (grp == "licita")],
                                            xr[val & (grp == "sin_etiqueta")]),
        "val_licitas_vs_marcados": domain_auc(xr[val & (grp == "licita")],
                                              xr[val & (g4 == "sin_etiqueta_marcado")]),
        "val_licitas_vs_no_marcados": domain_auc(xr[val & (grp == "licita")],
                                                 xr[val & (g4 == "sin_etiqueta_no_marcado")]),
        "val_ilicitas_vs_marcados": domain_auc(xr[val & (grp == "ilicita")],
                                               xr[val & (g4 == "sin_etiqueta_marcado")]),
        "val_licitas_vs_ilicitas": domain_auc(xr[val & (grp == "licita")],
                                              xr[val & (grp == "ilicita")]),
    }
    # control nulo: dos mitades al azar de las lícitas de validación
    li_v = xr[val & (grp == "licita")]
    perm = RNG.permutation(len(li_v))
    dom["control_licitas_mitad_vs_mitad"] = domain_auc(li_v[perm[: len(perm) // 2]],
                                                       li_v[perm[len(perm) // 2:]])
    res["clasificador_dominio_auc"] = dom

    # Tasa de marcado de B por decil de «parecido a sin etiqueta»: clasificador de dominio
    # entrenado en train (lícitas vs sin etiqueta), aplicado a los sin etiqueta de val.
    Xa, Xb = xr[tr & (grp == "licita")], xr[tr & (grp == "sin_etiqueta")]
    sel = RNG.choice(len(Xb), 40000, replace=False)
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(
        np.vstack([Xa, Xb[sel]]), np.r_[np.zeros(len(Xa)), np.ones(len(sel))])
    s_dom = clf.predict_proba(xr[val & (grp == "sin_etiqueta")])[:, 1]
    fl_unk = (g4[val & (grp == "sin_etiqueta")] == "sin_etiqueta_marcado")
    dec = pd.qcut(s_dom, 10, labels=False, duplicates="drop")
    tab = pd.DataFrame({"decil": dec, "marcado": fl_unk, "score": s_dom}).groupby("decil").agg(
        n=("marcado", "size"), tasa_marcado=("marcado", "mean"), score_medio=("score", "mean"))
    tab.to_csv(OUT / "marcado_por_decil_dominio.csv")
    res["marcado_por_decil_dominio"] = tab.reset_index().round(4).values.tolist()
    res["auc_score_dominio_predice_marca"] = float(roc_auc_score(fl_unk, s_dom))

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(tab.index + 1, tab["tasa_marcado"], color="#edae49")
    ax.set_xlabel("Decil del score «no se parece a las lícitas de train» (1 = más parecido)")
    ax.set_ylabel("Tasa de marcado de B")
    ax.set_xticks(range(1, len(tab) + 1))
    fig.tight_layout()
    fig.savefig(FIG / "marcado_por_decil_dominio.png", dpi=150)
    plt.close(fig)

    (OUT / "poblacion_cobertura.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                                  encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("psi_top15",)}, indent=1,
                     ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
