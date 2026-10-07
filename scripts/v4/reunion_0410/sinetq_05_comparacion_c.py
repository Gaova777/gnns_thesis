#!/usr/bin/env python
"""Paso 7 y controles: comparación con C, consenso entre los B, clasificadores sin grafo y
cotas por mezcla a partir del clasificador de dominio.

(a) Para cada B: qué fracción de los sin etiqueta que marca en validación (y en test) marca
    también cada modelo C (semilla 42, mismo arquitectura x balanceo; el mejor C; y mayoría
    de las 3 semillas de C de la misma configuración). Lift = tasa de C entre marcados de B /
    tasa de C entre no marcados de B. AUC del score de C para separar marcados de no marcados.
(b) Consenso: cuántos de los 9 B marcan a cada sin etiqueta de validación.
(c) Clasificadores sin grafo (HistGradientBoosting y logística) entrenados con las
    etiquetadas de train como B (lícitas vs ilícitas) y como C (lícitas + sin etiqueta vs
    ilícitas), umbral de F1 máximo en validación (como Trainer.calibrate_threshold).
    Si el de tipo B ya marca una fracción parecida de sin etiqueta, el fenómeno está en las
    features y en la elección de negativos, no en el paso de mensajes del GNN.
(d) Cotas por mezcla. Si los sin etiqueta fueran (1 - pi) lícitas con la MISMA distribución
    que las lícitas etiquetadas + pi otra cosa, cualquier clasificador cumpliría
    AUC(lícitas vs sin etiqueta) <= 0,5 + pi/2, o sea pi >= 2 AUC - 1. Análogamente, si una
    fracción rho de los marcados se distribuyera como las ilícitas etiquetadas,
    AUC(ilícitas vs marcados) <= 1 - rho/2, o sea rho <= 2 (1 - AUC).

Uso: ~/.local/bin/uv run --frozen python scripts/v4/reunion_0410/sinetq_05_comparacion_c.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

from sinetq_02_poblacion import domain_auc
from sinetq_common import OUT, ensure_dirs, group4, load_data, load_preds

BEST_C = "C|GraphSAGE|class_weighting|42"
RNG = np.random.default_rng(2026)


def f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.argmax(f1[:-1])])


def main() -> None:
    ensure_dirs()
    data, xr = load_data()
    xn = data.x.numpy()
    nodos, modelos, probs, flags = load_preds()
    keys = list(modelos["key"])
    split, grp = nodos["split"].values, nodos["group"].values
    res: dict = {}

    # ── (a) comparación con C ──────────────────────────────────────────────────
    rows = []
    for sp in ("val", "test"):
        unk = (split == sp) & (grp == "sin_etiqueta")
        for i, kb in enumerate(keys):
            if not kb.startswith("B|"):
                continue
            _, arch, bal, _ = kb.split("|")
            fb = flags[i][unk]
            cands = {"C_misma_config_s42": f"C|{arch}|{bal}|42", "C_mejor": BEST_C}
            for label, kc in cands.items():
                fc = flags[keys.index(kc)][unk]
                pc = probs[keys.index(kc)][unk]
                rows.append({"split": sp, "B": kb, "comparador": label, "C": kc,
                             "n_marcados_B": int(fb.sum()),
                             "tasa_C_en_todos_sinetq": float(fc.mean()),
                             "frac_marcados_B_tambien_C": float(fc[fb].mean()),
                             "tasa_C_en_no_marcados_B": float(fc[~fb].mean()),
                             "lift": float(fc[fb].mean() / max(fc[~fb].mean(), 1e-9)),
                             "frac_marcados_C_tambien_B": float(fb[fc].mean()) if fc.any()
                             else np.nan,
                             "auc_scoreC_marcadosB": float(roc_auc_score(fb, pc))})
            # mayoría de las 3 semillas de C de la misma configuración
            idx3 = [keys.index(f"C|{arch}|{bal}|{s}") for s in (42, 43, 44)]
            fc3 = flags[idx3][:, unk].sum(0) >= 2
            rows.append({"split": sp, "B": kb, "comparador": "C_mayoria_3_semillas",
                         "C": f"C|{arch}|{bal}|42-44", "n_marcados_B": int(fb.sum()),
                         "tasa_C_en_todos_sinetq": float(fc3.mean()),
                         "frac_marcados_B_tambien_C": float(fc3[fb].mean()),
                         "tasa_C_en_no_marcados_B": float(fc3[~fb].mean()),
                         "lift": float(fc3[fb].mean() / max(fc3[~fb].mean(), 1e-9)),
                         "frac_marcados_C_tambien_B": float(fb[fc3].mean()) if fc3.any()
                         else np.nan,
                         "auc_scoreC_marcadosB": np.nan})
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "comparacion_B_C.csv", index=False)
    res["comparacion_B_C_mediana"] = (comp.groupby(["split", "comparador"])
                                      [["frac_marcados_B_tambien_C", "tasa_C_en_todos_sinetq",
                                        "lift", "auc_scoreC_marcadosB"]]
                                      .median().round(4).reset_index().to_dict("records"))
    # recall de C sobre las ilícitas (escala)
    res["recall_ilicitas_C_mejor"] = {
        sp: float(flags[keys.index(BEST_C)][(split == sp) & (grp == "ilicita")].mean())
        for sp in ("val", "test")}

    # ── (b) consenso entre los B ──────────────────────────────────────────────
    bidx = [i for i, k in enumerate(keys) if k.startswith("B|")]
    good = [i for i in bidx if "|GCN|" not in keys[i]]
    unk = (split == "val") & (grp == "sin_etiqueta")
    cnt = flags[bidx][:, unk].sum(0)
    cnt6 = flags[good][:, unk].sum(0)
    res["consenso_B_val"] = {
        "distribucion_n_modelos_9": np.bincount(cnt, minlength=10).tolist(),
        "frac_marcado_por_al_menos_1": float((cnt >= 1).mean()),
        "frac_marcado_por_mayoria_5de9": float((cnt >= 5).mean()),
        "frac_marcado_por_los_9": float((cnt == 9).mean()),
        "frac_marcado_por_los_6_SAGE_TAGCN": float((cnt6 == 6).mean()),
        "n_marcado_por_los_6_SAGE_TAGCN": int((cnt6 == 6).sum())}
    jac = np.zeros((len(bidx), len(bidx)))
    for a, i in enumerate(bidx):
        for b, j in enumerate(bidx):
            fi, fj = flags[i][unk], flags[j][unk]
            jac[a, b] = (fi & fj).sum() / max((fi | fj).sum(), 1)
    pd.DataFrame(jac, index=[keys[i] for i in bidx], columns=[keys[i] for i in bidx]).round(
        3).to_csv(OUT / "jaccard_marcados_entre_B.csv")
    off = jac[~np.eye(len(bidx), dtype=bool)]
    res["consenso_B_val"]["jaccard_mediano_entre_B"] = float(np.median(off))
    # Acuerdo entre pares de B corregido por azar (kappa de Cohen), por grupo, en validación:
    # si los marcados fueran una población definida (fraude oculto coherente), los B buenos
    # coincidirían en ellos tanto como coinciden en las ilícitas.
    from itertools import combinations
    from sklearn.metrics import cohen_kappa_score
    kap = {}
    for g in ("ilicita", "licita", "sin_etiqueta"):
        m = (split == "val") & (grp == g)
        for name, idxs in (("6_SAGE_TAGCN", good), ("9_B", bidx)):
            ks = [cohen_kappa_score(flags[i][m], flags[j][m]) for i, j in combinations(idxs, 2)]
            jj = [(flags[i][m] & flags[j][m]).sum() / max((flags[i][m] | flags[j][m]).sum(), 1)
                  for i, j in combinations(idxs, 2)]
            kap[f"{g}_{name}"] = {"kappa_mediana": float(np.median(ks)),
                                  "kappa_min": float(np.min(ks)),
                                  "kappa_max": float(np.max(ks)),
                                  "jaccard_mediana": float(np.median(jj))}
    res["acuerdo_entre_B_val"] = kap

    # ── (c) clasificadores sin grafo ─────────────────────────────────────────
    tr = split == "train"
    lab_tr_B = tr & (grp != "sin_etiqueta")
    y_tr_B = (grp[lab_tr_B] == "ilicita").astype(int)
    all_tr = tr
    y_tr_C = (grp[all_tr] == "ilicita").astype(int)
    val_lab = (split == "val") & (grp != "sin_etiqueta")
    val_all = split == "val"
    sc = StandardScaler().fit(xn[tr])
    out_c = {}
    kb_best = keys.index("B|GraphSAGE|none|42")
    for clf_name in ("hgb", "logistica"):
        for mode, (Xm, ym, calib_m) in {
                "tipo_B": (lab_tr_B, y_tr_B, val_lab),
                "tipo_C": (all_tr, y_tr_C, val_all)}.items():
            if clf_name == "hgb":
                clf = HistGradientBoostingClassifier(max_iter=300, random_state=0,
                                                     class_weight="balanced")
                clf.fit(xr[Xm], ym)
                score = clf.predict_proba(xr)[:, 1]
            else:
                clf = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")
                clf.fit(np.clip(sc.transform(xn[Xm]), -10, 10), ym)
                score = clf.predict_proba(np.clip(sc.transform(xn), -10, 10))[:, 1]
            thr = f1_threshold((grp[calib_m] == "ilicita").astype(int), score[calib_m])
            f = score >= thr
            r = {"umbral": thr}
            for sp in ("train", "val", "test"):
                for g in ("licita", "ilicita", "sin_etiqueta"):
                    m = (split == sp) & (grp == g)
                    r[f"{sp}_{g}"] = float(f[m].mean())
            m = val_lab
            r["val_roc_auc_vs_licitas"] = float(roc_auc_score(grp[m] == "ilicita", score[m]))
            u = (split == "val") & (grp == "sin_etiqueta")
            fb = flags[kb_best][u]
            r["jaccard_con_mejor_B_sinetq_val"] = float((f[u] & fb).sum() / max((f[u] | fb)
                                                                                 .sum(), 1))
            r["frac_marcados_mejorB_tambien_este"] = float(f[u][fb].mean())
            out_c[f"{clf_name}_{mode}"] = r
    res["clasificadores_sin_grafo"] = out_c

    # ── (d) cotas por mezcla con el clasificador de dominio ──────────────────
    val = split == "val"
    a_lu = domain_auc(xr[val & (grp == "licita")], xr[val & (grp == "sin_etiqueta")])
    a_lu_tr = domain_auc(xr[tr & (grp == "licita")], xr[tr & (grp == "sin_etiqueta")])
    bounds = {"auc_licitas_vs_sinetq_val": a_lu, "pi_min_val": 2 * a_lu - 1,
              "auc_licitas_vs_sinetq_train": a_lu_tr, "pi_min_train": 2 * a_lu_tr - 1,
              "por_B": []}
    # variabilidad: 5 repeticiones con otra partición de CV y otra submuestra
    reps = [domain_auc(xr[val & (grp == "licita")], xr[val & (grp == "sin_etiqueta")], seed=s)
            for s in range(1, 6)]
    bounds["auc_licitas_vs_sinetq_val_rep_min_max"] = [float(min(reps)), float(max(reps))]
    for i in bidx:
        g4 = group4(nodos, flags[i])
        uf = val & (g4 == "sin_etiqueta_marcado")
        a = domain_auc(xr[val & (grp == "ilicita")], xr[uf])
        a2 = domain_auc(xr[val & (grp == "licita")], xr[uf])
        bounds["por_B"].append({"B": keys[i], "n_marcados": int(uf.sum()),
                                "auc_ilicitas_vs_marcados": a, "rho_max": 2 * (1 - a),
                                "auc_licitas_vs_marcados": a2})
    res["cotas_mezcla"] = bounds

    (OUT / "comparacion_c_controles.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
