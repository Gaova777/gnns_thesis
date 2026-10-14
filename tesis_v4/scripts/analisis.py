"""Tablas y pruebas del manuscrito v4 (corrida sin el cierre del mercado), sin GPU.

Unidad independiente: el modelo entrenado (arquitectura x semilla). Nunca se usan como
bloques independientes los escenarios de una misma semilla: comparten pesos iniciales y
salen casi idénticos (pseudorreplicación, ver results_v4/reunion_0710/VALIDACION.md).

  H1 (escenario)    Friedman, bloques = 9 modelos (3 arquitecturas que aprenden x 3
                    semillas), tratamientos = 6 escenarios; pesos por clase. Además TOST
                    pareado nativo frente a 1:200 con margen 0,05.
  H2 (arquitectura) Friedman, bloques = 3 semillas, tratamientos = 4 arquitecturas
                    (media de los 4 escenarios principales, pesos por clase); se informa
                    también en cuántas semillas queda primera cada arquitectura.
  H3 (pérdida)      prueba de signos pareada, 9 pares (arquitectura x semilla, media de los
                    4 escenarios principales): pesos por clase frente a focal loss y frente
                    a sin ajuste.

Salidas: tesis_v4/tablas/*.tex y tesis_v4/tablas/pruebas.csv.
Uso: uv run --frozen python tesis_v4/scripts/analisis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import estilo as E  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PRE = ROOT / "results_v4_precierre"
ROB = PRE / "robusta"
CURV = PRE / "curvas"
TAB = Path(__file__).resolve().parents[1] / "tablas"
APR = ["GraphSAGE", "GAT", "TAGCN"]
MAIN = ["1:10", "1:10_os", "1:20", "native"]


def c(v, d=3):
    """Número con coma decimal para LaTeX."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:.{d}f}".replace(".", "{,}")


def p_txt(p):
    if p < 0.001:
        return "$<0{,}001$"
    return c(p, 3)


def guardar(nombre, cuerpo):
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / f"{nombre}.tex").write_text(
        f"% Generado por tesis_v4/scripts/analisis.py. No editar a mano.\n{cuerpo}",
        encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
def tabla_particion():
    cls = pd.read_csv(ROOT / "data/raw/elliptic_txs_classes.csv")
    ts = pd.read_csv(ROOT / "data/raw/elliptic_txs_features.csv", header=None,
                     usecols=[0, 1], names=["txId", "t"])
    d = cls.merge(ts, on="txId")
    d["il"] = d["class"].astype(str) == "1"
    filas = []
    for nombre, part in (("Original", [("Entrenamiento", 1, 34), ("Validación", 35, 42),
                                       ("Prueba", 43, 49)]),
                         ("Sin el cierre", [("Entrenamiento", 1, 29), ("Validación", 30, 35),
                                            ("Prueba", 36, 42)])):
        for parte, a, b in part:
            s = d[(d.t >= a) & (d.t <= b)]
            il, neg = int(s.il.sum()), int((~s.il).sum())
            filas.append((nombre, parte, f"{a} a {b}", il, neg, neg / il))
    tot = d[(d.t <= 42)]
    cuerpo = ["\\begin{tabular}{llrrrr}", "\\toprule",
              "Partición & Parte & Pasos & Ilícitas & Negativas & Razón \\\\", "\\midrule"]
    previo = None
    for nombre, parte, pasos, il, neg, r in filas:
        if previo and nombre != previo:
            cuerpo.append("\\midrule")
        cuerpo.append(f"{nombre if nombre != previo else ''} & {parte} & {pasos} & "
                      f"{il:,} & {neg:,} & 1:{c(r, 1)} \\\\".replace(",", ".")
                      .replace("{.}", "{,}"))
        previo = nombre
    cuerpo += ["\\bottomrule", "\\end{tabular}"]
    guardar("particion", "\n".join(cuerpo) + "\n")
    print(f"pasos 1-42: {int(tot.il.sum())} ilícitas")


def tabla_rendimiento():
    met = pd.read_csv(CURV / "curvas_metricas_por_modelo.csv")
    nat = met[met.scenario == "native"]
    val = nat[nat.split == "val"].groupby(["arch", "balancing"]).pr_auc.mean()
    filas = ["\\begin{tabular}{llcccc}", "\\toprule",
             "Arquitectura & Pérdida elegida & PR-AUC val. & PR-AUC prueba & ROC-AUC prueba "
             "& Filtro \\\\", "\\midrule"]
    for arch in E.ARQ:
        loss = val.loc[arch].idxmax()
        v = nat[(nat.arch == arch) & (nat.balancing == loss)]
        tv, tt = v[v.split == "val"], v[v.split == "test"]
        gate = int(tv.quality_passed.sum())
        filas.append(f"{arch} & {E.NOMBRE_PERDIDA[loss]} & "
                     f"{c(tv.pr_auc.mean())} $\\pm$ {c(tv.pr_auc.std())} & "
                     f"{c(tt.pr_auc.mean())} $\\pm$ {c(tt.pr_auc.std())} & "
                     f"{c(tt.roc_auc.mean())} $\\pm$ {c(tt.roc_auc.std())} & {gate}/3 \\\\")
    prev_v = nat[nat.split == "val"].prevalence.mean()
    prev_t = nat[nat.split == "test"].prevalence.mean()
    filas += ["\\midrule",
              f"Azar & & {c(prev_v)} & {c(prev_t)} & 0{{,}}5 & \\\\",
              "\\bottomrule", "\\end{tabular}"]
    guardar("rendimiento_nativo", "\n".join(filas) + "\n")

    cfg = pd.read_csv(CURV / "curvas_metricas_por_config.csv")
    cfg = cfg[cfg.split == "val"]
    filas = ["\\begin{tabular}{ll" + "c" * len(E.ESC) + "}", "\\toprule",
             "Arquitectura & Pérdida & " + " & ".join(E.NOMBRE_ESC[s] for s in E.ESC)
             + " \\\\", "\\midrule"]
    for i, arch in enumerate(E.ARQ):
        if i:
            filas.append("\\midrule")
        for j, loss in enumerate(E.PERDIDA):
            d = cfg[(cfg.arch == arch) & (cfg.balancing == loss)].set_index("scenario")
            vals = " & ".join(c(d.pr_auc_mean.get(s, np.nan)) for s in E.ESC)
            filas.append(f"{arch if j == 0 else ''} & {E.NOMBRE_PERDIDA[loss]} & {vals} \\\\")
    filas += ["\\bottomrule", "\\end{tabular}"]
    guardar("rendimiento_escenarios", "\n".join(filas) + "\n")

    g = cfg.groupby(["arch", "balancing"]).n_gate_passed.sum().unstack()
    filas = ["\\begin{tabular}{lccc}", "\\toprule",
             "Arquitectura & " + " & ".join(E.NOMBRE_PERDIDA[p] for p in E.PERDIDA) + " \\\\",
             "\\midrule"]
    for arch in E.ARQ:
        filas.append(f"{arch} & " + " & ".join(f"{int(g.loc[arch, p])}/18" for p in E.PERDIDA)
                     + " \\\\")
    filas += ["\\bottomrule", "\\end{tabular}"]
    guardar("filtro_calidad", "\n".join(filas) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
def pruebas():
    m = pd.read_csv(ROB / "robusta_modelos.csv")
    filas = []

    # H1
    cw = m[(m.balancing == "class_weighting") & m.arch.isin(APR)]
    for ex in E.EXPL:
        for med in E.MEDIDA:
            t = cw[cw.explainer == ex].pivot_table(index=["arch", "seed"], columns="scenario",
                                                  values=med)[E.ESC].dropna()
            medias = t.mean()
            if t.shape[0] >= 2 and t.values.std() > 0:
                chi, p = stats.friedmanchisquare(*[t[s] for s in E.ESC])
                w = chi / (t.shape[0] * (len(E.ESC) - 1))
            else:
                chi, p, w = np.nan, np.nan, np.nan
            dif = (t["1:200_subil"] - t["native"]).values
            if dif.std() > 0:
                p1 = stats.ttest_1samp(dif, -0.05, alternative="greater").pvalue
                p2 = stats.ttest_1samp(dif, 0.05, alternative="less").pvalue
                tost = max(p1, p2)
            else:
                tost = 0.0 if abs(dif.mean()) < 0.05 else 1.0
            filas.append({"hip": "H1", "explainer": ex, "medida": med, "n": t.shape[0],
                          "estadistico": chi, "p": p, "W": w,
                          "rango_medias": medias.max() - medias.min(), "tost_p": tost,
                          **{f"m_{s}": medias[s] for s in E.ESC}})

    # H2
    h2 = m[(m.balancing == "class_weighting") & m.scenario.isin(MAIN)]
    for ex in E.EXPL:
        for med in E.MEDIDA:
            t = h2[h2.explainer == ex].groupby(["seed", "arch"])[med].mean().unstack()[E.ARQ]
            t = t.dropna()
            if t.values.std() > 0:
                chi, p = stats.friedmanchisquare(*[t[a] for a in E.ARQ])
                t3 = t[APR]
                chi3, p3 = stats.friedmanchisquare(*[t3[a] for a in APR])
            else:
                chi = p = chi3 = p3 = np.nan
            primera = t.idxmax(axis=1).value_counts().to_dict()
            filas.append({"hip": "H2", "explainer": ex, "medida": med, "n": t.shape[0],
                          "estadistico": chi, "p": p, "p_sin_gcn": p3,
                          "primera": "; ".join(f"{k} {v}" for k, v in primera.items()),
                          **{f"m_{a}": t[a].mean() for a in E.ARQ}})

    # H3
    h3 = m[m.arch.isin(APR) & m.scenario.isin(MAIN)]
    for ex in E.EXPL:
        for med in E.MEDIDA:
            t = h3[h3.explainer == ex].groupby(["arch", "seed", "balancing"])[med].mean() \
                .unstack()[E.PERDIDA].dropna()
            for otra in ("focal_loss", "none"):
                d = (t["class_weighting"] - t[otra]).values
                mayor = int((d > 0).sum())
                n = int((d != 0).sum())
                p = stats.binomtest(mayor, n, 0.5).pvalue if n else np.nan
                filas.append({"hip": "H3", "explainer": ex, "medida": med, "n": len(d),
                              "comparacion": f"class_weighting vs {otra}", "mayor": mayor,
                              "p": p, "dif_media": d.mean(),
                              **{f"m_{b}": t[b].mean() for b in E.PERDIDA}})
    df = pd.DataFrame(filas)
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / "pruebas.csv", index=False)

    # tabla H1 (estabilidad por escenario, las tres medidas)
    h1 = df[df.hip == "H1"]
    cuerpo = ["\\begin{tabular}{ll" + "c" * len(E.ESC) + "cc}", "\\toprule",
              "Medida & Explicador & " + " & ".join(E.NOMBRE_ESC[s] for s in E.ESC)
              + " & $p$ Friedman & $p$ TOST \\\\", "\\midrule"]
    for i, med in enumerate(E.MEDIDA):
        if i:
            cuerpo.append("\\midrule")
        for j, ex in enumerate(E.EXPL):
            r = h1[(h1.explainer == ex) & (h1.medida == med)].iloc[0]
            nm = {"sp_all": "Spearman global", "sp_nz": "Spearman $\\neq 0$",
                  "jac10": "Jaccard top-10"}[med]
            cuerpo.append(f"{nm if j == 0 else ''} & {E.NOMBRE_EXPL[ex]} & "
                          + " & ".join(c(r[f'm_{s}']) for s in E.ESC)
                          + f" & {p_txt(r.p) if not np.isnan(r.p) else '--'}"
                          f" & {p_txt(r.tost_p)} \\\\")
    cuerpo += ["\\bottomrule", "\\end{tabular}"]
    guardar("h1_escenario", "\n".join(cuerpo) + "\n")

    # tabla H2
    h2t = df[df.hip == "H2"]
    cuerpo = ["\\begin{tabular}{llccccccl}", "\\toprule",
              "Medida & Explicador & " + " & ".join(E.ARQ)
              + " & $p$ (4) & $p$ (3) & Primera \\\\", "\\midrule"]
    for i, med in enumerate(E.MEDIDA):
        if i:
            cuerpo.append("\\midrule")
        for j, ex in enumerate(["GNNExplainer", "ShapleyFeatures", "ExpectedGradients"]):
            r = h2t[(h2t.explainer == ex) & (h2t.medida == med)].iloc[0]
            nm = {"sp_all": "Spearman global", "sp_nz": "Spearman $\\neq 0$",
                  "jac10": "Jaccard top-10"}[med]
            cuerpo.append(f"{nm if j == 0 else ''} & {E.NOMBRE_EXPL[ex]} & "
                          + " & ".join(c(r[f'm_{a}']) for a in E.ARQ)
                          + f" & {p_txt(r.p)} & {p_txt(r.p_sin_gcn)} & "
                          f"{r.primera.replace(';', ',')} \\\\")
    cuerpo += ["\\bottomrule", "\\end{tabular}"]
    guardar("h2_arquitectura", "\n".join(cuerpo) + "\n")

    # tabla H3
    h3t = df[df.hip == "H3"]
    cuerpo = ["\\begin{tabular}{llccccc}", "\\toprule",
              "Medida & Explicador & Pesos por clase & Focal loss & Sin ajuste & "
              "PC $>$ FL & PC $>$ SA \\\\", "\\midrule"]
    for i, med in enumerate(E.MEDIDA):
        if i:
            cuerpo.append("\\midrule")
        for j, ex in enumerate(["GNNExplainer", "ShapleyFeatures", "ExpectedGradients"]):
            r1 = h3t[(h3t.explainer == ex) & (h3t.medida == med)
                     & (h3t.comparacion.str.endswith("focal_loss"))].iloc[0]
            r2 = h3t[(h3t.explainer == ex) & (h3t.medida == med)
                     & (h3t.comparacion.str.endswith("none"))].iloc[0]
            nm = {"sp_all": "Spearman global", "sp_nz": "Spearman $\\neq 0$",
                  "jac10": "Jaccard top-10"}[med]
            cuerpo.append(f"{nm if j == 0 else ''} & {E.NOMBRE_EXPL[ex]} & "
                          f"{c(r1.m_class_weighting)} & {c(r1.m_focal_loss)} & {c(r1.m_none)} & "
                          f"{r1.mayor}/9 ({p_txt(r1.p)}) & {r2.mayor}/9 ({p_txt(r2.p)}) \\\\")
    cuerpo += ["\\bottomrule", "\\end{tabular}"]
    guardar("h3_perdida", "\n".join(cuerpo) + "\n")
    return df


def tabla_control():
    p = ROB / "control_aleatorio.csv"
    if not p.exists():
        print("control: falta", p)
        return
    cc = pd.read_csv(p)
    g = cc.groupby("explainer").mean(numeric_only=True).reindex(E.EXPL)
    cuerpo = ["\\begin{tabular}{lcccccc}", "\\toprule",
              " & \\multicolumn{3}{c}{Spearman global} & \\multicolumn{3}{c}{Jaccard top-10} \\\\",
              "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
              "Explicador & Estab. entrenada & Estab. azar & Entrenada vs azar & "
              "Estab. entrenada & Estab. azar & Entrenada vs azar \\\\", "\\midrule"]
    for ex in E.EXPL:
        r = g.loc[ex]
        cuerpo.append(f"{E.NOMBRE_EXPL[ex]} & {c(r.estab_entrenado)} & {c(r.estab_azar)} & "
                      f"{c(r.rho_entrenado_azar)} & {c(r.jac10_estab_entrenado)} & "
                      f"{c(r.jac10_estab_azar)} & {c(r.jac10_entrenado_azar)} \\\\")
    cuerpo += ["\\bottomrule", "\\end{tabular}"]
    guardar("control", "\n".join(cuerpo) + "\n")
    print(f"control: {len(cc)} filas")


def tabla_pgexplainer():
    """PGExplainer explica aristas: Spearman y Jaccard top-20 entre réplicas sobre las aristas
    del subgrafo, por arquitectura y pérdida (escenarios principales, 3 semillas)."""
    s = pd.read_csv(PRE / "elliptic_v4_stability.csv")
    s = s[(s.explainer == "PGExplainer") & s.scenario.isin(MAIN)]
    g = s.groupby(["arch", "balancing"])[["spearman_edges", "jaccard_edges_topk"]].mean()
    cuerpo = ["\\begin{tabular}{l" + "cc" * 3 + "}", "\\toprule",
              " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{E.NOMBRE_PERDIDA[p]}}}"
                                 for p in E.PERDIDA) + " \\\\",
              "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
              "Arquitectura" + " & Spearman & Jaccard" * 3 + " \\\\", "\\midrule"]
    for arch in E.ARQ:
        cuerpo.append(arch + "".join(
            f" & {c(g.loc[(arch, p), 'spearman_edges'])} & "
            f"{c(g.loc[(arch, p), 'jaccard_edges_topk'])}" for p in E.PERDIDA) + " \\\\")
    cuerpo += ["\\bottomrule", "\\end{tabular}"]
    guardar("pgexplainer", "\n".join(cuerpo) + "\n")
    print("PGExplainer por pérdida:\n", s.groupby("balancing")[["spearman_edges",
                                                              "jaccard_edges_topk"]].mean())


def main() -> int:
    tabla_particion()
    tabla_rendimiento()
    df = pruebas()
    tabla_control()
    tabla_pgexplainer()
    pd.set_option("display.width", 220)
    print(df[df.hip == "H1"][["explainer", "medida", "n", "p", "W", "rango_medias", "tost_p"]]
          .round(4).to_string())
    print(df[df.hip == "H2"][["explainer", "medida", "n", "p", "p_sin_gcn", "primera"]
                             + [f"m_{a}" for a in E.ARQ]].round(3).to_string())
    print(df[df.hip == "H3"][["explainer", "medida", "comparacion", "mayor", "p", "dif_media"]]
          .round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
