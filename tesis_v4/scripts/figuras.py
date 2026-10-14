"""Figuras del manuscrito v4, desde los resultados guardados (sin GPU, sin entrenar).

Corrida principal: partición sin el cierre del mercado (train 1-29, val 30-35, test 36-42).
Corrida original (1-34 / 35-42 / 43-49): solo como evidencia del efecto del cierre.

Uso: uv run --frozen python tesis_v4/scripts/figuras.py [nombre ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import estilo as E  # noqa: E402
from estilo import plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PRE = ROOT / "results_v4_precierre"
ORI = ROOT / "results_v4"
CURV_PRE = PRE / "curvas"
CURV_ORI = ORI / "reunion_0410" / "curvas"
ROB = PRE / "robusta"


def rid(scen, arch, loss, seed):
    r = f"{scen}_{arch}_{loss}".replace(":", "-")
    return r if seed == 42 else f"{r}_s{seed}"


# ─────────────────────────────────────────────────────────────────────────────
def fig_particion():
    """Ilícitas por paso de tiempo y las dos particiones (original y sin el cierre)."""
    cls = pd.read_csv(ROOT / "data/raw/elliptic_txs_classes.csv")
    ts = pd.read_csv(ROOT / "data/raw/elliptic_txs_features.csv", header=None, usecols=[0, 1],
                     names=["txId", "t"])
    d = cls.merge(ts, on="txId")
    il = d[d["class"].astype(str) == "1"].groupby("t").size().reindex(range(1, 50), fill_value=0)
    li = d[d["class"].astype(str) == "2"].groupby("t").size().reindex(range(1, 50), fill_value=0)

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(E.ANCHO, 3.4), sharex=True,
                                 gridspec_kw={"height_ratios": [2.4, 1], "hspace": 0.22})
    ax.bar(il.index, il.values, width=0.8, color=E.OKABE["bermellon"], label="Ilícitas")
    ax.set_ylabel("Transacciones ilícitas")
    ax.axvline(42.5, color="k", lw=0.8, ls="--")
    ax.text(42.2, il.max() * 1.02, "cierre del mercado (t = 43)", fontsize=7.5, va="bottom",
            ha="right")
    ax.set_ylim(0, il.max() * 1.12)
    ax2 = ax.twinx()
    ax2.plot(li.index, li.values, color=E.OKABE["gris"], lw=0.9, label="Lícitas")
    ax2.set_ylabel("Lícitas", color=E.OKABE["gris"])
    ax2.tick_params(axis="y", colors=E.OKABE["gris"])
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2,
              fontsize=7.5)
    E.letra(ax, "a")

    tramos = {"Original": [(1, 34, "Entrenamiento"), (35, 42, "Validación"), (43, 49, "Prueba")],
              "Sin el cierre": [(1, 29, "Entrenamiento"), (30, 35, "Validación"),
                                (36, 42, "Prueba"), (43, 49, "No se usa")]}
    col = {"Entrenamiento": E.OKABE["azul"], "Validación": E.OKABE["naranja"],
           "Prueba": E.OKABE["verde"], "No se usa": "white"}
    for y, (nombre, tr) in enumerate(tramos.items()):
        for a, b, parte in tr:
            hatch = "////" if parte == "No se usa" else None
            bx.barh(1 - y, b - a + 1, left=a - 0.5, height=0.62, color=col[parte],
                    edgecolor="k" if hatch else "white", lw=0.5, hatch=hatch,
                    alpha=0.9 if not hatch else 1)
            if b - a >= 4:
                bx.text((a + b) / 2, 1 - y, parte, ha="center", va="center", fontsize=7,
                        color="white" if parte != "No se usa" else "k",
                        bbox=None if parte != "No se usa" else
                        dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    bx.set_yticks([1, 0], list(tramos))
    bx.set_ylim(-0.5, 1.5)
    bx.axvline(42.5, color="k", lw=0.8, ls="--")
    bx.grid(False)
    bx.set_xlim(0.3, 49.7)
    bx.set_xticks([1, 10, 20, 29, 34, 42, 49])
    bx.tick_params(axis="x", labelsize=7)
    bx.set_xlabel("Paso de tiempo")
    bx.text(-0.02, 1.04, "(b)", transform=bx.transAxes, ha="right", va="bottom",
            fontsize=9, fontweight="bold")
    return E.guardar(fig, "particion_temporal")


# ─────────────────────────────────────────────────────────────────────────────
def _labels(curv):
    z = np.load(curv / "scores" / "_labels.npz")
    return {"val": z["y_val"].astype(int), "test": z["y_test"].astype(int)}


def _curvas(curv, scen, arch, loss, split, y):
    out = []
    for s in (42, 43, 44):
        p = curv / "scores" / f"{rid(scen, arch, loss, s)}.npz"
        if p.exists():
            out.append(np.load(p)[f"p_{split}"].astype(np.float64))
    return out


def fig_curvas(split="test"):
    """PR y ROC en el nativo, la pérdida elegida por validación para cada arquitectura.

    Se dibuja la curva de cada semilla en trazo fino y la de la semilla 42 en trazo grueso.
    """
    lab = _labels(CURV_PRE)
    mejor = json.loads((CURV_PRE / "mejor_perdida_native.json").read_text())
    y = lab[split]
    prev = y.mean()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(E.ANCHO, 2.9))
    met = pd.read_csv(CURV_PRE / "curvas_metricas_por_modelo.csv")
    for arch in E.ARQ:
        loss = mejor[arch]
        ps = _curvas(CURV_PRE, "native", arch, loss, split, y)
        m = met[(met.scenario == "native") & (met.arch == arch) & (met.balancing == loss)
                & (met.split == split)]
        for i, p in enumerate(ps):
            pr, rc, _ = precision_recall_curve(y, p)
            fp, tp, _ = roc_curve(y, p)
            kw = dict(color=E.COLOR_ARQ[arch], ls=E.LINEA_ARQ[arch])
            if i == 0:
                lbl_pr = f"{arch} ({m.pr_auc.mean():.3f})".replace(".", ",")
                lbl_roc = f"{arch} ({m.roc_auc.mean():.3f})".replace(".", ",")
                ax.plot(rc, pr, lw=1.3, label=lbl_pr, **kw)
                bx.plot(fp, tp, lw=1.3, label=lbl_roc, **kw)
            else:
                ax.plot(rc, pr, lw=0.5, alpha=0.45, **kw)
                bx.plot(fp, tp, lw=0.5, alpha=0.45, **kw)
    ax.axhline(prev, color="k", lw=0.7, ls=(0, (2, 2)))
    ax.text(0.02, prev + 0.012, f"azar = {prev:.3f}".replace(".", ","), ha="left",
            va="bottom", fontsize=7)
    bx.plot([0, 1], [0, 1], color="k", lw=0.7, ls=(0, (2, 2)))
    ax.set(xlabel="Exhaustividad (recall)", ylabel="Precisión", xlim=(0, 1), ylim=(0, 1))
    bx.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(title="PR-AUC media", loc="upper right", title_fontsize=8)
    bx.legend(title="ROC-AUC media", loc="lower right", title_fontsize=8)
    for a in (ax, bx):
        E.coma(a)
        a.set_aspect("equal")
    E.letra(ax, "a")
    E.letra(bx, "b")
    fig.tight_layout(w_pad=2)
    return E.guardar(fig, f"curvas_nativo_{split}")


# ─────────────────────────────────────────────────────────────────────────────
def fig_cierre():
    """PR-AUC y ROC-AUC de prueba por arquitectura: con el cierre frente a sin el cierre."""
    filas = []
    for nombre, curv in (("Con el cierre (pasos 43 a 49)", CURV_ORI),
                         ("Sin el cierre (pasos 36 a 42)", CURV_PRE)):
        met = pd.read_csv(curv / "curvas_metricas_por_modelo.csv")
        nat = met[met.scenario == "native"]
        val = nat[nat.split == "val"].groupby(["arch", "balancing"]).pr_auc.mean()
        for arch in E.ARQ:
            loss = val.loc[arch].idxmax()
            t = nat[(nat.split == "test") & (nat.arch == arch) & (nat.balancing == loss)]
            filas.append({"corrida": nombre, "arch": arch, "pr": t.pr_auc.values,
                          "roc": t.roc_auc.values, "prev": t.prevalence.mean()})
    df = pd.DataFrame(filas)
    fig, axs = plt.subplots(1, 2, figsize=(E.ANCHO, 2.5))
    w = 0.38
    x = np.arange(len(E.ARQ))
    col = [E.OKABE["gris"], E.OKABE["azul"]]
    for j, (k, eti) in enumerate((("pr", "PR-AUC en prueba"), ("roc", "ROC-AUC en prueba"))):
        ax = axs[j]
        for i, nombre in enumerate(df.corrida.unique()):
            d = df[df.corrida == nombre].set_index("arch").loc[E.ARQ]
            med = [v.mean() for v in d[k]]
            ax.bar(x + (i - 0.5) * w, med, w, color=col[i], label=nombre)
            for xi, v in zip(x, d[k]):
                ax.plot(np.full(len(v), xi + (i - 0.5) * w), v, "o", ms=2.2, color="k",
                        alpha=0.7)
            if k == "pr":
                ax.hlines(d.prev.values, x + (i - 1) * w, x + i * w, color="k", lw=0.8,
                          ls=(0, (2, 1)))
        if k == "roc":
            ax.axhline(0.5, color="k", lw=0.8, ls=(0, (2, 1)))
            ax.set_ylim(0.3, 1)
        ax.set_xticks(x, E.ARQ)
        ax.set_ylabel(eti)
        E.coma(ax, "y")
        E.letra(ax, "ab"[j])
    axs[0].legend(loc="upper left", fontsize=7)
    fig.tight_layout(w_pad=2)
    return E.guardar(fig, "efecto_cierre")


# ─────────────────────────────────────────────────────────────────────────────
def fig_perdida_escenario():
    """PR-AUC de validación por escenario y pérdida, una columna por arquitectura."""
    cfg = pd.read_csv(CURV_PRE / "curvas_metricas_por_config.csv")
    cfg = cfg[cfg.split == "val"]
    prev = cfg.prevalence_mean.mean()
    fig, axs = plt.subplots(1, 4, figsize=(E.ANCHO, 2.3), sharey=True)
    x = np.arange(len(E.ESC))
    for ax, arch in zip(axs, E.ARQ):
        for loss in E.PERDIDA:
            d = cfg[(cfg.arch == arch) & (cfg.balancing == loss)].set_index("scenario")
            d = d.reindex(E.ESC)
            ax.errorbar(x, d.pr_auc_mean, yerr=d.pr_auc_std, color=E.COLOR_PERDIDA[loss],
                        ls=E.LINEA_PERDIDA[loss], marker="o", ms=2.8, capsize=1.5, lw=1,
                        elinewidth=0.6, label=E.NOMBRE_PERDIDA[loss])
        ax.axhline(prev, color="k", lw=0.6, ls=(0, (2, 2)))
        ax.set_xticks(x, [E.NOMBRE_ESC[s].replace(" + SMOTE", "\n+SMOTE") for s in E.ESC],
                      rotation=90, fontsize=7)
        ax.text(0.03, 0.97, arch, transform=ax.transAxes, va="top", fontsize=8.5,
                fontweight="bold")
        ax.set_ylim(0, 0.8)
        E.coma(ax, "y")
    axs[0].set_ylabel("PR-AUC en validación")
    h, l = axs[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.08, right=0.99, top=0.97, bottom=0.36, wspace=0.08)
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    return E.guardar(fig, "perdida_por_escenario")


# ─────────────────────────────────────────────────────────────────────────────
def _modelos():
    m = pd.read_csv(ROB / "robusta_modelos.csv")
    return m


def fig_estabilidad_escenario():
    """H1: estabilidad entre réplicas por escenario, pesos por clase, 3 arquitecturas.

    Cada punto es la media de 9 modelos (3 arquitecturas x 3 semillas); la barra es el
    intervalo bootstrap del 95 % sobre esos 9 modelos.
    """
    m = _modelos()
    m = m[(m.balancing == "class_weighting") & m.arch.isin(["GraphSAGE", "GAT", "TAGCN"])]
    rng = np.random.default_rng(2026)
    fig, axs = plt.subplots(1, 3, figsize=(E.ANCHO, 2.9), sharey=False)
    x = np.arange(len(E.ESC))
    for j, (ax, med) in enumerate(zip(axs, E.MEDIDA)):
        for k, ex in enumerate(E.EXPL):
            mu, lo, hi = [], [], []
            for sc in E.ESC:
                v = m[(m.explainer == ex) & (m.scenario == sc)][med].dropna().values
                b = rng.choice(v, (2000, len(v))).mean(1)
                mu.append(v.mean())
                lo.append(np.quantile(b, 0.025))
                hi.append(np.quantile(b, 0.975))
            mu, lo, hi = map(np.array, (mu, lo, hi))
            ax.errorbar(x + (k - 1.5) * 0.08, mu, yerr=[mu - lo, hi - mu],
                        color=E.COLOR_EXPL[ex], marker=E.MARCA_EXPL[ex], ms=3, lw=0.9,
                        capsize=1.2, elinewidth=0.6, label=E.NOMBRE_EXPL[ex])
        ax.set_xticks(x, [E.NOMBRE_ESC[s].replace(" + SMOTE", "\n+SMOTE") for s in E.ESC],
                      rotation=90, fontsize=7)
        ax.set_title(f"({'abc'[j]}) " + E.NOMBRE_MEDIDA[med], fontsize=8.5)
        ax.set_ylim(0.5 if med != "sp_all" else 0.88, 1.01)
        E.coma(ax, "y", dec=2)
    axs[0].set_ylabel("Estabilidad entre réplicas")
    h, l = axs[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.09, right=0.99, top=0.92, bottom=0.33, wspace=0.28)
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    return E.guardar(fig, "estabilidad_por_escenario")


def fig_estabilidad_arquitectura():
    """H2: GNNExplainer, pesos por clase, escenarios principales; un punto por semilla."""
    m = _modelos()
    m = m[(m.balancing == "class_weighting") & (m.explainer == "GNNExplainer")
          & m.scenario.isin(["1:10", "1:10_os", "1:20", "native"])]
    s = m.groupby(["arch", "seed"])[E.MEDIDA].mean().reset_index()
    fig, axs = plt.subplots(1, 3, figsize=(E.ANCHO, 2.2))
    x = np.arange(len(E.ARQ))
    for j, (ax, med) in enumerate(zip(axs, E.MEDIDA)):
        for i, arch in enumerate(E.ARQ):
            v = s[s.arch == arch][med].values
            ax.plot(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v,
                    marker=E.MARCA_ARQ[arch], ls="none", color=E.COLOR_ARQ[arch], ms=4)
            ax.hlines(v.mean(), i - 0.25, i + 0.25, color="k", lw=1)
        ax.set_xticks(x, ["GCN", "Graph-\nSAGE", "GAT", "TAGCN"], fontsize=7.5)
        ax.set_title(f"({'abc'[j]}) " + E.NOMBRE_MEDIDA[med], fontsize=8.5)
        E.coma(ax, "y")
    axs[0].set_ylabel("Estabilidad (GNNExplainer)")
    fig.tight_layout(w_pad=1)
    return E.guardar(fig, "estabilidad_por_arquitectura")


def fig_control():
    """Control de pesos al azar: acuerdo entre la explicación del modelo entrenado y la de la
    misma arquitectura sin entrenar, junto a la estabilidad entre réplicas."""
    c = pd.read_csv(ROB / "control_aleatorio.csv")
    g = c.groupby("explainer").mean(numeric_only=True).reindex(E.EXPL)
    fig, axs = plt.subplots(1, 2, figsize=(E.ANCHO, 2.7), sharey=True)
    x = np.arange(len(E.EXPL))
    w = 0.26
    pares = [(("estab_entrenado", "estab_azar", "rho_entrenado_azar"), "Spearman (165 variables)"),
             (("jac10_estab_entrenado", "jac10_estab_azar", "jac10_entrenado_azar"),
              "Jaccard top-10")]
    nombres = ["Estabilidad, red entrenada", "Estabilidad, red al azar",
               "Acuerdo entrenada frente a azar"]
    cols = [E.OKABE["azul"], E.OKABE["celeste"], E.OKABE["bermellon"]]
    for j, (ax, (cols_d, tit)) in enumerate(zip(axs, pares)):
        for i, (k, n, cc) in enumerate(zip(cols_d, nombres, cols)):
            ax.bar(x + (i - 1) * w, g[k], w, color=cc, label=n)
        ax.set_xticks(x, ["GNN-\nExplainer", "Shapley", "Integrated\nGradients",
                          "Expected\nGradients"], fontsize=7)
        ax.set_title(f"({'ab'[j]}) " + tit, fontsize=8.5)
        ax.set_ylim(0, 1.05)
        E.coma(ax, "y")
    axs[0].set_ylabel("Valor medio (12 modelos)")
    h, l = axs[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.09, right=0.99, top=0.92, bottom=0.30, wspace=0.08)
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    return E.guardar(fig, "control_pesos_azar")


def fig_acuerdo_explicadores():
    """Acuerdo entre explicadores sobre el mismo modelo (3 arquitecturas, escenarios
    principales), con Spearman global y con Jaccard top-10."""
    o = pd.read_csv(ROB / "robusta_otros.csv")
    o = o[o.tipo == "explicadores"]
    fig, axs = plt.subplots(1, 2, figsize=(E.ANCHO, 2.9))
    n = len(E.EXPL)
    for j, (ax, med) in enumerate(zip(axs, ["sp_all", "jac10"])):
        M = np.eye(n)
        for a in range(n):
            for b in range(n):
                if a != b:
                    d = o[((o.a == E.EXPL[a]) & (o.b == E.EXPL[b]))
                          | ((o.a == E.EXPL[b]) & (o.b == E.EXPL[a]))]
                    M[a, b] = d[med].mean()
        im = ax.imshow(M, vmin=0, vmax=1, cmap="Blues")
        for a in range(n):
            for b in range(n):
                ax.text(b, a, f"{M[a, b]:.2f}".replace(".", ","), ha="center", va="center",
                        fontsize=7.5, color="white" if M[a, b] > 0.6 else "k")
        etis = ["GNN-\nExplainer", "Shapley", "Integrated\nGradients", "Expected\nGradients"]
        ax.set_xticks(range(n), etis, fontsize=7)
        ax.set_yticks(range(n), etis if j == 0 else [""] * n, fontsize=7)
        ax.grid(False)
        ax.set_title(f"({'ab'[j]}) " + E.NOMBRE_MEDIDA[med], fontsize=8.5)
    fig.subplots_adjust(left=0.13, right=0.88, top=0.92, bottom=0.14, wspace=0.06)
    cb = fig.colorbar(im, ax=axs, shrink=0.85, pad=0.02)
    E.coma(cb.ax, "y")
    return E.guardar(fig, "acuerdo_explicadores")


FIGURAS = {"particion": fig_particion, "curvas_test": lambda: fig_curvas("test"),
           "curvas_val": lambda: fig_curvas("val"), "cierre": fig_cierre,
           "perdida": fig_perdida_escenario, "h1": fig_estabilidad_escenario,
           "h2": fig_estabilidad_arquitectura, "control": fig_control,
           "explicadores": fig_acuerdo_explicadores}


def main() -> int:
    E.aplicar()
    pedidas = sys.argv[1:] or list(FIGURAS)
    for k in pedidas:
        try:
            print(k, "->", FIGURAS[k]())
        except FileNotFoundError as e:
            print(k, "-> falta", e.filename)
    return 0


if __name__ == "__main__":
    sys.exit(main())
