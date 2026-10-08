"""
Tabla plana de PR-AUC y ROC-AUC por configuración (media ± sd de 3 semillas), pedida por
Cristian el 04-oct. Lee curvas_metricas_por_config.csv (salida de curvas.py) y la compuerta
de configuración de configs48/configs48_metricas.csv. Escribe PR_AUC_todas.{csv,md} en
results_v4/reunion_0410/curvas/. Solo CPU.

Uso: ~/.local/bin/uv run --frozen python scripts/v4/reunion_0410/pr_auc_tabla.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CUR = ROOT / "results_v4" / "reunion_0410" / "curvas"
CFG = ROOT / "results_v4" / "reunion_0410" / "configs48" / "configs48_metricas.csv"
SCENS = ["native", "1:10", "1:10_os", "1:20", "1:1"]


def ms(m, s, d=3):
    return f"{m:.{d}f} ± {s:.{d}f}"


def main():
    c = pd.read_csv(CUR / "curvas_metricas_por_config.csv")
    g = pd.read_csv(CFG)[["scenario", "arch", "balancing", "g_cfg_mean"]]
    w = c.pivot_table(index=["scenario", "arch", "balancing"], columns="split",
                      values=["pr_auc_mean", "pr_auc_std", "roc_auc_mean", "roc_auc_std",
                              "n_gate_passed", "prevalence_mean"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index().merge(g, on=["scenario", "arch", "balancing"], how="left")
    w["_o"] = w.scenario.map({s: i for i, s in enumerate(SCENS)})
    w = w.sort_values(["_o", "arch", "balancing"]).drop(columns="_o")
    out = pd.DataFrame({
        "scenario": w.scenario, "arch": w.arch, "balancing": w.balancing,
        "PR-AUC val": [ms(a, b) for a, b in zip(w.pr_auc_mean_val, w.pr_auc_std_val)],
        "PR-AUC test": [ms(a, b) for a, b in zip(w.pr_auc_mean_test, w.pr_auc_std_test)],
        "ROC-AUC val": [ms(a, b) for a, b in zip(w.roc_auc_mean_val, w.roc_auc_std_val)],
        "ROC-AUC test": [ms(a, b) for a, b in zip(w.roc_auc_mean_test, w.roc_auc_std_test)],
        "compuerta": [f"{int(n)}/3" for n in w.n_gate_passed_val],
        "compuerta config": ["pasa" if x else "no" for x in w.g_cfg_mean],
    })
    out.to_csv(CUR / "PR_AUC_todas.csv", index=False)
    pv, pt = w.prevalence_mean_val.iloc[0], w.prevalence_mean_test.iloc[0]
    n_main = int(w[w.scenario != "1:1"].g_cfg_mean.sum())
    lines = [
        f"# PR-AUC de las {len(out)} configuraciones (media ± sd de 3 semillas)", "",
        f"Azar: {pv:.3f} en validación y {pt:.4f} en test".replace(".", ",") + f" (la prevalencia). Curvas completas "
        "en `figuras/curvas_*_{val,test}.png`. Las 48 principales son native, 1:10, 1:10_os y "
        "1:20; el 1:1 va al anexo. «compuerta» = semillas que pasan la compuerta por modelo; "
        "«compuerta config» = la media de las 3 semillas pasa (val F1 ≥ 0,30 y MCC ≥ 0,15). "
        f"Pasan {n_main} de las 48 principales.", "",
        "| " + " | ".join(out.columns) + " |",
        "|" + "---|" * len(out.columns)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in out.itertuples(index=False)]
    (CUR / "PR_AUC_todas.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(out)} configuraciones; pasan {n_main}/48 principales")


if __name__ == "__main__":
    main()
