#!/usr/bin/env python
"""Paso 0 del plan v4: comparar los modos de etiqueta A / B / C (solo entrenamiento).

Pregunta (reunión con Cristian, 28-sep-2026): ¿se pueden tratar los nodos sin etiqueta como
lícitos? Si sí, todo el análisis de estabilidad corre sobre C («completo»: negativos =
lícitas + sin etiqueta), que es el caso más real y conserva el desbalance real.

Entrada: los meta.json del escenario nativo (semilla 42) de cada modo. Cada modelo ya trae
``cross_label_eval`` (scripts/train_matrix.py): sus métricas en VARAS COMUNES, que no
dependen del modo con que se entrenó:
  vs_licit    ilícitas vs lícitas revisadas     (la tarea de B)
  vs_unknown  ilícitas vs sin etiqueta          (la tarea de A, v3)
  vs_all      ilícitas vs lícitas + sin etiqueta (la tarea de C)
  flag_rate   fracción de cada clase marcada como ilícita con el umbral de validación

Lo que se mira (propuesta; se discute con Cristian antes de fijarlo):
  E1  En la vara común vs_licit (test), C rinde parecido a B: |mediana(C − B)| de PR-AUC
      ≤ 0,05 sobre las 12 configuraciones pareadas (arquitectura × balanceo).
      → Tratar los sin etiqueta como lícitos no le quita al modelo la capacidad de separar
        ilícitas de lícitas reales.
  E2  Los modelos B (que nunca vieron un sin etiqueta) marcan como ilícitos a una fracción de
      los sin etiqueta parecida a la de las lícitas (y muy lejos de la de las ilícitas).
      → Vistos por un modelo que no los conoce, los sin etiqueta se comportan como lícitos.
  E3  A y C coinciden (A ≈ C en vs_all): los 26.432 lícitos que A deja fuera no cambian la
      frontera, porque los sin etiqueta ya se parecen a ellos.
Salida: label_comparison.csv (una fila por modelo) y label_comparison.md (resumen).

Uso:
  uv run --frozen python scripts/v4/compare_label_modes.py \
     --models licit_unknown=results_models_v4 --models licit=results_models_v4_labels/licit \
     --models unknown=results_models_v4_labels/unknown --out-dir results_v4_labels
  uv run --frozen python scripts/v4/compare_label_modes.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

MODE_NAME = {"licit_unknown": "C (lícitas + sin etiqueta)", "licit": "B (lícitas)",
             "unknown": "A (sin etiqueta, v3)"}
MODE_ORDER = ["licit_unknown", "licit", "unknown"]
E1_MARGIN = 0.05


def load_rows(models: dict[str, Path], scenario: str, seed: int) -> pd.DataFrame:
    rows = []
    for mode, d in models.items():
        for p in sorted(Path(d).glob("*_meta.json")):
            m = json.loads(p.read_text(encoding="utf-8"))
            if m.get("scenario") != scenario or int(m.get("seed", -1)) != seed:
                continue
            got_mode = m.get("label_mode")
            if got_mode != mode:
                raise SystemExit(f"{p}: label_mode {got_mode!r} != {mode!r} (¿directorio cruzado?)")
            ce = m.get("cross_label_eval")
            if not ce:
                raise SystemExit(f"{p}: sin cross_label_eval (¿entrenado antes del 28-sep?)")
            r = {"mode": mode, "run_id": m["run_id"], "arch": m["architecture"],
                 "balancing": m["balancing"], "gate_passed": m.get("quality_passed"),
                 "own_val_pr_auc": m["val_metrics"]["pr_auc"],
                 "own_test_pr_auc": m["test_metrics"]["pr_auc"]}
            for split in ("val", "test"):
                for yard in ("vs_licit", "vs_unknown", "vs_all"):
                    for k in ("pr_auc", "f1", "recall", "precision"):
                        r[f"{split}_{yard}_{k}"] = ce[split][yard][k]
                for cls in ("illicit", "licit", "unknown"):
                    r[f"{split}_flag_{cls}"] = ce[split]["flag_rate"][cls]
                    r[f"{split}_meanp_{cls}"] = ce[split]["mean_prob"][cls]
            rows.append(r)
    return pd.DataFrame(rows)


def _paired(df: pd.DataFrame, a: str, b: str, col: str) -> dict:
    pa = df[df["mode"] == a].set_index(["arch", "balancing"])[col]
    pb = df[df["mode"] == b].set_index(["arch", "balancing"])[col]
    both = pd.concat([pa.rename("a"), pb.rename("b")], axis=1).dropna()
    if both.empty:
        return {"n": 0}
    diff = both["a"] - both["b"]
    out = {"n": int(len(diff)), "median_diff": float(diff.median()),
           "mean_diff": float(diff.mean()), "min": float(diff.min()), "max": float(diff.max())}
    if len(diff) >= 5 and (diff != 0).any():
        from scipy.stats import wilcoxon
        out["wilcoxon_p"] = float(wilcoxon(both["a"], both["b"]).pvalue)
    return out


def summarize(df: pd.DataFrame) -> str:
    L = ["# Paso 0 — ¿los sin etiqueta se pueden tratar como lícitos?", ""]
    L.append("Escenario nativo, semilla 42. Varas comunes en **test** (ts 43-49); el umbral de "
             "cada modelo se calibró en su propia validación.")
    L.append("")
    L.append("| Modo | n | PR-AUC vs lícitas | PR-AUC vs sin etiqueta | PR-AUC vs todo | "
             "recall ilícitas | marcadas: lícitas | marcadas: sin etiqueta |")
    L.append("|---|---|---|---|---|---|---|---|")
    for mode in MODE_ORDER:
        d = df[df["mode"] == mode]
        if d.empty:
            continue
        L.append(f"| {MODE_NAME[mode]} | {len(d)} | {d.test_vs_licit_pr_auc.median():.3f} | "
                 f"{d.test_vs_unknown_pr_auc.median():.3f} | {d.test_vs_all_pr_auc.median():.3f} | "
                 f"{d.test_vs_all_recall.median():.3f} | {d.test_flag_licit.median():.2%} | "
                 f"{d.test_flag_unknown.median():.2%} |")
    L.append("")
    L.append("_Medianas sobre las configuraciones (arquitectura × balanceo)._")
    L.append("")

    L.append("## E1 · C rinde como B separando ilícitas de lícitas reales")
    e1 = _paired(df, "licit_unknown", "licit", "test_vs_licit_pr_auc")
    if e1.get("n"):
        ok = abs(e1["median_diff"]) <= E1_MARGIN
        L.append(f"- Mediana(C − B) de PR-AUC vs lícitas = {e1['median_diff']:+.3f} "
                 f"(rango {e1['min']:+.3f} a {e1['max']:+.3f}, n = {e1['n']}"
                 + (f", Wilcoxon p = {e1['wilcoxon_p']:.3f}" if "wilcoxon_p" in e1 else "")
                 + f"). Criterio |mediana| ≤ {E1_MARGIN}: **{'se cumple' if ok else 'NO se cumple'}**.")
    else:
        L.append("- Faltan modelos de C o de B.")
    L.append("")

    L.append("## E2 · Un modelo que nunca vio los sin etiqueta los trata como lícitos")
    b = df[df["mode"] == "licit"]
    if not b.empty:
        fl, fu, fi = (b.test_flag_licit.median(), b.test_flag_unknown.median(),
                      b.test_flag_illicit.median())
        L.append(f"- Modelos B, fracción marcada como ilícita en test: lícitas {fl:.2%} · "
                 f"sin etiqueta {fu:.2%} · ilícitas {fi:.2%}.")
        L.append(f"- Probabilidad media de ilícita: lícitas {b.test_meanp_licit.median():.3f} · "
                 f"sin etiqueta {b.test_meanp_unknown.median():.3f} · "
                 f"ilícitas {b.test_meanp_illicit.median():.3f}.")
        L.append("- Lectura: si «sin etiqueta» queda cerca de «lícitas» y lejos de «ilícitas», "
                 "los sin etiqueta son en su mayoría lícitos; el exceso sobre las lícitas "
                 "acota el fraude no detectado que esconden.")
    else:
        L.append("- Faltan modelos de B.")
    L.append("")

    L.append("## E3 · A y C llegan a lo mismo")
    e3 = _paired(df, "licit_unknown", "unknown", "test_vs_all_pr_auc")
    if e3.get("n"):
        L.append(f"- Mediana(C − A) de PR-AUC vs todo = {e3['median_diff']:+.3f} "
                 f"(rango {e3['min']:+.3f} a {e3['max']:+.3f}, n = {e3['n']}).")
    else:
        L.append("- Faltan modelos de C o de A.")
    L.append("")
    L.append("## Compuerta de calidad (validación propia de cada modo)")
    for mode in MODE_ORDER:
        d = df[df["mode"] == mode]
        if not d.empty:
            L.append(f"- {MODE_NAME[mode]}: {int(d.gate_passed.sum())}/{len(d)} pasan.")
    return "\n".join(L) + "\n"


def run(models: dict[str, Path], out_dir: Path, scenario: str = "native", seed: int = 42) -> str:
    df = load_rows(models, scenario, seed)
    if df.empty:
        raise SystemExit("No hay meta.json que comparar.")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "label_comparison.csv", index=False)
    md = summarize(df)
    (out_dir / "label_comparison.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"-> {out_dir / 'label_comparison.csv'} · {out_dir / 'label_comparison.md'}")
    return md


def _selftest() -> None:
    rng = np.random.default_rng(0)
    tmp = Path(tempfile.mkdtemp(prefix="label_cmp_selftest_"))
    models = {}
    for mode in MODE_ORDER:
        d = tmp / mode
        d.mkdir()
        models[mode] = d
        for arch in ("GCN", "GraphSAGE", "GAT", "TAGCN"):
            for bal in ("none", "class_weighting", "focal_loss"):
                def yard():
                    return {"pr_auc": float(rng.uniform(.5, .8)), "f1": .5, "recall": .6,
                            "precision": .5, "mcc": .4, "n_illicit": 10, "n_negative": 100}
                split = {"vs_licit": yard(), "vs_unknown": yard(), "vs_all": yard(),
                         "flag_rate": {"illicit": .6, "licit": .02, "unknown": .03},
                         "mean_prob": {"illicit": .7, "licit": .05, "unknown": .06}}
                meta = {"run_id": f"native_{arch}_{bal}", "scenario": "native", "seed": 42,
                        "label_mode": mode, "architecture": arch, "balancing": bal,
                        "quality_passed": True, "val_metrics": {"pr_auc": .7},
                        "test_metrics": {"pr_auc": .6},
                        "cross_label_eval": {"threshold": .5, "val": split, "test": split}}
                (d / f"native_{arch}_{bal}_meta.json").write_text(json.dumps(meta))
    md = run(models, tmp / "out")
    assert "E1" in md and "B (lícitas)" in md and (tmp / "out" / "label_comparison.csv").exists()
    print(f"SELFTEST OK ({tmp})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--models", action="append", default=[],
                   help="modo=directorio_de_modelos (repetible)")
    p.add_argument("--out-dir", default="results_v4_labels")
    p.add_argument("--scenario", default="native")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        _selftest()
        return 0
    models = {}
    for spec in a.models:
        mode, _, d = spec.partition("=")
        if mode not in MODE_NAME or not d:
            p.error(f"--models {spec!r}: usa modo=dir con modo en {list(MODE_NAME)}")
        models[mode] = Path(d)
    if not models:
        p.error("falta --models")
    run(models, Path(a.out_dir), a.scenario, a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
