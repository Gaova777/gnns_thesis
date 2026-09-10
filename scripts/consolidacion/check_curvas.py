# -*- coding: utf-8 -*-
"""Comprobacion del volcado de curvas PR/ROC de reeval_rocauc.py.

No necesita checkpoints ni el dataset: valida `curve_points_on_grid` sobre
probabilidades sinteticas que imitan el desbalance de Elliptic (validacion con
prevalencia ~4%, test con ~0,4%). Sirve para verificar el codigo en una maquina
sin GPU antes de lanzar `reeval_rocauc.py --dump-curves` en la que si los tiene.

    uv run python scripts/consolidacion/check_curvas.py

Comprueba cuatro cosas:
  1. El area bajo la curva remuestreada coincide con el AP y el ROC-AUC reales.
  2. La curva PR arranca en precision 1 en recall 0 y termina en la prevalencia.
     Este es el punto que fallaba: sklearn emite dos puntos en recall 0 y
     `np.interp` se quedaba con el degenerado, dejando la curva boca abajo justo
     en el borde izquierdo, que es donde mira el ojo.
  3. La curva ROC es monotona no decreciente.
  4. Un split sin positivos devuelve NaN en lugar de reventar.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

AQUI = Path(__file__).resolve().parent


def cargar_funcion():
    """Extrae las dos funciones puras de reeval_rocauc.py sin importar torch."""
    src = (AQUI / "reeval_rocauc.py").read_text(encoding="utf-8")
    ini = src.index("def curve_points_on_grid")
    fin = src.index("@torch.no_grad()")
    ns = {}
    exec(
        "import numpy as np\n"
        "from sklearn.metrics import precision_recall_curve, roc_curve\n" + src[ini:fin],
        ns,
    )
    return ns["curve_points_on_grid"]


def main():
    f = cargar_funcion()
    rng = np.random.default_rng(0)
    fallos = []

    for etiqueta, n, prev, sep in [("validacion", 20_000, 0.045, 1.7),
                                   ("test", 40_000, 0.004, 0.35)]:
        y = (rng.random(n) < prev).astype(int)
        p = 1 / (1 + np.exp(-(rng.normal(0, 1, n) + y * sep)))
        grid, pr, roc = f(p, y)
        ap, ra = average_precision_score(y, p), roc_auc_score(y, p)
        area_pr, area_roc = np.trapezoid(pr, grid), np.trapezoid(roc, grid)
        prevalencia = y.mean()

        print(f"  {etiqueta}: AP={ap:.4f} (area {area_pr:.4f}) | "
              f"ROC={ra:.4f} (area {area_roc:.4f}) | "
              f"PR[recall=0]={pr[0]:.3f} PR[recall=1]={pr[-1]:.4f} "
              f"(prevalencia {prevalencia:.4f})")

        if abs(area_roc - ra) > 0.01:
            fallos.append(f"{etiqueta}: el area ROC remuestreada se aleja del ROC-AUC real")
        if abs(area_pr - ap) > 0.01:
            fallos.append(f"{etiqueta}: el area PR remuestreada se aleja del AP real")
        if pr[0] < 0.99:
            fallos.append(f"{etiqueta}: la curva PR no arranca en precision 1 en recall 0")
        if abs(pr[-1] - prevalencia) > 0.01:
            fallos.append(f"{etiqueta}: la curva PR no termina en la prevalencia")
        if not np.all(np.diff(roc) >= -1e-9):
            fallos.append(f"{etiqueta}: la curva ROC no es monotona")
        if np.isnan(pr).any() or np.isnan(roc).any():
            fallos.append(f"{etiqueta}: aparecieron NaN en un split valido")

    grid, pr, roc = f(np.array([0.1, 0.2, 0.3]), np.array([0, 0, 0]))
    print(f"  split sin positivos: NaN en las dos curvas = "
          f"{bool(np.isnan(pr).all() and np.isnan(roc).all())}")
    if not (np.isnan(pr).all() and np.isnan(roc).all()):
        fallos.append("un split sin positivos deberia devolver NaN")

    if fallos:
        print("\nFALLOS:")
        for x in fallos:
            print("  -", x)
        sys.exit(1)
    print("\nOK: el volcado de curvas es fiel a las metricas autoritativas.")


if __name__ == "__main__":
    main()
