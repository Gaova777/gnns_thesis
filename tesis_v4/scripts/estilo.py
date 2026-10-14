"""Estilo común de las figuras del manuscrito (no de las diapositivas).

- Letra serif Nimbus Roman, clon métrico de Times, la misma familia que usa el texto
  (mathptmx), a 9 pt para que en la página se lea igual que una nota al pie.
- Ancho de página de 6,27 in (A4 con márgenes de 1 in); media página 3,05 in.
- Paleta Okabe e Ito (legible con daltonismo y en escala de grises por el tipo de línea).
- Salida PDF vectorial con fuentes incrustadas (Type 42), sin título dentro de la figura:
  el título va en el \\caption.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ANCHO = 6.27
MEDIO = 3.05
FIG = Path(__file__).resolve().parents[1] / "figuras"

OKABE = {"negro": "#000000", "naranja": "#E69F00", "celeste": "#56B4E9", "verde": "#009E73",
         "amarillo": "#F0E442", "azul": "#0072B2", "bermellon": "#D55E00", "purpura": "#CC79A7",
         "gris": "#7F7F7F"}

ARQ = ["GCN", "GraphSAGE", "GAT", "TAGCN"]
COLOR_ARQ = {"GCN": OKABE["gris"], "GraphSAGE": OKABE["azul"], "GAT": OKABE["bermellon"],
             "TAGCN": OKABE["verde"]}
LINEA_ARQ = {"GCN": ":", "GraphSAGE": "-", "GAT": "--", "TAGCN": "-."}
MARCA_ARQ = {"GCN": "s", "GraphSAGE": "o", "GAT": "^", "TAGCN": "D"}

PERDIDA = ["class_weighting", "focal_loss", "none"]
NOMBRE_PERDIDA = {"class_weighting": "Pesos por clase", "focal_loss": "Focal loss",
                  "none": "Sin ajuste"}
COLOR_PERDIDA = {"class_weighting": OKABE["azul"], "focal_loss": OKABE["naranja"],
                 "none": OKABE["gris"]}
LINEA_PERDIDA = {"class_weighting": "-", "focal_loss": "--", "none": ":"}

EXPL = ["GNNExplainer", "ShapleyFeatures", "IntegratedGradients", "ExpectedGradients"]
NOMBRE_EXPL = {"GNNExplainer": "GNNExplainer", "ShapleyFeatures": "Shapley",
               "IntegratedGradients": "Integrated Gradients",
               "ExpectedGradients": "Expected Gradients"}
COLOR_EXPL = {"GNNExplainer": OKABE["azul"], "ShapleyFeatures": OKABE["naranja"],
              "IntegratedGradients": OKABE["verde"], "ExpectedGradients": OKABE["purpura"]}
MARCA_EXPL = {"GNNExplainer": "o", "ShapleyFeatures": "s", "IntegratedGradients": "^",
              "ExpectedGradients": "D"}

ESC = ["1:10", "1:10_os", "1:20", "native", "1:100_subil", "1:200_subil"]
NOMBRE_ESC = {"1:10": "1:10", "1:10_os": "1:10 + SMOTE", "1:20": "1:20",
              "native": "Nativo", "1:100_subil": "1:100", "1:200_subil": "1:200"}

MEDIDA = ["sp_all", "sp_nz", "jac10"]
NOMBRE_MEDIDA = {"sp_all": r"Spearman (165 variables)",
                 "sp_nz": r"Spearman (variables $\neq 0$)",
                 "jac10": r"Jaccard top-10"}


def aplicar() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.linewidth": 0.4, "grid.color": "#D9D9D9",
        "axes.axisbelow": True, "lines.linewidth": 1.2, "lines.markersize": 4,
        "legend.frameon": False, "legend.handlelength": 2.2,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.formatter.use_locale": False,
    })


def coma(ax, ejes: str = "xy", dec: int | None = None) -> None:
    """Separador decimal con coma, como en el texto en español."""
    def f(v, _):
        s = f"{v:.{dec}f}" if dec is not None else f"{v:g}"
        return s.replace(".", ",")
    if "x" in ejes:
        ax.xaxis.set_major_formatter(FuncFormatter(f))
    if "y" in ejes:
        ax.yaxis.set_major_formatter(FuncFormatter(f))


def letra(ax, s: str) -> None:
    """Letra de panel (a), (b)... arriba a la izquierda, fuera del área de datos."""
    ax.text(-0.02, 1.04, f"({s})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontweight="bold")


def guardar(fig, nombre: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / f"{nombre}.pdf"
    fig.savefig(p)
    fig.savefig(FIG / f"{nombre}.png", dpi=200)
    plt.close(fig)
    return p
