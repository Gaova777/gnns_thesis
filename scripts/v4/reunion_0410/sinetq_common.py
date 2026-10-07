"""Utilidades comunes del análisis «¿por qué B marca a los sin etiqueta?» (reunión 04-oct).

Todo es de solo lectura sobre los artefactos v4: carga Elliptic con el loader del proyecto,
reconstruye los modelos desde sus meta.json y escribe solo en
results_v4/reunion_0410/sin_etiqueta/.

Definición de «marcado» (idéntica a scripts/train_matrix.py::cross_label_eval):
  p = softmax(model(x, edge_index))[:, 1] sobre el grafo completo, en modo eval,
  marcado  <=>  p >= meta["calibrated_threshold"] (umbral calibrado en la validación propia).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.loader import ILLICIT, LICIT, UNKNOWN, apply_label_mode, load_elliptic  # noqa: E402
from src.data.preprocessing import preprocess  # noqa: E402
from src.training.trainer import build_model  # noqa: E402

OUT = ROOT / "results_v4" / "reunion_0410" / "sin_etiqueta"
FIG = OUT / "figuras"
SPLITS = {"train": (1, 34), "val": (35, 42), "test": (43, 49)}
MODEL_DIRS = {
    "B": ROOT / "results_models_v4_labels" / "licit",
    "A": ROOT / "results_models_v4_labels" / "unknown",
    "C": ROOT / "results_models_v4",
}
GROUP_NAMES = {LICIT: "licita", ILLICIT: "ilicita", UNKNOWN: "sin_etiqueta"}

# Nombres de las 165 columnas de data.x (PyG descarta txId y el timestep):
# columnas 2..94 del CSV = 93 locales (L01..L93); 95..166 = 72 agregadas (A01..A72).
FEATURE_NAMES = [f"L{k:02d}" for k in range(1, 94)] + [f"A{k:02d}" for k in range(1, 73)]


def ensure_dirs() -> None:
    FIG.mkdir(parents=True, exist_ok=True)


def load_data():
    """Devuelve (data, x_raw). data.x queda normalizado como en el entrenamiento
    (RobustScaler ajustado con todos los nodos de ts 1-34 + clip ±10), idéntico en A/B/C."""
    data = load_elliptic(root=str(ROOT / "data"))
    x_raw = data.x.clone().numpy()
    apply_label_mode(data, "licit")  # y3 (verdad de 3 clases) no depende del modo
    preprocess(data, train_range=SPLITS["train"], val_range=SPLITS["val"],
               test_range=SPLITS["test"])
    assert data.x.shape[1] == len(FEATURE_NAMES), data.x.shape
    return data, x_raw


def split_of(ts: np.ndarray) -> np.ndarray:
    s = np.full(ts.shape, "", dtype=object)
    for name, (lo, hi) in SPLITS.items():
        s[(ts >= lo) & (ts <= hi)] = name
    return s


def list_metas(mode: str, seeds=(42,), scenario: str = "native") -> list[dict]:
    out = []
    for p in sorted(MODEL_DIRS[mode].glob("*_meta.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        if m.get("scenario") != scenario or int(m.get("seed", -1)) not in seeds:
            continue
        m["_dir"] = str(MODEL_DIRS[mode])
        m["_mode"] = mode
        out.append(m)
    return out


def build_from_meta(meta: dict, in_channels: int, device: str):
    bp = meta.get("best_params", {}) or {}
    kw = {}
    if meta["architecture"] == "GAT" and "heads" in bp:
        kw["heads"] = bp["heads"]
    if meta["architecture"] == "TAGCN" and "K" in bp:
        kw["K"] = bp["K"]
    model = build_model(meta["architecture"], in_channels=in_channels,
                        hidden_channels=bp.get("hidden_dim", 128),
                        num_layers=bp.get("num_layers", 2),
                        dropout=bp.get("dropout", 0.3), **kw)
    ckpt = Path(meta["_dir"]) / meta["checkpoint"]
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def pick_device(min_free_gib: float = 3.0) -> str:
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free / 2**30 >= min_free_gib:
            return "cuda"
    return "cpu"


def psi(ref: np.ndarray, act: np.ndarray, n_bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index con cortes en los deciles de la referencia.

    PSI = sum_i (a_i - e_i) * ln(a_i / e_i). Los cortes repetidos (masas puntuales,
    frecuentes en Elliptic) se colapsan; los bins vacíos se suavizan con eps.
    """
    ref = ref[np.isfinite(ref)]
    act = act[np.isfinite(act)]
    qs = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)[1:-1]))
    e = np.bincount(np.searchsorted(qs, ref, side="right"), minlength=len(qs) + 1) / len(ref)
    a = np.bincount(np.searchsorted(qs, act, side="right"), minlength=len(qs) + 1) / len(act)
    e = np.clip(e, eps, None)
    a = np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def fmt_pct(x: float, nd: int = 2) -> str:
    return f"{100 * x:.{nd}f}".replace(".", ",") + " %"


def fmt_num(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def fmt_int(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


BEST_B = "B|GraphSAGE|none|42"   # mejor B por PR-AUC de validación (0,859)
GROUP4 = ["licita", "ilicita", "sin_etiqueta_marcado", "sin_etiqueta_no_marcado"]
GROUP4_LABEL = {"licita": "Lícita", "ilicita": "Ilícita",
                "sin_etiqueta_marcado": "Sin etiqueta marcado",
                "sin_etiqueta_no_marcado": "Sin etiqueta no marcado"}
GROUP4_COLOR = {"licita": "#2a7ab9", "ilicita": "#d1495b",
                "sin_etiqueta_marcado": "#edae49", "sin_etiqueta_no_marcado": "#8d99ae"}


def load_preds():
    """Devuelve (nodos DataFrame, modelos DataFrame, probs [M x N], flags [M x N])."""
    import pandas as pd
    nodos = pd.read_csv(OUT / "nodos.csv.gz")
    modelos = pd.read_csv(OUT / "modelos.csv")
    z = np.load(OUT / "probs.npz", allow_pickle=True)
    probs = z["probs"]
    assert list(z["keys"]) == list(modelos["key"])
    flags = probs >= modelos["threshold"].values[:, None]
    return nodos, modelos, probs, flags


def group4(nodos, flag: np.ndarray) -> np.ndarray:
    g = nodos["group"].values.astype(object).copy()
    unk = g == "sin_etiqueta"
    g[unk & flag] = "sin_etiqueta_marcado"
    g[unk & ~flag] = "sin_etiqueta_no_marcado"
    return g
