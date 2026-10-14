"""Piso de Spearman que fija el bloque de variables nulas (corrida sin el cierre).

Para cada una de las 30 transacciones explicadas se simulan 200 pares de explicaciones que solo
coinciden en dejar empatadas al final las variables que valen 0 tras el escalado y ordenan las
demás al azar. La media del Spearman entre esos pares es el acuerdo que la medida global da sin
que las explicaciones compartan nada más. Uso: uv run --frozen python tesis_v4/scripts/piso_spearman.py
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.loader import apply_label_mode, load_elliptic  # noqa: E402
from src.data.preprocessing import preprocess  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    d = load_elliptic(root=str(ROOT / "data"))
    apply_label_mode(d, "licit_unknown")
    preprocess(d, train_range=(1, 29), val_range=(30, 35), test_range=(36, 42))
nodes = json.loads((ROOT / "results_v4_precierre/explain_nodes_v4.json").read_text())["nodes"]
rng = np.random.default_rng(0)
piso, ceros = [], []
for n in nodes:
    nz = (d.x[n] != 0).numpy()
    ceros.append(1 - nz.mean())
    v = [spearmanr(np.where(nz, rng.random(nz.size) + 1, 0.0),
                   np.where(nz, rng.random(nz.size) + 1, 0.0)).statistic for _ in range(200)]
    piso.append(np.mean(v))
print(f"fracción de variables nulas: media {np.mean(ceros):.3f} (de {min(ceros):.3f} a {max(ceros):.3f})")
print(f"piso de Spearman: media {np.mean(piso):.3f} (de {min(piso):.3f} a {max(piso):.3f})")
