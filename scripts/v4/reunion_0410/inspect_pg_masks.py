"""Inspección de las máscaras de aristas de PGExplainer guardadas en results_v4/rankings.

Solo lee. Para cada modelo calcula, sobre los nodos con >= 2 aristas:
  * rango de valores de la máscara, |diferencia| máxima entre aristas de un mismo nodo y réplica
    (si es ~1e-6 el orden es ruido numérico),
  * fracción de máscaras saturadas (> 0,99 o < 0,01),
  * acuerdo del orden de aristas entre réplicas (el Spearman de 2 aristas es ±1).
Escribe results_v4/reunion_0410/pgexpl_gcn/pg_masks_summary.csv
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results_v4/reunion_0410/pgexpl_gcn"
rows = []
for f in sorted(glob.glob(str(ROOT / "results_v4/rankings/*__PGExplainer.npz"))):
    z = np.load(f, allow_pickle=True)
    ptr = z["edge_ptr"]; edge = z["edge"]; R = int(z["n_replicas"])
    nodes = z["nodes"]
    gaps, vals, spread_rep, n2 = [], [], [], 0
    for i in range(len(nodes)):
        a, b = int(ptr[i]), int(ptr[i + 1])
        if b - a < 2:
            continue
        m = edge[:, a:b] if edge.shape[0] == R else edge[a:b].T
        n2 += 1
        vals.append(m.ravel())
        # diferencia máxima entre aristas dentro de cada réplica
        gaps.append(np.ptp(m, axis=1))
        # variación de la misma arista entre réplicas
        spread_rep.append(np.ptp(m, axis=0))
    if not vals:
        continue
    v = np.concatenate(vals); g = np.concatenate(gaps); s = np.concatenate(spread_rep)
    rows.append(dict(run_id=str(z["run_id"]), arch=str(z["arch"]), scenario=str(z["scenario"]),
                     balancing=str(z["balancing"]), seed=int(z["seed"]), nodes_2plus=n2,
                     mask_mean=v.mean(), mask_min=v.min(), mask_max=v.max(),
                     frac_sat_hi=(v > 0.99).mean(), frac_sat_lo=(v < 0.01).mean(),
                     within_gap_median=np.median(g), within_gap_p90=np.quantile(g, .9),
                     frac_gap_lt_1e3=(g < 1e-3).mean(), frac_gap_lt_1e5=(g < 1e-5).mean(),
                     between_rep_spread_median=np.median(s)))
df = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / "pg_masks_summary.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
cols = ["mask_mean", "frac_sat_hi", "frac_sat_lo", "within_gap_median", "frac_gap_lt_1e3",
        "frac_gap_lt_1e5", "between_rep_spread_median"]
print(df.groupby("balancing")[cols].mean().round(4))
print(df.groupby(["arch", "balancing"])[cols].mean().round(4))
