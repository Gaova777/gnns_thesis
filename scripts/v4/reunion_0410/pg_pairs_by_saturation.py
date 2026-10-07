"""Descompone el Spearman de aristas de PGExplainer (results_v4/rankings) por tipo de par de
réplicas según la saturación de la máscara: ambas réplicas «bajas» (media < 0,5), ambas
«altas», o mixtas; y cuenta los pares descartados por vector constante (empate exacto en
float32, típicamente 1,0 = 1,0). Solo lee. Salida: pg_pairs_by_saturation.csv"""
import glob
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results_v4/reunion_0410/pgexpl_gcn"
rows = []
meta = pd.read_csv(ROOT / "results_v4/elliptic_v4_stability.csv")
gate = {(r.run_id): bool(r.gate_passed) for r in meta[meta.explainer == "PGExplainer"].itertuples()}
for f in sorted(glob.glob(str(ROOT / "results_v4/rankings/*__PGExplainer.npz"))):
    z = np.load(f, allow_pickle=True)
    ptr, E = z["edge_ptr"], z["edge"]
    rid = str(z["run_id"])
    for i in range(len(z["nodes"])):
        a, b = int(ptr[i]), int(ptr[i + 1])
        if b - a < 2:
            continue
        M = E[:, a:b]
        for p, q in combinations(range(M.shape[0]), 2):
            hp, hq = M[p].mean() > 0.5, M[q].mean() > 0.5
            kind = "alta-alta" if hp and hq else ("baja-baja" if not hp and not hq else "mixta")
            const = np.ptp(M[p]) == 0 or np.ptp(M[q]) == 0
            rho = np.nan if const else stats.spearmanr(M[p], M[q]).statistic
            rows.append(dict(run_id=rid, arch=str(z["arch"]), scenario=str(z["scenario"]),
                             balancing=str(z["balancing"]), seed=int(z["seed"]),
                             gate=gate.get(rid), node=int(z["nodes"][i]), n_edges=b - a,
                             kind=kind, constant=const, rho=rho))
df = pd.DataFrame(rows)
df.to_csv(OUT / "pg_pairs_by_saturation.csv", index=False)
pd.set_option("display.width", 250)
for sub, name in ((df, "todos"), (df[df.gate == True], "compuerta")):  # noqa: E712
    print(f"== {name}")
    print(sub.groupby(["balancing", "kind"]).agg(n=("rho", "size"), frac_const=("constant", "mean"),
                                                 rho=("rho", "mean")).round(3).unstack())
    print(sub.groupby("balancing").agg(frac_const=("constant", "mean"), rho=("rho", "mean")).round(3))
