"""Prueba de ALINEAMIENTO modelo↔tipología del eje sintético v4.

Pregunta: ¿el modelo clasifica USANDO las aristas de la tipología? Si no, la plausibilidad
(¿el explicador señala la tipología?) confunde falla del explicador con falla del modelo: un
explicador fiel a un modelo que usa un atajo NO debe señalar la tipología.

Protocolo (por arquitectura; modelo entrenado sobre el grafo completo, CPU basta):
  (a) PR-AUC en test con el grafo completo;
  (b) PR-AUC en test quitando TODAS las aristas de tipología (typology_edge > 0, ambos sentidos);
  (c) PR-AUC en test quitando el MISMO número de aristas no dirigidas elegidas al azar
      (uniforme sobre todas las aristas; media de --random-reps sorteos).
  caída_b = a − b ; caída_c = a − c ; margen = caída_b − caída_c.
  Criterio «alineado»: margen ≥ --threshold (default 0,10 de PR-AUC).
El control (c) descuenta el efecto genérico de perder aristas (menos mensaje, grados distintos
en la propagación); lo que sobra es atribuible específicamente a la tipología.
Criterio estricto (``aligned_strict``): además, la habilidad residual sin tipología
  (b − prevalencia) / (a − prevalencia) ≤ 0,5, es decir, al menos la mitad de lo que el modelo
  sabe por encima del azar depende de las aristas de tipología. Motivo: en el grafo LEGADO el
  margen también sale ≥ 0,10 (quitar la tipología deja nodos casi aislados), pero el modelo
  conserva buena parte de su PR-AUC con los atajos de x; el criterio estricto lo delata.
Informativo: (d) PR-AUC de una regresión logística sobre x SIN grafo (si es alta, las features
del nodo bastan: atajo a nivel de grafo, independiente de la arquitectura).

Uso:
  uv run --frozen python phase1/alignment_check.py --archs GCN --epochs 150
  uv run --frozen python phase1/alignment_check.py --legacy     # grafo con atajos (control)
Salida: JSON (por arquitectura) en --out y resumen en stdout. Exit 0 siempre que corra; el
veredicto está en el JSON ("aligned": true/false).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v4_common as C  # noqa: E402


def undirected_pairs(ei: torch.Tensor):
    """Índices [0, E/2) si el grafo está simetrizado como cat([ei, ei.flip(0)]); si no, None."""
    E = ei.size(1)
    if E % 2 == 0 and torch.equal(ei[:, : E // 2].flip(0), ei[:, E // 2:]):
        return E // 2
    return None


def drop_random(ei: torch.Tensor, n_drop_und: int, rng: np.random.Generator):
    E = ei.size(1)
    half = undirected_pairs(ei)
    keep = torch.ones(E, dtype=torch.bool)
    if half is not None:
        idx = torch.as_tensor(rng.choice(half, size=min(n_drop_und, half), replace=False))
        keep[idx] = False; keep[idx + half] = False
    else:
        keep[torch.as_tensor(rng.choice(E, size=min(n_drop_und, E), replace=False))] = False
    return ei[:, keep]


def node_only_pr_auc(d) -> float:
    tr, te = d.train_mask.numpy(), d.test_mask.numpy()
    X, y = d.x.numpy(), d.y.numpy()
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[tr], y[tr])
    return float(average_precision_score(y[te], clf.predict_proba(X[te])[:, 1]))


def check_arch(d, arch, epochs, patience, reps, threshold, seed, device):
    t0 = time.time()
    model, m = C.train_model(d, arch, epochs=epochs, patience=patience, seed=seed, device=device)
    ei = d.edge_index
    typ = d.typology_edge > 0
    n_typ_directed = int(typ.sum())
    half = undirected_pairs(ei)
    n_drop_und = n_typ_directed // 2 if half is not None else n_typ_directed
    a = C.pr_auc(model, d, "test_mask", device=device)
    b = C.pr_auc(model, d, "test_mask", edge_index=ei[:, ~typ], device=device)
    rng = np.random.default_rng(seed)
    cs = [C.pr_auc(model, d, "test_mask", edge_index=drop_random(ei, n_drop_und, rng), device=device)
          for _ in range(reps)]
    c = float(np.mean(cs))
    drop_b, drop_c = a - b, a - c
    margin = drop_b - drop_c
    prev = float(d.y[d.test_mask].float().mean())
    resid = (b - prev) / (a - prev) if a - prev > 1e-6 else float("nan")
    return dict(architecture=arch, val_pr_auc=m["val_pr_auc"], epochs_run=m["epochs_run"],
                pr_auc_full=round(a, 4), pr_auc_no_typology=round(b, 4),
                pr_auc_random_drop=round(c, 4), pr_auc_random_drop_sd=round(float(np.std(cs)), 4),
                drop_typology=round(drop_b, 4), drop_random=round(drop_c, 4),
                margin=round(margin, 4), threshold=threshold, aligned=bool(margin >= threshold),
                residual_skill_frac=round(resid, 4),
                aligned_strict=bool(margin >= threshold and resid == resid and resid <= 0.5),
                edges_removed_directed=n_typ_directed, edges_total_directed=int(ei.size(1)),
                seconds=round(time.time() - t0, 1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archs", nargs="+", default=C.ARCHS)
    ap.add_argument("--data", default=None, help=".pt ya generado (si no, se genera)")
    ap.add_argument("--graph-seed", type=int, default=42)
    ap.add_argument("--legacy", action="store_true", help="generador legado (con atajos)")
    ap.add_argument("--signature-shift", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--random-reps", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--model-seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results_phase1_v4/alignment.json")
    a = ap.parse_args(argv)

    if a.data:
        d = C.load_graph(a.data)
    else:
        d = C.make_graph(seed=a.graph_seed, legacy=a.legacy, signature_shift=a.signature_shift)
    params = getattr(d, "generator_params", {"legacy_pt": a.data})
    out = dict(generator=params, criterion=f"(drop_typology - drop_random) >= {a.threshold}",
               node_only_logreg_pr_auc=round(node_only_pr_auc(d), 4),
               test_prevalence=round(float(d.y[d.test_mask].float().mean()), 4), results={})
    print(f"grafo: N={d.num_nodes} E={d.edge_index.size(1)} params={params}")
    print(f"(d) PR-AUC sin grafo (logística sobre x): {out['node_only_logreg_pr_auc']}  "
          f"prevalencia test={out['test_prevalence']}")
    for arch in a.archs:
        r = check_arch(d, arch, a.epochs, a.patience, a.random_reps, a.threshold, a.model_seed, a.device)
        out["results"][arch] = r
        print(f"  {arch:10} full={r['pr_auc_full']:.3f}  sin_tipología={r['pr_auc_no_typology']:.3f}  "
              f"azar={r['pr_auc_random_drop']:.3f}±{r['pr_auc_random_drop_sd']:.3f}  "
              f"margen={r['margin']:+.3f} resid={r['residual_skill_frac']:.2f}  → "
              f"{'ALINEADO' if r['aligned'] else 'NO alineado'}"
              f"{' (estricto)' if r['aligned_strict'] else ' (no estricto)'}  ({r['seconds']}s)")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"escrito {a.out}")
    return out


if __name__ == "__main__":
    main()
