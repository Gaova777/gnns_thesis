"""Piezas compartidas del eje sintético v4 (alignment_check.py y run_phase1_v4.py).

Deliberadamente NO depende de ``src/training`` ni de ``src/data/imbalance`` (se están
reescribiendo para Elliptic v4): el entrenamiento y los escenarios del eje sintético viven aquí,
pequeños y fijos. Sí usa ``src/models`` (arquitecturas) y ``src/balancing`` (pérdidas).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import RobustScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "phase1")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.balancing.losses import get_loss_function  # noqa: E402
from synthetic_aml_generator import V4_DEFAULTS, generate_aml_graph  # noqa: E402

ARCHS = ["GCN", "GraphSAGE", "GAT", "TAGCN"]
NUM_LAYERS = 2
TAGCN_K = 3
GAT_HEADS = 4


# --------------------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------------------
def scale(d):
    """RobustScaler ajustado en train y recorte a ±10 (mismo tratamiento que el eje legado)."""
    sc = RobustScaler().fit(d.x[d.train_mask].numpy())
    d.x = torch.clamp(torch.tensor(sc.transform(d.x.numpy()), dtype=torch.float), -10, 10)
    return d


def make_graph(seed: int = 42, legacy: bool = False, **overrides):
    kw = {} if legacy else dict(V4_DEFAULTS)
    kw.update({k: v for k, v in overrides.items() if v is not None})
    return scale(generate_aml_graph(seed=seed, **kw))


def load_graph(path: str):
    return scale(torch.load(path, weights_only=False))


# --------------------------------------------------------------------------------------
# Escenarios de desbalance — niveles REALES
# --------------------------------------------------------------------------------------
SCENARIO_RATIO = {"1:1": 1.0, "1:10": 0.1, "1:50": 0.02, "1:100": 0.01, "natural": None}


def native_ratio(d) -> float:
    m = d.train_mask
    return float((d.y[m] == 1).sum()) / max(1.0, float((d.y[m] == 0).sum()))


def collapsing_scenarios(d, scenarios, tol: float = 2.0) -> list[str]:
    """Escenarios cuyo objetivo cae a menos de ``tol``× del ratio nativo: no son un nivel
    distinto. En el grafo legado (nativo ≈1:5) y en el v4 (≈1:6,2), 1:10 exigía MÁS lícitos de
    los que hay y src/data/imbalance lo devolvía IGUAL al nativo; make_scenario ya lo alcanza
    recortando ilícitos, pero queda a 1,6× del nativo → se descarta por defecto."""
    nat = native_ratio(d)
    out = []
    for s in scenarios:
        r = SCENARIO_RATIO[s]
        if r is None:
            continue
        if max(r, nat) / min(r, nat) < tol:
            out.append(s)
    return out


def make_scenario(d, scenario: str, seed: int = 42):
    """Submuestrea la máscara de TRAIN para alcanzar el ratio ilícito:lícito pedido.

    Submuestrea la clase que sobra (lícitos si el objetivo es más balanceado que el nativo,
    ilícitos si es más desbalanceado) → nunca colapsa en silencio al nativo. Val/test intactos.
    """
    r = SCENARIO_RATIO[scenario]
    if r is None:
        return d
    rng = np.random.RandomState(seed)
    m = d.train_mask
    ill = torch.where(m & (d.y == 1))[0].numpy()
    lic = torch.where(m & (d.y == 0))[0].numpy()
    if len(ill) / len(lic) > r:            # sobran ilícitos
        ill = rng.choice(ill, size=max(1, int(round(len(lic) * r))), replace=False)
    else:                                  # sobran lícitos
        lic = rng.choice(lic, size=min(len(lic), int(round(len(ill) / r))), replace=False)
    d2 = d.clone()
    nm = torch.zeros_like(m)
    nm[torch.as_tensor(ill)] = True
    nm[torch.as_tensor(lic)] = True
    d2.train_mask = nm
    return d2


# --------------------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------------------
def receptive_hops(arch: str, num_layers: int = NUM_LAYERS, K: int = TAGCN_K) -> int:
    """Campo receptivo en saltos. TAGConv(K) agrega hasta K saltos POR CAPA."""
    return num_layers * K if arch == "TAGCN" else num_layers


def build(arch: str, in_ch: int, hidden: int = 64, dropout: float = 0.3):
    from src.models.gat import GAT
    from src.models.gcn import GCN
    from src.models.sage import GraphSAGE
    from src.models.tagcn import TAGCN
    common = dict(in_channels=in_ch, hidden_channels=hidden, num_layers=NUM_LAYERS, dropout=dropout)
    if arch == "GAT":
        return GAT(heads=GAT_HEADS, **common)
    if arch == "TAGCN":
        return TAGCN(K=TAGCN_K, **common)
    return {"GCN": GCN, "GraphSAGE": GraphSAGE}[arch](**common)


@torch.no_grad()
def proba(model, x, ei):
    model.eval()
    return torch.softmax(model(x, ei), dim=1)[:, 1]


def pr_auc(model, d, mask_name="test_mask", edge_index=None, device="cpu"):
    ei = (d.edge_index if edge_index is None else edge_index).to(device)
    p = proba(model, d.x.to(device), ei).cpu().numpy()
    m = getattr(d, mask_name).numpy()
    return float(average_precision_score(d.y.numpy()[m], p[m]))


def train_model(d, arch, balancing="none", epochs=200, patience=30, seed=42, device="cpu",
                lr=0.01, weight_decay=5e-4, hidden=64):
    """Entrenamiento full-batch con early stopping en VAL PR-AUC (restaura el mejor estado)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = build(arch, d.x.size(1), hidden=hidden).to(device)
    loss_fn = get_loss_function(balancing, labels=d.y, mask=d.train_mask, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    x, ei, y = d.x.to(device), d.edge_index.to(device), d.y.to(device)
    tm = d.train_mask.to(device)
    best, best_state, bad, ep = -1.0, None, 0, 0
    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        loss = loss_fn(model(x, ei)[tm], y[tm])
        loss.backward(); opt.step()
        v = pr_auc(model, d, "val_mask", device=device)
        if v > best + 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"val_pr_auc": round(best, 4), "test_pr_auc": round(pr_auc(model, d, device=device), 4),
                   "epochs_run": ep}


@torch.no_grad()
def val_threshold(model, d, device="cpu") -> float:
    """Umbral que maximiza F1 en validación. Con pérdida sin ponderar y desbalance, argmax
    (0,5) deja 0 positivos predichos y la celda queda sin nodos que explicar; el umbral
    calibrado define los «verdaderos positivos» igual para todas las celdas."""
    from sklearn.metrics import precision_recall_curve
    p = proba(model, d.x.to(device), d.edge_index.to(device)).cpu().numpy()
    m = d.val_mask.numpy()
    pr, rc, th = precision_recall_curve(d.y.numpy()[m], p[m])
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    return float(th[int(np.argmax(f1))]) if len(th) else 0.5


@torch.no_grad()
def predict(model, d, threshold, device="cpu"):
    return (proba(model, d.x.to(device), d.edge_index.to(device)).cpu() >= threshold).long()


@torch.no_grad()
def tp_val_nodes(model, d, n, device="cpu", seed=42, threshold=0.5):
    pred = proba(model, d.x.to(device), d.edge_index.to(device)).cpu() >= threshold
    iv = torch.where(d.val_mask & (d.y == 1))[0]
    tp = iv[pred[iv]]
    if len(tp) == 0:
        return []
    rng = np.random.RandomState(seed)
    return rng.choice(tp.numpy(), size=min(n, len(tp)), replace=False).tolist()
