"""
plausibility.py — Métrica de plausibilidad para explicaciones XAI sobre grafos con
tipologías de lavado inyectadas (item 7 de la tesis).

Mide cuánto coincide la explicación de un nodo ilícito (subgrafo de edges y/o ranking de
features que produce el explainer) con el GROUND-TRUTH de la tipología a la que pertenece.
Es el "puente" con la estabilidad: para cada celda se reporta estabilidad (Spearman de
features, igual que Elliptic) Y plausibilidad, y se pregunta si una predice la otra.

Requiere que el Data tenga:
  - data.typology_node : LongTensor [N]  (0=ninguna, >0 = id de tipología del nodo)
  - data.typology_edge : LongTensor [E]  (0=ninguna, >0 = id de tipología del edge)
  - data.edge_index    : LongTensor [2,E]

El explainer entrega, por nodo objetivo:
  - edge_mask : FloatTensor [E]  (importancia por edge, sobre el edge_index del subgrafo)
  - node_feat_mask / feature_importance : FloatTensor [F]  (importancia por feature)

Convención: se evalúa SOLO sobre nodos ilícitos correctamente clasificados (true positives),
igual que la estabilidad, para no medir plausibilidad de explicaciones de predicciones erróneas.
"""
import numpy as np
import torch


def _prf(pred_set, true_set):
    """precision, recall, f1 entre dos conjuntos de índices."""
    if len(pred_set) == 0 and len(true_set) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_set) == 0 or len(true_set) == 0:
        return 0.0, 0.0, 0.0
    tp = len(pred_set & true_set)
    prec = tp / len(pred_set)
    rec = tp / len(true_set)
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def edge_plausibility(edge_mask, sub_edge_index, sub_edge_typ, node_typ_id, top_k=None):
    """
    OPCIÓN 1 (primaria): plausibilidad de edges.
    Compara los top-k edges por importancia contra los edges que pertenecen a la tipología
    del nodo (sub_edge_typ == node_typ_id). Devuelve precision/recall/f1.

    edge_mask       : [E_sub] importancia por edge en el subgrafo.
    sub_edge_typ    : [E_sub] id de tipología de cada edge del subgrafo (0=ninguna).
    node_typ_id     : id de tipología del nodo objetivo (>0).
    top_k           : nº de edges a tomar como "explicación". Si None, usa el nº real de
                      edges de tipología presentes (comparación balanceada).
    """
    E = len(edge_mask)
    true_edges = set(np.where(sub_edge_typ == node_typ_id)[0].tolist())
    if len(true_edges) == 0:
        return None  # el subgrafo no contiene edges de la tipología -> no evaluable
    k = top_k if top_k is not None else len(true_edges)
    k = min(k, E)
    pred_edges = set(np.argsort(-np.asarray(edge_mask))[:k].tolist())
    return dict(zip(("precision", "recall", "f1"), _prf(pred_edges, true_edges)))


def subgraph_overlap(pred_node_set, true_node_set):
    """
    OPCIÓN 2 (alternativa): overlap de nodos (Jaccard / IoU) entre el subgrafo explicado y
    el subgrafo de la tipología.
    """
    pred, true = set(pred_node_set), set(true_node_set)
    if not pred and not true:
        return 1.0
    union = pred | true
    return len(pred & true) / len(union) if union else 0.0


def feature_plausibility(feat_importance, typology_feature_idx, top_k=None):
    """
    OPCIÓN 3 (secundaria, puente con la estabilidad de features):
    compara el ranking de features del explainer contra el conjunto de features que
    DEFINEN la tipología (índices conocidos por construcción del dataset). precision@k.

    feat_importance     : [F] importancia por feature.
    typology_feature_idx: iterable de índices de features que codifican la tipología.
    top_k               : si None, usa len(typology_feature_idx).
    """
    true_f = set(int(i) for i in typology_feature_idx)
    if not true_f:
        return None
    k = top_k if top_k is not None else len(true_f)
    k = min(k, len(feat_importance))
    pred_f = set(np.argsort(-np.asarray(feat_importance))[:k].tolist())
    p, r, f = _prf(pred_f, true_f)
    return dict(precision=p, recall=r, f1=f)


def aggregate(per_node_scores):
    """Promedia una lista de dicts (o floats) ignorando None (nodos no evaluables)."""
    vals = [s for s in per_node_scores if s is not None]
    if not vals:
        return None
    if isinstance(vals[0], dict):
        keys = vals[0].keys()
        return {k: float(np.mean([v[k] for v in vals])) for k in keys}, len(vals)
    return float(np.mean(vals)), len(vals)
