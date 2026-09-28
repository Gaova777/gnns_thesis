"""
Generador sintético de tipologías de lavado de dinero para estudio de plausibilidad XAI.
Alternativa en Python puro a AMLSim: control DIRECTO de la densidad (requisito crítico
para que la plausibilidad a nivel de subgrafo sea medible, a diferencia de Elliptic).

Inyecta 3 tipologías con ground-truth por nodo y por edge:
  - STRUCTURING (smurfing): un origen reparte a muchos intermediarios en montos pequeños.
  - LAYERING: cadena de transferencias por cuentas intermedias.
  - FAN_IN / FAN_OUT: muchos->uno (recolección) / uno->muchos (distribución).

Salida: objeto PyG Data con x, edge_index, y (lícito/ilícito), máscaras train/val/test,
y ground-truth de tipología (typology_node, typology_edge).

MODO v4 SIN ATAJOS (``shortcut_free=True``; lo activa ``V4_DEFAULTS`` y ``--v4``)
--------------------------------------------------------------------------------
El generador legado (``shortcut_free=False``, default de la función para no desincronizar
los CSV del Capítulo 5) deja tres atajos que permiten acertar SIN usar la tipología:
  1. los distractores solo salen de nodos de patrón → todo ilícito tiene out-degree ≥ 3,
     el fondo ≈ 1,2: el grado (feature 0-2 de x) delata la clase;
  2. las variables de grado cuentan esos distractores (mismo atajo, en x);
  3. cada tipología sube su feature-firma +1,5 σ: separable nodo a nodo.
En v4:
  1'. distractores (tipología NONE) emitidos por TODOS los nodos con la misma tasa
      (Poisson(n_distractors)) y destino uniforme sobre todos los nodos;
  2'. «negativos difíciles»: por cada estructura de patrón se inyecta un GEMELO LÍCITO con la
      misma forma y tamaño (estrella, cadena, fan-in, fan-out) → los grados de los ilícitos y
      de estos lícitos tienen la misma distribución (``data.hard_negative`` los marca);
  3'. firma atenuada (``signature_shift``, default v4 = +0,3 σ): nodo a nodo casi no separa
      (AUC ≈ 0,58 para su tipología), pero AGREGADA sobre los miembros de la estructura sí.
      La única forma de clasificar bien es agregar a través de las aristas de la tipología,
      que es justo lo que la plausibilidad mide. ``phase1/alignment_check.py`` lo verifica.
``shortcut_report(data)`` mide el AUC de un clasificador de una sola variable (grado, firma).
"""
import numpy as np
import torch
from torch_geometric.data import Data

TYPOLOGIES = {"NONE": 0, "STRUCTURING": 1, "LAYERING": 2, "FAN_IN": 3, "FAN_OUT": 4}

# Preset del eje sintético v4 (sin atajos). El default de generate_aml_graph sigue siendo el
# legado para que phase1/random_baseline_plausibility.py y los CSV del Cap. 5 no cambien.
V4_DEFAULTS = dict(shortcut_free=True, signature_shift=0.3, hard_negative_frac=1.0,
                   n_distractors=1.5)


def _add_edge(edges, edge_typ, src, dst, typ):
    edges.append((src, dst)); edge_typ.append(TYPOLOGIES[typ])


def generate_aml_graph(
    n_background=8000,       # cuentas lícitas de fondo
    n_structuring=25,        # nº de patrones structuring a inyectar
    n_layering=45,           # subido de 25: las cadenas son cortas, así que layering quedaba
                             # subrepresentado (~4 nodos en el muestreo de 30 TP). Con 45 cadenas
                             # más largas queda parejo con las otras 3 tipologías.
    n_fanin=20,
    n_fanout=20,
    struct_fanout=(8, 20),   # cada structuring reparte a entre 8 y 20 intermediarios
    layer_len=(8, 15),       # cada layering: cadena de 8 a 15 saltos (subido de 5-12)
    fan_size=(10, 25),       # fan-in/out: entre 10 y 25 contrapartes
    bg_degree=1.2,           # grado medio del fondo (disperso, como transacciones normales)
    n_distractors=3,         # edges de ruido (typ=NONE) de cada nodo de patrón hacia el
                             # fondo. Necesario para que la plausibilidad de edges discrimine:
                             # sin distractores el subgrafo es 100% patrón y todo top-k acierta.
    symmetrize=True,         # CRÍTICO: aristas no dirigidas. Con dirigido el receptive field
                             # cae a ~2 nodos (los patrones estrella/cadena quedan invisibles al
                             # message passing dirigido) y la plausibilidad de subgrafo NO es
                             # medible. Simetrizar sube la mediana a ~17 y hace medible el item 7.
    seed=42,
    shortcut_free=False,     # v4: quita los tres atajos (ver docstring del módulo)
    signature_shift=None,    # desplazamiento de la feature-firma; None → 1.5 (legado)
    hard_negative_frac=1.0,  # (solo v4) nº de gemelos lícitos por estructura de patrón
):
    if signature_shift is None:
        signature_shift = 1.5
    rng = np.random.default_rng(seed)
    edges, edge_typ = [], []
    node_typ = {}            # id -> typology id (solo para nodos de patrón)
    illicit = set()
    nid = n_background        # los primeros n_background son fondo lícito; patrones empiezan aquí

    node_inst = {}           # id -> nº de instancia de patrón (GT por instancia; no toca el RNG)
    inst = [-1]

    def new_node(typ):
        nonlocal nid
        i = nid; nid += 1; node_typ[i] = TYPOLOGIES[typ]; illicit.add(i)
        node_inst[i] = inst[0]; return i

    # ---- fondo lícito: transacciones aleatorias dispersas ----
    n_bg_edges = int(n_background * bg_degree)
    for _ in range(n_bg_edges):
        a, b = rng.integers(0, n_background, size=2)
        if a != b:
            _add_edge(edges, edge_typ, int(a), int(b), "NONE")

    # ---- STRUCTURING: origen -> muchos intermediarios (montos pequeños) ----
    for _ in range(n_structuring):
        inst[0] += 1
        src = new_node("STRUCTURING")
        k = rng.integers(*struct_fanout)
        for _ in range(k):
            inter = new_node("STRUCTURING")
            _add_edge(edges, edge_typ, src, inter, "STRUCTURING")
            # el intermediario reenvía a una cuenta colectora (patrón completo)
            _add_edge(edges, edge_typ, inter, src, "STRUCTURING") if rng.random() < 0.3 else None

    # ---- LAYERING: cadena larga de transferencias ----
    for _ in range(n_layering):
        inst[0] += 1
        L = rng.integers(*layer_len)
        chain = [new_node("LAYERING") for _ in range(L)]
        for a, b in zip(chain[:-1], chain[1:]):
            _add_edge(edges, edge_typ, a, b, "LAYERING")

    # ---- FAN_IN: muchos -> uno ----
    for _ in range(n_fanin):
        inst[0] += 1
        collector = new_node("FAN_IN")
        k = rng.integers(*fan_size)
        for _ in range(k):
            s = new_node("FAN_IN")
            _add_edge(edges, edge_typ, s, collector, "FAN_IN")

    # ---- FAN_OUT: uno -> muchos ----
    for _ in range(n_fanout):
        inst[0] += 1
        src = new_node("FAN_OUT")
        k = rng.integers(*fan_size)
        for _ in range(k):
            d = new_node("FAN_OUT")
            _add_edge(edges, edge_typ, src, d, "FAN_OUT")

    # ---- DISTRACTORES: conectar nodos de patrón al fondo lícito con edges de ruido
    #      (typology=NONE). Sin esto, el subgrafo de un nodo de tipología es 100% edges de
    #      patrón y CUALQUIER selección top-k acierta -> la plausibilidad de edges no
    #      discrimina entre un buen explainer y uno aleatorio. Con distractores, el
    #      vecindario mezcla edges de patrón con ruido de fondo, y solo un explainer que
    #      señala el patrón obtiene precision/recall altos. n_distractors por nodo de patrón.
    hard_neg = set()
    if not shortcut_free:
        pattern_nodes = list(node_typ.keys())
        for pn in pattern_nodes:
            for _ in range(int(n_distractors)):
                bg = int(rng.integers(0, n_background))
                _add_edge(edges, edge_typ, pn, bg, "NONE")
    else:
        # ---- v4 (2'): GEMELOS LÍCITOS con la misma forma que cada patrón ----
        rng_hn = np.random.default_rng(seed + 7)

        def new_licit():
            nonlocal nid
            i = nid; nid += 1; hard_neg.add(i); return i

        def n_twins(n):
            return int(round(n * hard_negative_frac))
        for _ in range(n_twins(n_structuring)):
            src = new_licit()
            for _ in range(rng_hn.integers(*struct_fanout)):
                inter = new_licit()
                _add_edge(edges, edge_typ, src, inter, "NONE")
                if rng_hn.random() < 0.3:
                    _add_edge(edges, edge_typ, inter, src, "NONE")
        for _ in range(n_twins(n_layering)):
            chain = [new_licit() for _ in range(rng_hn.integers(*layer_len))]
            for a, b in zip(chain[:-1], chain[1:]):
                _add_edge(edges, edge_typ, a, b, "NONE")
        for _ in range(n_twins(n_fanin)):
            collector = new_licit()
            for _ in range(rng_hn.integers(*fan_size)):
                _add_edge(edges, edge_typ, new_licit(), collector, "NONE")
        for _ in range(n_twins(n_fanout)):
            src = new_licit()
            for _ in range(rng_hn.integers(*fan_size)):
                _add_edge(edges, edge_typ, src, new_licit(), "NONE")
        # ---- v4 (1'): distractores desde TODOS los nodos, misma tasa, destino uniforme ----
        n_all = nid
        n_out = rng_hn.poisson(float(n_distractors), size=n_all)
        for u in range(n_all):
            for _ in range(int(n_out[u])):
                v = int(rng_hn.integers(0, n_all))
                if v != u:
                    _add_edge(edges, edge_typ, u, v, "NONE")

    N = nid
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    e_typ = torch.tensor(edge_typ, dtype=torch.long)
    # instancia de patrón por arista (-1 = ninguna): la de su origen si la arista es de tipología
    e_inst = torch.tensor([node_inst[u] if t > 0 else -1 for (u, _), t in zip(edges, edge_typ)],
                          dtype=torch.long)

    # ---- features de nodo: agregados de flujo (análogos en espíritu a Elliptic) ----
    out_deg = torch.zeros(N); in_deg = torch.zeros(N)
    out_deg.scatter_add_(0, ei[0], torch.ones(ei.size(1)))
    in_deg.scatter_add_(0, ei[1], torch.ones(ei.size(1)))
    rngx = np.random.default_rng(seed + 1)
    # montos: patrón structuring/fan tiene montos pequeños; layering montos medianos; fondo variado
    amt = torch.tensor(rngx.lognormal(3.0, 1.0, size=N), dtype=torch.float)
    feats = torch.stack([
        in_deg, out_deg, in_deg + out_deg,
        (in_deg + 1) / (out_deg + 1),           # razón in/out
        amt, amt * (in_deg + out_deg),          # volumen aproximado
    ], dim=1)                                    # índices 0-5

    # ---- FEATURES-FIRMA POR TIPOLOGÍA (índices 6-9) ----
    # Cada tipología tiene UNA feature que se eleva SOLO en sus nodos (señal) más ruido.
    # Esto habilita la plausibilidad de FEATURES (puente con el Spearman de Elliptic):
    # un buen explainer debe señalar la feature-firma del nodo. Índices fijos y conocidos.
    sig = torch.tensor(rngx.normal(0, 1, size=(N, 4)), dtype=torch.float)  # base de ruido
    for i, t in node_typ.items():
        sig[i, t - 1] += signature_shift    # legado 1.5; v4 0.3. Firma ATENUADA (+1.5, antes +4): deja de ser trivialmente
                                # separable → plausibilidad de features no trivial (encargo re-corrida B)
    # índices 6,7,8,9 = firma de STRUCTURING, LAYERING, FAN_IN, FAN_OUT respectivamente

    # padding con ruido (para dar dimensionalidad tipo Elliptic), índices 10-19
    extra = torch.tensor(rngx.normal(0, 1, size=(N, 10)), dtype=torch.float)
    x = torch.cat([feats, sig, extra], dim=1)

    # ---- etiquetas ----
    y = torch.zeros(N, dtype=torch.long)
    typ_node = torch.zeros(N, dtype=torch.long)
    for i, t in node_typ.items():
        y[i] = 1; typ_node[i] = t

    # ---- split temporal simulado: fondo en train, patrones repartidos ----
    idx = torch.arange(N)
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    perm = torch.tensor(rng.permutation(N))
    n_tr = int(0.6 * N); n_va = int(0.2 * N)
    train_mask[perm[:n_tr]] = True
    val_mask[perm[n_tr:n_tr + n_va]] = True
    test_mask[perm[n_tr + n_va:]] = True

    # ---- simetrización (por defecto): imprescindible para que la tipología caiga en el
    #      receptive field. Se duplican los edges (u,v)->(u,v),(v,u) y su etiqueta de
    #      tipología se preserva en ambas direcciones. ----
    if symmetrize:
        ei_rev = ei.flip(0)
        ei = torch.cat([ei, ei_rev], dim=1)
        e_typ = torch.cat([e_typ, e_typ], dim=0)
        e_inst = torch.cat([e_inst, e_inst], dim=0)

    data = Data(x=x, edge_index=ei, y=y)
    data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
    data.typology_node = typ_node
    data.typology_edge = e_typ
    inst_node = torch.full((N,), -1, dtype=torch.long)
    for i, k in node_inst.items():
        inst_node[i] = k
    data.pattern_instance_node = inst_node      # GT por INSTANCIA (no solo por tipología)
    data.pattern_instance_edge = e_inst
    # mapeo tipología -> índice de su feature-firma (para plausibilidad de features).
    # STRUCTURING(1)->6, LAYERING(2)->7, FAN_IN(3)->8, FAN_OUT(4)->9
    data.typology_feature_index = {1: 6, 2: 7, 3: 8, 4: 9}
    hn = torch.zeros(N, dtype=torch.bool)
    if hard_neg:
        hn[torch.tensor(sorted(hard_neg))] = True
    data.hard_negative = hn
    data.generator_params = dict(shortcut_free=bool(shortcut_free),
                                 signature_shift=float(signature_shift),
                                 hard_negative_frac=float(hard_negative_frac) if shortcut_free else 0.0,
                                 n_distractors=float(n_distractors), seed=int(seed))
    return data


def _auc(score, y):
    from sklearn.metrics import roc_auc_score
    a = float(roc_auc_score(y, score))
    return max(a, 1.0 - a)      # un clasificador de una variable puede usar cualquier signo


def shortcut_report(data):
    """AUC de clasificadores de UNA sola variable (ilícito vs lícito, todos los nodos).

    Umbral de «sin atajo» usado en v4: AUC ≤ 0,65. Además se reporta la firma evaluada solo
    sobre su tipología (ilícitos de la tipología t vs todos los lícitos), que es su uso real.
    """
    x = data.x.numpy(); y = data.y.numpy()
    names = ["in_deg", "out_deg", "total_deg", "in_out_ratio", "amount", "volume"]
    rep = {n: _auc(x[:, i], y) for i, n in enumerate(names)}
    tn = data.typology_node.numpy()
    for t, fi in data.typology_feature_index.items():
        sel = (tn == t) | (y == 0)
        rep[f"sig_{t}_own_typology"] = _auc(x[sel, fi], (tn[sel] == t).astype(int))
    rep["sig_max_all"] = _auc(x[:, 6:10].max(1), y)
    return rep


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4", action="store_true", help="preset sin atajos (V4_DEFAULTS)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--signature-shift", type=float, default=None)
    ap.add_argument("--report", action="store_true", help="imprime el AUC de una variable")
    ap.add_argument("--out", default="amlsim_synthetic_v0.pt")
    a = ap.parse_args()
    kw = dict(V4_DEFAULTS) if a.v4 else {}
    if a.signature_shift is not None:
        kw["signature_shift"] = a.signature_shift
    d = generate_aml_graph(seed=a.seed, **kw)
    print("N nodos:", d.num_nodes, "| edges:", d.edge_index.size(1), "| feats:", d.x.size(1))
    print("ilícitos:", int(d.y.sum().item()), f"({100*d.y.float().mean():.1f}%)")
    if a.report:
        print(json.dumps({k: round(v, 3) for k, v in shortcut_report(d).items()}, indent=1))
    torch.save(d, a.out)
