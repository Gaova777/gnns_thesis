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
"""
import numpy as np
import torch
from torch_geometric.data import Data

TYPOLOGIES = {"NONE": 0, "STRUCTURING": 1, "LAYERING": 2, "FAN_IN": 3, "FAN_OUT": 4}


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
):
    rng = np.random.default_rng(seed)
    edges, edge_typ = [], []
    node_typ = {}            # id -> typology id (solo para nodos de patrón)
    illicit = set()
    nid = n_background        # los primeros n_background son fondo lícito; patrones empiezan aquí

    def new_node(typ):
        nonlocal nid
        i = nid; nid += 1; node_typ[i] = TYPOLOGIES[typ]; illicit.add(i); return i

    # ---- fondo lícito: transacciones aleatorias dispersas ----
    n_bg_edges = int(n_background * bg_degree)
    for _ in range(n_bg_edges):
        a, b = rng.integers(0, n_background, size=2)
        if a != b:
            _add_edge(edges, edge_typ, int(a), int(b), "NONE")

    # ---- STRUCTURING: origen -> muchos intermediarios (montos pequeños) ----
    for _ in range(n_structuring):
        src = new_node("STRUCTURING")
        k = rng.integers(*struct_fanout)
        for _ in range(k):
            inter = new_node("STRUCTURING")
            _add_edge(edges, edge_typ, src, inter, "STRUCTURING")
            # el intermediario reenvía a una cuenta colectora (patrón completo)
            _add_edge(edges, edge_typ, inter, src, "STRUCTURING") if rng.random() < 0.3 else None

    # ---- LAYERING: cadena larga de transferencias ----
    for _ in range(n_layering):
        L = rng.integers(*layer_len)
        chain = [new_node("LAYERING") for _ in range(L)]
        for a, b in zip(chain[:-1], chain[1:]):
            _add_edge(edges, edge_typ, a, b, "LAYERING")

    # ---- FAN_IN: muchos -> uno ----
    for _ in range(n_fanin):
        collector = new_node("FAN_IN")
        k = rng.integers(*fan_size)
        for _ in range(k):
            s = new_node("FAN_IN")
            _add_edge(edges, edge_typ, s, collector, "FAN_IN")

    # ---- FAN_OUT: uno -> muchos ----
    for _ in range(n_fanout):
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
    pattern_nodes = list(node_typ.keys())
    for pn in pattern_nodes:
        for _ in range(int(n_distractors)):
            bg = int(rng.integers(0, n_background))
            _add_edge(edges, edge_typ, pn, bg, "NONE")

    N = nid
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    e_typ = torch.tensor(edge_typ, dtype=torch.long)

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
        sig[i, t - 1] += 1.5    # firma ATENUADA (+1.5, antes +4): deja de ser trivialmente
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

    data = Data(x=x, edge_index=ei, y=y)
    data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
    data.typology_node = typ_node
    data.typology_edge = e_typ
    # mapeo tipología -> índice de su feature-firma (para plausibilidad de features).
    # STRUCTURING(1)->6, LAYERING(2)->7, FAN_IN(3)->8, FAN_OUT(4)->9
    data.typology_feature_index = {1: 6, 2: 7, 3: 8, 4: 9}
    return data


if __name__ == "__main__":
    d = generate_aml_graph()
    print("N nodos:", d.num_nodes, "| edges:", d.edge_index.size(1), "| feats:", d.x.size(1))
    print("ilícitos:", int(d.y.sum().item()), f"({100*d.y.float().mean():.1f}%)")
    torch.save(d, "amlsim_synthetic_v0.pt")
