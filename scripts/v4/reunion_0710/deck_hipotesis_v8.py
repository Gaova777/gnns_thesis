"""Deck de hipótesis, versión 8: correcciones de Cristian del 7-oct (THE-43).

Parte de la versión VIVA del artefacto (la 7, publicada por otra sesión) y aplica encima:
  - rendimiento con pesos por clase en las cuatro arquitecturas (láminas 5 y 6);
  - lámina nueva: por qué la compuerta se decide en validación y no en test;
  - curvas con la leyenda de colores arriba y «sin ajuste» en vez de «sin balanceo»;
  - H1: el porqué con el mecanismo y el acuerdo entre escenarios, y los escenarios de estrés
    1:100 y 1:200 (si ya hay resultados);
  - H2: lámina nueva con la ecuación de cada capa;
  - H3: redacción nueva con los explicadores de variables y lámina nueva del explicador
    adicional (Integrated Gradients y Expected Gradients);
  - cierre con el candidato optimizado (THE-36), resumen y plan al día.

Las cifras salen de results_v4/reunion_0410/ y results_v4/reunion_0710/ (correr antes
scripts/v4/reunion_0710/analisis.py). Uso:
  uv run --frozen python scripts/v4/reunion_0710/deck_hipotesis_v8.py SRC.html OUT_DIR
"""
import base64
import io
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
FIG = REPO / "results_v4/reunion_0410/curvas/figuras/deck"
R4 = REPO / "results_v4/reunion_0410"
R7 = REPO / "results_v4/reunion_0710"
SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

LOGO = ('<div class="logo" role="img" aria-label="Universidad Tecnológica de Pereira, '
        'Facultad de Ingenierías"></div>')
REF = dict(
    saito="Saito y Rehmsmeier (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLoS ONE.",
    massey="Massey, F. J. (1951). The Kolmogorov-Smirnov test for goodness of fit. JASA, 46(253), 68-78.",
    cawley="Cawley, G. y Talbot, N. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. JMLR, 11, 2079-2107.",
    elkan="Elkan, C. (2001). The foundations of cost-sensitive learning. IJCAI · Dal Pozzolo, A. et al. (2015). Calibrating probability with undersampling. IEEE SSCI.",
    gnns="Kipf y Welling (2017), ICLR · Hamilton et al. (2017), NeurIPS · Veličković et al. (2018), ICLR · Du et al. (2017), arXiv.",
    focal="Lin, T.-Y. et al. (2017). Focal loss for dense object detection. ICCV.",
    ig="Sundararajan, M. et al. (2017). Axiomatic attribution for deep networks. ICML · Erion, G. et al. (2021). Nature Machine Intelligence, 3, 620-631.",
    krishna="Krishna, S. et al. (2022). The disagreement problem in explainable machine learning: a practitioner's perspective. arXiv:2202.01602.",
    bergstra="Bergstra, J. et al. (2011). Algorithms for hyper-parameter optimization. NeurIPS.",
    weber="Weber, M. et al. (2019). Anti-money laundering in Bitcoin: experimenting with graph convolutional networks. arXiv:1908.02591.",
    yuan="Yuan, H. et al. (2023). Explainability in graph neural networks: a taxonomic survey. IEEE TPAMI, 45(5), 5782-5799.",
)


def c(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def uri(name, h):
    im = Image.open(FIG / name).convert("RGB")
    w = round(im.width * h / im.height)
    b = io.BytesIO()
    im.resize((w, h), Image.LANCZOS).save(b, "JPEG", quality=86, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def notas(ns):
    return "".join(f'<div data-paso="{k}" data-cue="{cue}"><p data-voz="{v}">{t}</p></div>'
                   for k, (cue, v, t) in enumerate(ns))


def lamina(seccion, titular, cuerpo, pie, ref, ns, cls="lc"):
    pasos = len(ns) - 1
    pie_html = f'\n    <p class="pie">{pie}</p>' if pie else ""
    return (f'<section class="slide lamina {cls}" data-pasos="{pasos}" aria-label="Lámina 0">\n'
            f'    {LOGO}\n    <p class="seccion">{seccion}</p>\n'
            f'    <h2 class="titular">{titular}</h2>\n{cuerpo}{pie_html}\n'
            f'    <div class="banda"></div>\n    <p class="ref">{ref}</p>\n'
            f'    <div class="num">0</div>\n'
            f'    <aside class="notas" hidden>{notas(ns)}</aside>\n  </section>')


# ── fuente: la versión viva ───────────────────────────────────────────────────
h = SRC.read_text(encoding="utf8")
h = h[h.index("<body>") + len("<body>"):].lstrip("\n")
h = h[:h.rindex("</body>")].rstrip()
m = list(re.finditer(r'<section\b[^>]*>.*?</section>', h, flags=re.S))
PRE, POST = h[:m[0].start()], h[m[-1].end():]
S = {i + 1: x.group(0) for i, x in enumerate(m)}   # numeración de la versión 7
assert len(S) == 26, len(S)


def rep(n, old, new, cnt=1):
    assert S[n].count(old) == cnt, (n, S[n].count(old), old[:70])
    S[n] = S[n].replace(old, new)


def titulo(n, t):
    S[n], k = re.subn(r'<h2 class="titular">.*?</h2>', f'<h2 class="titular">{t}</h2>', S[n],
                      flags=re.S)
    assert k == 1


def set_notas(n, ns):
    S[n], k = re.subn(r'<aside class="notas" hidden>.*?</aside>',
                      lambda _: f'<aside class="notas" hidden>{notas(ns)}</aside>', S[n],
                      flags=re.S)
    assert k == 1
    S[n] = re.sub(r'data-pasos="\d+"', f'data-pasos="{len(ns) - 1}"', S[n], count=1)


def set_ref(n, t):
    S[n], k = re.subn(r'<p class="ref">.*?</p>', lambda _: f'<p class="ref">{t}</p>', S[n],
                      flags=re.S)
    assert k == 1


def cuerpo(n, nuevo, pie=None):
    """Reemplaza todo lo que hay entre el titular y la banda."""
    a = S[n].index("</h2>") + len("</h2>")
    b = S[n].index('<div class="banda">')
    pie_html = f'\n    <p class="pie">{pie}</p>' if pie else ""
    S[n] = S[n][:a] + "\n" + nuevo + pie_html + "\n    " + S[n][b:]


def imagen(n, name, h_px, estilo):
    S[n], k = re.subn(r'(<img class="fig" src=")data:image/jpeg;base64,[^"]+(" alt="[^"]*" style=")[^"]*(")',
                      lambda mm: mm.group(1) + uri(name, h_px) + mm.group(2) + estilo + mm.group(3),
                      S[n])
    assert k == 1, (n, k)


# ── datos ────────────────────────────────────────────────────────────────────
cur = pd.read_csv(R4 / "curvas/curvas_metricas_por_config.csv")
curm = pd.read_csv(R4 / "curvas/curvas_metricas_por_modelo.csv")
cfg = pd.read_csv(R7 / "config_compuerta.csv")
stab = pd.read_csv(R7 / "estabilidad_config.csv")
esc = pd.read_csv(R7 / "acuerdo_escenarios.csv")
sem = pd.read_csv(R7 / "acuerdo_semillas_nativo.csv")
expl = pd.read_csv(R7 / "acuerdo_explicadores.csv")
MAIN = ["1:10", "1:10_os", "1:20", "native"]
STRESS = [s for s in ["1:100_subil", "1:200_subil"]
          if (stab[(stab.scenario == s) & (stab.explainer == "ExpectedGradients")].shape[0] >= 12
              and stab[(stab.scenario == s) & (stab.explainer == "GNNExplainer")].shape[0] >= 12)]
APR = ["GraphSAGE", "GAT", "TAGCN"]
ARQ = ["GraphSAGE", "GAT", "TAGCN", "GCN"]
CW = "class_weighting"
ESC_NOM = {"1:10": "1:10", "1:10_os": "1:10 + SMOTE", "1:20": "1:20",
           "native": "Nativo (≈1:40)", "1:100_subil": "1:100", "1:200_subil": "1:200"}


def met(arch, split, col, scen="native", bal=CW):
    r = cur[(cur.scenario == scen) & (cur.arch == arch) & (cur.balancing == bal)
            & (cur.split == split)].iloc[0]
    return r[col + "_mean"], r[col + "_std"]


# ═════════════════════════════════════════════════════════════════════════════
# «sin balanceo» → «sin ajuste» (Cristian: el submuestreo ya es un balanceo)
# ═════════════════════════════════════════════════════════════════════════════
for n in S:
    S[n] = (S[n].replace("sin balanceo", "sin ajuste").replace("Sin balanceo", "Sin ajuste"))

# ═════════════════════════════════════════════════════════════════════════════
# Lámina 4: leer las curvas (se agrega qué pasa bajo la diagonal)
# ═════════════════════════════════════════════════════════════════════════════
rep(4, "Dice si el modelo separa las dos clases. La diagonal es el azar.",
    "Dice si el modelo separa las dos clases. La diagonal es el azar; por debajo de ella "
    "el modelo ordena al revés.")
rep(4, "Arriba a la izquierda es bueno; la diagonal es tirar una moneda.",
    "Arriba a la izquierda es bueno; la diagonal es tirar una moneda. Una curva por debajo "
    "de la diagonal no es peor que el azar: el modelo ordena al revés, y eso pasa cuando la "
    "población de prueba ya no se parece a la de entrenamiento.")

# ═════════════════════════════════════════════════════════════════════════════
# Lámina 5: resumen con pesos por clase en las cuatro
# ═════════════════════════════════════════════════════════════════════════════
pv = {a: met(a, "val", "pr_auc")[0] for a in ARQ}
rt = {a: met(a, "test", "roc_auc")[0] for a in ARQ}
titulo(5, "Con la misma pérdida en las cuatro, GCN es la única que no aprende en validación; "
          "en test ninguna se separa del azar")
cuerpo(5, f'''<img class="fig" src="{uri("resumen_native_pesos_por_clase.png", 1080)}" alt="Desbalance real, pesos por clase en las cuatro arquitecturas. Arriba ROC, abajo PR; izquierda validación, derecha test." style="left:1224px;top:330px;height:540px;width:616px">
<p class="llamada media" data-desde="1" style="left:80px;top:330px;width:1094px">Validación: GraphSAGE va arriba en todo el recall. GCN queda sola abajo: PR-AUC {c(pv["GCN"], 2)} frente a {c(min(pv[a] for a in APR), 2)} a {c(max(pv[a] for a in APR), 2)} de las otras tres.</p>
<p class="llamada media" data-desde="2" style="left:80px;top:530px;width:1094px">Test: las cuatro curvas PR quedan pegadas a la línea de azar y las ROC se juntan entre {c(min(rt.values()), 2)} y {c(max(rt.values()), 2)}.</p>
<p class="llamada fuerte" data-desde="3" style="left:80px;top:730px;width:1094px">El orden de validación no se conserva en test: por eso la estabilidad se mide en validación.</p>
<p class="pie junto" style="left:1224px;top:882px;width:616px">Figura 2. Desbalance real (≈1:40), pesos por clase. Línea: media de 3 semillas. Sombra: mínimo y máximo. Puntos: umbral elegido en cada semilla.</p>''')
set_notas(5, [
    ("Resumen", "J", "Primero el resumen: el escenario con el desbalance real y la misma pérdida, pesos por clase, en las cuatro arquitecturas. Así comparamos arquitecturas sin mezclar pérdidas. La línea es la media de tres semillas, la sombra va del mínimo al máximo y cada punto es el umbral con el que se reporta el F1."),
    ("Clic 1", "J", "En validación GraphSAGE está arriba en todo el recall y su sombra es angosta: las tres semillas coinciden. GCN queda sola abajo."),
    ("Clic 2", "J", "En test todo se aplasta contra el azar y las cuatro curvas ROC quedan juntas. Aquí GCN ya no se distingue de las demás, y es lo que explicamos en dos láminas."),
    ("Clic 3", "J", "Como el orden no se conserva en test, medimos la estabilidad en validación, donde el modelo sí aprendió."),
])

# ═════════════════════════════════════════════════════════════════════════════
# Lámina 6: tabla con pesos por clase, GCN al final y marcada
# ═════════════════════════════════════════════════════════════════════════════
def fila(a, desde):
    v = lambda sp, col, d=2: "{} ± {}".format(c(met(a, sp, col)[0], d), c(met(a, sp, col)[1], d))
    nom = a if a != "GCN" else 'GCN<br><span style="font-weight:400;font-size:21px">no pasa la compuerta</span>'
    return (f'<tr data-desde="{desde}"><td class="h">{nom}</td><td>{v("val", "roc_auc")}</td>'
            f'<td>{v("val", "pr_auc")}</td><td>{v("val", "f1")}</td><td>{v("val", "ks")}</td>'
            f'<td>{v("test", "roc_auc")}</td><td>{v("test", "pr_auc", 3)}</td>'
            f'<td>{v("test", "f1")}</td><td>{v("test", "ks")}</td></tr>')


orden = sorted(APR, key=lambda a: -pv[a])
titulo(6, "Las cuatro métricas coinciden: tres arquitecturas aprenden en validación, GCN no, "
          "y todas caen al azar en test")
cuerpo(6, '<table class="tabla met"><thead><tr><th>Arquitectura</th><th>ROC-AUC val</th>'
          '<th>PR-AUC val</th><th>F1 val</th><th>KS val</th><th>ROC-AUC test</th>'
          '<th>PR-AUC test</th><th>F1 test</th><th>KS test</th></tr></thead><tbody>'
          + "".join(fila(a, 1) for a in orden) + fila("GCN", 2)
          + '<tr class="azar" data-desde="3"><td>Azar</td><td>0,50</td><td>0,02</td><td>0,05</td>'
            '<td>0</td><td>0,50</td><td>0,006</td><td>0,01</td><td>0</td></tr></tbody></table>',
       "Tabla 1. Desbalance real (≈1:40), pesos por clase en las cuatro arquitecturas, ordenadas "
       "por PR-AUC de validación. Media ± desviación estándar de 3 semillas. F1 en el umbral "
       "elegido en validación; KS: máxima distancia entre las tasas de ilícitas y lícitas "
       "detectadas.")
set_notas(6, [
    ("Tabla", "J", "Las curvas con sus números, antes de aplicar el filtro. ROC-AUC va con PR-AUC, y F1 con KS, porque con clases desbalanceadas una sola métrica no basta."),
    ("Clic 1", "J", "En validación GraphSAGE, GAT y TAGCN están muy por encima del azar en las cuatro métricas."),
    ("Clic 2", "J", "GCN apenas se separa del azar en validación: por eso no pasa la compuerta y no entra a las hipótesis. De aquí en adelante hablamos de las otras tres."),
    ("Clic 3", "J", "En test el F1 es casi cero y el PR-AUC queda en el azar para todas. El KS de test sigue alto, pero se alcanza con muchas falsas alarmas, así que no sirve para operar."),
])

# ═════════════════════════════════════════════════════════════════════════════
# Lámina nueva: la compuerta se decide en validación
# ═════════════════════════════════════════════════════════════════════════════
def puntos(split, col, lo, hi, x0, x1, y0=70, paso=72):
    X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    out = []
    for k, a in enumerate(ARQ):
        y = y0 + k * paso
        d = curm[(curm.scenario == "native") & (curm.balancing == CW) & (curm.arch == a)
                 & (curm.split == split)][col].values
        cls = "il" if a == "GCN" else "marca"
        out.append(f'<path class="guia" d="M{x0} {y}L{x1} {y}"/>')
        out.append(f'<path class="bigote" style="stroke-width:6;opacity:.35" d="M{X(d.min()):.1f} {y}L{X(d.max()):.1f} {y}"/>')
        out += [f'<circle class="{cls}" cx="{X(v):.1f}" cy="{y}" r="11"/>' for v in d]
    return "".join(out), X


def panel(titulo_p, split, col, lo, hi, ticks, azar, ox, etiqueta):
    x0, x1 = ox + 190, ox + 510
    pts, X = puntos(split, col, lo, hi, x0, x1)
    s = [f'<text class="t tb" x="{ox + 190}" y="24" font-size="27">{titulo_p}</text>']
    for k, a in enumerate(ARQ):
        s.append(f'<text class="{"t tb" if a == "GCN" else "t"}" x="{ox}" y="{79 + k * 72}" font-size="28">{a}</text>')
    s.append(pts)
    s.append(f'<path class="eje" d="M{x0} 330L{x1} 330"/>')
    for t in ticks:
        s.append(f'<path class="eje" d="M{X(t):.1f} 330L{X(t):.1f} 340"/>'
                 f'<text class="t3 tc" x="{X(t):.1f}" y="368" font-size="22">{c(t, 2)}</text>')
    s.append(f'<path class="azar" d="M{X(azar):.1f} 44L{X(azar):.1f} 330"/>'
             f'<text class="t3" x="{X(azar) + 8:.1f}" y="56" font-size="20">azar</text>')
    s.append(f'<text class="t3 tc" x="{(x0 + x1) / 2}" y="404" font-size="22">{etiqueta}</text>')
    return "".join(s)


g_val = curm[(curm.scenario == "native") & (curm.balancing == CW) & (curm.split == "val")]
g_tst = curm[(curm.scenario == "native") & (curm.balancing == CW) & (curm.split == "test")]
rng = lambda d, a, col: (d[d.arch == a][col].min(), d[d.arch == a][col].max())
gcn_t, gat_t = rng(g_tst, "GCN", "roc_auc"), rng(g_tst, "GAT", "roc_auc")
COMPUERTA = lamina(
    "<b>Resultados:</b> rendimiento de los modelos",
    "La compuerta se decide en validación: escoger con test premiaría la suerte de la semilla",
    f'''<svg class="fig" data-desde="1" viewBox="0 0 1100 420" style="left:80px;top:350px;width:1100px;height:420px" role="img" aria-label="Por semilla: PR-AUC de validación y ROC-AUC de test de las cuatro arquitecturas con pesos por clase. En validación GCN queda separada; en test sus semillas se mezclan con las de las demás."><g class="aparece">{panel("Validación · PR-AUC", "val", "pr_auc", 0.0, 0.5, [0, 0.25, 0.5], 0.024, 0, "decide la compuerta")}</g><g class="aparece" style="--d:.4s">{panel("Test · ROC-AUC", "test", "roc_auc", 0.5, 0.75, [0.5, 0.6, 0.7], 0.5, 570, "es el reporte, no se usa para escoger")}</g></svg>
<p class="llamada media" data-desde="2" style="left:80px;top:790px;width:1100px">En validación las tres semillas de GCN quedan lejos de las demás. En test van de {c(gcn_t[0], 2)} a {c(gcn_t[1], 2)} y se mezclan con las de GAT ({c(gat_t[0], 2)} a {c(gat_t[1], 2)}): ese orden lo decide la semilla.</p>
<div class="ficha" data-desde="3" style="left:1240px;top:340px;width:600px;height:400px"><p class="etq">Por qué en validación</p><p class="txt">Test es la única prueba que nadie ha tocado. Si se usa para escoger, lo que se reporta incluye la suerte de los que salieron bien.</p><p class="txt">La compuerta pregunta si el modelo aprendió algo que valga la pena explicar. Eso se contesta en el periodo que el modelo vio.</p></div>
<div class="veredicto" data-desde="4" style="left:1240px;top:760px;width:600px"><p class="etq">Lo que se concluye</p><p class="txt">GCN <b>no pasa</b>. Que en test no quede última no es aprendizaje: cerca del azar manda la semilla.</p></div>''',
    "Figura 3. Un punto por semilla; la barra clara une el mínimo y el máximo. Desbalance real, "
    "pesos por clase. Azar: PR-AUC 0,024 en validación, ROC-AUC 0,5.",
    REF["cawley"],
    [("Compuerta", "A", "Una pregunta que nos hicieron: si GCN no pasa la compuerta, ¿por qué en test no es la peor? Y detrás de esa hay otra: ¿por qué filtramos con validación y no con test?"),
     ("Clic 1 · por semilla", "A", "Aquí cada punto es una semilla. A la izquierda, validación: GCN, en naranja, queda separada de las otras tres. A la derecha, test: los puntos de las cuatro se mezclan."),
     ("Clic 2 · lectura", "A", "En test la diferencia entre arquitecturas es del mismo tamaño que la diferencia entre semillas de una misma arquitectura. Un modelo que no aprendió el periodo de entrenamiento no tiene nada que perder cuando la población cambia; los que sí aprendieron pierden lo aprendido."),
     ("Clic 3 · el principio", "A", "El motivo de fondo no depende de este dataset. Si escogemos modelos con test, test deja de ser una prueba honesta y terminamos reportando a los que tuvieron suerte. Escoger en validación también tiene sesgo, pero deja test intacto."),
     ("Clic 4 · conclusión", "A", "Por eso GCN no pasa, y por eso el resultado que reportamos como honesto es el de test, aunque sea malo."),
     ])

# ═════════════════════════════════════════════════════════════════════════════
# Curvas por escenario: figuras nuevas (leyenda arriba) y texto de referencia
# ═════════════════════════════════════════════════════════════════════════════
CURVAS = {7: "curvas_native_val.png", 8: "curvas_native_test.png", 9: "curvas_1-10_val.png",
          10: "curvas_1-10_test.png", 11: "curvas_1-10_os_val.png", 12: "curvas_1-10_os_test.png",
          13: "curvas_1-20_val.png", 14: "curvas_1-20_test.png",
          25: "anexo_curvas_1-1_val.png", 26: "anexo_curvas_1-1_test.png"}
for n, name in CURVAS.items():
    imagen(n, name, 1140, "left:812px;top:322px;height:570px;width:1028px")
    S[n] = re.sub(r'(<p class="llamada[^"]*"[^>]*style="left:80px;top:\d+px;width:)574px',
                  r'\g<1>682px', S[n])
    S[n], k = re.subn(r'<p class="pie junto" style="[^"]*">',
                      '<p class="pie junto" style="left:812px;top:900px;width:1028px">', S[n])
    assert k == 1, n
    S[n] = S[n].replace(" Color: forma de compensar el desbalance.", "")
    S[n] = S[n].replace(" Línea: media de 3 semillas. Sombra: mínimo y máximo de las 3. Puntos: umbral elegido en cada semilla.", " Línea: media de 3 semillas; sombra: mínimo y máximo; puntos: umbral de cada semilla.")
rep(7, "La curva azul (sin ajuste) queda por debajo",
    "La curva azul (sin ajuste) es la referencia. Queda por debajo")
rep(9, "En GAT, sin ajuste supera a las dos variantes que compensan.",
    "En GAT, sin ajuste supera a las otras dos: el submuestreo ya redujo el desbalance y "
    "queda menos que compensar.")
rep(9, 'data-cue="1-10 · val"><p data-voz="J">', 'data-cue="1-10 · val"><p data-voz="J">'
    "Ojo con el nombre: aquí «sin ajuste» no quiere decir sin balancear, porque el submuestreo "
    "ya es una forma de balanceo. Por eso las tres curvas se acercan: con 10 negativos por "
    "ilícita queda menos desbalance que compensar. ")

# ═════════════════════════════════════════════════════════════════════════════
# H1
# ═════════════════════════════════════════════════════════════════════════════
titulo(15, "H1: esperábamos que más desbalance volviera inestables las explicaciones; "
           "entre 1:10 y ≈1:40 no pasó")

# Comparación justa: se cruza también la semilla, porque los modelos de la misma semilla
# comparten inicialización e hiperparámetros y quedan casi iguales (coseno de pesos 0,90 a 1).
rb = pd.read_csv(R7 / "robusta_semillas.csv")
rb = rb[rb.arch.isin(APR) & (rb.explainer == "GNNExplainer") & rb.scenario.isin(MAIN)]
gg = lambda tipo, cw: rb[(rb.tipo == tipo) & ((rb.balancing == CW) == cw)]
AG = {(t, cw): gg(t, cw).sp_all.mean() for t in rb.tipo.unique() for cw in (True, False)}
AGJ = {(t, cw): gg(t, cw).jac10.mean() for t in rb.tipo.unique() for cw in (True, False)}
T_REF, T_CRUZ, T_MISMA = ("mismo escenario, otra semilla", "otro escenario, otra semilla",
                          "otro escenario, misma semilla")
XA = lambda v: 420 + (v - 0.80) / 0.20 * 560
filas = [("Mismo escenario, otra semilla", "la referencia: volver a entrenar el modelo",
          rb[rb.tipo == T_REF].sp_all.mean(), "cat-n"),
         ("Otro escenario y otra semilla", "pesos por clase", AG[(T_CRUZ, True)], "barra"),
         ("Otro escenario y otra semilla", "sin ajuste y focal loss", AG[(T_CRUZ, False)], "barra-s")]
bars = []
for k, (a, b, v, cls) in enumerate(filas):
    y = 40 + k * 104
    bars.append(
        f'<text class="t" x="0" y="{y + 26}" font-size="29">{a}</text>'
        f'<text class="t3" x="0" y="{y + 56}" font-size="21">{b}</text>'
        f'<rect class="{cls} crece" x="420" y="{y}" width="{XA(v) - 420:.1f}" height="50" style="--d:{k * .15:.2f}s"/>'
        f'<g class="aparece" style="--d:{.8 + k * .15:.2f}s"><text class="t tb" x="{XA(v) + 14:.1f}" y="{y + 36}" font-size="30">{c(v, 2)}</text></g>')
eje = '<path class="eje" d="M420 360L980 360"/>' + "".join(
    f'<path class="eje" d="M{XA(t):.1f} 360L{XA(t):.1f} 370"/><text class="t3 tc" x="{XA(t):.1f}" y="398" font-size="23">{c(t, 2)}</text>'
    for t in (0.80, 0.85, 0.90, 0.95, 1.00))
titulo(16, "H1: submuestrear negativos cambia la proporción de clases, no lo que el modelo "
           "aprende del fraude")
cuerpo(16, f'''<p class="etiqueta" data-desde="1" style="left:80px;top:330px">5 · Por qué dio eso</p><svg class="fig" data-desde="1" viewBox="0 0 1080 440" style="left:80px;top:380px;width:1080px;height:440px" role="img" aria-label="Acuerdo entre las explicaciones del mismo nodo: mismo escenario con otra semilla {c(filas[0][2], 2)}; otro escenario y otra semilla, con pesos por clase {c(filas[1][2], 2)} y con otras pérdidas {c(filas[2][2], 2)}.">{"".join(bars)}{eje}<text class="t3 tc" x="700" y="432" font-size="23">¿señalan las mismas variables? (Spearman entre las dos explicaciones del mismo nodo)</text></svg><p class="llamada media" data-desde="2" style="left:80px;top:835px;width:1080px">Cambiar además el escenario no mueve la explicación más de lo que ya la mueve cambiar la semilla.</p><div class="ficha" data-desde="2" style="left:1220px;top:330px;width:620px;height:400px"><p class="etq">El mecanismo</p><p class="txt">Quitar negativos al azar cambia la proporción de clases, no cómo son las ilícitas. Con pesos por clase la pérdida es la misma en los cuatro escenarios:</p><p class="txt" style="font-size:25px;color:var(--tinta);white-space:nowrap"><b>L = ½·media(ilícitas) + ½·media(negativos)</b></p><p class="txt">Con la misma semilla los modelos quedan casi iguales (acuerdo {c(AG[(T_MISMA, True)], 2)}).</p></div><div class="veredicto" data-desde="3" style="left:1220px;top:750px;width:620px"><p class="etq">6 · En qué impacta</p><p class="txt">Con estos escenarios H1 <b>no podía fallar</b>. Para mover la explicación hay que cambiar lo que el modelo sabe del fraude.</p></div>''',
       "Figura 12. GNNExplainer, GraphSAGE, GAT y TAGCN, 30 nodos. Las explicaciones de 1:10, "
       "1:10 + SMOTE y 1:20 se comparan con las del modelo nativo entrenado con otra semilla.")
set_ref(16, REF["elkan"])
set_notas(16, [
    ("H1 · por qué", "J", "¿Por qué no cambió? Esta vez lo medimos directamente."),
    ("Clic 1 · acuerdo", "J", "Tomamos el mismo nodo y preguntamos si dos modelos señalan las mismas variables. La referencia, en gris, es entrenar el mismo modelo en el mismo escenario con otra semilla: 0,94. Si además de la semilla cambiamos el escenario, el acuerdo es el mismo: 0,94 con pesos por clase y 0,92 con las otras pérdidas."),
    ("Clic 2 · mecanismo", "J", "O sea que el escenario no agrega nada a lo que ya mueve la semilla. La razón es que quitar negativos al azar cambia cuántos hay de cada clase, pero no cómo son las ilícitas. Con pesos por clase la función de pérdida es literalmente la misma en los cuatro escenarios: la mitad del peso para las ilícitas y la mitad para los negativos. De hecho, con la misma semilla los cuatro modelos terminan casi idénticos. Una advertencia honesta: la teoría dice que sin ajuste el submuestreo solo desplaza el puntaje en una constante, pero en nuestros modelos sin ajuste eso no se cumple, porque entrenan pocas épocas y no llegan al óptimo."),
    ("Clic 3 · impacto", "J", "La consecuencia: con estos escenarios H1 no podía fallar. No es que el desbalance no importe; es que submuestrear negativos no cambia lo que el modelo aprende del fraude. Para poner a prueba la hipótesis hay que tocar el fraude, y eso es lo que hacen los escenarios 1:100 y 1:200."),
])

# ═════════════════════════════════════════════════════════════════════════════
# H2: lámina nueva con la ecuación de cada capa
# ═════════════════════════════════════════════════════════════════════════════
titulo(17, "H2: esperábamos diferencias de estabilidad entre arquitecturas; las tres que "
           "aprenden explican igual")
sub = lambda t: f"<sub>{t}</sub>"
ECS = [
    ("GCN", f"h{sub('i')} = σ( Σ{sub('j ∈ N(i) ∪ {i}')} W·h{sub('j')} / √(d{sub('i')}d{sub('j')}) )",
     "No. El nodo y sus vecinos comparten W y se promedian", "GCN"),
    ("GraphSAGE", f"h{sub('i')} = σ( W{sub('1')}·h{sub('i')} + W{sub('2')}·media{sub('j ∈ N(i)')} h{sub('j')} )",
     f"Sí: W{sub('1')} es solo para el nodo", "GraphSAGE"),
    ("GAT", f"h{sub('i')} = σ( Σ{sub('j ∈ N(i) ∪ {i}')} α{sub('ij')}·W·h{sub('j')} )",
     f"Sí: la atención α{sub('ii')} se aprende", "GAT"),
    ("TAGCN", f"h{sub('i')} = σ( Σ{sub('k = 0…K')} (A{'<sup>k</sup>'}H){sub('i')}·W{sub('k')} )",
     f"Sí: el término k = 0 es el nodo con W{sub('0')}", "TAGCN"),
]
ECUACIONES = lamina(
    "<b>Resultados:</b> H2, arquitectura",
    "H2: la diferencia está en una pieza de la capa: si el nodo tiene o no un peso propio",
    '<table class="tabla" style="top:340px"><thead><tr><th style="width:230px">Arquitectura</th>'
    '<th style="width:640px">Cómo calcula la capa</th><th>¿Peso propio para el nodo?</th>'
    '<th style="width:230px;text-align:right">PR-AUC val</th></tr></thead><tbody>'
    + "".join(f'<tr data-desde="{1 if a == "GCN" else 2}"><td class="h">{a}</td>'
              f'<td style="font-size:29px;color:var(--tinta)">{eq}</td><td>{own}</td>'
              f'<td class="r" style="text-align:right">{c(pv[key], 2)}</td></tr>'
              for a, eq, own, key in ECS)
    + '</tbody></table>'
    '<div class="veredicto" data-desde="3" style="left:80px;top:745px;width:1760px">'
    '<p class="etq">Lo que explica</p><p class="txt">En Elliptic la señal está en las variables '
    'del nodo y solo el 12 % de los vecinos de una ilícita es ilícito. Las tres capas que '
    'conservan al nodo llegan al mismo techo de rendimiento. GCN lo mezcla con sus vecinos y '
    '<b>pierde la señal</b>: por eso es la única que no aprende.</p></div>',
    "Tabla 2. h: representación del nodo i; N(i): sus vecinos; d: grado; W: pesos que se "
    "aprenden. PR-AUC de validación con desbalance real y pesos por clase (azar 0,024).",
    REF["gnns"],
    [("H2 · ecuaciones", "A", "Nos pidieron el porqué con las ecuaciones. La diferencia entre las cuatro arquitecturas, para este problema, cabe en una pregunta: cuando la capa calcula la representación de un nodo, ¿ese nodo tiene un peso propio o se mezcla con sus vecinos?"),
     ("Clic 1 · GCN", "A", "GCN suma al nodo y a sus vecinos con la misma matriz W y normaliza por el grado. No puede tratar al nodo distinto de sus vecinos."),
     ("Clic 2 · las otras tres", "A", "GraphSAGE tiene una matriz solo para el nodo. GAT aprende cuánta atención darle al propio nodo. TAGCN tiene el término de cero saltos, que es el nodo con su propia matriz. Las tres pueden conservar lo que dice el nodo."),
     ("Clic 3 · lectura", "A", "En Elliptic la señal del fraude está en las variables del propio nodo y los vecinos casi no ayudan. Las tres que conservan al nodo llegan al mismo techo de rendimiento. Ojo: eso no quiere decir que señalen las mismas variables; entre arquitecturas solo coinciden de 2 a 5 de las 10 variables más importantes. El experimento de control de la lámina anterior lo confirma: a GCN le agregamos un peso propio y sube de 0,21 a 0,46."),
     ])

# ═════════════════════════════════════════════════════════════════════════════
# H3: redacción nueva y los explicadores de variables
# ═════════════════════════════════════════════════════════════════════════════
x = stab.merge(cfg[["scenario", "arch", "balancing", "gate_05"]], on=["scenario", "arch", "balancing"])
x = x[x.gate_05 & x.scenario.isin(MAIN)]
h3 = x.groupby(["explainer", "balancing"]).y.mean()
GRUPOS = [("GNNExplainer", "GNNExplainer", "165 variables"),
          ("ShapleyFeatures", "Shapley", "165 variables"),
          ("ExpectedGradients", "Expected Gradients", "165 variables"),
          ("PGExplainer", "PGExplainer", "2 o 3 aristas")]
Y = lambda v: 370 - v * 340
gb = []
for g, (ex, nom, obj) in enumerate(GRUPOS):
    gx = 130 + g * 225
    for k, (bal, cls) in enumerate((("none", "cat-n"), ("class_weighting", "cat-c"), ("focal_loss", "cat-f"))):
        v = h3[(ex, bal)]
        bx = gx + k * 58
        gb.append(f'<rect class="{cls} creceY" x="{bx}" y="{Y(v):.1f}" width="52" height="{370 - Y(v):.1f}" style="--d:{k * .1:.2f}s"/>'
                  f'<text class="t tb tc" x="{bx + 26}" y="{Y(v) - 10:.1f}" font-size="22">{c(v, 2)}</text>')
    gb.append(f'<text class="t tb tc" x="{gx + 84}" y="406" font-size="25">{nom}</text>'
              f'<text class="t3 tc" x="{gx + 84}" y="434" font-size="21">{obj}</text>')
guia = "".join(f'<path class="guia" d="M100 {Y(t):.0f}L1030 {Y(t):.0f}"/><text class="t3 te" x="88" y="{Y(t) + 8:.0f}" font-size="22">{c(t, 2)}</text>'
               for t in (0.25, 0.5, 0.75, 1.0))
ley = "".join(f'<rect class="{cls}" x="{130 + k * 250}" y="456" width="26" height="26" rx="4"/><text class="t" x="{166 + k * 250}" y="478" font-size="23">{t}</text>'
              for k, (cls, t) in enumerate((("cat-n", "sin ajuste"), ("cat-c", "pesos por clase"), ("cat-f", "focal loss"))))
titulo(19, "H3: la pérdida no mueve la estabilidad cuando se explican variables; solo cambia "
           "PGExplainer, que ordena aristas")
cuerpo(19, f'''<div class="ficha" data-desde="1" style="left:80px;top:330px;width:660px;height:300px"><p class="etq">1 · Qué esperábamos</p><p class="afirma">Que la pérdida no moviera la estabilidad</p><svg width="560" height="120" viewBox="0 0 560 120"><path class="eje" d="M10 6L10 104L540 104"/><rect class="cat-n creceY" x="120" y="30" width="80" height="74"/><rect class="cat-c creceY" x="240" y="30" width="80" height="74" style="--d:.1s"/><rect class="cat-f creceY" x="360" y="30" width="80" height="74" style="--d:.2s"/></svg></div><div class="ficha" data-desde="2" style="left:80px;top:650px;width:660px;height:280px"><p class="etq">2 · Por qué</p><p class="txt">La pérdida cambia cuánto pesa cada error al entrenar. Mueve el umbral del modelo, no las variables que usa para decidir.</p></div><p class="etiqueta" data-desde="3" style="left:800px;top:330px">3 · Qué dio</p><svg class="fig" data-desde="3" viewBox="0 0 1040 490" style="left:820px;top:360px;width:998px;height:470px" role="img" aria-label="Estabilidad por pérdida. GNNExplainer {c(h3[("GNNExplainer", "none")], 2)}, {c(h3[("GNNExplainer", CW)], 2)}, {c(h3[("GNNExplainer", "focal_loss")], 2)}. PGExplainer {c(h3[("PGExplainer", "none")], 2)}, {c(h3[("PGExplainer", CW)], 2)}, {c(h3[("PGExplainer", "focal_loss")], 2)}.">{guia}<path class="eje" d="M100 20L100 370L1030 370"/><text class="t3 te" x="88" y="378" font-size="22">0,00</text><g class="aparece" style="--d:.2s">{"".join(gb)}</g>{ley}</svg><div class="veredicto" data-desde="4" style="left:800px;top:838px;width:1040px;padding:12px 28px"><p class="txt" style="font-size:29px"><b>Sí</b> era lo esperado en los tres que explican variables. <b>No</b> en PGExplainer, y es el instrumento.</p></div>''',
       "Figura 15. Media de las configuraciones que aprenden, sin el 1:1. Bajo cada explicador, lo "
       "que ordena. Integrated Gradients es determinista: su estabilidad vale 1 por construcción.")
set_ref(19, REF["focal"])
set_notas(19, [
    ("H3", "J", "Tercera hipótesis: la función de pérdida con la que se compensa el desbalance."),
    ("Clic 1 · qué esperábamos", "J", "Esperábamos que no moviera la estabilidad."),
    ("Clic 2 · por qué", "J", "Porque la pérdida solo cambia cuánto pesa cada error durante el entrenamiento. Eso desplaza el umbral del modelo, pero no cambia qué variables llevan la señal."),
    ("Clic 3 · qué dio", "J", "Aquí hay que leer con cuidado, porque la vez pasada se entendió al revés. Los tres primeros grupos son explicadores que ordenan las 165 variables del nodo: GNNExplainer, Shapley y el nuevo, Expected Gradients. En los tres, las tres barras quedan a la misma altura: la pérdida no mueve nada. El cuarto grupo es PGExplainer, que no ordena variables sino aristas. Ahí sí hay una caída grande con pesos por clase."),
    ("Clic 4 · veredicto", "J", "Entonces: en los explicadores de variables salió lo esperado. Lo de PGExplainer no es un efecto de la pérdida sobre la explicación; es un problema del instrumento, y lo mostramos en la siguiente."),
])
rep(20, "Por qué class weighting satura más que las otras pérdidas todavía lo estamos verificando.",
    "Y la prueba de que es el instrumento: el mismo modelo, explicado dos veces con el mismo "
    "código, dio 0,04 y 0,42.")
rep(20, "H3 se responde con GNNExplainer y Shapley", "H3 se responde con los explicadores de variables")

# ── Lámina nueva: el explicador adicional ────────────────────────────────────
y4 = stab[stab.scenario.isin(MAIN) & stab.arch.isin(APR) & (stab.balancing == CW)]
er = y4.groupby("explainer").y.mean()
es = y4.groupby("explainer").cs.mean()
ea = expl[expl.arch.isin(APR) & expl.scenario.isin(MAIN)].groupby(["expl_a", "expl_b"]).rho.mean()
par = lambda a, b: ea.get((a, b), ea.get((b, a)))
FAM = [("GNNExplainer", "GNNExplainer", "optimiza una máscara"),
       ("ShapleyFeatures", "Shapley", "muestrea coaliciones"),
       ("IntegratedGradients", "Integrated Gradients", "gradientes, línea base fija"),
       ("ExpectedGradients", "Expected Gradients", "gradientes, línea base al azar")]
tab = "".join(
    f'<tr data-desde="1"><td class="h">{nom}<br><span style="font-weight:400;font-size:21px">{fam}</span></td>'
    f'<td class="r" style="text-align:right">{"1 (por construcción)" if ex == "IntegratedGradients" else c(er[ex], 2)}</td>'
    f'<td class="r" style="text-align:right">{c(es[ex], 2)}</td></tr>' for ex, nom, fam in FAM)
tab += (f'<tr data-desde="1"><td class="h" style="color:var(--tinta-3)">PGExplainer<br><span style="font-weight:400;font-size:21px">ordena aristas</span></td>'
        f'<td style="text-align:right">{c(er["PGExplainer"], 2)}</td><td style="text-align:right">{c(es["PGExplainer"], 2)}</td></tr>')
CEL, X0, Y0 = 150, 330, 70
mat = []
for i, (ea_, na, _) in enumerate(FAM):
    if i < len(FAM) - 1:
        mat.append(f'<text class="t te" x="{X0 + (i + 1) * CEL - 16}" y="{Y0 + i * 84 + 50}" font-size="23">{na}</text>')
    if i > 0:
        mat.append(f'<text class="t3 tc" x="{X0 + i * CEL + CEL / 2}" y="{Y0 - 14}" font-size="19">{na.replace("Integrated Gradients", "Integrated G.").replace("Expected Gradients", "Expected G.")}</text>')
    for j, (eb_, nb, _) in enumerate(FAM):
        if j <= i:
            continue
        v = par(ea_, eb_)
        op = 0.15 + 0.85 * (v - 0.3) / 0.7
        mat.append(f'<rect class="barra" x="{X0 + j * CEL + 4}" y="{Y0 + i * 84 + 4}" width="{CEL - 8}" height="76" rx="8" style="opacity:{op:.2f}"/>'
                   f'<text class="{"tw" if op > .55 else "t"} tb tc" x="{X0 + j * CEL + CEL / 2}" y="{Y0 + i * 84 + 52}" font-size="29">{c(v, 2)}</text>')
EXPLICADORES = lamina(
    "<b>Resultados:</b> el efecto del explicador",
    "Con un explicador más se ve el efecto del explicador: los cuatro son estables, pero no "
    "señalan las mismas variables",
    '<table class="tabla" style="left:80px;top:340px;width:800px"><thead><tr><th>Explicador</th>'
    '<th style="text-align:right">Entre repeticiones</th><th style="text-align:right">Entre semillas</th>'
    f'</tr></thead><tbody>{tab}</tbody></table>'
    f'<p class="etiqueta" data-desde="2" style="left:960px;top:340px">¿Señalan las mismas variables?</p>'
    f'<svg class="fig" data-desde="2" viewBox="0 0 940 420" style="left:940px;top:380px;width:900px;height:402px" role="img" aria-label="Acuerdo entre explicadores sobre el mismo modelo y los mismos nodos: GNNExplainer con Integrated Gradients {c(par("GNNExplainer", "IntegratedGradients"), 2)}; GNNExplainer con Shapley {c(par("GNNExplainer", "ShapleyFeatures"), 2)}; Shapley con Integrated Gradients {c(par("ShapleyFeatures", "IntegratedGradients"), 2)}."><g class="aparece">{"".join(mat)}</g></svg>'
    '<div class="veredicto" data-desde="3" style="left:940px;top:790px;width:900px"><p class="etq">Lo que agrega</p>'
    '<p class="txt">Estable no es lo mismo que coincidir. Cada explicador responde otra pregunta: '
    'hay que decir <b>con cuál</b> se explicó.</p></div>',
    "Tabla 3 y Figura 17. Pesos por clase, GraphSAGE, GAT y TAGCN, escenarios 1:10 a ≈1:40. "
    "Estabilidad y acuerdo: Spearman sobre las 165 variables, promedio de 30 nodos.",
    REF["ig"],
    [("Explicador adicional", "A", "Nos dijo que si un explicador resultó inapropiado, debía haber otro que lo compensara, para poder hablar del efecto del explicador. Agregamos uno de una familia que faltaba: gradientes."),
     ("Clic 1 · estabilidad", "A", "Integrated Gradients reparte la predicción entre las variables siguiendo el gradiente desde una transacción de referencia. Con una referencia fija es determinista: repetirlo da siempre lo mismo, y su estabilidad entre repeticiones vale uno por construcción. Por eso usamos también la versión con referencia al azar, Expected Gradients, que sí tiene ruido: 0,92. La columna de la derecha es más exigente: mismo nodo, mismo escenario, pero el modelo entrenado con otra semilla. Ahí Expected Gradients baja a 0,77 y los demás se quedan en 0,94 a 0,97."),
     ("Clic 2 · acuerdo", "A", "Y esto es lo nuevo. Tomamos el mismo modelo y los mismos nodos y preguntamos si dos explicadores señalan las mismas variables. GNNExplainer e Integrated Gradients coinciden bastante, 0,84. Pero Shapley coincide con ellos apenas 0,47 a 0,58."),
     ("Clic 3 · lectura", "A", "Cada explicador es estable consigo mismo y aun así no cuentan la misma historia, porque cada uno responde una pregunta distinta: uno busca la máscara mínima que conserva la predicción, otro compara contra el promedio del vecindario, otro contra una transacción lícita. Para auditar, el explicador es parte del resultado y hay que declararlo."),
     ])

# ═════════════════════════════════════════════════════════════════════════════
# Cierre: el candidato optimizado (THE-36)
# ═════════════════════════════════════════════════════════════════════════════
ot = pd.read_csv(R4 / "optuna_top/comparacion_config.csv")
oe = pd.read_csv(R4 / "optuna_top/estabilidad.csv")


def _ot(ver, scen, col):
    r = ot[(ot.version == ver) & (ot.scenario == scen)]
    return float(r[col].iloc[0])


pr_col = [k for k in ot.columns if k.startswith("val_pr_auc") and "sd" not in k and "std" not in k][0]
YB = lambda v: 330 - v / 0.6 * 300
cb = []
for g, sc in enumerate(MAIN):
    gx = 120 + g * 215
    for k, (ver, cls) in enumerate((("matriz_8", "cat-n"), ("top_150_prauc", "cat-c"))):
        v = _ot(ver, sc, pr_col)
        cb.append(f'<rect class="{cls} creceY" x="{gx + k * 74}" y="{YB(v):.1f}" width="68" height="{330 - YB(v):.1f}" style="--d:{k * .2:.2f}s"/>'
                  f'<text class="t tb tc" x="{gx + k * 74 + 34}" y="{YB(v) - 10:.1f}" font-size="24">{c(v, 2)}</text>')
    cb.append(f'<text class="t tc" x="{gx + 71}" y="366" font-size="25">{ESC_NOM[sc]}</text>')
ge = oe[oe.explainer == "GNNExplainer"] if "explainer" in oe.columns else oe
CANDIDATO = lamina(
    "<b>Cierre:</b> el candidato con optimización completa",
    "Con la optimización completa el candidato sube de 0,46 a 0,50 en validación, y sus "
    "explicaciones siguen igual de estables",
    f'''<svg class="fig" data-desde="1" viewBox="0 0 1000 430" style="left:80px;top:350px;width:1000px;height:430px" role="img" aria-label="PR-AUC de validación de GraphSAGE con pesos por clase: fase de selección frente a optimización completa, en los cuatro escenarios."><path class="eje" d="M90 20L90 330L990 330"/>{"".join(f'<path class="guia" d="M90 {YB(t):.0f}L990 {YB(t):.0f}"/><text class="t3 te" x="78" y="{YB(t) + 8:.0f}" font-size="22">{c(t, 1)}</text>' for t in (0.2, 0.4, 0.6))}<g class="aparece">{"".join(cb)}</g><rect class="cat-n" x="200" y="392" width="26" height="26" rx="4"/><text class="t" x="236" y="414" font-size="23">fase de selección</text><rect class="cat-c" x="500" y="392" width="26" height="26" rx="4"/><text class="t" x="536" y="414" font-size="23">optimización completa</text></svg>
<p class="llamada media" data-desde="1" style="left:80px;top:800px;width:1000px">GraphSAGE con pesos por clase, PR-AUC de validación. Las tres semillas coinciden (desviación ≤ 0,02).</p>
<div class="ficha" data-desde="2" style="left:1160px;top:340px;width:680px;height:400px"><p class="etq">Lo que no cambia</p><p class="txt"><b>Test:</b> PR-AUC 0,011 a 0,012, igual que antes. La caída no depende de cuánto se optimice.</p><p class="txt"><b>Estabilidad:</b> GNNExplainer 0,95 frente a 0,96 de la fase de selección; Shapley 0,99 en las dos.</p></div>
<div class="veredicto" data-desde="3" style="left:1160px;top:760px;width:680px"><p class="etq">Lo que se concluye</p><p class="txt">La matriz selecciona y el candidato optimizado confirma: <b>H1 a H3 se sostienen</b> con el mejor modelo.</p></div>''',
    "Figura 20. Media de 3 semillas. La optimización completa usa una búsqueda 19 veces más "
    "larga de hiperparámetros; azar 0,024.",
    REF["bergstra"],
    [("Candidato", "J", "Nos pidió que la matriz fuera la fase de selección y que al candidato que saliera de ahí le diéramos una optimización completa. El candidato es GraphSAGE con pesos por clase."),
     ("Clic 1 · rendimiento", "J", "Con la optimización completa el PR-AUC de validación sube de 0,44 a 0,46 hasta 0,50 a 0,51, en los cuatro escenarios y con las tres semillas de acuerdo."),
     ("Clic 2 · lo que no cambia", "J", "En test no cambia nada: sigue en el azar. Eso descarta que la caída en test fuera falta de optimización. Y la estabilidad de las explicaciones tampoco cambia: 0,95 frente a 0,96."),
     ("Clic 3 · conclusión", "J", "Entonces las conclusiones de las tres hipótesis no dependían de haber usado modelos poco optimizados. Se sostienen con el mejor modelo que pudimos entrenar."),
     ])

# ═════════════════════════════════════════════════════════════════════════════
# Escenarios de estrés (solo si ya hay resultados)
# ═════════════════════════════════════════════════════════════════════════════
ESTRES = None
if len(STRESS) == 2:
    SEIS = MAIN + STRESS
    tr = cfg[cfg.arch.isin(APR) & (cfg.balancing == CW)]
    prs = tr.groupby("scenario").val_pr_auc.mean()
    prsd = tr.groupby("scenario").val_pr_auc_sd.mean()
    nil = tr.groupby("scenario").n_train_illicit.first()
    st3 = stab[stab.arch.isin(APR) & (stab.balancing == CW)]
    gr = st3[st3.explainer == "GNNExplainer"].groupby("scenario").y.mean()
    eg = st3[st3.explainer == "ExpectedGradients"].groupby("scenario").y.mean()
    gcs = st3[st3.explainer == "GNNExplainer"].groupby("scenario").cs.mean()
    ecs = st3[st3.explainer == "ExpectedGradients"].groupby("scenario").cs.mean()
    acs = esc[esc.arch.isin(APR) & (esc.balancing == CW) & (esc.explainer == "GNNExplainer")].groupby("scenario").rho.mean()
    cols = "".join(f'<th style="text-align:right">{ESC_NOM[s]}</th>' for s in SEIS)
    def fila_e(nom, sub_, ser, d=2, fmt=None, desde=1):
        cel = "".join(f'<td class="{"r" if s in STRESS else ""}" style="text-align:right">'
                      f'{(fmt(ser[s]) if fmt else c(ser[s], d)) if s in ser.index else "ref."}</td>' for s in SEIS)
        return (f'<tr data-desde="{desde}"><td class="h">{nom}<br><span style="font-weight:400;font-size:21px">{sub_}</span></td>{cel}</tr>')
    miles = lambda v: f"{int(v):,}".replace(",", ".")
    cuerpo_e = ('<table class="tabla met" style="top:330px"><thead><tr><th>Pesos por clase, 3 arquitecturas</th>'
                + cols + '</tr></thead><tbody>'
                + fila_e("Ilícitas para entrenar", "de 3.462", nil, fmt=miles, desde=1)
                + fila_e("PR-AUC de validación", "¿aprende?", prs, desde=1)
                + fila_e("Desviación entre semillas", "del PR-AUC", prsd, desde=1)
                + fila_e("GNNExplainer", "entre repeticiones", gr, 3, desde=2)
                + fila_e("Expected Gradients", "entre repeticiones", eg, 3, desde=2)
                + fila_e("GNNExplainer", "entre semillas del modelo", gcs, 3, desde=3)
                + fila_e("Expected Gradients", "entre semillas del modelo", ecs, 3, desde=3)
                + fila_e("Acuerdo con el nativo", "mismas variables, GNNExplainer", acs, 3, desde=3)
                + '</tbody></table>')
    ESTRES = dict(cuerpo=cuerpo_e, prs=prs, prsd=prsd, gr=gr, eg=eg, gcs=gcs, ecs=ecs, acs=acs)

# ═════════════════════════════════════════════════════════════════════════════
# Resumen y plan
# ═════════════════════════════════════════════════════════════════════════════
rep(23, "<td>Mismo fraude, modelo que rinde parecido</td><td>Se pueden submuestrear negativos</td>",
    "<td>Submuestrear negativos solo desplaza el puntaje; las variables son las mismas</td>"
    "<td>Para probar H1 hay que cambiar lo que el modelo sabe del fraude</td>")
rep(23, '<td class="r">Igual (0,93 en los cuatro)</td>', '<td class="r">Igual entre 1:10 y ≈1:40 (0,93)</td>')
rep(23, "<td>La señal está en el nodo; GCN la diluye</td>",
    "<td>La señal está en el nodo; GCN no tiene peso propio para él</td>")
rep(23, '<td class="r">Como esperábamos en GNNExplainer y Shapley; PGExplainer ≈ 0</td><td>PGExplainer satura en nodos de 2 aristas</td><td>PGExplainer como limitación</td>',
    '<td class="r">Sin efecto en los explicadores de variables; PGExplainer ≈ 0</td><td>La pérdida mueve el umbral, no las variables; PGExplainer ordena 2 aristas</td><td>PGExplainer como limitación; un explicador de gradientes lo reemplaza</td>')
rep(24, '<span class="estado">Optimización completa cuando salga el candidato</span>',
    '<span class="estado ok">✓ Candidato con optimización completa</span>'
    '<span class="estado ok">✓ Explicador de gradientes</span>'
    + ('<span class="estado ok">✓ Escenarios 1:100 y 1:200</span>' if ESTRES else
       '<span class="estado">Escenarios 1:100 y 1:200 corriendo</span>')
    + '<span class="estado">Partición antes del cierre del mercado</span>')
rep(24, "<p>H1, H2 y H3 con las seis preguntas</p>",
    "<p>H1, H2 y H3 con las seis preguntas y su porqué</p><p>El efecto del explicador</p>")

# ═════════════════════════════════════════════════════════════════════════════
# Ensamblar, renumerar láminas, figuras y tablas
# ═════════════════════════════════════════════════════════════════════════════
orden_final = [S[i] for i in range(1, 7)] + [COMPUERTA] + [S[i] for i in range(7, 17)]
if ESTRES:
    orden_final.append("@@ESTRES@@")
orden_final += [S[17], S[18], ECUACIONES, S[19], S[20], EXPLICADORES, S[21], S[22], CANDIDATO,
                S[23], S[24], S[25], S[26]]
ESTRES_SLOT = orden_final.index("@@ESTRES@@") if ESTRES else None
if __name__ == "__main__":
    extra = {}
    hook = OUT / "estres_lamina.py"      # la lámina de estrés se redacta al ver los datos
    if ESTRES and hook.exists():
        exec(hook.read_text(encoding="utf8"), {"ESTRES": ESTRES, "lamina": lamina, "REF": REF,
                                                 "c": c, "extra": extra})
    if ESTRES:
        orden_final[ESTRES_SLOT] = extra.get("lamina") or lamina(
            "<b>Resultados:</b> H1, desbalance", "H1: escenarios 1:100 y 1:200",
            ESTRES["cuerpo"], "", REF["weber"], [("Estrés", "J", "")] * 4)
    nfig = ntab = 0
    out = []
    for k, s in enumerate(orden_final, 1):
        s = re.sub(r'aria-label="Lámina \d+"', f'aria-label="Lámina {k}"', s, count=1)
        s = re.sub(r'<div class="num">\d+</div>', f'<div class="num">{k}</div>', s, count=1)

        def renum(mm):
            global nfig, ntab
            t = mm.group(2)
            has_t, has_f = "Tabla" in t[:12], "Figura" in t
            if t.startswith("Tabla") and "y Figura" in t[:22]:
                ntab += 1; nfig += 1
                return mm.group(1) + re.sub(r'^Tabla \d+ y Figura \d+\.', f'Tabla {ntab} y Figura {nfig}.', t)
            if t.startswith("Tabla"):
                ntab += 1
                return mm.group(1) + re.sub(r'^Tabla \d+\.', f'Tabla {ntab}.', t)
            if t.startswith("Figura"):
                nfig += 1
                return mm.group(1) + re.sub(r'^Figura \d+\.', f'Figura {nfig}.', t)
            return mm.group(0)
        s = re.sub(r'(<p class="pie[^"]*"[^>]*>)(.*?)(?=</p>)', renum, s, flags=re.S)
        out.append(s)
    html = PRE + "\n\n  ".join(out) + POST.replace("textContent = '—'", "textContent = ''")
    assert "—" not in "".join(orden_final[1:]) and "–" not in "".join(orden_final[1:])
    (OUT / "hipotesis.html").write_text(html, encoding="utf8")
    (OUT / "preview.html").write_text(
        '<!doctype html><html><head><meta charset="utf8"><style>.ayuda{display:none!important}'
        '</style></head><body>' + html + '</body></html>', encoding="utf8")
    print(f"{len(out)} láminas, {nfig} figuras, {ntab} tablas, {len(html) / 1e6:.2f} MB -> {OUT}")
    for k, s in enumerate(out, 1):
        t = re.search(r'<h[12][^>]*>(.*?)</h[12]>', s, flags=re.S)
        print(k, re.sub(r'<[^>]+>', '', t.group(1))[:110])
