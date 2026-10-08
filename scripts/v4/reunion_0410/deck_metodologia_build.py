"""Deck de introducción y metodología («Estabilidad XAI en GNNs») con las correcciones de la
reunión con Cristian del 4-oct (THE-17, puntos B1 a B12 del acta).

Parte de la versión publicada 1790822022-19dc (10 láminas) y aplica:
  B1  lámina de entrada para público general (transacciones, patologías económicas, redes
      neuronales), con referencias
  B2  la alerta de la GNN con etiqueta visible (ilícita) y sin «caja negra» en esa lámina
  B3  una cita (paper o informe) en cada lámina, en la banda naranja
  B4  lámina que introduce qué es explicar y qué es un explicador, en contexto de negocio
  B5  «corrida» pasa a «ejecución»
  B6  sección arriba como «Sección: frase corta»; el titular queda como subtítulo
  B7  desbalance genérico (≈1:40) y pie simple pegado a la figura en la lámina de objetivos
  B8  lámina de Elliptic: qué es, cuándo, para qué se usa y por qué lo escogimos, con el texto
      a la izquierda y la figura compacta a la derecha
  B9  A, B y C en orden; C por efectos prácticos, con el resultado medido
  B10 la lámina de muestreo dice que es submuestreo y sobremuestreo, por qué y por qué SMOTE;
      1:20 reemplaza a 1:1 (que pasa al anexo)
  A3  sin Optuna: misma configuración de hiperparámetros y 3 semillas como incertidumbre
  B11 la estabilidad se define en la lámina donde aparece
Uso: ~/.local/bin/uv run --frozen python scripts/v4/reunion_0410/deck_metodologia_build.py
"""
import re
from pathlib import Path

TR = Path("/home/juan/.claude/projects/-home-juan-Escritorio-gnn-thesis-gnns-thesis/"
          "9d9b6fed-2b52-4084-9cd8-473897b3a22f/tool-results")
SRC = TR / "artifact-06d068e9-1790822022-19dc.html"
OUT = Path("/tmp/claude-1000/-home-juan-Escritorio-gnn-thesis-gnns-thesis/"
           "9d9b6fed-2b52-4084-9cd8-473897b3a22f/scratchpad/met")
OUT.mkdir(parents=True, exist_ok=True)

h = SRC.read_text(encoding="utf8")
h = h[h.index("<body>") + len("<body>"):].lstrip("\n").replace("</body></html>", "").rstrip()


def rep(a, b, n=1):
    global h
    assert h.count(a) == n, (h.count(a), a[:80])
    h = h.replace(a, b)


def sec(num):
    m = re.search(r'<section class="slide lamina l%d".*?</section>' % num, h, re.S)
    return m.group(0)


def put(num, new):
    global h
    h = h.replace(sec(num), new)


def edit(num, a, b):
    s = sec(num)
    assert s.count(a) == 1, (num, s.count(a), a[:80])
    put(num, s.replace(a, b))


# ── Citas (todas verificadas)
C = dict(
    intro="UNODC (2011), informe sobre flujos financieros ilícitos · Cheng, D. et al. (2025). Graph "
          "neural networks for financial fraud detection: a review. Frontiers of Computer Science.",
    weber="Weber, M. et al. (2019). Anti-money laundering in Bitcoin: experimenting with graph "
          "convolutional networks for financial forensics. KDD Workshop on Anomaly Detection in Finance.",
    gilmer="Gilmer, J. et al. (2017). Neural message passing for quantum chemistry. ICML · Kipf, T. y "
           "Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR.",
    explicar="Doshi-Velez, F. y Kim, B. (2017). Towards a rigorous science of interpretable machine "
             "learning. arXiv:1702.08608 · Ying, R. et al. (2019). GNNExplainer. NeurIPS.",
    agarwal="Agarwal, C. et al. (2023). Evaluating explainability for graph neural networks. "
            "Scientific Data, 10, 144.",
    alarab="Alarab, I. y Prakoonwit, S. (2022). Effect of data resampling on feature importance in "
           "imbalanced blockchain data. Data Science and Management, 5, 66-76.",
    lorenz="Lorenz, J. et al. (2020). Machine learning methods to detect money laundering in the Bitcoin "
           "blockchain in the presence of label scarcity. ICAIF.",
    smote="Chawla, N. V. et al. (2002). SMOTE: synthetic minority over-sampling technique. JAIR, 16 · "
          "Zhao, T. et al. (2021). GraphSMOTE. WSDM.",
    gnns="Kipf y Welling (2017), ICLR · Hamilton et al. (2017), NeurIPS · Veličković et al. (2018), "
         "ICLR · Du et al. (2017), arXiv:1710.10370.",
    medir="Spearman, C. (1904). The proof and measurement of association between two things. American "
          "Journal of Psychology, 15 · Jaccard, P. (1912). New Phytologist, 11.",
)

CSS = '''
/* Reunión del 4-oct: sección arriba («Sección: frase corta»), cita en la banda naranja */
.seccion { position: absolute; right: 80px; top: 52px; margin: 0; max-width: 1380px; font-size: 36px; font-weight: 400; color: var(--modelo); text-align: right; }
.seccion b { font-weight: 700; }
.ref { position: absolute; left: 80px; bottom: 0; height: 56px; max-width: 1660px; display: flex; align-items: center; margin: 0; font-size: 19px; color: #2a1606; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pie.junto { top: auto; right: auto; font-size: 24px; }
.col { position: absolute; margin: 0; padding: 22px 26px; border-radius: 16px; background: var(--particion); }
.col h3 { margin: 0 0 12px; font-size: 32px; color: var(--modelo); }
.col p { margin: 0; font-size: 27px; line-height: 1.3; color: var(--tinta-2); }
.col p + p { margin-top: 12px; }
.ficha { position: absolute; margin: 0; padding: 20px 28px; border: 2px solid #d9dee2; border-radius: 16px; background: #fff; }
.ficha .etq { margin: 0 0 10px; }
.ficha .txt { margin: 0; font-size: 29px; line-height: 1.3; color: var(--tinta-2); }
.ficha .txt + .txt { margin-top: 12px; }
.ficha .ok { color: #1f7a4d; font-weight: 700; }
.ficha .no { color: #b3261e; font-weight: 700; }
.veredicto { position: absolute; margin: 0; padding: 18px 28px; border-radius: 14px; background: var(--modelo); color: #fff; }
.veredicto .etq { margin: 0 0 6px; color: #c9d6da; }
.veredicto .txt { margin: 0; font-size: 31px; line-height: 1.25; }
.veredicto .txt b { color: #ffd2a8; }
.blq { position: absolute; left: 80px; width: 800px; margin: 0; }
.blq .etq { margin: 0 0 4px; }
.blq .enun { margin: 0; font-size: 29px; line-height: 1.28; color: var(--tinta); }
'''
rep("</style>", CSS + "</style>")

# ── B6 y B3: sección arriba y cita en la banda, en las láminas existentes
SECC = {2: "<b>Introducción:</b> el lavado como patrón de red",
        3: "<b>Marco teórico:</b> cómo decide una GNN",
        4: "<b>Marco teórico:</b> qué es la estabilidad",
        5: "<b>Objetivos:</b> pregunta, hipótesis y objetivo",
        6: "<b>Metodología:</b> el conjunto de datos Elliptic",
        7: "<b>Metodología:</b> qué cuenta como negativo",
        8: "<b>Metodología:</b> submuestreo y sobremuestreo",
        9: "<b>Metodología:</b> matriz experimental",
        10: "<b>Metodología:</b> cómo se mide la estabilidad"}
REF = {2: "weber", 3: "gilmer", 4: "agarwal", 5: "alarab", 6: "weber", 7: "lorenz", 8: "smote",
       9: "gnns", 10: "medir"}
for n in SECC:
    s = sec(n)
    s = s.replace('<h2 class="titular">', f'<p class="seccion">{SECC[n]}</p>\n    <h2 class="titular">', 1)
    s = s.replace('<div class="banda"></div>', f'<div class="banda"></div>\n    <p class="ref">{C[REF[n]]}</p>', 1)
    put(n, s)

# ── Lámina 2 (B2): sin «caja negra» y la alerta con etiqueta
edit(2, "El lavado de dinero es un patrón de red, y la GNN que lo detecta es una caja negra",
     "El lavado de dinero es un patrón de red, y una GNN puede aprender a reconocerlo")
edit(2, '<text class="tw2 tc" x="1205" y="322" font-size="30">caja negra</text>',
     '<text class="tw2 tc" x="1205" y="318" font-size="24">red neuronal</text>'
     '<text class="tw2 tc" x="1205" y="346" font-size="24">de grafos</text>')
edit(2, '<text class="t tc tb" x="1585" y="266" font-size="44">Alerta</text><text class="t2 tc" x="1585" y="306" font-size="28">transacción marcada</text>',
     '<text class="t tc tb" x="1585" y="266" font-size="44">Ilícita</text><text class="t2 tc" x="1585" y="306" font-size="26">alerta de fraude</text>')
edit(2, "una red neuronal de grafos la marca como ilícita sin decir por qué.",
     "una red neuronal de grafos la clasifica como ilícita y emite una alerta.")
edit(2, "Para detectar esas formas se usan redes neuronales de grafos, o GNN: reciben la red completa y marcan la transacción sospechosa. El problema es que entregan la alerta sin decir por qué.",
     "Para detectar esas formas se usan redes neuronales de grafos, o GNN: reciben la red completa y clasifican cada transacción como lícita o ilícita. Aquí la marca como ilícita: es una alerta de fraude.")
edit(2, "Y en un banco esa pregunta no es opcional: el analista tiene que justificar cada alerta. Para responderla hay que ver qué pasa dentro de la caja, y eso lo explica Alejandro.",
     "Y en un banco esa pregunta no es opcional: el analista tiene que justificar cada alerta. Para responderla hay que ver cómo decide la GNN, y eso lo explica Alejandro.")

# ── Lámina 4 (B5 y B11): ejecución y la definición de estabilidad a la vista
s = sec(4).replace(">Corrida 1<", ">Ejecución 1<").replace(">Corrida 2<", ">Ejecución 2<")
s = s.replace("Dos corridas del mismo explicador", "Dos ejecuciones del mismo explicador")
s = s.replace("Ejemplo ilustrativo de dos corridas de un explicador", "Ejemplo ilustrativo de dos ejecuciones de un explicador")
s = s.replace('data-cue="Clic 1 · corrida 1"', 'data-cue="Clic 1 · ejecución 1"').replace('data-cue="Clic 2 · corrida 2"', 'data-cue="Clic 2 · ejecución 2"')
s = s.replace("En una primera corrida", "En una primera ejecución").replace("Si lo corremos otra vez", "Si lo ejecutamos otra vez")
s = s.replace("Esa coincidencia entre corridas es lo que llamamos estabilidad.", "Esa coincidencia entre ejecuciones es lo que llamamos estabilidad.")
a = '<p class="llamada" data-desde="4" style="left:80px;top:884px;width:1760px;--d:0s">Ante un auditor, la misma alerta tiene que tener la misma razón.</p>'
assert s.count(a) == 1
s = s.replace(a, '<p class="llamada media" data-desde="3" style="left:80px;top:872px;width:1760px;--d:.4s"><b>Estabilidad:</b> cuánto coinciden dos ejecuciones del explicador sobre la misma alerta (1 = señalan lo mismo; 0 = nada en común).</p>')
s = s.replace('data-pasos="4"', 'data-pasos="3"', 1)
s = s.replace('<div data-paso="4" data-cue="Clic 4 · por qué importa"><p data-voz="J">Y esto importa más allá de la técnica: ante un auditor, la misma alerta tiene que tener la misma razón. Si la explicación cambia cada vez que se calcula, no sirve para justificar nada.</p></div>', '')
s = s.replace("Aquí coinciden 2 de 10, un 0,2. Esa coincidencia entre ejecuciones es lo que llamamos estabilidad.</p>",
              "Aquí coinciden 2 de 10, un 0,2. Esa coincidencia entre ejecuciones es lo que llamamos estabilidad: 1 si señalan lo mismo y 0 si no tienen nada en común.</p><p data-voz=\"J\">Y esto importa más allá de la técnica: ante un auditor, la misma alerta tiene que tener la misma razón.</p>")
put(4, s)

# ── Lámina 5 (B7): desbalance genérico y pie pegado a la figura
s = sec(5)
for a, b in [('<text class="t tc" x="200" y="482" font-size="28">1:1</text>', '<text class="t tc" x="200" y="482" font-size="28">1:10</text>'),
             ('<text class="t tc" x="450" y="482" font-size="28">1:10</text>', '<text class="t tc" x="450" y="482" font-size="28">1:20</text>'),
             ('<text class="t tc" x="700" y="482" font-size="28">1:38,4</text>', '<text class="t tc" x="700" y="482" font-size="28">≈1:40</text>'),
             ('<p class="pie">Figura 4. Qué se esperaría ver si la hipótesis es cierta y si no hay efecto (esquema, no datos). Redacción provisional, pendiente de revisión con el director.</p>',
              '<p class="pie junto" style="left:1000px;top:902px;width:840px">Figura 4. Esquema, no datos: con la hipótesis la estabilidad baja al crecer el desbalance.</p>')]:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
put(5, s)

# ── Lámina 6 (B8): Elliptic, con texto a la izquierda y figura compacta a la derecha
def barra(y, w, cls, etq, d):
    return (f'<rect class="seg {cls} crece" x="0" y="{y}" width="{w}" height="54" rx="3" style="--d:{d}s"/>'
            f'<text class="t aparece" x="{w + 14 if w < 600 else 14}" y="{y + 37}" font-size="26" style="--d:{d + .3}s"'
            f'{" fill=#fff" if False else ""}>{etq}</text>')

FW = 840
ill, lic, unk = 4545, 42019, 157205
sc = (FW - 10) / unk
svg6 = f'''<svg class="fig" viewBox="0 0 {FW} 520" style="left:1000px;top:330px;width:{FW}px;height:520px" role="img" aria-label="Elliptic: 4.545 ilícitas, 42.019 lícitas y 157.205 sin etiqueta; se entrena con los pasos 1 a 34, se valida con 35 a 42 y se prueba con 43 a 49.">
<text class="t2" x="0" y="28" font-size="26">203.769 transacciones · 234.355 flujos · 49 pasos de tiempo</text>
<g data-desde="1">
<rect class="seg seg-il crece" x="0" y="60" width="{ill * sc:.1f}" height="54" rx="3" style="--d:0s"/><text class="t tb aparece" x="{ill * sc + 14:.1f}" y="97" font-size="26" style="--d:.3s">Ilícitas · 4.545 (2,2 %)</text>
<rect class="seg seg-li crece" x="0" y="128" width="{lic * sc:.1f}" height="54" rx="3" style="--d:.2s"/><text class="t tb aparece" x="{lic * sc + 14:.1f}" y="165" font-size="26" style="--d:.5s">Lícitas · 42.019 (20,6 %)</text>
<rect class="seg crece" x="0" y="196" width="{unk * sc:.1f}" height="54" rx="3" style="--d:.4s"/><text class="t tb aparece" x="14" y="233" font-size="26" style="--d:.7s">Sin etiqueta · 157.205 (77,1 %)</text>
</g>
<g data-desde="2">
<text class="t2" x="0" y="310" font-size="26">Pasos de tiempo →</text>
<rect class="particion crece" x="0" y="330" width="{FW * 34 / 49 - 4:.1f}" height="70" rx="3" style="--d:0s"/>
<text class="t tc aparece" x="{(FW * 34 / 49) / 2:.1f}" y="373" font-size="26" style="--d:.4s">Entrenamiento · 1 a 34</text>
<text class="t tc aparece" x="{(FW * 34 / 49) / 2:.1f}" y="440" font-size="24" style="--d:.5s"><tspan class="punto-il">●</tspan> 3.462 ilícitas</text>
<rect class="particion crece" x="{FW * 34 / 49:.1f}" y="330" width="{FW * 8 / 49 - 4:.1f}" height="70" rx="3" style="--d:.5s"/>
<text class="t tc aparece" x="{FW * 38 / 49:.1f}" y="373" font-size="22" style="--d:.8s">Val. 35-42</text>
<text class="t tc aparece" x="{FW * 38 / 49:.1f}" y="440" font-size="24" style="--d:.9s"><tspan class="punto-il">●</tspan> 914</text>
<rect class="particion crece" x="{FW * 42 / 49:.1f}" y="330" width="{FW * 7 / 49:.1f}" height="70" rx="3" style="--d:.9s"/>
<text class="t tc aparece" x="{FW * 45.5 / 49:.1f}" y="373" font-size="22" style="--d:1.2s">Test 43-49</text>
<text class="t tc aparece" x="{FW * 45.5 / 49:.1f}" y="440" font-size="24" style="--d:1.3s"><tspan class="punto-il">●</tspan> 169</text>
<text class="t2 aparece" x="0" y="500" font-size="24" style="--d:1.5s">Se aprende con el pasado y se evalúa en el futuro.</text>
</g>
</svg>'''
blqs = [("Qué es", "Un grafo de transacciones reales de Bitcoin unidas por los flujos de pago, con 166 variables por transacción."),
        ("Cuándo y quién", "Lo publicaron en 2019 la empresa Elliptic, IBM y el MIT, junto con un primer estudio con GCN."),
        ("Para qué se usa", "Es el conjunto de referencia para detectar lavado con modelos sobre grafos: es el mayor grafo público de transacciones de cripto con etiquetas."),
        ("Por qué lo escogimos", "Datos reales, estructura de red, orden en el tiempo y etiquetas: lo necesario para entrenar una GNN y medir sus explicaciones.")]
ys = [330, 470, 610, 790]
cuerpo6 = "".join(f'<div class="blq" style="top:{y}px"><p class="etq">{e}</p><p class="enun">{t}</p></div>' for (e, t), y in zip(blqs, ys))
notas6 = ('<aside class="notas" hidden><div data-paso="0" data-cue="Qué es Elliptic"><p data-voz="J">Para medir esto usamos Elliptic. '
          'Es un grafo de transacciones reales de Bitcoin, unidas por los flujos de pago, que publicaron en 2019 la empresa Elliptic, IBM y el MIT. '
          'Desde entonces es el conjunto de referencia para detectar lavado con modelos sobre grafos, y lo escogimos porque trae lo que necesitamos: '
          'datos reales, la red, el orden en el tiempo y etiquetas.</p></div>'
          '<div data-paso="1" data-cue="Clic 1 · cómo está compuesto"><p data-voz="J">Solo el 2,2 % está etiquetado como ilícito y el 20,6 % como lícito. '
          'El 77 % no tiene etiqueta: nadie lo revisó.</p></div>'
          '<div data-paso="2" data-cue="Clic 2 · la partición temporal"><p data-voz="J">Respetamos el orden del tiempo: entrenamos con los pasos 1 a 34, '
          'ajustamos con los 35 a 42 y evaluamos con los 43 a 49. Así se reproduce lo que pasa en un banco. Y queda una pregunta: ¿qué hacemos con ese 77 % sin etiqueta? Alejandro.</p></div></aside>')
put(6, f'''<section class="slide lamina l6" data-pasos="2" aria-label="Lámina 6">
    <div class="logo" role="img" aria-label="Universidad Tecnológica de Pereira, Facultad de Ingenierías"></div>
    <p class="seccion">{SECC[6]}</p>
    <h2 class="titular">Elliptic es el mayor grafo público de transacciones de Bitcoin con etiquetas, y por eso lo usamos</h2>
    {cuerpo6}
    {svg6}
    <p class="pie junto" style="left:1000px;top:862px;width:{FW}px">Figura 5. Elliptic Data Set, Weber et al. (2019). Conteos verificados al cargar los datos; ilícitas etiquetadas por partición.</p>
    <div class="banda"></div>
    <p class="ref">{C["weber"]}</p>
    <div class="num">6</div>
    {notas6}
  </section>''')

# ── Lámina 7 (B9): A, B, C en orden; C por efectos prácticos y con el resultado
s = sec(7)
s = s.replace("Antes de medir, hay que justificar que las transacciones sin etiqueta se pueden tratar como lícitas",
              "Las transacciones sin etiqueta se tratan como lícitas por efectos prácticos, como lo haría un banco")
s = s.replace('aria-label="Tres formas de armar la clase negativa en entrenamiento: A solo sin etiqueta (1:30,7), C lícitas más sin etiqueta (1:38,4) y B solo lícitas (1:7,6)."',
              'aria-label="Tres formas de armar la clase negativa en entrenamiento: A solo sin etiqueta (1:30,7), B solo lícitas (1:7,6) y C lícitas más sin etiqueta (≈1:40)."')
# filas en orden A, B, C (de arriba abajo) y en ese orden de aparición
fc = re.search(r'<g class="fila fila-c">.*?</g></g>', s, re.S).group(0)
fb = re.search(r'<g class="fila fila-b">.*?</g></g>', s, re.S).group(0)
fc2 = fc.replace('y="246"', 'y="366"').replace('y="200"', 'y="320"').replace('y="247"', 'y="367"').replace('data-desde="3"', 'data-desde="3"').replace(">1:38,4<", ">≈1:40<")
fb2 = fb.replace('y="366"', 'y="246"').replace('y="320"', 'y="200"').replace('y="367"', 'y="247"').replace('data-desde="1"', 'data-desde="2"')
s = s.replace(fc, "@@C@@").replace(fb, "@@B@@").replace("@@C@@", fb2).replace("@@B@@", fc2)
fa = re.search(r'<g class="fila fila-a">.*?</g></g>', s, re.S).group(0)
s = s.replace(fa, fa.replace('data-desde="2"', 'data-desde="1"'))
s = s.replace('<p class="llamada media" data-desde="4" style="left:80px;top:772px;width:1760px;--d:0s"><b>Criterio propuesto:</b> si los modelos C rinden como los B en la misma prueba (|Δ PR-AUC| ≤ 0,05), todo el análisis sigue con C. <b>Resultado:</b> pendiente de la corrida del paso 0.</p>',
              '<p class="llamada media" data-desde="4" style="left:80px;top:772px;width:1760px;--d:0s"><b>Resultado:</b> C rinde como B (diferencia de PR-AUC −0,007; criterio ≤ 0,05). El 15,6 % de sin etiqueta que marca B no es fraude escondido: a lo sumo el 1,2 % se parece al fraude conocido.</p>')
s = s.replace('<p class="llamada" data-desde="3" style="left:80px;top:884px;width:1760px;--d:1.4s">C es el caso real: el fraude que nadie detectó no puede estar etiquetado, y su desbalance es el real.</p>',
              '<p class="llamada media" data-desde="3" style="left:80px;top:884px;width:1760px;--d:1.4s">Se escoge C por efectos prácticos: los sin etiqueta son 4 de cada 5 negativos y en su mayoría deben ser lícitos.</p>')
s = re.sub(r'<aside class="notas" hidden>.*?</aside>', '<aside class="notas" hidden>'
           '<div data-paso="0" data-cue="Solo las ilícitas, iguales en las tres filas"><p data-voz="A">Esa es la primera decisión del diseño. La clase positiva siempre son las ilícitas: 3.462 en entrenamiento. La pregunta es contra qué las comparamos.</p></div>'
           '<div data-paso="1" data-cue="Clic 1 · opción A"><p data-voz="A">Opción A: solo contra las sin etiqueta. Es lo que hacía la versión anterior y nos sirve de contraste.</p></div>'
           '<div data-paso="2" data-cue="Clic 2 · opción B"><p data-voz="A">Opción B: solo contra las lícitas revisadas. Es lo más limpio, pero deja por fuera cuatro de cada cinco negativos y su desbalance, 1 a 7,6, no es el real.</p></div>'
           '<div data-paso="3" data-cue="Clic 3 · opción C"><p data-voz="A">Opción C: lícitas y sin etiqueta juntas. No sabemos si todas las sin etiqueta son lícitas, igual que no sabemos si son ilícitas; pero como el fraude es la clase rara, la gran mayoría deben ser lícitas. Por efectos prácticos las tratamos como lícitas, que es lo que haría un banco.</p></div>'
           '<div data-paso="4" data-cue="Clic 4 · el resultado"><p data-voz="J">¿Y cómo sabemos que no estamos metiendo fraude como lícito?</p><p data-voz="A">Lo medimos. C rinde como B en la misma prueba. Y aunque B marca al 15,6 % de las sin etiqueta, eso no es fraude escondido: las lícitas etiquetadas no se parecen a las sin etiqueta, B nunca vio esa población, y a lo sumo un 1,2 % de lo que marca se parece al fraude conocido.</p></div>'
           '</aside>', s, flags=re.S)
put(7, s)

# ── Lámina 8 (B10): submuestreo y sobremuestreo; 1:20 en lugar de 1:1
s = sec(8)
s = s.replace("Ningún escenario quita ilícitas: se reducen los negativos o se crean ilícitas sintéticas",
              "Para cambiar el desbalance sin perder fraude, se submuestrean negativos y se sobremuestrean ilícitas con SMOTE")
s = s.replace('<text class="t tb" x="0" y="73" font-size="30">Nativo · 1:38,4</text>', '<text class="t tb" x="0" y="73" font-size="30">Nativo · ≈1:40</text>')
r11 = re.search(r'<g data-desde="2">\s*<text class="t tb" x="0" y="293" font-size="30">1:1</text>.*?3\.462 : 3\.462</text>\s*</g>', s, re.S).group(0)
r20 = ('<g data-desde="2">\n<text class="t tb" x="0" y="293" font-size="30">1:20</text>\n'
       '<rect class="il" x="330" y="250" width="29.22" height="64" rx="3"/>\n'
       '<g class="encoge" style="--k:1.9180;--d:.45s"><rect class="li" x="361.22" y="250" width="116.3" height="64" rx="3"/>'
       '<rect class="un" x="479.52" y="250" width="468.04" height="64" rx="3"/></g>\n'
       '<text class="t2 aparece" x="963.56" y="293" font-size="28" style="--d:1.45s">3.462 : 69.240</text>\n</g>')
s = s.replace(r11, r20)
s = s.replace("nativo 1:38,4; 1:10 y 1:1 quitando negativos", "nativo ≈1:40; 1:10 y 1:20 quitando negativos")
s = s.replace('<p class="llamada" data-desde="4" style="left:80px;top:884px;width:1760px;--d:0s">1:10 con y sin SMOTE tienen la misma razón: compararlos aísla el efecto de la técnica.</p>',
              '<p class="llamada media" data-desde="4" style="left:80px;top:862px;width:1760px;--d:0s"><b>Por qué SMOTE:</b> crea ilícitas nuevas entre dos reales parecidas en vez de repetir las mismas, y en el grafo la nueva hereda las aristas de su origen.</p>')
s = s.replace("Figura 7. Entrenamiento en modo C; validación y prueba no se tocan. Negativos submuestreados de forma estratificada (≈ 19,9 % lícitas). SMOTE (Chawla et al., 2002) con copia de aristas, línea base de GraphSMOTE (Zhao et al., 2021).",
              "Figura 7. Solo cambia el entrenamiento. Negativos tomados al azar, estratificados por origen (≈ 19,9 % lícitas). El 1:1 pasa al anexo.")
s = re.sub(r'<aside class="notas" hidden>.*?</aside>', '<aside class="notas" hidden>'
           '<div data-paso="0" data-cue="El escenario nativo"><p data-voz="A">Con C, el desbalance real es de 1 a 40, más o menos. Para estudiar su efecto construimos escenarios con una regla: nunca quitar ilícitas, porque son la señal que el modelo tiene que aprender.</p></div>'
           '<div data-paso="1" data-cue="Clic 1 · submuestreo a 1:10"><p data-voz="A">Primera técnica, el submuestreo: para llegar a 1 a 10 tomamos al azar una parte de los negativos, con la misma mezcla de lícitas y sin etiqueta que en el nativo. Es aleatorio a propósito: un método que eligiera qué negativos quitar cambiaría también qué ve el modelo.</p></div>'
           '<div data-paso="2" data-cue="Clic 2 · submuestreo a 1:20"><p data-voz="A">Lo mismo para 1 a 20, un punto intermedio entre 1 a 10 y el nativo. El 1 a 1 lo pasamos al anexo: con clases iguales, las tres pérdidas se vuelven la misma y no sirve para compararlas.</p></div>'
           '<div data-paso="3" data-cue="Clic 3 · sobremuestreo con SMOTE"><p data-voz="A">Segunda técnica, el sobremuestreo: en vez de quitar negativos, agregamos ilícitas. Usamos SMOTE, la técnica estándar: crea una ilícita nueva en un punto intermedio entre una real y una de sus cinco más parecidas, y la nueva copia las aristas de la real.</p></div>'
           '<div data-paso="4" data-cue="Clic 4 · por qué SMOTE"><p data-voz="J">¿Por qué no copiar las ilícitas tal cual? Porque el modelo memorizaría casos repetidos. Y 1 a 10 con y sin SMOTE tienen la misma razón, así que compararlos aísla el efecto de la técnica.</p></div>'
           '</aside>', s, flags=re.S)
put(8, s)

# ── Lámina 9 (A3): sin Optuna, semillas como incertidumbre, compuerta por configuración
s = sec(9)
s = s.replace("Solo los modelos que aprenden a detectar el fraude pasan a ser explicados",
              "La matriz es una fase de selección: solo las configuraciones que aprenden pasan a ser explicadas")
s = s.replace('<text class="t tc" x="455" y="218" font-size="30">1:1</text>', '<text class="t tc" x="455" y="218" font-size="30">1:20</text>')
s = s.replace(">× 3 semillas<", ">× 3 semillas (incertidumbre)<")
s = s.replace('<text class="t2 tc" x="1110" y="285" font-size="32">× 3 semillas (incertidumbre)</text>',
              '<text class="t2 tc" x="1090" y="285" font-size="28">× 3 semillas (incertidumbre)</text>')
s = s.replace(">Solo los que pasan se explican<", ">Solo se explican las que aprenden<")
s = s.replace(">Compuerta en validación: F1 ≥ 0,30 y MCC ≥ 0,15<", ">Media de 3 semillas en validación: F1 ≥ 0,30 y MCC ≥ 0,15<")
s = s.replace(">Cuántos pasan: pendiente de la corrida v4<", ">Pasan 28 de 48 configuraciones (media de 3 semillas)<")
s = s.replace('<p class="llamada" data-desde="3" style="left:80px;top:740px;width:1000px;--d:1s">Las cuatro arquitecturas reciben el mismo presupuesto de búsqueda: 8 ensayos de Optuna y 150 épocas.</p>',
              '<p class="llamada" data-desde="3" style="left:80px;top:740px;width:1000px;--d:1s">Todas usan la misma configuración de hiperparámetros; las 3 semillas miden la variación estadística.</p>')
s = s.replace("Figura 8. Matriz de la versión 4. El umbral de decisión se calibra solo con validación; la compuerta usa F1 y MCC de validación.",
              "Figura 8. Matriz de la versión 4. El umbral se calibra solo con validación. El candidato que salga de esta fase se optimiza a fondo.")
s = s.replace("Cada una con tres semillas: 144 modelos. Y todas las arquitecturas reciben el mismo presupuesto de búsqueda, para que la comparación sea justa.",
              "Cada configuración se entrena con tres semillas. Las semillas no son modelos distintos: son la incertidumbre de la configuración. Todas usan la misma configuración de hiperparámetros, para que la comparación sea justa.")
s = s.replace("Por eso hay una compuerta de calidad sobre validación, F1 de al menos 0,30 y MCC de al menos 0,15. Solo los modelos que la pasan se explican.",
              "Por eso hay una compuerta de calidad sobre validación, F1 de al menos 0,30 y MCC de al menos 0,15, sobre la media de las tres semillas. Solo se explican las configuraciones que la pasan, y de esta fase sale el candidato que luego se optimiza a fondo.")
s = s.replace('<p data-voz="nota">Cuántos modelos pasan: pendiente de la corrida v4.</p>', '')
put(9, s)

# ── Lámina 10: ejecución y cierre
s = sec(10).replace('<p data-voz="nota">Las láminas de resultados se agregan cuando termine la corrida v4.</p>',
                    '<p data-voz="nota">Los resultados siguen en el deck de hipótesis.</p>')
put(10, s)


# ── B1: lámina de entrada, después de la portada
def lamina(clase, pasos, secc, tit, cuerpo, ref, notas, num_ph="0"):
    nt = "".join(f'<div data-paso="{k}" data-cue="{c}"><p data-voz="{v}">{t}</p></div>' for k, (c, v, t) in enumerate(notas))
    return f'''
  <section class="slide lamina {clase}" data-pasos="{pasos}" aria-label="Lámina">
    <div class="logo" role="img" aria-label="Universidad Tecnológica de Pereira, Facultad de Ingenierías"></div>
    <p class="seccion">{secc}</p>
    <h2 class="titular">{tit}</h2>
{cuerpo}
    <div class="banda"></div>
    <p class="ref">{ref}</p>
    <div class="num">{num_ph}</div>
    <aside class="notas" hidden>{nt}</aside>
  </section>
'''


cols = [("Transacciones", ["Bancos, pasarelas de pago y criptomonedas como Bitcoin registran cada pago como una transacción entre cuentas.",
                           "En Bitcoin ese registro es público: se ve quién le paga a quién, aunque no se sepa quién es."]),
        ("Patologías económicas", ["Fraude: engañar para quedarse con dinero ajeno, como en el robo de identidad.",
                                   "Lavado de dinero: hacer que dinero ilícito parezca legítimo. La ONU estimó 1,6 billones de dólares lavados en 2009, el 2,7 % del PIB mundial."]),
        ("Redes neuronales", ["Bancos y empresas de análisis de blockchain usan aprendizaje automático para marcar transacciones sospechosas.",
                              "Las redes neuronales de grafos (GNN) se estudian cada vez más porque leen la red de pagos completa."])]
cuerpo1 = "".join(f'<div class="col" data-desde="{i + 1}" style="left:{80 + i * 595}px;top:340px;width:520px;height:440px"><h3>{t}</h3>' + "".join(f"<p>{p}</p>" for p in ps) + "</div>"
                  for i, (t, ps) in enumerate(cols))
s_intro = lamina("li", 3, "<b>Introducción:</b> contexto",
                 "La economía se mueve en transacciones, y la banca ya usa redes neuronales para vigilarlas",
                 cuerpo1, C["intro"],
                 [("Contexto", "J", "Antes de entrar en la tesis, un poco de contexto."),
                  ("Clic 1 · transacciones", "J", "Toda la economía se mueve en transacciones: pagos entre cuentas de bancos, de pasarelas de pago o de criptomonedas como Bitcoin. En Bitcoin, además, el registro es público: cualquiera puede ver quién le paga a quién, aunque no sepa quién es."),
                  ("Clic 2 · patologías", "J", "Sobre esas transacciones aparecen patologías económicas. El fraude, como el robo de identidad, y el lavado de dinero, que es hacer pasar dinero ilícito por legítimo. Según la ONU, en 2009 se lavaron cerca de 1,6 billones de dólares, casi el 3 % del PIB mundial."),
                  ("Clic 3 · redes neuronales", "J", "Para vigilarlas, bancos y empresas de análisis usan aprendizaje automático. Entre esos modelos, las redes neuronales de grafos se estudian cada vez más, porque leen la red de pagos completa. Veamos por qué eso importa en el lavado.")])
rep('  <section class="slide lamina l2"', s_intro + '\n  <section class="slide lamina l2"')

# ── B4: qué es explicar, antes de la lámina de la explicación que cambia
cuerpo2 = ('<p class="etq" style="position:absolute;left:80px;top:330px">Un banco cambia su modelo y empieza a alertar distinto. Hay que saber por qué antes de bloquear clientes.</p>'
           '<div class="ficha" data-desde="1" style="left:80px;top:390px;width:820px;height:330px"><p class="etq">Modelo A · explicación</p>'
           '<p class="txt">Alerta a clientes mayores de 60 años con pagos inusuales: es el patrón del robo de identidad, que afecta más a personas mayores.</p>'
           '<p class="txt"><span class="ok">✓ Tiene sentido de negocio:</span> se puede revisar y defender ante un auditor.</p></div>'
           '<div class="ficha" data-desde="2" style="left:1020px;top:390px;width:820px;height:330px"><p class="etq">Modelo B · explicación</p>'
           '<p class="txt">Alerta por una mezcla de variables v17, v93 y v128 sin relación aparente. Acierta igual que A.</p>'
           '<p class="txt"><span class="no">✗ No se entiende por qué decide:</span> no se puede saber si alerta por la razón correcta.</p></div>'
           '<div class="veredicto" data-desde="3" style="left:80px;top:756px;width:1760px"><p class="etq">Qué es un explicador</p>'
           '<p class="txt">Un algoritmo que señala qué variables y qué conexiones pesaron en una decisión. Sin él, la GNN es una <b>caja negra</b>: da la alerta, pero no la razón.</p></div>'
           '<p class="pie">Figura 4. Ejemplo ilustrativo de dos modelos con el mismo desempeño y explicaciones distintas.</p>')
s_expl = lamina("le", 3, "<b>Marco teórico:</b> qué es explicar",
                "Explicar es decir por qué el modelo alertó, en términos que el negocio pueda revisar",
                cuerpo2, C["explicar"],
                [("Qué es explicar", "A", "Antes de medir explicaciones, qué es explicar. Imaginen que un banco cambia su modelo y empieza a alertar distinto: antes de bloquear clientes, alguien tiene que entender por qué."),
                 ("Clic 1 · modelo A", "A", "Tomemos dos modelos igual de buenos. El primero explica sus alertas así: clientes mayores con pagos inusuales, que es el patrón del robo de identidad. Eso tiene sentido de negocio y se puede defender."),
                 ("Clic 2 · modelo B", "A", "El segundo acierta igual, pero sus alertas dependen de una mezcla de variables sin relación aparente. No sabemos si acierta por la razón correcta, y eso no se puede llevar a un auditor ni vender a un cliente."),
                 ("Clic 3 · el explicador", "A", "Lo que nos permite ver esa diferencia es un explicador: un algoritmo que señala qué variables y qué conexiones pesaron en cada decisión. Sin él, la GNN es una caja negra. Y si vamos a usar sus explicaciones, primero tienen que ser confiables.")])
rep('  <section class="slide lamina l4"', s_expl + '\n  <section class="slide lamina l4"')

# ── B5 global: «corrida» en lo que se ve y en las notas
for a, b in [("corridas", "ejecuciones"), ("Corridas", "Ejecuciones"), ("corrida", "ejecución"), ("Corrida", "Ejecución")]:
    h = h.replace(a, b)

# Sin rayas en el texto visible
h = h.replace("(pasos 1–34)", "(pasos 1 a 34)")
h = h.replace('font-size="26">—</text>', 'font-size="26">no</text>')

# ── Renumerar láminas y figuras
it = iter(range(2, 200)); h = re.sub(r'<div class="num">\d+</div>', lambda m: f'<div class="num">{next(it)}</div>', h)
it = iter(range(1, 200)); h = re.sub(r'(<p class="pie[^"]*"[^>]*>)Figura \d+\.', lambda m: f'{m.group(1)}Figura {next(it)}.', h)
it = iter(range(2, 200)); h = re.sub(r'aria-label="Lámina( \d+)?">', lambda m: f'aria-label="Lámina {next(it)}">', h)

vis = re.sub(r'<script.*?</script>', '', re.sub(r'src="data:[^"]+"', '', h), flags=re.S)
vis = re.sub(r'/\*.*?\*/', '', vis, flags=re.S)
for bad in ["—", "–", ".,", "orrida", "Optuna", "caja negra</text>", "38,4"]:
    assert bad not in vis, bad
(OUT / "metodologia.html").write_text(h, encoding="utf8")
(OUT / "preview.html").write_text("<!doctype html><html><head><meta charset=utf8></head><body>" + h + "</body></html>", encoding="utf8")
print(round(len(h) / 1e6, 2), "MB", len(re.findall(r'<section class="slide', h)), "láminas")
