"""Deck de hipótesis v4 con las indicaciones de Cristian (reunión 4-oct).
Parte de la versión publicada 1 (13 láminas) y aplica: título de sección arriba con el
formato «Sección: frase corta» (el titular queda como subtítulo), una cita en cada lámina en
la banda naranja, pie de figura pegado a su figura, desbalance genérico (≈1:40), sin Optuna,
definir antes de usar (cómo leer las curvas), ROC-AUC con PR-AUC y F1 con KS, y el 1:1 al anexo.
Desde el 7-oct (THE-35) el 1:20 entra al análisis principal: láminas de H1 a H3 y de curvas con 1:20."""
import base64, io, re
import pandas as pd
from PIL import Image
HERE="/tmp/claude-1000/-home-juan-Escritorio-gnn-thesis-gnns-thesis/9d9b6fed-2b52-4084-9cd8-473897b3a22f/scratchpad/hip/"
SRC="/home/juan/.claude/projects/-home-juan-Escritorio-gnn-thesis-gnns-thesis/9d9b6fed-2b52-4084-9cd8-473897b3a22f/tool-results/artifact-f737ed96-1791248821-e835.html"
REPO="/home/juan/Escritorio/gnn_thesis/gnns_thesis/"
FIG=REPO+"results_v4/reunion_0410/curvas/figuras/deck/"
CSV=REPO+"results_v4/reunion_0410/curvas/curvas_metricas_por_config.csv"

def uri(name, h):
    im=Image.open(FIG+name).convert("RGB"); w=round(im.width*h/im.height)
    b=io.BytesIO(); im.resize((w,h),Image.LANCZOS).save(b,"JPEG",quality=86,optimize=True)
    return "data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode()

# Citas cortas (todas reales) para que quepan en la banda naranja
C=dict(
 fawcett="Fawcett, T. (2006). An introduction to ROC analysis. Pattern Recognition Letters, 27(8), 861-874.",
 saito="Saito y Rehmsmeier (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLoS ONE.",
 davis="Davis, J. y Goadrich, M. (2006). The relationship between precision-recall and ROC curves. ICML, 233-240.",
 massey="Massey, F. J. (1951). The Kolmogorov-Smirnov test for goodness of fit. JASA, 46(253), 68-78.",
 weber="Weber, M. et al. (2019). Anti-money laundering in Bitcoin: experimenting with graph convolutional networks. arXiv:1908.02591.",
 chawla="Chawla, N. V. et al. (2002). SMOTE: synthetic minority over-sampling technique. JAIR, 16, 321-357.",
 yuan="Yuan, H. et al. (2023). Explainability in graph neural networks: a taxonomic survey. IEEE TPAMI, 45(5), 5782-5799.",
 agarwal="Agarwal, C. et al. (2023). Evaluating explainability for graph neural networks. Scientific Data, 10, 144.",
 lakens="Lakens, D. (2017). Equivalence tests: a practical primer. Social Psychological and Personality Science, 8(4), 355-362.",
 gnns="Kipf y Welling (2017), ICLR · Hamilton et al. (2017), NeurIPS · Veličković et al. (2018), ICLR · Du et al. (2017), arXiv.",
 hamilton="Hamilton, W. et al. (2017). Inductive representation learning on large graphs. NeurIPS, 1024-1034.",
 focal="Lin, T.-Y. et al. (2017). Focal loss for dense object detection. ICCV · King, G. y Zeng, L. (2001). Political Analysis.",
 pg="Ying, R. et al. (2019). GNNExplainer. NeurIPS · Luo, D. et al. (2020). Parameterized explainer for GNN. NeurIPS.",
 lundberg="Lundberg, S. y Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS.",
)

h=open(SRC,encoding="utf8").read()
i=h.index("<body>")+len("<body>"); h=h[i:].lstrip("\n").replace("</body></html>","").rstrip()

# ── 1. Título de sección arriba: «Sección: frase corta»; el titular queda como subtítulo
SUB={"Qué medimos":"qué es la estabilidad","Cómo respondemos":"cómo respondemos las hipótesis",
 "H1 · desbalance":"H1, desbalance","H2 · arquitectura":"H2, arquitectura","H3 · balanceo":"H3, balanceo",
 "Calidad · estable frente a correcto":"estable frente a correcto","Las cuatro preguntas":"las cuatro preguntas",
 "Láminas y documento":"láminas y documento"}
def sec(m):
    s,sub=m.group(1),m.group(2)
    return f'<p class="seccion"><b>{s}:</b> {SUB[sub]}</p>'
h,n=re.subn(r'<p class="seccion">([^<]+)</p>\s*<p class="subsec">([^<]+)</p>',sec,h); assert n==12,n
CSS='''
/* Indicaciones de Cristian (4-oct): sección arriba como «Sección: frase corta» */
.seccion { top: 52px; font-size: 36px; font-weight: 400; letter-spacing: 0; text-transform: none; max-width: 1380px; }
.seccion b { font-weight: 700; }
.ref { font-size: 20px; max-width: 1660px; }
.pie.junto { top: auto; right: auto; font-size: 24px; }
.lc .col { padding: 20px 24px; }
.lc .col p { font-size: 26px; }
.tabla.met td, .tabla.met th { text-align: right; white-space: nowrap; }
.tabla.met td:first-child, .tabla.met th:first-child { text-align: left; }
.tabla.met td { padding: 14px 12px; font-size: 25px; }
.tabla.met th { font-size: 19px; padding: 0 12px 10px; }
.tabla.met tr.azar td { color: var(--tinta-3); font-style: italic; }
'''
h=h.replace("</style>",CSS+"</style>",1)

# ── 2. Una cita (paper) en cada lámina; las que no la tenían o estaban truncadas
def setref(n,txt):
    global h
    pat=re.compile(r'(<section class="slide lamina l%d".*?<p class="ref">)(.*?)(</p>)'%n,re.S)
    h,k=pat.subn(lambda m:m.group(1)+txt+m.group(3),h); assert k==1,n
for n,k in {2:"agarwal",3:"lakens",4:"chawla",5:"weber",6:"gnns",7:"hamilton",8:"focal",9:"pg",10:"agarwal",11:"lundberg",12:"yuan",13:"weber"}.items():
    setref(n,C[k])

# ── 3. Sin Optuna en las láminas
h=h.replace("Candidato top con Optuna completo","Candidato top con optimización completa de hiperparámetros")
h=h.replace("Optuna completo cuando salga el candidato","Optimización completa cuando salga el candidato")
# THE-32 y THE-33 cerrados (7-oct): cifras nuevas en las láminas de H2 y H3
for x,y in [("Por qué class weighting lo empeora más: en verificación.",
             "El mismo modelo, explicado dos veces, da 0,04 y 0,42: la diferencia entre pérdidas cabe en ese ruido."),
            ("Con un peso propio, GCN pasa de 0,21 a 0,46.",
             "Con un peso propio pasa de 0,21 a 0,46, como Skip-GCN en Weber et al. (2019).")]:
    assert h.count(x)==1,x; h=h.replace(x,y)
assert "Optuna" not in re.sub(r'<aside.*?</aside>','',h,flags=re.S)

# ── 3b. 1:20 reemplaza al 1:1 en el análisis principal (7-oct, THE-35): cifras desde los CSV
U=pd.read_csv(REPO+"results_v4/reunion_0410/configs48/configs48_units_estabilidad.csv")
U=U[U["filter"]=="cfg_media_sin1:1"]
G=U[U.explainer=="GNNExplainer"]
CF=pd.read_csv(REPO+"results_v4/reunion_0410/configs48/configs48_metricas.csv")
EXF=pd.read_csv(REPO+"results_v4/reunion_0410/configs48/configs48_friedman_exacto.csv")
def pex(ex,hy):
    r=EXF[(EXF["filter"]=="cfg_media_sin1:1")&(EXF.explainer==ex)&(EXF.hypothesis==hy)].iloc[0]
    return r.p_exact, int(r.n_blocks)
def c(x,d=3): return f"{x:.{d}f}".replace(".",",")
def X(v): return 300+(v-0.80)*3300
def subst(n, pat, new, cnt=1):
    """Reemplaza dentro de la lámina l{n} del original."""
    global h
    m=re.search(r'<section class="slide lamina l%d".*?</section>'%n,h,re.S); sl=m.group(0)
    sl2,k=re.subn(pat,lambda _:new,sl,flags=re.S); assert k==cnt,(n,pat[:50],k)
    h=h.replace(sl,sl2)
def barras(rows, centro, y0=48, paso=72):
    """Barras horizontales de estabilidad (media ± sd) con el margen de equivalencia ±0,05."""
    out=[f'<g class="aparece" style="--d:1.1s"><rect class="margen" x="{X(centro)-165:.1f}" y="{y0-18}" width="330.0" height="{paso*len(rows)}" rx="6"/><text class="t2 tc" x="{X(centro):.1f}" y="{y0-26}" font-size="24">margen de equivalencia ±0,05</text></g>']
    for i,(lab,sub,m,sd) in enumerate(rows):
        y=y0+i*paso; cy=y+22; d=0.15*i
        out.append(f'<text class="t" x="0" y="{y+28}" font-size="29">{lab}</text><text class="t3" x="0" y="{y+56}" font-size="21">{sub}</text>'
                   f'<rect class="barra-s crece" x="300" y="{y}" width="{X(m)-300:.1f}" height="44" style="--d:{d:.2f}s"/>'
                   f'<g class="aparece" style="--d:{0.8+d:.2f}s"><path class="bigote" d="M{X(m-sd):.1f} {cy}L{X(m+sd):.1f} {cy}M{X(m-sd):.1f} {cy-12}L{X(m-sd):.1f} {cy+12}M{X(m+sd):.1f} {cy-12}L{X(m+sd):.1f} {cy+12}"/>'
                   f'<circle cx="{X(m):.1f}" cy="{cy}" r="9" class="marca"/><text class="t tb" x="{X(m+sd)+16:.1f}" y="{cy+11}" font-size="30">{c(m)}</text></g>')
    return "".join(out)
def eje(y):
    t=f'<path class="eje" d="M300 {y}L960 {y}"/>'
    for k in range(5):
        x=300+k*165; t+=f'<path class="eje" d="M{x:.1f} {y}L{x:.1f} {y+10}"/><text class="t3 tc" x="{x:.1f}" y="{y+38}" font-size="23">{c(0.80+0.05*k,2)}</text>'
    return t

# Lámina 3: el embudo pasa a 60 → 48 (sin 1:1, con 1:20) → las que aprenden
NG=int(CF[(CF.scenario!="1:1")].g_cfg_mean.sum())
porA={a:int(CF[(CF.scenario!="1:1")&(CF.arch==a)].g_cfg_mean.sum()) for a in ["GraphSAGE","TAGCN","GAT","GCN"]}
subst(3,r'sobre las 20 configuraciones que aprenden',f'sobre las {NG} configuraciones que aprenden')
subst(3,r'aria-label="De 48 configuraciones.*?">',f'aria-label="De 60 configuraciones, 48 sin el 1:1 y {NG} que aprenden: GraphSAGE {porA["GraphSAGE"]}, TAGCN {porA["TAGCN"]}, GAT {porA["GAT"]} y GCN {porA["GCN"]}.">')
subst(3,r'font-size="72">48</text>(.*?)4 arquitecturas × 4 escenarios','font-size="72">60</text><text class="t" x="128" y="60" font-size="29">configuraciones</text><text class="t3" x="128" y="94" font-size="23">4 arquitecturas × 5 escenarios')
subst(3,r'font-size="72">36</text><text class="t" x="188" y="210" font-size="29">sin el 1:1</text>','font-size="72">48</text><text class="t" x="188" y="210" font-size="29">sin el 1:1 (entra el 1:20)</text>')
subst(3,r'font-size="72">20</text>',f'font-size="72">{NG}</text>')
subst(3,r'<text class="t2" x="150" y="476".*?GCN 0 de 9</text>',
      f'<text class="t2" x="130" y="476" font-size="24">GraphSAGE {porA["GraphSAGE"]} de 12</text><text class="t2" x="390" y="476" font-size="24">TAGCN {porA["TAGCN"]} de 12</text><text class="t2" x="610" y="476" font-size="24">GAT {porA["GAT"]} de 12</text><text class="t2" x="790" y="476" font-size="24">GCN {porA["GCN"]} de 12</text>')
subst(3,r'Ya no contamos 144 modelos: son 48 configuraciones con tres semillas cada una\. Sacamos el 1:1 al anexo, porque ahí las tres pérdidas son la misma\.',
      'Ya no contamos modelos: son 60 configuraciones con tres semillas cada una. Sacamos el 1:1 al anexo, porque ahí las tres pérdidas son la misma, y en su lugar entra el 1:20: quedan 48.')
subst(3,r'Quedan 20\. GCN no tiene ninguna\.',f'Quedan {NG}. GCN no tiene ninguna.')

# Lámina 4 (H1): cuatro escenarios
ESC4=[("1:10","1:10"),("1:10_os","1:10 + SMOTE"),("1:20","1:20"),("native","Nativo (≈1:40)")]
st={k:(G[G.scenario==k].y.mean(),G[G.scenario==k].y.std(),len(G[G.scenario==k])) for k,_ in ESC4}
rows=[(lab,f"{st[k][2]} configuraciones",st[k][0],st[k][1]) for k,lab in ESC4]
svg4=(f'<svg class="fig" data-desde="3" viewBox="0 0 1040 440" style="left:800px;top:380px;width:1040px;height:440px" role="img" aria-label="Estabilidad de GNNExplainer: '
      +", ".join(f"{lab} {c(st[k][0])}" for k,lab in ESC4)+'. Las cuatro caben en el margen de ±0,05.">'
      +barras(rows,st["native"][0])+eje(345)+'<text class="t3 tc" x="630.0" y="425" font-size="23">estabilidad de GNNExplainer (media ± sd entre configuraciones)</text></svg>')
subst(4,r'<svg class="fig" data-desde="3".*?</svg>',svg4)
subst(4,r'Las tres son equivalentes \(TOST','Las cuatro son equivalentes (TOST')
subst(4,r'Pero la estabilidad sale igual en los tres escenarios: 0,928, 0,928 y 0,931,',
      'Pero la estabilidad sale igual en los cuatro escenarios: '+", ".join(c(st[k][0]) for k,_ in ESC4[:-1])+" y "+c(st["native"][0])+',')

# Lámina 5 (H1, por qué): cuatro filas de composición
pr={k:CF[(CF.scenario==k)&CF.g_cfg_mean].val_pr_auc_mean.mean() for k,_ in ESC4}
roc=[CF[(CF.scenario==k)&CF.g_cfg_mean].val_roc_auc_mean.mean() for k,_ in ESC4]
NEG={"1:10":34620,"1:10_os":69240,"1:20":69240,"native":132803}; K=584.8/132803
svg5=['<svg class="fig" data-desde="1" viewBox="0 0 1080 430" style="left:80px;top:380px;width:1080px;height:430px" role="img" aria-label="Composición del entrenamiento: las mismas 3.462 ilícitas en los cuatro escenarios y 34.620, 69.240 (con SMOTE), 69.240 o 132.803 negativos. PR-AUC de validación '
      +", ".join(c(pr[k],2) for k,_ in ESC4)+'."><text class="t3 tb" x="250" y="26" font-size="22" letter-spacing="1">LO QUE VE EL MODELO AL ENTRENAR</text><text class="t3 tb tc" x="975" y="26" font-size="22" letter-spacing="1">PR-AUC VAL</text>']
for i,(k,lab) in enumerate(ESC4):
    y=56+i*78; d=0.15*i; x0=267.2
    r=f'<text class="t" x="0" y="{y+32}" font-size="30">{lab}</text><rect class="il" x="250" y="{y}" width="15.2" height="46"/>'
    if k=="1:10_os":
        r+=f'<rect class="smote" x="266.2" y="{y+1}" width="13.2" height="44"/>'; x0=282.5
    r+=(f'<rect class="un crece" x="{x0}" y="{y}" width="{NEG[k]*K:.1f}" height="46" style="--d:{d:.2f}s"/><text class="t" x="{x0+12}" y="{y+31}" font-size="23">{NEG[k]:,} negativos</text>'.replace(",",".")
        +f'<text class="t tb tc aparece" x="975" y="{y+34}" font-size="34" style="--d:{0.6+d:.2f}s">{c(pr[k],2)}</text>')
    svg5.append(r)
svg5.append('<rect class="il" x="250" y="388" width="26" height="26"/><text class="t2" x="288" y="410" font-size="23">ilícitas: las mismas 3.462 siempre</text><rect class="smote" x="660" y="388" width="26" height="26"/><text class="t2" x="698" y="410" font-size="23">ilícitas sintéticas</text></svg>')
subst(5,r'<svg class="fig" data-desde="1".*?</svg>',"".join(svg5))
subst(5,r'El fraude es el mismo en los tres escenarios y el modelo rinde parecido \(ROC-AUC 0,89 a 0,90\)',
      f'El fraude es el mismo en los cuatro escenarios y el modelo rinde parecido (ROC-AUC {c(min(roc),2)} a {c(max(roc),2)})')
subst(5,r'Con 1:20 \(en ejecución\) la escalera queda 1:10, 1:20, ≈1:40\.',
      f'Con el 1:20 la escalera queda 1:10, 1:20, ≈1:40, y el 1:20 también es equivalente al nativo (diferencia de {c(abs(st["1:20"][0]-st["native"][0]))}).')
subst(5,r'Y el rendimiento en validación es parecido: 0,38, 0,38 y 0,34\.',
      'Y el rendimiento en validación es parecido: '+", ".join(c(pr[k],2) for k,_ in ESC4[:-1])+" y "+c(pr["native"],2)+'.')
subst(5,r'Por eso agregamos el 1:20, que ya está corriendo\.',
      'Por eso agregamos el 1:20, que queda en medio, y da lo mismo.')

# Lámina 6 (H2): tres arquitecturas con las 28 configuraciones
AR=["GraphSAGE","GAT","TAGCN"]
sa={a:(G[G.arch==a].y.mean(),G[G.arch==a].y.std(),len(G[G.arch==a])) for a in AR}
svg6=(f'<svg class="fig" data-desde="3" viewBox="0 0 1040 540" style="left:800px;top:380px;width:1040px;height:540px" role="img" aria-label="'
      +", ".join(f"{a} {c(sa[a][0])}" for a in AR)+', dentro del margen. GCN no aprende en ninguna configuración.">'
      +barras([(a,f"{sa[a][2]} configuraciones",sa[a][0],sa[a][1]) for a in AR],G.y.mean(),y0=48,paso=85)
      +'<text class="t3" x="0" y="337" font-size="31">GCN</text><text class="t3" x="300" y="337" font-size="27">ninguna configuración aprende: no se explica</text>'
      +eje(410)+'</svg>')
subst(6,r'<svg class="fig" data-desde="3".*?</svg>',svg6)
subst(6,r'Las tres que aprenden quedan juntas: 0,938, 0,932 y 0,919, dentro del margen\.',
      'Las tres que aprenden quedan juntas: '+", ".join(c(sa[a][0]) for a in AR[:-1])+" y "+c(sa["TAGCN"][0])+', dentro del margen. Con Shapley la prueba detecta diferencias, pero son menores de cinco milésimas: equivalentes para efectos prácticos.')

# Lámina 8 (H3): barras con las 28 configuraciones y el p exacto
P=U[U.explainer=="PGExplainer"]
gb={b:G[G.balancing==b].y.mean() for b in ["none","class_weighting","focal_loss"]}
pb={b:P[P.balancing==b].y.mean() for b in ["none","class_weighting","focal_loss"]}
pg,nbg=pex("GNNExplainer","H3"); pp,nbp=pex("PGExplainer","H3")
def bar(x,v,cl,d):
    hh=v*340; return f'<rect class="{cl} creceY" x="{x}" y="{370-hh:.1f}" width="100" height="{hh:.1f}" style="--d:{d:.2f}s"/><text class="t tb tc" x="{x+50}" y="{358-hh:.1f}" font-size="28">{c(v,2)}</text>'
CL={"none":"cat-n","class_weighting":"cat-c","focal_loss":"cat-f"}
gG="".join(bar(150+120*i,gb[b],CL[b],0.12*i) for i,b in enumerate(CL))
gP="".join(bar(600+120*i,pb[b],CL[b],0.12*i) for i,b in enumerate(CL))
subst(8,r'aria-label="GNNExplainer: 0,931.*?">',
      f'aria-label="GNNExplainer: {c(gb["none"])}, {c(gb["class_weighting"])} y {c(gb["focal_loss"])}, sin efecto (p exacto {c(pg,2)}). PGExplainer: {c(pb["none"])}, {c(pb["class_weighting"])} y {c(pb["focal_loss"])} (p exacto {c(pp)}).">')
subst(8,r'<g class="aparece" style="--d:0.2s">.*?<text class="t tb tc" x="320" y="408"',f'<g class="aparece" style="--d:0.2s">{gG}<text class="t tb tc" x="320" y="408"')
subst(8,r'p exacto = 0,27: sin efecto',f'p exacto = {c(pg,2)}: sin efecto')
subst(8,r'<g class="aparece" style="--d:0.6s">.*?<text class="t tb tc" x="770" y="408"',f'<g class="aparece" style="--d:0.6s">{gP}<text class="t tb tc" x="770" y="408"')
subst(8,r'p exacto = 0,07, caída grande',f'p exacto = {c(pp)}: caída grande')
subst(8,r'PGExplainer: ≈ 0 con class weighting\.',f'PGExplainer ({c(pb["class_weighting"],2)}).')
subst(8,r'Friedman exacto por permutación \(4 bloques\)',f'Friedman exacto por permutación ({nbg} bloques)')
subst(8,r'GNNExplainer ya no muestra efecto: el p exacto es 0,27\. Antes daba 0,0498, pero era porque promediábamos solo las semillas que pasaban, justo lo de la semilla con suerte, y además era la aproximación asintótica con solo cuatro bloques\. Lo que sí se ve grande es la caída de PGExplainer con class weighting: casi cero\. Con cuatro bloques no alcanza significancia, p exacto 0,07, pero la caída aparece en GraphSAGE y en TAGCN\.',
      f'GNNExplainer ya no muestra efecto: el p exacto es {c(pg,2)} y las diferencias caben en el margen de equivalencia. Antes daba 0,0498, pero era porque promediábamos solo las semillas que pasaban, justo lo de la semilla con suerte, y además era la aproximación asintótica con pocos bloques. Lo que sí se ve grande es la caída de PGExplainer con class weighting: {c(pb["class_weighting"],2)}. Con el 1:20 ya hay {nbp} bloques y es significativa, p exacto {c(pp)}.')

# Láminas 12 y 13
subst(12,r'Igual \(0,93 en los tres\)','Igual (0,93 en los cuatro)')
subst(13,r'<span class="estado">1:20 en ejecución</span><span class="estado">Análisis de los sin etiqueta \(PSI, PCA, grafo\) en curso</span>',
      '<span class="estado ok">✓ 1:20 en el análisis principal</span><span class="estado ok">✓ Análisis de los sin etiqueta</span>')
subst(13,r'✓ Curvas ROC y PR de las 48 configuraciones','✓ Curvas ROC y PR de las 60 configuraciones')

# ── 4. Láminas nuevas
TOP=330
def notas(ns): return "".join(f'<div data-paso="{k}" data-cue="{c}"><p data-voz="J">{t}</p></div>' for k,(c,t) in enumerate(ns))
def lamina(sec, tit, cuerpo, ref, ns, pasos):
    return f'''
  <section class="slide lamina lc" data-pasos="{pasos}" aria-label="Lámina">
    <div class="logo" role="img" aria-label="Universidad Tecnológica de Pereira, Facultad de Ingenierías"></div>
    <p class="seccion">{sec}</p>
    <h2 class="titular">{tit}</h2>
{cuerpo}
    <div class="banda"></div>
    <p class="ref">{ref}</p>
    <div class="num">0</div>
    <aside class="notas" hidden>{notas(ns)}</aside>
  </section>
'''
def con_figura(sec, tit, img, aspect, fh, llamadas, pie, ref, ns):
    """Texto a la izquierda, figura a la derecha y su pie justo debajo de ella."""
    fw=round(fh*aspect); fx=1840-fw; tw=fx-80-50
    ll=""; y=TOP
    for i,(t,cl) in enumerate(llamadas,1):
        ll+=f'<p class="llamada {cl}" data-desde="{i}" style="left:80px;top:{y}px;width:{tw}px">{t}</p>\n'
        y+=200
    cuerpo=(f'<img class="fig" src="{uri(img,fh*2)}" alt="{pie}" style="left:{fx}px;top:{TOP}px;height:{fh}px;width:{fw}px">\n'
            f'{ll}<p class="pie junto" style="left:{fx}px;top:{TOP+fh+12}px;width:{fw}px">Figura 0. {pie}</p>')
    return lamina(sec,tit,cuerpo,ref,ns,len(llamadas))

A_CUR=3804/1876; A_RES=2451/2150
SR="<b>Resultados:</b> rendimiento de los modelos"
LEE=" Línea: media de 3 semillas. Sombra: mínimo y máximo de las 3. Puntos: umbral elegido en cada semilla."
LOG=" En test la precisión va en escala logarítmica."

# 4a. Cómo leer las curvas (definir antes de usar)
cols=[("El umbral","El modelo da a cada transacción un puntaje de sospecha. El umbral decide desde qué puntaje se alerta.","Cada punto de la curva es el mismo modelo con otro umbral. El modelo final es un solo punto."),
 ("Curva ROC","Eje x: lícitas alertadas por error (FPR). Eje y: ilícitas detectadas (TPR).","Dice si el modelo separa las dos clases. La diagonal es el azar."),
 ("Curva PR","Eje x: ilícitas detectadas (recall). Eje y: de las alertas, cuántas son ilícitas (precisión).","Dice si aprende la clase minoritaria. El azar es la prevalencia: 2,4 % en validación."),
 ("Cómo se leen","Siempre juntas: una ve lo que la otra no. Si una curva va arriba en todo, ese modelo domina; si se cruzan, el área sola no decide.","Curva suave: aprendió. Curva con picos: suerte de una ejecución. En las figuras, la línea es la media de 3 semillas, la sombra su mínimo y máximo, y los puntos el umbral elegido.")]
cuerpo="".join(f'<div class="col" data-desde="{i}" style="left:{80+i0*445}px;top:{TOP}px;width:400px;height:560px"><h3>{t}</h3><p>{a}</p><p>{b}</p></div>' for i0,(t,a,b) in enumerate(cols) for i in [i0+1])
s_leer=lamina("<b>Metodología:</b> cómo leer las curvas ROC y PR",
 "Cada punto de una curva es el mismo modelo con otro umbral de alerta", cuerpo, C["fawcett"],
 [("Cómo leer","Antes de los resultados, cómo se leen las curvas, porque el área bajo la curva es un resumen de la curva entera."),
  ("Clic 1 · umbral","El modelo no dice sí o no: da un puntaje. El umbral decide qué se alerta. Cada umbral da un punto de la curva, y el modelo que usaríamos en el banco es un solo punto."),
  ("Clic 2 · ROC","La ROC dice si el modelo separa lícitas de ilícitas. Arriba a la izquierda es bueno; la diagonal es tirar una moneda."),
  ("Clic 3 · PR","La PR dice si aprende la clase rara. Con 2,4 % de ilícitas, alertar al azar da 2,4 % de precisión."),
  ("Clic 4 · lectura","Se leen juntas. Si una curva va arriba en todo el rango, ese modelo es mejor; si se cruzan, depende del umbral. Y una curva suave indica aprendizaje; los picos, suerte.")],4)

# 4b. Resumen nativo
s_res=con_figura(SR,"En validación GraphSAGE con pesos por clase es la mejor curva en todo umbral; en test ninguna se separa del azar",
 "resumen_native_mejor_por_arquitectura.png",A_RES,540,
 [("Validación: la curva de GraphSAGE con pesos por clase va arriba en todo el recall y su sombra es la más angosta.","media"),
  ("Test: las cuatro curvas PR quedan pegadas a la línea de azar y GAT cae bajo la diagonal ROC.","media"),
  ("El orden de validación no se conserva en test: por eso medimos la estabilidad en validación.","fuerte")],
 "Desbalance real (≈1:40), mejor pérdida de cada arquitectura. Arriba ROC, abajo PR; izquierda validación, derecha test.",
 C["saito"],
 [("Resumen","Primero el resumen: el escenario con el desbalance real y, para cada arquitectura, la pérdida que mejor le fue en validación."),
  ("Clic 1","En validación GraphSAGE con pesos por clase está arriba en todo el recall, y su sombra es angosta: las tres semillas coinciden."),
  ("Clic 2","En test todo se aplasta contra el azar. GAT incluso queda bajo la diagonal: ordena al revés."),
  ("Clic 3","Como el orden no se conserva en test, medimos la estabilidad en validación, donde el modelo sí aprendió.")])

# 4c. Tabla con las cuatro métricas: ROC-AUC con PR-AUC y F1 con KS
d=pd.read_csv(CSV); best={"GCN":"focal_loss","GraphSAGE":"class_weighting","GAT":"focal_loss","TAGCN":"class_weighting"}
LL={"none":"sin balanceo","class_weighting":"pesos por clase","focal_loss":"focal loss"}
def v(a,sp,m):
    r=d[(d.scenario=="native")&(d.arch==a)&(d.balancing==best[a])&(d.split==sp)].iloc[0]
    return f'{r[m+"_mean"]:.2f} ± {r[m+"_std"]:.2f}'.replace(".",",")
mets=[("roc_auc","ROC-AUC"),("pr_auc","PR-AUC"),("f1","F1"),("ks","KS")]
th="".join(f"<th>{n} {('val' if sp=='val' else 'test')}</th>" for sp in ("val","test") for _,n in mets)
filas="".join(f'<tr data-desde="{1 if a!="GCN" else 2}"><td class="h">{a}<br><span style="font-weight:400;font-size:21px">{LL[best[a]]}</span></td>'+"".join(f"<td>{v(a,sp,m)}</td>" for sp in ("val","test") for m,_ in mets)+"</tr>" for a in best)
filas+='<tr class="azar" data-desde="3"><td>Azar</td><td>0,50</td><td>0,02</td><td>0,05</td><td>0</td><td>0,50</td><td>0,01</td><td>0,01</td><td>0</td></tr>'
cuerpo=(f'<table class="tabla met"><thead><tr><th>Arquitectura</th>{th}</tr></thead><tbody>{filas}</tbody></table>'
        f'<p class="pie">Tabla 1. Desbalance real (≈1:40), la mejor pérdida de cada arquitectura según validación. Media ± desviación estándar de 3 semillas. F1 en el umbral elegido en validación; KS: máxima distancia entre las tasas de ilícitas y lícitas detectadas. Azar de F1 y PR-AUC: la prevalencia.</p>')
s_tab=lamina(SR,"Las cuatro métricas dicen lo mismo: los modelos aprenden en validación y caen al azar en test",cuerpo,C["massey"],
 [("Tabla","Las curvas con sus números. ROC-AUC va con PR-AUC, y F1 con KS, porque con clases desbalanceadas una sola métrica no basta."),
  ("Clic 1","En validación GraphSAGE, GAT y TAGCN están muy por encima del azar en las cuatro."),
  ("Clic 2","GCN apenas se separa del azar: por eso no pasa la compuerta."),
  ("Clic 3","En test el F1 es casi cero y el PR-AUC queda en el azar. El KS de test sigue alto, pero se alcanza con muchas falsas alarmas, así que no sirve para operar.")],3)

ESC=[("native","con el desbalance real (≈1:40)"),("1-10","1:10"),("1-10_os","1:10 con SMOTE"),("1-20","1:20")]
porsc=[
 ("native","val","Con el desbalance real, no compensar el desbalance da la peor curva en las cuatro arquitecturas",
  [("La curva azul (sin balanceo) queda por debajo de las otras dos en todo el recall en 7 de los 8 pares.","media"),
   ("GCN tiene las sombras más anchas: sus semillas no coinciden y ninguno de sus modelos pasa la compuerta.","media"),
   ("Los puntos caen en el codo de la ROC: el umbral elegido está bien puesto para validación.","media")],C["saito"],
  ["Desbalance real en validación: arriba la ROC, abajo la PR, una columna por arquitectura; el color es la forma de compensar el desbalance.",
   "Sin balanceo, en azul, va abajo en todo el rango. Es el único escenario donde la pérdida ordena las curvas sin cruces.",
   "GCN tiene sombras anchas: las semillas dan curvas distintas.",
   "Los puntos son el umbral elegido en cada semilla: caen en el codo, donde se detecta mucho con pocas falsas alarmas."]),
 ("native","test","En test, con el desbalance real, ninguna curva PR se separa del azar y dos ROC quedan por debajo de él",
  [("Las curvas PR van pegadas a la línea de azar: a lo sumo el doble del azar.","media"),
   ("GraphSAGE y TAGCN sin balanceo ordenan al revés: su ROC queda bajo la diagonal.","media"),
   ("Los puntos caen cerca del origen: el umbral de validación casi no alerta en test.","media")],C["weber"],
  ["El mismo escenario en test, que son los meses posteriores.",
   "Las curvas PR quedan sobre la línea de azar.",
   "Dos modelos ordenan al revés: lo que aprendieron como fraude apunta al lado contrario en los últimos meses. Weber y colegas reportan este quiebre tras el cierre de un mercado oscuro.",
   "El umbral de validación casi no marca nada en test."]),
 ("1-10","val","Al dejar 10 lícitas por cada ilícita, compensar el desbalance deja de ayudar y las curvas se cruzan",
  [("En GAT, sin balanceo supera a las dos variantes que compensan.","media"),
   ("GraphSAGE con pesos por clase sigue siendo la mejor curva.","media"),
   ("En TAGCN las curvas se cruzan: el área sola no basta para ordenarlas.","media")],C["davis"],
  ["Ahora con 10 lícitas por cada ilícita.",
   "El balanceo deja de ayudar: el submuestreo ya corrigió buena parte del desbalance.",
   "GraphSAGE con pesos por clase sigue arriba.",
   "En TAGCN las curvas se cruzan: según el umbral, gana una u otra."]),
 ("1-10","test","En test, con 1:10, se repite la caída al azar; pesos por clase es la que mejor resiste en GraphSAGE",
  [("Las curvas PR vuelven a quedar sobre la línea de azar.","media"),
   ("GraphSAGE con pesos por clase tiene la ROC más alta; TAGCN con focal loss cae bajo la diagonal.","media"),
   ("GCN sin balanceo tiene el área PR más alta, pero su sombra es enorme: es suerte de una semilla.","media")],C["weber"],
  ["1:10 en test.","Mismo patrón: pegadas al azar.",
   "Pesos por clase es la más robusta al cambio en el tiempo; focal loss en TAGCN se invierte.",
   "El área más alta es de GCN, pero la sombra muestra que la da una sola semilla con suerte."]),
 ("1-10_os","val","Con SMOTE las curvas quedan casi iguales que en 1:10: GraphSAGE arriba y GCN abajo",
  [("GraphSAGE con pesos por clase y con focal loss quedan prácticamente empatadas arriba.","media"),
   ("La sombra ancha de TAGCN con pesos por clase es una semilla que no aprende.","media"),
   ("GCN queda abajo con las tres pérdidas.","media")],C["chawla"],
  ["1:10 con SMOTE, que crea ilícitas sintéticas.",
   "GraphSAGE sigue arriba con las dos pérdidas que compensan.",
   "La sombra ancha de TAGCN es una semilla que no aprendió.",
   "GCN otra vez abajo."]),
 ("1-10_os","test","En test, SMOTE no evita la caída al azar y dos configuraciones quedan por debajo de él",
  [("Las curvas PR vuelven a quedar sobre la línea de azar.","media"),
   ("Lo poco que se separa del azar está en tasas de falsas alarmas altas, que no sirven para operar.","media"),
   ("GraphSAGE sin balanceo y TAGCN con focal loss ordenan al revés.","media")],C["weber"],
  ["Y en test.","Las PR otra vez en el azar.",
   "Lo que se separa del azar está a la derecha de la ROC, con muchas falsas alarmas.",
   "SMOTE no cambia la caída en test."]),
 ("1-20","val","Con 20 lícitas por cada ilícita, GraphSAGE con pesos por clase repite el resultado del desbalance real",
  [("GraphSAGE con pesos por clase vuelve a ser la mejor curva (PR-AUC 0,46, igual que con el desbalance real).","media"),
   ("Las curvas se cruzan en GAT y TAGCN: como en 1:10, compensar el desbalance ya no ordena las pérdidas.","media"),
   ("TAGCN con focal loss es la segunda y la que menos varía entre semillas: su sombra casi no se ve.","media")],C["davis"],
  ["El 1:20, que pidió Cristian para tener un punto entre 1:10 y el desbalance real.",
   "GraphSAGE con pesos por clase da lo mismo que con el desbalance real: 0,46.",
   "En GAT y TAGCN las curvas se cruzan, como en 1:10.",
   "TAGCN con focal loss queda segunda, con una sombra casi invisible: las tres semillas coinciden."]),
 ("1-20","test","En test, con 1:20, se repite la caída al azar y GraphSAGE sin balanceo vuelve a ordenar al revés",
  [("Las curvas PR quedan sobre la línea de azar: a lo sumo el doble del azar.","media"),
   ("GraphSAGE sin balanceo queda bajo la diagonal ROC (0,35), como con el desbalance real.","media"),
   ("Pesos por clase vuelve a ser la que mejor resiste: ROC-AUC 0,72 en GraphSAGE.","media")],C["weber"],
  ["Y el 1:20 en test.","Otra vez pegadas al azar.",
   "GraphSAGE sin balanceo vuelve a ordenar al revés.",
   "Pesos por clase vuelve a ser la más robusta al cambio en el tiempo."]),
]
NOM={"native":"con el desbalance real (≈1:40)","1-10":"1:10","1-10_os":"1:10 con SMOTE","1-20":"1:20"}
def escena(sc,sp,tit,ll,ref,ns,pref="",secc=SR):
    pie=f"Curvas ROC (arriba) y PR (abajo) {NOM.get(sc,'1:1')}, en {'validación' if sp=='val' else 'test'}. Color: forma de compensar el desbalance."+LEE+(LOG if sp=="test" else "")
    cues=[f"{sc} · {sp}"]+[f"Clic {k}" for k in range(1,len(ns))]
    return con_figura(secc,tit,f"{pref}curvas_{sc}_{sp}.png",A_CUR,560,ll,pie,ref,list(zip(cues,ns)))
s_esc=[escena(*x) for x in porsc]
SA="<b>Anexo:</b> escenario 1:1"
anexo=[escena("1-1","val","Con 1:1, sin balanceo y pesos por clase dan la misma curva, y ningún modelo pasa la compuerta",
  [("Con clases iguales los pesos valen lo mismo: las curvas azul y naranja coinciden.","media"),
   ("Rinde menos que 1:10 en GraphSAGE y GAT: ve pocas lícitas y pierde precisión.","media")],C["saito"],
  ["El 1:1 queda en el anexo, como acordamos.","Sin balanceo y pesos por clase coinciden, porque con clases iguales el peso es el mismo.",
   "Y rinde menos que 1:10: al ver tan pocas lícitas pierde precisión a la prevalencia real."],"anexo_",SA),
 escena("1-1","test","Con 1:1, en test se repite la caída al azar",
  [("Ninguna curva PR se separa de la línea de azar.","media")],C["weber"],
  ["En test, lo mismo que en los demás escenarios.","Ninguna curva se separa del azar en la zona útil."],"anexo_",SA)]

marca='  <section class="slide lamina l4"'; assert h.count(marca)==1
h=h.replace(marca,s_leer+s_res+s_tab+"".join(s_esc)+"\n"+marca)
h=h.replace("</main>","".join(anexo)+"</main>",1)

# ── 5. Renumerar láminas y figuras
it=iter(range(2,200)); h=re.sub(r'<div class="num">\d+</div>',lambda m:f'<div class="num">{next(it)}</div>',h)
it=iter(range(1,200)); h=re.sub(r'(<p class="pie[^"]*"[^>]*>)Figura \d+\.',lambda m:f'{m.group(1)}Figura {next(it)}.',h)
it=iter(range(2,200)); h=re.sub(r'aria-label="Lámina( \d+)?">',lambda m:f'aria-label="Lámina {next(it)}">',h)

vis=re.sub(r'<script.*?</script>','',re.sub(r'src="data:[^"]+"','',h),flags=re.S)
for bad in ["—","–",".,","corrida","38,4"]:
    assert bad not in re.sub(r'/\*.*?\*/','',vis,flags=re.S), bad
open(HERE+"hipotesis.html","w",encoding="utf8").write(h)
open(HERE+"preview.html","w",encoding="utf8").write("<!doctype html><html><head><meta charset=utf8></head><body>"+h+"</body></html>")
print(round(len(h)/1e6,2),"MB", len(re.findall(r'<section class="slide',h)),"láminas")
