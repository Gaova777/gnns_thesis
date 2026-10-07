"""Deck de hipótesis v4 con las indicaciones de Cristian (reunión 4-oct).
Parte de la versión publicada 1 (13 láminas) y aplica: título de sección arriba con el
formato «Sección: frase corta» (el titular queda como subtítulo), una cita en cada lámina en
la banda naranja, pie de figura pegado a su figura, desbalance genérico (≈1:40), sin Optuna,
definir antes de usar (cómo leer las curvas), ROC-AUC con PR-AUC y F1 con KS, y el 1:1 al anexo."""
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

ESC=[("native","con el desbalance real (≈1:40)"),("1-10","1:10"),("1-10_os","1:10 con SMOTE")]
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
]
NOM={"native":"con el desbalance real (≈1:40)","1-10":"1:10","1-10_os":"1:10 con SMOTE"}
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
