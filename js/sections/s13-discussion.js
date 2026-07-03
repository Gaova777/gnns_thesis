// Section 13: Discusion profunda e implicaciones
export function getSection() {
  const html = `
<h1 class="section-title">13 — Discusion e implicaciones</h1>
<p class="section-subtitle">Que significan estos numeros, por que importan, y como se comparan con el estado del arte. Interpretacion completa para la defensa.</p>

<h2>1. El hallazgo central — <em>accuracy no predice estabilidad XAI</em></h2>

<div class="info-box important">
  <p class="info-title">La hipotesis inicial era que el mejor modelo daria las mejores explicaciones. <strong>Es falsa.</strong></p>
  <p>GraphSAGE alcanza val F1 = 0.46 (el mejor predictor) pero produce explicaciones con Spearman = 0.41 (el peor entre las 4 arquitecturas). GAT es el caso opuesto: val F1 = 0.37 pero Spearman = 0.64. Este tradeoff es <strong>el finding mas provocativo del estudio</strong>.</p>
</div>

<h3>Por que GAT produce explicaciones mas estables</h3>

<p>Hipotesis del estudio:</p>
<ul>
  <li><strong>El mecanismo de atencion actua como regularizador implicito de las atribuciones</strong>. Los coeficientes de atencion se aprenden desde los datos con una funcion softmax sobre vecinos — esto impone una estructura de sparsity natural en como el modelo integra informacion.</li>
  <li>Cuando GNNExplainer optimiza un mask sobre edges/features, los gradientes en GAT llegan a una superficie de optimizacion <strong>mas restringida</strong> que en GraphSAGE (donde el aggregation es mean ponderado igualmente).</li>
  <li>El resultado es que el mask converge consistentemente a la misma region de soluciones — Spearman alto.</li>
</ul>

<p>Esta interpretacion es <strong>consistente con literatura de regularizacion en atencion</strong> (Jain & Wallace 2019, "Attention is not Explanation"; Wiegreffe & Pinter 2019, "Attention is not not Explanation"). El debate sobre si la atencion es o no explicacion valida recibe un nuevo dato: en GNNs, la atencion parece dar explicaciones <em>mas estables</em> que arquitecturas sin atencion, independientemente de su validez como "explicacion causal".</p>

<h3>Por que GraphSAGE es el peor explicador pese a ser el mejor predictor</h3>

<p>El sampling de vecinos de GraphSAGE (idea original del paper, Hamilton 2017) introduce estocasticidad en el mensaje pasaje. Aunque esto mejora generalizacion (por eso predice mejor), tambien significa que:</p>
<ul>
  <li>El GNN <strong>no usa exactamente los mismos vecinos cada vez</strong>, incluso con mismo grafo (depende del sampling estocastico).</li>
  <li>GNNExplainer, al optimizar un mask, busca identificar edges clave — pero en GraphSAGE esos edges cambian de rueda en rueda durante training, desdibujando la senal.</li>
  <li>El resultado: masks mas difusos, rankings de features menos concentrados, Spearman entre replicas mas bajo.</li>
</ul>

<p><strong>Implicacion practica</strong>: si necesitas un modelo para produccion cuyas explicaciones sean <em>auditables</em> (compliance, banca, legal), GAT es la opcion — aunque sacrifiques 9 puntos de val F1. La confianza en las explicaciones puede ser mas importante que 9 puntos de accuracy.</p>

<h2>2. El peak en 1:50 — sorpresa interpretada</h2>

<div class="info-box concept">
  <p class="info-title">Esperabamos decaimiento monotonico. Obtuvimos una curva con peak en 1:50.</p>
  <p>Spearman = 0.42 (1:1) → 0.53 (1:10) → <strong>0.59 (1:50 peak)</strong> → 0.24 (1:100 colapso). La estabilidad XAI no decae linealmente — tiene un <em>sweet spot</em>.</p>
</div>

<h3>Interpretacion del peak</h3>

<p>Dos fuerzas compiten:</p>
<ol>
  <li><strong>Fuerza 1 — mas datos, mejor aprendizaje, mejor explicabilidad</strong>: en 1:1 (solo 6,924 nodos de training), el modelo tiene muy pocos datos para converger. Las explicaciones reflejan esta sub-optimizacion — masks variables entre replicas porque el modelo mismo es sub-entrenado.</li>
  <li><strong>Fuerza 2 — imbalance extremo degrada el gradiente sobre minoritarios</strong>: en 1:100, el gradiente de la clase ilicita es tan dominado por licitos que el modelo aprende features genericas — explicaciones ruidosas y poco concentradas.</li>
</ol>

<p><strong>El sweet spot 1:50 balancea ambos</strong>: suficiente data de entrenamiento (107,434 nodos) pero imbalance no tan extremo que degrade la senal de fraude. Con class_weighting o focal_loss adaptandose al imbalance, el modelo encuentra patrones estables que GNNExplainer puede extraer reproduciblemente.</p>

<h3>El colapso en 1:100</h3>

<p>El drop de 0.59 a 0.24 (-60%) entre 1:50 y 1:100 es <strong>el momento donde la explicabilidad se vuelve no-confiable</strong>. Implicaciones practicas:</p>
<ul>
  <li>Para deploy en entornos con <strong>imbalance cercano al real</strong> (Bitcoin tiene ~2.8% ilicitos globales, parecido a 1:35), Spearman esta en territorio seguro (0.50+).</li>
  <li>Para aplicaciones con imbalance extremo (fraude sofisticado, &lt;1% en ciertos cluster), las explicaciones <strong>pierden confiabilidad medible</strong>.</li>
  <li>Esto sugiere un <strong>umbral operativo</strong>: deploy XAI-validado hasta ~1:50, con caveats documentados para casos mas extremos.</li>
</ul>

<h2>3. Jaccard = 1.000 universal — contribucion metodologica</h2>

<div class="info-box concept">
  <p class="info-title">34/34 runs de GNNExplainer y PGExplainer con Jaccard = 1.000</p>
  <p>Esto no es "estabilidad perfecta". Es evidencia de que <strong>las mascaras de aristas en GNN XAI son deterministicas dado el modelo</strong> — todos los runs con el mismo modelo convergen al mismo top-K de aristas.</p>
</div>

<h3>Por que esto es una contribucion</h3>

<p>Varios papers recientes sobre estabilidad XAI en GNNs usan Jaccard como metrica principal. Sus resultados tipicos: Jaccard = 0.85-1.0, reportado como "buena estabilidad". <strong>Nuestro estudio revela que este valor es un artefacto del metodo, no evidencia de robustez</strong>.</p>

<p>Razones tecnicas:</p>
<ul>
  <li>GNNExplainer con <code>edge_mask_type="object"</code> optimiza un unico vector de mask. Dos optimizaciones partiendo de inits diferentes convergen al mismo optimo local porque la loss tiene un minimo dominante.</li>
  <li>PGExplainer, aun con entrenamiento inestable, produce outputs similares porque sus pesos quedan cerca del init aleatorio en todas las replicas.</li>
  <li>La variabilidad real existe, pero vive en los <strong>valores continuos de importancia</strong>, no en la seleccion discreta de top-K aristas.</li>
</ul>

<p><strong>Recomendacion metodologica del estudio</strong>: para evaluar estabilidad de explicaciones en GNNs, Spearman sobre rankings de features debe ser la metrica primaria. Jaccard sobre subgraphs puede reportarse como metrica secundaria pero es inadecuada como medida principal.</p>

<h2>4. Comparacion con estado del arte</h2>

<table class="data-table">
  <thead>
    <tr>
      <th>Trabajo</th>
      <th>Modelo</th>
      <th>Dataset</th>
      <th>Metrica reportada</th>
      <th>Valor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Weber et al. 2019</strong> (original)</td>
      <td>GCN</td>
      <td>Elliptic (test)</td>
      <td>F1 illicit</td>
      <td>~0.41</td>
    </tr>
    <tr>
      <td><strong>Weber et al. 2019</strong></td>
      <td>Random Forest</td>
      <td>Elliptic (test)</td>
      <td>F1 illicit</td>
      <td>~0.79</td>
    </tr>
    <tr>
      <td><strong>Pareja et al. 2020</strong></td>
      <td>EvolveGCN (temporal)</td>
      <td>Elliptic (test)</td>
      <td>F1 illicit</td>
      <td>~0.89</td>
    </tr>
    <tr>
      <td><strong>arXiv:2602.23599 (2026)</strong></td>
      <td>GraphSAGE + GraphNorm</td>
      <td>Elliptic (val)</td>
      <td>F1 illicit</td>
      <td>~0.85</td>
    </tr>
    <tr>
      <td style="background:rgba(16,185,129,0.1);"><strong>Esta tesis</strong></td>
      <td>GraphSAGE (mejor)</td>
      <td>Elliptic (val)</td>
      <td>F1 illicit</td>
      <td>0.53</td>
    </tr>
    <tr>
      <td style="background:rgba(16,185,129,0.1);"><strong>Esta tesis</strong></td>
      <td>GCN baseline</td>
      <td>Elliptic (val)</td>
      <td>F1 illicit</td>
      <td>0.31</td>
    </tr>
    <tr>
      <td style="background:rgba(16,185,129,0.1);"><strong>Esta tesis</strong></td>
      <td>GNNExplainer (todos)</td>
      <td>Elliptic</td>
      <td><strong>Spearman estabilidad</strong></td>
      <td><strong>0.24 - 0.79</strong></td>
    </tr>
  </tbody>
</table>

<div class="info-box success">
  <p class="info-title">Contribucion unica de esta tesis</p>
  <p>La literatura previa se enfoca en <em>accuracy de prediccion</em>. Esta tesis es de los pocos estudios que cuantifica sistematicamente <strong>la estabilidad XAI</strong> como variable de salida en un diseno factorial completo (escenario &times; arquitectura &times; balancing &times; explainer). Los numeros de prediccion estan alineados con vanilla GCN/GraphSAGE (no con SOTA), pero eso es consistente con el objetivo: el estudio es sobre <em>estabilidad</em>, no sobre <em>maxima prediccion</em>.</p>
</div>

<h2>5. Implicaciones para la practica</h2>

<h3>Recomendaciones operativas extraidas del estudio</h3>

<table class="data-table">
  <thead>
    <tr>
      <th>Escenario de deploy</th>
      <th>Arquitectura recomendada</th>
      <th>Balancing</th>
      <th>Explainer</th>
      <th>Spearman esperado</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Imbalance moderado (1:10-1:50), prioriza prediccion</td>
      <td>GraphSAGE</td>
      <td>Class Weighting / Focal Loss</td>
      <td>GNNExplainer</td>
      <td>~0.42-0.47</td>
    </tr>
    <tr>
      <td>Imbalance moderado, <strong>prioriza auditabilidad</strong></td>
      <td><strong>GAT o TAGCN</strong></td>
      <td>Class Weighting / Focal Loss</td>
      <td>GNNExplainer</td>
      <td><strong>0.62-0.79</strong></td>
    </tr>
    <tr>
      <td>Imbalance extremo (&ge;1:100)</td>
      <td>GraphSAGE</td>
      <td>Class Weighting (unica que pasa)</td>
      <td>GNNExplainer con caveat</td>
      <td>~0.24 (BAJA)</td>
    </tr>
    <tr>
      <td>Escenario balanceado (1:1)</td>
      <td>GraphSAGE</td>
      <td>None o Class Weighting</td>
      <td>GNNExplainer</td>
      <td>~0.43</td>
    </tr>
    <tr>
      <td>Cualquier deploy</td>
      <td>Evitar</td>
      <td>—</td>
      <td><strong>NO usar PGExplainer</strong> (PyG 2.7)</td>
      <td>0.00 (degenerado)</td>
    </tr>
  </tbody>
</table>

<h3>Para compliance bancaria y AML</h3>

<p>Las regulaciones emergentes (EU AI Act, guidance de reguladores financieros) requieren <em>explicabilidad auditada</em> para sistemas automatizados de clasificacion de transacciones. Este estudio indica:</p>
<ul>
  <li>Un sistema XAI confiable para AML requiere <strong>Spearman &ge; 0.50</strong> — nuestro estudio muestra que esto es alcanzable con GAT + class_weighting en imbalance moderado.</li>
  <li>En imbalance cercano al 1:100 (realismo de Bitcoin global), los explainers pierden confiabilidad — <strong>los modelos deben complementarse con validacion humana</strong>.</li>
  <li>PGExplainer, a pesar de su popularidad en literatura, no es viable para deploy con la implementacion actual de PyG.</li>
</ul>

<h2>6. Limitaciones del estudio</h2>

<div class="info-box warning">
  <p class="info-title">Limitaciones que deben reconocerse en la tesis</p>
  <ol>
    <li><strong>Tamano de muestra por grupo</strong>: n=1 en algunas celdas (ej. 1:100 GNNExplainer). Kruskal-Wallis no alcanza significancia formal — el analisis se sustenta en effect sizes.</li>
    <li><strong>Temporal shift no resuelto</strong>: GNNs estaticas no cruzan el shift de Elliptic. No podemos comparar directamente con papers que usan arquitecturas temporales.</li>
    <li><strong>GraphSMOTE no implementada</strong>: queda como trabajo futuro. El tercer tratamiento de balancing es "none" en lugar de GraphSMOTE.</li>
    <li><strong>PGExplainer implementacion</strong>: el fallo numerico puede ser fixeable con modificaciones arquitectonicas no exploradas en esta tesis.</li>
    <li><strong>SHAP Concentration no reportada</strong>: la metrica de parsimonia propuesta originalmente fue omitida al revelarse redundante frente a Jaccard=1.0 universal.</li>
    <li><strong>Un solo dataset (Elliptic)</strong>: los findings pueden no generalizar a otros dominios de aplicacion de GNNs.</li>
  </ol>
</div>

<h2>7. Trabajo futuro</h2>

<ol>
  <li><strong>Mas seeds aleatorios</strong>: repetir cada configuracion con 5 seeds para llevar n por celda a 5+ y confirmar Kruskal-Wallis con potencia adecuada.</li>
  <li><strong>Arquitecturas temporales</strong>: reproducir el estudio con EvolveGCN o TGN para ver si el temporal modeling altera la relacion accuracy-estabilidad.</li>
  <li><strong>GraphSMOTE con implementacion moderna</strong>: evaluar si oversampling sintetico cambia la estabilidad XAI.</li>
  <li><strong>Fix a PGExplainer</strong>: gradient clipping agresivo + batch normalization en la red parametrica; evaluar si la convergencia mejora.</li>
  <li><strong>Otros datasets</strong>: Amazon fraud, Reddit communities, OGB-Products — verificar generalizacion del tradeoff accuracy-estabilidad.</li>
  <li><strong>XAI-aware training</strong>: ¿se puede entrenar un modelo optimizando conjuntamente accuracy y estabilidad de explicaciones? El tradeoff observado sugiere que es un problema de optimizacion multi-objetivo no trivial.</li>
</ol>

<h2>8. Veredicto final — es presentable para tesis de maestria?</h2>

<div class="info-box success">
  <p class="info-title">SI, con la narrativa correcta.</p>
  <p>La tesis tiene <strong>3 contribuciones defendibles</strong>:</p>
  <ol>
    <li><strong>Cuantificacion empirica del tradeoff accuracy-estabilidad XAI en GNNs</strong> — rank correlation negativa entre accuracy y Spearman (no reportada antes en Elliptic).</li>
    <li><strong>Identificacion del peak de estabilidad en 1:50</strong> — el sweet spot donde la explicabilidad es maxima, con colapso a 1:100.</li>
    <li><strong>Contribucion metodologica sobre Jaccard</strong> — revelar que Jaccard=1.0 universal es artefacto, no estabilidad; recomendar Spearman sobre rankings como metrica primaria.</li>
  </ol>
  <p style="margin-top:0.5rem;">A esto se agregan <em>hallazgos secundarios</em>: la degeneracion de PGExplainer, el rol de attention como regularizador, las recomendaciones operativas para deploy.</p>
</div>

<h3>El elevator pitch de la tesis</h3>

<div class="quote-box" style="padding:1.2rem; background:rgba(99,102,241,0.1); border-left:4px solid var(--c-primary); margin:1rem 0; border-radius:4px;">
  <p style="font-style:italic; margin:0;">
    "La estabilidad de explicaciones XAI en Graph Neural Networks para deteccion de fraude no esta monotonicamente correlacionada con la calidad de prediccion — tiene un peak en imbalance 1:50 y colapsa -60% en 1:100. Disrumpiendo la intuicion, el mejor predictor (GraphSAGE) es el peor explicador, mientras que GAT — con attention — produce las explicaciones mas reproducibles pese a menor accuracy. Esto sugiere que para aplicaciones con requerimientos de auditabilidad, la eleccion arquitectonica debe priorizar estabilidad XAI, no accuracy maxima."
  </p>
</div>
`;
  const quiz = [
    {
      question: "Cual es la interpretacion propuesta para que GAT produzca explicaciones mas estables que GraphSAGE?",
      options: [
        "GAT es una arquitectura mas nueva",
        "El mecanismo de attention actua como regularizador implicito de las atribuciones",
        "GAT tiene mas parametros",
        "Es coincidencia estadistica"
      ],
      correct: 1,
      explanation: "La hipotesis del estudio es que los coeficientes de attention (softmax sobre vecinos) imponen una estructura de sparsity que restringe el espacio de soluciones del mask de GNNExplainer, llevandolo consistentemente a la misma region - Spearman alto."
    },
    {
      question: "Por que 1:50 es un 'sweet spot' de estabilidad XAI en lugar de un decaimiento monotonico?",
      options: [
        "Porque tiene mas aristas",
        "Porque balancea dos fuerzas: suficientes datos de entrenamiento vs imbalance no-degradante",
        "Por un error experimental",
        "Porque 1:50 es un numero magico"
      ],
      correct: 1,
      explanation: "1:1 tiene pocos datos (modelo sub-entrenado, explicaciones variables). 1:100 tiene imbalance toxico (gradientes minoritarios degradados). 1:50 balancea: 107k nodos de training + imbalance manejable con class_weighting = condiciones optimas para aprender patrones estables."
    },
    {
      question: "Cual es la recomendacion metodologica del estudio respecto a Jaccard?",
      options: [
        "Usar Jaccard siempre",
        "Reemplazar Jaccard por Spearman sobre rankings de features como metrica primaria de estabilidad XAI en GNNs",
        "Nunca usar Jaccard",
        "Calcular ambos y promediar"
      ],
      correct: 1,
      explanation: "Jaccard=1.0 universal revela que los edge masks son deterministicos dado el modelo - no miden robustez real. Spearman sobre rankings de features captura la variabilidad continua donde si existe senal informativa. El estudio recomienda Spearman como metrica primaria."
    },
    {
      question: "Para un deploy en AML con requisitos de auditabilidad, que recomienda el estudio?",
      options: [
        "Usar GraphSAGE por su mejor accuracy",
        "Usar GAT aunque tenga menor accuracy, porque sus explicaciones son mas estables (Spearman 0.62-0.79)",
        "No usar GNNs, usar Random Forest",
        "Entrenar sin balancing"
      ],
      correct: 1,
      explanation: "GAT sacrifica ~9 puntos de val F1 pero gana 54% en Spearman vs GraphSAGE. Para compliance bancaria que requiere explicaciones auditables, la reproducibilidad de las explicaciones es mas critica que accuracy absoluta."
    },
    {
      question: "Cuales son las 3 contribuciones principales defendibles de la tesis?",
      options: [
        "Dataset nuevo, modelo nuevo, metrica nueva",
        "Tradeoff accuracy-estabilidad, peak en 1:50, recomendacion de Spearman sobre Jaccard",
        "Solo mejorar la accuracy del paper original",
        "Crear un nuevo explainer"
      ],
      correct: 1,
      explanation: "Los 3 aportes son: (1) cuantificacion empirica del tradeoff accuracy-estabilidad XAI (rank correlation negativa), (2) identificacion del peak de estabilidad en 1:50 con colapso en 1:100, (3) contribucion metodologica documentando que Jaccard=1.0 es artefacto y recomendando Spearman como metrica primaria."
    }
  ];
  return { html, quiz };
}
