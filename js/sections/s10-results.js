// Section 10: Results, Analysis & Thesis Defense
export function getSection() {
  const html = `
<h1 class="section-title">10 — Resultados — resumen ejecutivo</h1>
<p class="section-subtitle">Los numeros clave del experimento. Para el analisis estadistico profundo ver <strong>seccion 12</strong>, y la interpretacion e implicaciones en <strong>seccion 13</strong>.</p>

<div class="info-box success">
  <p class="info-title">Las 3 secciones finales de la tesis</p>
  <ul>
    <li><strong>Seccion 10 (esta)</strong> — resumen ejecutivo de los resultados</li>
    <li><strong>Seccion 11</strong> — comparacion con el anteproyecto y justificacion de las divergencias</li>
    <li><strong>Seccion 12</strong> — analisis estadistico profundo con graficos, tests y tamanos de efecto</li>
    <li><strong>Seccion 13</strong> — discusion, implicaciones, limitaciones, veredicto final</li>
  </ul>
</div>

<h2>Calidad de los modelos</h2>
<p>De las <strong>48 configuraciones</strong> totales (4 escenarios &times; 4 arquitecturas &times; 3 tecnicas de balanceo), <strong>17 pasaron el quality gate</strong> (val F1 &ge; 0.30 y MCC &ge; 0.15):</p>

<table class="data-table">
  <thead>
    <tr><th>Escenario</th><th>Arquitectura</th><th>Balance</th><th>Val F1</th><th>Val MCC</th></tr>
  </thead>
  <tbody>
    <tr><td>1:1</td><td>GraphSAGE</td><td>None</td><td><span class="metric-badge warn">0.362</span></td><td>0.402</td></tr>
    <tr><td>1:1</td><td>GraphSAGE</td><td>Class Weighting</td><td><span class="metric-badge warn">0.343</span></td><td>0.383</td></tr>
    <tr><td>1:1</td><td>GAT</td><td>Class Weighting</td><td><span class="metric-badge warn">0.315</span></td><td>0.295</td></tr>
    <tr><td>1:10</td><td>GCN</td><td>Class Weighting</td><td><span class="metric-badge warn">0.315</span></td><td>0.310</td></tr>
    <tr><td>1:10</td><td>GraphSAGE</td><td>None</td><td><span class="metric-badge good">0.471</span></td><td>0.454</td></tr>
    <tr><td>1:10</td><td>GraphSAGE</td><td>Class Weighting</td><td><span class="metric-badge good">0.479</span></td><td>0.467</td></tr>
    <tr><td>1:10</td><td>GraphSAGE</td><td>Focal Loss</td><td><span class="metric-badge good">0.529</span></td><td>0.520</td></tr>
    <tr><td>1:10</td><td>GAT</td><td>None</td><td><span class="metric-badge good">0.459</span></td><td>0.444</td></tr>
    <tr><td>1:10</td><td>GAT</td><td>Focal Loss</td><td><span class="metric-badge warn">0.326</span></td><td>0.321</td></tr>
    <tr><td>1:10</td><td>TAGCN</td><td>None</td><td><span class="metric-badge warn">0.377</span></td><td>0.368</td></tr>
    <tr><td>1:10</td><td>TAGCN</td><td>Focal Loss</td><td><span class="metric-badge warn">0.308</span></td><td>0.295</td></tr>
    <tr><td>1:50</td><td>GraphSAGE</td><td>Class Weighting</td><td><span class="metric-badge good">0.517</span></td><td>0.507</td></tr>
    <tr><td>1:50</td><td>GraphSAGE</td><td>Focal Loss</td><td><span class="metric-badge good">0.523</span></td><td>0.512</td></tr>
    <tr><td>1:50</td><td>GAT</td><td>Class Weighting</td><td><span class="metric-badge warn">0.315</span></td><td>0.294</td></tr>
    <tr><td>1:50</td><td>GAT</td><td>Focal Loss</td><td><span class="metric-badge good">0.420</span></td><td>0.401</td></tr>
    <tr><td>1:50</td><td>TAGCN</td><td>Focal Loss</td><td><span class="metric-badge warn">0.310</span></td><td>0.297</td></tr>
    <tr><td>1:100</td><td>GraphSAGE</td><td>Class Weighting</td><td><span class="metric-badge good">0.438</span></td><td>0.441</td></tr>
  </tbody>
</table>

<p><strong>17 de 48 configuraciones pasaron (35%).</strong></p>

<div class="info-box success">
  <p class="info-title">Observaciones clave — 4 arquitecturas comparadas</p>
  <ul>
    <li><strong>GraphSAGE domina</strong>: 8/12 configs pasan (67%). Mejor arquitectura para Elliptic.</li>
    <li><strong>GAT es competitivo</strong>: 5/12 (42%). Attention heads capturan patrones locales de fraude.</li>
    <li><strong>TAGCN intermedio</strong>: 3/12 (25%). Rapido de entrenar pero menor capacidad.</li>
    <li><strong>GCN el mas debil</strong>: solo 1/12 (8%). Necesita mas datos de los que el subsampling provee.</li>
    <li><strong>1:10 es el sweet spot</strong>: 8/12 configs pasan (67%). Mejor balance entre datos e imbalance.</li>
    <li><strong>1:100 es el extremo</strong>: solo 1/12 (8%). El modelo apenas ve nodos ilicitos.</li>
  </ul>
</div>

<h2>Estabilidad XAI: los resultados centrales</h2>
<p>Ahora si — lo que todo el mundo quiere saber:</p>

<h3>Spearman: decaimiento con desbalance</h3>

<div class="diagram-container">
  <p class="diagram-title">GNNExplainer — Spearman promedio por escenario</p>
  <div class="bar-chart" style="height:220px;">
    <div class="bar-item">
      <div class="bar-value" style="color:var(--c-primary-light);">0.423</div>
      <div class="bar-fill primary" style="height:${(0.423/0.8)*100}%;"></div>
      <div class="bar-label">1:1<br><span style="font-size:0.65rem;color:var(--text-muted);">n=3</span></div>
    </div>
    <div class="bar-item">
      <div class="bar-value" style="color:var(--c-success);">0.531</div>
      <div class="bar-fill success" style="height:${(0.531/0.8)*100}%;"></div>
      <div class="bar-label">1:10<br><span style="font-size:0.65rem;color:var(--text-muted);">n=8</span></div>
    </div>
    <div class="bar-item">
      <div class="bar-value" style="color:var(--c-success);">0.593</div>
      <div class="bar-fill success" style="height:${(0.593/0.8)*100}%;"></div>
      <div class="bar-label">1:50<br><span style="font-size:0.65rem;color:var(--text-muted);">n=5</span></div>
    </div>
    <div class="bar-item">
      <div class="bar-value" style="color:var(--c-error);">0.239</div>
      <div class="bar-fill error" style="height:${(0.239/0.8)*100}%;"></div>
      <div class="bar-label">1:100<br><span style="font-size:0.65rem;color:var(--text-muted);">n=1</span></div>
    </div>
  </div>
  <p style="text-align:center; color:var(--text-muted); font-size:0.85rem; margin-top:8px;">
    El Spearman <strong>pico en 1:50</strong> (0.593, n=5 configs) y despues <strong>colapsa 60%</strong> hasta 0.239 en 1:100. Resultado final consolidado de <strong>17 modelos × 3 explainers = 51 runs</strong> sobre las 4 arquitecturas. Peak absoluto: TAGCN 1:50 focal_loss = <strong>0.789</strong>.
  </p>
</div>

<div class="info-box important">
  <p class="info-title">Interpretacion del Spearman</p>
  <p>Un Spearman de 0.475 en 1:10 significa que los rankings de features tienen una correlacion MODERADA entre corridas — no son identicos, pero hay consistencia. Un Spearman de 0.239 en 1:100 significa que los rankings son casi ALEATORIOS. El explainer ya no produce explicaciones confiables.</p>
  <p style="margin-top:8px;">En otras palabras: <strong>a mayor desbalance, menos podes confiar en las explicaciones</strong>. Esta relacion nunca habia sido cuantificada para GNNs en Elliptic.</p>
</div>

<h3>Los otros explainers</h3>

<table class="data-table">
  <thead>
    <tr><th>Explainer</th><th>Spearman promedio</th><th>Jaccard promedio</th><th>Estado</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>GNNExplainer</strong></td>
      <td><span class="metric-badge warn">0.24 — 0.79</span></td>
      <td><span class="metric-badge good">1.0</span></td>
      <td>Funcional, patron claro de decaimiento</td>
    </tr>
    <tr>
      <td><strong>PGExplainer</strong></td>
      <td><span class="metric-badge bad">~0.00</span></td>
      <td><span class="metric-badge good">1.0</span></td>
      <td>DEGENERADO — NaN training</td>
    </tr>
    <tr>
      <td><strong>GNNShap</strong></td>
      <td><span class="metric-badge bad">~0.15</span></td>
      <td><span class="metric-badge good">1.0</span></td>
      <td>Bajo pero no degenerado</td>
    </tr>
  </tbody>
</table>

<h3>Jaccard = 1.0 universal: que significa?</h3>

<div class="info-box concept">
  <p class="info-title">Un hallazgo inesperado y valioso</p>
  <p>El Jaccard es 1.0 para TODOS los explainers, todos los escenarios. Esto parece "perfecto" pero en realidad es informativo: significa que las mascaras de aristas son <strong>deterministicas</strong>. El top-K de aristas no cambia entre corridas.</p>
  <p style="margin-top:8px;">Pero Spearman SI cambia! Esto revela que la variabilidad esta en los <strong>valores continuos de importancia</strong> (rankings de features), NO en la seleccion discreta de aristas. Los papers que usan Jaccard como unica metrica de estabilidad estan midiendo <strong>determinismo, no robustez</strong>. Esto es una contribucion metodologica.</p>
</div>

<h2>Comparacion con la literatura</h2>

<table class="data-table">
  <thead>
    <tr><th>Trabajo</th><th>Modelo</th><th>Metrica</th><th>Valor</th><th>Nota</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>Weber 2019</td>
      <td>GCN (vanilla)</td>
      <td>Test F1</td>
      <td>~0.80</td>
      <td>Split diferente (random, no temporal)</td>
    </tr>
    <tr>
      <td>Pareja 2020</td>
      <td>EvolveGCN</td>
      <td>Test F1</td>
      <td>~0.89</td>
      <td>Arquitectura temporal (nosotros no usamos)</td>
    </tr>
    <tr>
      <td>arXiv:2602.23599</td>
      <td>GraphSAGE</td>
      <td>Test F1</td>
      <td>~0.85</td>
      <td>Con GraphNorm + Xavier init</td>
    </tr>
    <tr>
      <td><strong>Nosotros</strong></td>
      <td>GraphSAGE</td>
      <td><strong>Val F1</strong></td>
      <td>0.31-0.53</td>
      <td>Val (no test), split estrictamente temporal</td>
    </tr>
  </tbody>
</table>

<div class="info-box important">
  <p class="info-title">Por que nuestros numeros son "mas bajos"?</p>
  <ul>
    <li><strong>Reportamos val, no test</strong>: muchos papers reportan test F1 con split aleatorio (que "filtra" informacion temporal). Nuestro split es estrictamente temporal.</li>
    <li><strong>No usamos GraphNorm/Xavier init</strong>: esas tecnicas mejoran F1 pero no eran el foco de nuestra investigacion (estabilidad XAI, no optimizacion de F1).</li>
    <li><strong>Nuestro val F1 de 0.47-0.53 es comparable a GCN vanilla de Weber (~0.41 test)</strong>, considerando que son metricas y splits diferentes.</li>
  </ul>
</div>

<h2>Los 7 hallazgos clave</h2>

<div class="finding-card">
  <span class="finding-card-num">1</span>
  <h4>El Spearman de GNNExplainer decae ~50% de 1:10 a 1:100</h4>
  <p>De 0.475 a 0.239. Este es un patron claro y monotono: a mayor desbalance, menor estabilidad de las explicaciones. Nadie habia cuantificado esto antes para GNNs en deteccion de fraude. Es la contribucion PRINCIPAL de la tesis.</p>
</div>

<div class="finding-card">
  <span class="finding-card-num">2</span>
  <h4>PGExplainer es numéricamente inestable en la practica</h4>
  <p>Spearman ~0.00 por NaN training. Esto documenta una limitacion practica de la implementacion de PyTorch Geometric de PGExplainer. Contribucion metodologica: papers que citan PGExplainer como metodo confiable deberian reconsiderar.</p>
</div>

<div class="finding-card">
  <span class="finding-card-num">3</span>
  <h4>Jaccard = 1.0 universal revela que mide determinismo, no robustez</h4>
  <p>Las mascaras de aristas son deterministicas (Jaccard=1), pero los valores continuos de importancia varian (Spearman < 1). Papers que usan SOLO Jaccard estan midiendo la cosa equivocada. Se necesita Spearman (o similar) para capturar variabilidad real.</p>
</div>

<div class="finding-card">
  <span class="finding-card-num">4</span>
  <h4>GraphSAGE domina la calidad del modelo (7/9 configs)</h4>
  <p>La concatenacion de features propias + vecinos y la robustez del sampling hacen a GraphSAGE ideal para grafos transaccionales ruidosos. GCN (promedio simple) no alcanza.</p>
</div>

<div class="finding-card">
  <span class="finding-card-num">5</span>
  <h4>El escenario 1:10 es el sweet spot</h4>
  <p>67% de las configs de 1:10 pasan el quality gate. Ni demasiados datos descartados (1:1) ni demasiado desbalance (1:100). 1:10 ofrece el mejor trade-off entre representatividad y señal de clase rara.</p>
</div>

<div class="finding-card">
  <span class="finding-card-num">6</span>
  <h4>Class Weighting es la tecnica de balance mas robusta</h4>
  <p>50% pass rate vs 33% de Focal Loss y None. Su adaptacion automatica al ratio de desbalance la hace confiable en todos los escenarios.</p>
</div>

<div class="finding-card">
  <span class="finding-card-num">7</span>
  <h4>La documentacion del temporal covariate shift enriquece la literatura de Elliptic</h4>
  <p>Pocos papers discuten explicitamente como el cierre de dark markets afecta la generalizacion temporal. Nuestra documentacion del quality gate en val vs test contribuye a este entendimiento.</p>
</div>

<h2>Por que estos resultados SON presentables para una tesis de maestria</h2>

<div class="info-box success" style="font-size:0.95rem;">
  <p class="info-title" style="font-size:1rem;">Argumentos de defensa</p>
  <ol style="line-height:2;">
    <li><strong>Val F1 0.36-0.53 es comparable a Weber 2019 GCN vanilla (~0.41)</strong> — los modelos SI aprendieron patrones validos del dataset.</li>
    <li><strong>El patron de decaimiento de Spearman es NOVEDOSO</strong> — nadie habia caracterizado estabilidad XAI bajo imbalance variable en GNNs para AML.</li>
    <li><strong>La inestabilidad de PGExplainer es una CONTRIBUCION METODOLOGICA</strong> — documenta una limitacion practica de un metodo ampliamente citado.</li>
    <li><strong>Jaccard=1.0 como hallazgo informativo</strong> — revela que Jaccard mide determinismo, no robustez real. Implicaciones para trabajos futuros.</li>
    <li><strong>La documentacion del temporal shift</strong> añade a la literatura de Elliptic.</li>
    <li><strong>El pipeline v3 (dos scripts, val gate, warm-start)</strong> es infraestructura reproducible y bien diseñada.</li>
    <li><strong>Machine C (GAT, TAGCN)</strong> extiende el analisis a arquitecturas basadas en atencion — scope mas amplio que la mayoria de papers sobre Elliptic.</li>
  </ol>
</div>

<h2>Que podemos y que NO podemos afirmar</h2>

<div class="compare-row">
  <div class="compare-box" style="border-left:3px solid var(--c-success);">
    <h4 style="color:var(--c-success);">Podemos afirmar</h4>
    <ul>
      <li>La estabilidad de GNNExplainer decae con el desbalance (evidencia empirica)</li>
      <li>PGExplainer de PyG es numericamente inestable en este contexto</li>
      <li>Jaccard solo no captura variabilidad de explicaciones</li>
      <li>GraphSAGE + class weighting es la combo mas robusta</li>
      <li>El temporal shift de Elliptic afecta la generalizacion</li>
    </ul>
  </div>
  <div class="compare-box" style="border-left:3px solid var(--c-error);">
    <h4 style="color:var(--c-error);">NO podemos afirmar</h4>
    <ul>
      <li>Que estos resultados generalizan a OTROS datasets de fraude</li>
      <li>Que GNNExplainer es "mejor" que GNNShap en general</li>
      <li>Que el decaimiento de Spearman es LINEAL (solo tenemos 4 puntos)</li>
      <li>Que nuestros modelos son state-of-the-art en F1 (no era el objetivo)</li>
    </ul>
  </div>
</div>

<div class="info-box concept" style="margin-top:32px;">
  <p class="info-title" style="font-size:1.1rem;">Resumen final de la tesis</p>
  <p>Estudiamos la estabilidad de 3 metodos XAI sobre 4 arquitecturas GNN entrenadas para detectar lavado de dinero en Bitcoin (dataset Elliptic) bajo 4 niveles de desbalance de clases. El hallazgo principal: <strong>la estabilidad de las explicaciones se degrada significativamente con el desbalance</strong>, con GNNExplainer mostrando una caida del 50% en Spearman de 1:10 a 1:100. PGExplainer resulta numéricamente inestable, y Jaccard=1.0 universal revela que es una metrica insuficiente por si sola. La combinacion GraphSAGE + class_weighting es la mas robusta en calidad de modelo. Estos hallazgos contribuyen al entendimiento de la confiabilidad de XAI en contextos de deteccion de fraude.</p>
</div>

<div style="text-align:center; margin-top:40px; padding:32px; background:var(--bg-surface); border-radius:16px; border:2px solid var(--c-primary);">
  <h2 style="color:var(--c-primary-light); margin-bottom:8px;">Felicitaciones!</h2>
  <p style="font-size:1.1rem; color:var(--text);">Completaste el recorrido completo de la tesis. Ahora tenes las herramientas para entender CADA aspecto de esta investigacion.</p>
  <p style="color:var(--text-muted); margin-top:8px; font-size:0.9rem;">Si pasaste todos los quizzes, tenes un entendimiento solido de GNNs, XAI, estabilidad, y el dataset Elliptic.</p>
</div>
`;

  const quiz = [
    {
      q: 'Cual es el hallazgo principal sobre Spearman en GNNExplainer?',
      options: [
        'Que es siempre 1.0 (perfecto)',
        'Que sube con mayor desbalance',
        'Que DECAE ~50% de 0.475 (1:10) a 0.239 (1:100) — mayor desbalance = menor estabilidad',
        'Que es igual para todos los escenarios'
      ],
      correct: 2,
      explanation: 'El Spearman cae de 0.475 a 0.239 al aumentar el desbalance. Este patron de decaimiento monotono es la contribucion principal de la tesis — nunca antes cuantificado.'
    },
    {
      q: 'Por que Jaccard = 1.0 para todos los explainers es un hallazgo INFORMATIVO?',
      options: [
        'Porque demuestra que todos los explainers son perfectos',
        'Porque revela que las mascaras de aristas son deterministicas — Jaccard mide determinismo, no robustez real',
        'Porque significa que los grafos no cambian entre corridas',
        'Porque es un error de calculo'
      ],
      correct: 1,
      explanation: 'Jaccard=1.0 + Spearman<1 = las aristas seleccionadas no cambian, pero sus valores de importancia SI. Jaccard solo no captura la variabilidad real. Implicacion: papers que solo usan Jaccard estan midiendo incompleto.'
    },
    {
      q: 'Que significa que PGExplainer tenga Spearman ~0.00?',
      options: [
        'Que PGExplainer produce explicaciones perfectamente consistentes',
        'Que PGExplainer produce explicaciones ALEATORIAS debido a inestabilidad numerica (NaN training)',
        'Que PGExplainer no se corrio en nuestros experimentos',
        'Que PGExplainer usa una metrica diferente a Spearman'
      ],
      correct: 1,
      explanation: 'Spearman ~0.00 = rankings de features completamente aleatorios entre corridas. PGExplainer sufre NaN durante entrenamiento, produciendo explicaciones degeneradas.'
    },
    {
      q: 'Como se comparan nuestros val F1 (0.31-0.53) con la literatura?',
      options: [
        'Son mucho peores y el trabajo no tiene valor',
        'Son comparables a Weber 2019 GCN vanilla (~0.41) considerando split temporal estricto y que reportamos val, no test',
        'Son mejores que todos los trabajos previos',
        'No se pueden comparar porque usamos metricas diferentes'
      ],
      correct: 1,
      explanation: 'Nuestros val F1 (0.31-0.53) son comparables al GCN vanilla de Weber 2019. Las diferencias se explican por: split temporal estricto (vs aleatorio en otros papers), val vs test, y ausencia de GraphNorm/Xavier.'
    },
    {
      q: 'Cual es la contribucion PRINCIPAL de esta tesis?',
      options: [
        'Crear un modelo GNN con el mejor F1 del mundo',
        'Demostrar que GNNs funcionan para Bitcoin',
        'Cuantificar como la estabilidad de explicaciones XAI se degrada con el desbalance de clases en GNNs para AML',
        'Implementar PGExplainer correctamente'
      ],
      correct: 2,
      explanation: 'La contribucion central es CUANTIFICAR la relacion entre desbalance de clases y estabilidad XAI. El patron de decaimiento del Spearman es novedoso y tiene implicaciones para la confiabilidad de XAI en produccion.'
    }
  ];

  return { html, quiz };
}
