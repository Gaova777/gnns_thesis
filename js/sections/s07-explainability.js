// Section 7: Explainability (XAI)
export function getSection() {
  const html = `
<h1 class="section-title">07 — Explicabilidad (XAI): el POR QUE detras de cada decision</h1>
<p class="section-subtitle">Un modelo que dice "esta transaccion es fraude" sin explicar por que es tan util como un medico que da un diagnostico sin justificacion. Aca entramos al corazon de la tesis.</p>

<h2>El problema de la caja negra</h2>
<p>Una GNN toma como entrada un grafo con miles de nodos, features y conexiones, y produce una prediccion: "ilicita" o "licita". Pero <strong>no explica POR QUE</strong>.</p>

<p>Imaginate este escenario:</p>

<div class="info-box important">
  <p class="info-title">Escenario real</p>
  <p>Un banco usa una GNN para monitorear transacciones. La GNN marca una transaccion de $50,000 como "sospechosa de lavado". El oficial de compliance pregunta: "por que?". Y el modelo... no puede responder. Es un tensor de miles de parametros que, mediante multiplicaciones de matrices y activaciones no lineales, llego a esa conclusion. Pero no puede señalar QUE features o QUE conexiones lo llevaron ahi.</p>
</div>

<p>Esto es un problema real por multiples razones:</p>
<ul>
  <li><strong>Regulacion</strong>: la EU AI Act (2024) exige que sistemas de IA de alto riesgo (como deteccion de fraude) provean explicaciones de sus decisiones.</li>
  <li><strong>Confianza</strong>: un analista no va a actuar sobre una prediccion que no entiende.</li>
  <li><strong>Debugging</strong>: sin explicaciones, no podes saber si el modelo aprendio patrones reales o artefactos de los datos.</li>
  <li><strong>Justicia</strong>: si el modelo discrimina injustamente, necesitas ver POR QUE para corregirlo.</li>
</ul>

<p>La solucion: <strong>XAI (eXplainable Artificial Intelligence)</strong> — metodos que generan explicaciones POST-HOC de las decisiones de un modelo. Nosotros probamos tres:</p>

<h2>Metodo 1: GNNExplainer (Ying et al., 2019)</h2>

<div class="info-box concept">
  <p class="info-title">La idea en una frase</p>
  <p>GNNExplainer aprende una <strong>mascara</strong> sobre las aristas y features que maximiza la prediccion del modelo. En otras palabras: encuentra el subconjunto MINIMO de aristas y features que, por si solo, es suficiente para que el modelo haga la misma prediccion.</p>
</div>

<p>Como funciona:</p>
<ol>
  <li>Toma un nodo v y su prediccion Y</li>
  <li>Inicializa una mascara continua M sobre las aristas (valores entre 0 y 1)</li>
  <li>Optimiza M para que, aplicando la mascara (multiplicando aristas &times; M), la prediccion del modelo se mantenga lo mas cercana posible a Y</li>
  <li>Agrega un termino de regularizacion para que la mascara sea lo mas dispersa posible (pocas aristas activas = explicacion simple)</li>
</ol>

<div class="formula-box">
  <span class="formula-label">Objetivo de GNNExplainer</span>
  max<sub>M</sub> MI(Y, (A &odot; M, X)) - &lambda; ||M||<sub>1</sub>
  <span class="formula-annotation">
    <strong>MI</strong> = Informacion Mutua entre la prediccion Y y el grafo enmascarado<br>
    <strong>A &odot; M</strong> = adyacencia multiplicada elemento a elemento por la mascara<br>
    <strong>||M||<sub>1</sub></strong> = norma L1 de la mascara (fuerza a que sea dispersa)<br>
    <strong>&lambda;</strong> = balance entre fidelidad y simplicidad
  </span>
</div>

<div class="diagram-container">
  <p class="diagram-title">GNNExplainer: mascara sobre aristas</p>
  <svg class="graph-svg" viewBox="0 0 400 250" xmlns="http://www.w3.org/2000/svg">
    <!-- All edges (dim) -->
    <line x1="200" y1="40" x2="80" y2="130" stroke="#94a3b8" stroke-width="1.5" opacity="0.2"/>
    <line x1="200" y1="40" x2="320" y2="130" stroke="#94a3b8" stroke-width="1.5" opacity="0.2"/>
    <line x1="80" y1="130" x2="320" y2="130" stroke="#94a3b8" stroke-width="1.5" opacity="0.2"/>
    <line x1="80" y1="130" x2="140" y2="220" stroke="#94a3b8" stroke-width="1.5" opacity="0.2"/>
    <line x1="320" y1="130" x2="260" y2="220" stroke="#94a3b8" stroke-width="1.5" opacity="0.2"/>

    <!-- Highlighted edges (important for explanation) -->
    <line x1="200" y1="40" x2="80" y2="130" stroke="#f59e0b" stroke-width="4" opacity="0.9"/>
    <line x1="80" y1="130" x2="140" y2="220" stroke="#f59e0b" stroke-width="3" opacity="0.7"/>
    <line x1="200" y1="40" x2="320" y2="130" stroke="#f59e0b" stroke-width="2.5" opacity="0.5"/>

    <!-- Mask values -->
    <text x="125" y="75" fill="#f59e0b" font-size="11" font-weight="700">M=0.92</text>
    <text x="95" y="185" fill="#f59e0b" font-size="11" font-weight="700">M=0.78</text>
    <text x="275" y="75" fill="#f59e0b" font-size="11" font-weight="700">M=0.45</text>

    <!-- Nodes -->
    <circle cx="200" cy="40" r="20" fill="#ef4444"/>
    <text x="200" y="40" text-anchor="middle" dominant-baseline="central" fill="white" font-weight="700" font-size="11">v*</text>
    <circle cx="80" cy="130" r="18" fill="#6366f1"/>
    <text x="80" y="130" text-anchor="middle" dominant-baseline="central" fill="white" font-weight="700" font-size="10">B</text>
    <circle cx="320" cy="130" r="18" fill="#6366f1"/>
    <text x="320" y="130" text-anchor="middle" dominant-baseline="central" fill="white" font-weight="700" font-size="10">C</text>
    <circle cx="140" cy="220" r="18" fill="#6366f1"/>
    <text x="140" y="220" text-anchor="middle" dominant-baseline="central" fill="white" font-weight="700" font-size="10">D</text>
    <circle cx="260" cy="220" r="18" fill="#6366f1"/>
    <text x="260" y="220" text-anchor="middle" dominant-baseline="central" fill="white" font-weight="700" font-size="10">E</text>
  </svg>
  <p style="text-align:center; color:var(--text-muted); font-size:0.85rem; margin-top:8px;">
    El nodo rojo v* es el que queremos explicar. Las aristas amarillas (con valor de mascara M) son las que GNNExplainer identifica como importantes. Aristas con M alto son mas relevantes para la prediccion.
  </p>
</div>

<p><strong>Tipo</strong>: Optimizacion local (una explicacion por nodo)<br>
<strong>Pro</strong>: Fundamentado teoricamente. Produce mascaras interpretables.<br>
<strong>Contra</strong>: Lento (optimiza POR nodo). La inicializacion aleatoria introduce variabilidad.</p>

<h2>Metodo 2: PGExplainer (Luo et al., 2020)</h2>

<div class="info-box concept">
  <p class="info-title">La idea en una frase</p>
  <p>En vez de optimizar una mascara para CADA nodo por separado, PGExplainer entrena una <strong>red neuronal auxiliar</strong> que aprende a generar mascaras para CUALQUIER nodo. Es "parametrico" — el conocimiento se generaliza.</p>
</div>

<p>Como funciona:</p>
<ol>
  <li>Entrena un MLP pequeño que toma las embeddings de un par de nodos (u, v) y predice la importancia de la arista entre ellos</li>
  <li>El MLP se entrena sobre un conjunto de nodos de entrenamiento</li>
  <li>Despues, puede generar explicaciones para nodos NUEVOS sin reentrenamiento</li>
</ol>

<p><strong>Tipo</strong>: Parametrico (una red entrenada genera todas las explicaciones)<br>
<strong>Pro</strong>: Rapido una vez entrenado. Puede generalizar a nodos no vistos.<br>
<strong>Contra</strong>: El entrenamiento puede ser inestable. En nuestra experiencia, sufre de <strong>NaN loss</strong> con frecuencia.</p>

<div class="info-box danger">
  <p class="info-title">Hallazgo critico: PGExplainer es inestable</p>
  <p>En nuestros experimentos, PGExplainer frecuentemente produce <strong>NaN</strong> (Not a Number) durante su entrenamiento, resultado de inestabilidad numerica en la implementacion de PyTorch Geometric. Cuando esto pasa, las explicaciones generadas son basura: importancias uniformes o aleatorias. Este hallazgo es en si mismo una CONTRIBUCION metodologica — documenta una limitacion practica de un metodo citado 1000+ veces.</p>
</div>

<h2>Metodo 3: GNNShap (basado en Shapley values)</h2>

<div class="info-box concept">
  <p class="info-title">La idea en una frase</p>
  <p>GNNShap aplica los <strong>valores de Shapley</strong> de teoria de juegos para calcular la contribucion de cada feature a la prediccion. Cada feature recibe un "puntaje de importancia" basado en su contribucion marginal promedio.</p>
</div>

<p>Shapley values vienen de la teoria de juegos cooperativos. Imaginate un equipo de jugadores (features) que produce un resultado (prediccion). El valor de Shapley de cada jugador es su contribucion promedio al resultado, considerando TODAS las posibles combinaciones de jugadores.</p>

<div class="formula-box">
  <span class="formula-label">Valor de Shapley (simplificado)</span>
  &phi;<sub>i</sub> = &Sigma;<sub>S &sube; F\\{i}</sub> [ |S|!(|F|-|S|-1)! / |F|! ] &times; [ f(S &cup; {i}) - f(S) ]
  <span class="formula-annotation">
    <strong>&phi;<sub>i</sub></strong> = importancia de la feature i<br>
    <strong>S</strong> = cada subconjunto posible de features SIN la feature i<br>
    <strong>f(S &cup; {i}) - f(S)</strong> = cuanto cambia la prediccion al agregar la feature i<br>
    Se promedia sobre TODOS los subconjuntos posibles — por eso es costoso
  </span>
</div>

<p><strong>Tipo</strong>: Teoria de juegos (fundamentacion axiomatica)<br>
<strong>Pro</strong>: Es el unico metodo con garantias teoricas de consistencia, eficiencia y simetria. Satisface los axiomas de Shapley.<br>
<strong>Contra</strong>: <strong>Computacionalmente MUY caro</strong>. Con 165 features, el calculo exacto es intratable. Se usan aproximaciones (muestreo).</p>

<h2>Comparacion de los tres metodos</h2>

<table class="data-table">
  <thead>
    <tr><th>Aspecto</th><th>GNNExplainer</th><th>PGExplainer</th><th>GNNShap</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Enfoque</strong></td>
      <td>Optimizacion de mascara</td>
      <td>Red neuronal parametrica</td>
      <td>Valores de Shapley</td>
    </tr>
    <tr>
      <td><strong>Que explica</strong></td>
      <td>Aristas + features</td>
      <td>Aristas</td>
      <td>Features</td>
    </tr>
    <tr>
      <td><strong>Granularidad</strong></td>
      <td>Por nodo (local)</td>
      <td>Global (entrenado), local (inferencia)</td>
      <td>Por nodo (local)</td>
    </tr>
    <tr>
      <td><strong>Velocidad</strong></td>
      <td>Lento (optimiza por nodo)</td>
      <td>Rapido en inferencia</td>
      <td>Muy lento (muestreo)</td>
    </tr>
    <tr>
      <td><strong>Fundamento teorico</strong></td>
      <td>Info mutua</td>
      <td>Info mutua (parametrica)</td>
      <td>Axiomas de Shapley</td>
    </tr>
    <tr>
      <td><strong>Estabilidad (lo que medimos)</strong></td>
      <td><span class="metric-badge warn">Media</span></td>
      <td><span class="metric-badge bad">Muy baja (NaN)</span></td>
      <td><span class="metric-badge warn">Baja pero no degenerada</span></td>
    </tr>
  </tbody>
</table>

<h2>Que produce cada metodo?</h2>
<p>Independientemente del metodo, la salida es alguna forma de <strong>ranking de importancia</strong>:</p>
<ul>
  <li><strong>GNNExplainer</strong>: un score por arista (M[e] entre 0 y 1) + un score por feature. Las aristas con M alto son las "importantes".</li>
  <li><strong>PGExplainer</strong>: un score por arista, generado por la red auxiliar.</li>
  <li><strong>GNNShap</strong>: un valor de Shapley por feature (&phi;<sub>i</sub>). Features con |&phi;| alto son las mas influyentes.</li>
</ul>

<p>Estos rankings son los que vamos a comparar para medir ESTABILIDAD en la proxima seccion.</p>

<div class="info-box concept">
  <p class="info-title">Resumen de la seccion</p>
  <p>XAI = metodos para explicar POR QUE un modelo toma una decision. Usamos 3: GNNExplainer (mascara optimizada por nodo), PGExplainer (red neuronal que genera mascaras, pero inestable/NaN), y GNNShap (valores de Shapley, costoso pero teoricamente solido). Cada uno produce rankings de importancia de aristas o features, que son la materia prima para medir estabilidad.</p>
</div>
`;

  const quiz = [
    {
      q: 'Por que es importante que un modelo de deteccion de fraude pueda EXPLICAR sus decisiones?',
      options: [
        'Porque las explicaciones hacen que el modelo sea mas rapido',
        'Por regulacion (EU AI Act), confianza del analista, debugging y justicia',
        'Porque sin explicaciones el modelo no puede entrenar',
        'Porque los grafos son demasiado grandes para visualizar'
      ],
      correct: 1,
      explanation: 'La EU AI Act exige explicaciones para IA de alto riesgo. Ademas, analistas necesitan justificacion para actuar, y debugging requiere entender QUE aprendio el modelo.'
    },
    {
      q: 'Como funciona GNNExplainer?',
      options: [
        'Entrena una red neuronal separada para generar explicaciones',
        'Calcula los valores de Shapley de cada feature',
        'Optimiza una mascara sobre aristas/features que preserve la prediccion original con minimas aristas',
        'Remueve nodos uno por uno y mide el cambio en la prediccion'
      ],
      correct: 2,
      explanation: 'GNNExplainer optimiza una mascara continua M que, al aplicarse sobre las aristas, mantiene la prediccion original usando el minimo de aristas posible.'
    },
    {
      q: 'Que diferencia a PGExplainer de GNNExplainer?',
      options: [
        'PGExplainer es mas lento',
        'PGExplainer usa una red neuronal ENTRENADA que genera mascaras para cualquier nodo, en vez de optimizar individualmente',
        'PGExplainer solo funciona con GAT',
        'PGExplainer no genera mascaras, solo rankings'
      ],
      correct: 1,
      explanation: 'PGExplainer entrena un MLP auxiliar sobre un conjunto de nodos. Despues, puede generar explicaciones para nodos nuevos sin reoptimizar. Es "parametrico" vs el enfoque "por nodo" de GNNExplainer.'
    },
    {
      q: 'Que son los valores de Shapley que usa GNNShap?',
      options: [
        'Un tipo de funcion de activacion para GNNs',
        'Los pesos aprendidos de la red neuronal',
        'La contribucion marginal promedio de cada feature a la prediccion, considerando todas las combinaciones',
        'Los valores propios de la matriz de adyacencia'
      ],
      correct: 2,
      explanation: 'Los Shapley values vienen de teoria de juegos: miden cuanto contribuye cada "jugador" (feature) al resultado (prediccion), promediando sobre TODAS las coaliciones posibles.'
    },
    {
      q: 'Cual de los tres metodos es el mas costoso computacionalmente?',
      options: [
        'GNNExplainer (optimiza por nodo)',
        'PGExplainer (entrena una red auxiliar)',
        'GNNShap (muestrea combinaciones de 165 features)',
        'Todos cuestan lo mismo'
      ],
      correct: 2,
      explanation: 'GNNShap calcula Shapley values sobre 165 features. El calculo exacto es exponencial (2^165 subconjuntos). Incluso con aproximaciones por muestreo, es el mas lento de los tres.'
    }
  ];

  return { html, quiz };
}
