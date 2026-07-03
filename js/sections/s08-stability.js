// Section 8: Stability — The Core Research Question
export function getSection() {
  const html = `
<h1 class="section-title">08 — Estabilidad: la pregunta central de la tesis</h1>
<p class="section-subtitle">Si corro el mismo explainer dos veces sobre el mismo modelo... obtengo la MISMA explicacion? Si la respuesta es "no", tenemos un problema serio.</p>

<h2>LA pregunta de investigacion</h2>

<div class="info-box important" style="font-size:1.05rem; padding:24px;">
  <p class="info-title" style="font-size:1.1rem;">Pregunta central de la tesis</p>
  <p><strong>"Que tan estables son las explicaciones generadas por metodos XAI sobre GNNs entrenadas en el dataset Elliptic, y como varia esa estabilidad segun el nivel de desbalance de clases?"</strong></p>
</div>

<p>Desempaquemos esto:</p>
<ul>
  <li><strong>Estables</strong> = si corro el explainer 5 veces con distintas semillas aleatorias, obtengo explicaciones similares?</li>
  <li><strong>Mismos metodos XAI</strong> = GNNExplainer, PGExplainer, GNNShap</li>
  <li><strong>Mismas GNNs</strong> = el modelo no cambia entre corridas. Solo cambia la aleatoriedad del explainer.</li>
  <li><strong>Segun nivel de desbalance</strong> = comparar 1:1, 1:10, 1:50, 1:100</li>
</ul>

<h2>Por que importa la estabilidad?</h2>

<p>Pensalo asi:</p>

<div class="info-box danger">
  <p class="info-title">Escenario catastrofico</p>
  <p>Un oficial de compliance corre el explainer sobre una transaccion sospechosa. GNNExplainer dice: "las aristas con nodos X, Y, Z son las importantes — estas transacciones conectadas son la razon del flag". El oficial investiga X, Y, Z.</p>
  <p style="margin-top:8px;">Al dia siguiente, OTRO oficial corre el mismo explainer sobre la misma transaccion. Pero ahora dice: "las aristas con nodos A, B, C son las importantes". Investiga A, B, C — completamente distintos.</p>
  <p style="margin-top:8px;">Si la explicacion cambia dependiendo de cuando la corres, <strong>no se puede usar para tomar decisiones</strong>. Es como un medico que hoy dice "cancer" y mañana "resfrio" mirando el mismo estudio.</p>
</div>

<p>La estabilidad no es un "nice to have". Es un <strong>REQUISITO</strong> para que las explicaciones tengan valor practico. Y nadie habia estudiado sistematicamente como el desbalance de clases afecta esta estabilidad en GNNs. Esa es la contribucion de la tesis.</p>

<h2>Como medimos estabilidad?</h2>
<p>Usamos dos metricas complementarias:</p>

<h3>1. Indice de Jaccard (estabilidad de subgrafos)</h3>
<p>El indice de Jaccard mide la superposicion entre dos conjuntos. Lo usamos para comparar los <strong>top-K aristas mas importantes</strong> entre corridas.</p>

<div class="formula-box">
  <span class="formula-label">Indice de Jaccard</span>
  J(A, B) = |A &cap; B| / |A &cup; B|
  <span class="formula-annotation">
    <strong>A</strong> = top-K aristas de la corrida 1<br>
    <strong>B</strong> = top-K aristas de la corrida 2<br>
    <strong>J = 1.0</strong>: ambas corridas seleccionan exactamente las mismas aristas<br>
    <strong>J = 0.0</strong>: cero aristas en comun
  </span>
</div>

<div class="info-box concept">
  <p class="info-title">Analogia: comparar listas de supermercado</p>
  <p>Imaginate que le pedis a dos personas que hagan la lista de los 10 items mas importantes del supermercado. Jaccard mide cuantos items comparten. Si las listas son identicas, Jaccard = 1. Si no comparten NADA, Jaccard = 0.</p>
</div>

<p>Para cada nodo explicado, corremos el explainer 5 veces con semillas distintas, obtenemos 5 conjuntos de top-K aristas, y calculamos el Jaccard promedio entre todos los pares (10 pares posibles con 5 corridas).</p>

<h3>2. Correlacion de Spearman (estabilidad de rankings de features)</h3>
<p>Spearman mide si dos rankings de features mantienen el mismo <strong>orden</strong>. No importa el valor absoluto de importancia — importa si feature 47 siempre esta en el top y feature 3 siempre esta al fondo.</p>

<div class="formula-box">
  <span class="formula-label">Correlacion de Spearman</span>
  &rho; = 1 - (6 &Sigma; d<sub>i</sub><sup>2</sup>) / (n(n<sup>2</sup>-1))
  <span class="formula-annotation">
    <strong>d<sub>i</sub></strong> = diferencia de rango de la feature i entre dos corridas<br>
    <strong>n</strong> = numero de features<br>
    <strong>&rho; = 1.0</strong>: mismo orden exacto<br>
    <strong>&rho; = 0.0</strong>: sin correlacion (orden aleatorio)<br>
    <strong>&rho; = -1.0</strong>: orden totalmente invertido
  </span>
</div>

<div class="info-box concept">
  <p class="info-title">Analogia: ranking de peliculas</p>
  <p>Pedile a alguien que rankee sus 10 peliculas favoritas el lunes. Despues pedile que lo haga de nuevo el viernes. Spearman mide que tan similares son los dos rankings. Si "El Padrino" es #1 ambas veces y "Toy Story" es #5 ambas veces, la correlacion es alta. Si los rankings cambian completamente, la correlacion es baja.</p>
</div>

<h2>El protocolo de replicas estocasticas</h2>
<p>Nuestro protocolo experimental es:</p>

<div class="timeline">
  <div class="timeline-item">
    <div class="timeline-item-title">1. Modelo fijo</div>
    <div class="timeline-item-desc">Tomamos un modelo GNN ya entrenado. Este NO cambia entre corridas. La aleatoriedad viene del explainer, no del modelo.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-item-title">2. Seleccion de nodos</div>
    <div class="timeline-item-desc">Seleccionamos nodos de test para explicar (tipicamente los que el modelo predijo como ilicitos).</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-item-title">3. Cinco replicas con semillas distintas</div>
    <div class="timeline-item-desc">Corremos el explainer 5 veces sobre los mismos nodos, cambiando solo la semilla aleatoria (random seed). La semilla afecta la inicializacion de la mascara (GNNExplainer), el muestreo (GNNShap), etc.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-item-title">4. Comparacion par a par</div>
    <div class="timeline-item-desc">Para cada nodo, calculamos Jaccard y Spearman entre todas las combinaciones de pares de las 5 corridas (10 pares = C(5,2)).</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-item-title">5. Promedios</div>
    <div class="timeline-item-desc">Promediamos Jaccard y Spearman sobre todos los nodos y pares para obtener un score unico por configuracion.</div>
  </div>
</div>

<h2>Que esperabamos vs que encontramos (adelanto)</h2>

<div class="compare-row">
  <div class="compare-box">
    <h4>Lo que esperabamos</h4>
    <ul>
      <li>GNNExplainer: estabilidad media-alta (metodo maduro)</li>
      <li>PGExplainer: alta estabilidad (parametrico, deberia generalizar)</li>
      <li>GNNShap: alta estabilidad (fundamentado en axiomas)</li>
      <li>A mayor desbalance, menor estabilidad (intuicion)</li>
    </ul>
  </div>
  <div class="compare-box">
    <h4>Lo que encontramos</h4>
    <ul>
      <li>GNNExplainer: Spearman ~0.47 a 1:10 pero decae a 0.24 a 1:100</li>
      <li>PGExplainer: Spearman ~0.00 (DEGENERADO por NaN)</li>
      <li>GNNShap: Spearman ~0.15 (bajo pero no degenerado)</li>
      <li>Jaccard = 1.0 UNIVERSAL (deterministico en mascaras)</li>
      <li>El patron de decaimiento es CLARO y NOVEDOSO</li>
    </ul>
  </div>
</div>

<div class="info-box concept">
  <p class="info-title">Resumen de la seccion</p>
  <p>La estabilidad mide si un explainer produce la misma explicacion bajo diferentes semillas aleatorias. La medimos con Jaccard (solapamiento de aristas top-K) y Spearman (consistencia de rankings de features). Usamos 5 replicas estocasticas por nodo. El hallazgo clave: la estabilidad DECAE con el desbalance, GNNExplainer muestra un patron claro de degradacion, PGExplainer es inestable por diseño, y Jaccard=1.0 es universal (lo cual es informativo en si mismo).</p>
</div>
`;

  const quiz = [
    {
      q: 'Que significa que una explicacion XAI sea "estable"?',
      options: [
        'Que el modelo base no cambia de prediccion',
        'Que al correr el MISMO explainer multiples veces sobre el MISMO modelo, las explicaciones son similares',
        'Que la explicacion funciona en diferentes datasets',
        'Que la explicacion es rapida de generar'
      ],
      correct: 1,
      explanation: 'Estabilidad = consistencia bajo variabilidad estocastica. Si corro el explainer con semillas diferentes sobre el mismo modelo/nodo, deberia obtener explicaciones similares.'
    },
    {
      q: 'Que mide el indice de Jaccard en nuestro contexto?',
      options: [
        'La precision del modelo GNN',
        'La similitud entre dos conjuntos de aristas importantes (top-K) generados por corridas distintas del explainer',
        'La distancia entre nodos en el grafo',
        'El numero de features relevantes'
      ],
      correct: 1,
      explanation: 'Jaccard = |A interseccion B| / |A union B|, donde A y B son los top-K aristas de dos corridas distintas. 1.0 = identicas, 0.0 = sin solapamiento.'
    },
    {
      q: 'Que mide la correlacion de Spearman?',
      options: [
        'Si dos rankings de features mantienen el mismo ORDEN relativo, independientemente de los valores absolutos',
        'La distancia euclidiana entre vectores de features',
        'La correlacion entre la prediccion del modelo y la etiqueta real',
        'La velocidad del explainer'
      ],
      correct: 0,
      explanation: 'Spearman mide concordancia de ORDEN. Si feature A siempre esta por encima de feature B en el ranking de importancia, Spearman es alto. Valores: -1 a +1.'
    },
    {
      q: 'Por que usamos 5 replicas con diferentes semillas?',
      options: [
        'Porque 5 es el minimo para calcular media y varianza con significancia',
        'Para capturar la variabilidad estocastica del explainer y medir cuanto varian las explicaciones',
        'Porque cada semilla corresponde a una arquitectura GNN diferente',
        'Porque el dataset tiene 5 clases'
      ],
      correct: 1,
      explanation: '5 replicas con semillas distintas generan 10 pares para comparar (C(5,2)=10). El modelo es fijo — solo cambia la aleatoriedad del explainer. Asi medimos la varianza inherente al metodo.'
    }
  ];

  return { html, quiz };
}
