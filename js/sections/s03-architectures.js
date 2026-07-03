// Section 3: GNN Architectures
export function getSection() {
  const html = `
<h1 class="section-title">03 — Las 4 arquitecturas GNN que usamos</h1>
<p class="section-subtitle">No todas las GNNs agregan informacion de la misma forma. Cada arquitectura tiene su propia idea sobre COMO combinar los mensajes de los vecinos.</p>

<p>En la seccion anterior vimos la idea general de message passing. Ahora vamos a ver cuatro variantes concretas, cada una con su personalidad. Las cuatro son las que usamos en la tesis.</p>

<div class="arch-cards">
  <div class="arch-card">
    <div class="arch-card-title">GCN</div>
    <div class="arch-card-year">Kipf & Welling, 2017</div>
    <div class="arch-card-analogy">"Promediá a todos tus vecinos por igual"</div>
    <p>La mas simple y clasica. Base de todo.</p>
  </div>
  <div class="arch-card">
    <div class="arch-card-title">GraphSAGE</div>
    <div class="arch-card-year">Hamilton et al., 2017</div>
    <div class="arch-card-analogy">"Samplea algunos vecinos y combinalos"</div>
    <p>Escalable a grafos enormes.</p>
  </div>
  <div class="arch-card">
    <div class="arch-card-title">GAT</div>
    <div class="arch-card-year">Velickovic et al., 2018</div>
    <div class="arch-card-analogy">"Prestale MAS atencion a los vecinos importantes"</div>
    <p>Atencion aprendida sobre el grafo.</p>
  </div>
  <div class="arch-card">
    <div class="arch-card-title">TAGCN</div>
    <div class="arch-card-year">Du et al., 2017</div>
    <div class="arch-card-analogy">"Mira vecinos de 1, 2, 3... saltos con pesos distintos"</div>
    <p>Filtros polinomiales multi-hop.</p>
  </div>
</div>

<h2>GCN — Graph Convolutional Network</h2>
<p>La GCN es la arquitectura mas simple y una de las mas citadas en la historia de las GNNs. Publicada por Kipf y Welling en 2017, se basa en una simplificacion de la convolucion espectral sobre grafos.</p>

<div class="formula-box">
  <span class="formula-label">Formula GCN</span>
  H<sup>(l+1)</sup> = &sigma;( D&#x0302;<sup>-1/2</sup> A&#x0302; D&#x0302;<sup>-1/2</sup> H<sup>(l)</sup> W<sup>(l)</sup> )
  <span class="formula-annotation">
    <strong>A&#x0302;</strong> = A + I (adyacencia + auto-loops)<br>
    <strong>D&#x0302;</strong> = matriz diagonal de grados<br>
    <strong>H<sup>(l)</sup></strong> = features de todos los nodos en la capa l<br>
    <strong>W<sup>(l)</sup></strong> = pesos aprendibles<br>
    La normalizacion D&#x0302;<sup>-1/2</sup> A&#x0302; D&#x0302;<sup>-1/2</sup> es lo que hace que sea un "promedio ponderado por grado"
  </span>
</div>

<div class="info-box concept">
  <p class="info-title">Analogia: el estudiante que promedia</p>
  <p>Imaginate que cada persona en un grupo de estudio tiene una opinion (su vector de features). En cada ronda, cada persona reemplaza su opinion por el PROMEDIO de las opiniones de sus amigos (y la suya propia, gracias al self-loop). Despues de varias rondas, las opiniones se "suavizan" — personas conectadas terminan con representaciones similares. El problema? GCN trata a TODOS los vecinos por igual. El amigo experto pesa lo mismo que el que no sabe nada.</p>
</div>

<p><strong>Ventajas</strong>: Simple, eficiente, facil de implementar. Buen baseline.<br>
<strong>Desventajas</strong>: Trata a todos los vecinos igual. Sufre de "over-smoothing" con muchas capas (todos los nodos terminan con la misma representacion).</p>

<h2>GraphSAGE — SAmple and aggreGatE</h2>
<p>GraphSAGE fue una respuesta directa al problema de escalabilidad de GCN. En grafos con millones de nodos, multiplicar por la adyacencia completa es prohibitivo. La solucion? <strong>No uses todos los vecinos — sampleá algunos</strong>.</p>

<div class="formula-box">
  <span class="formula-label">Formula GraphSAGE</span>
  h<sub>v</sub><sup>(l+1)</sup> = &sigma;( W<sup>(l)</sup> &middot; CONCAT( h<sub>v</sub><sup>(l)</sup>, AGG({ h<sub>u</sub><sup>(l)</sup> : u &isin; SAMPLE(N(v), k) }) ) )
  <span class="formula-annotation">
    <strong>SAMPLE(N(v), k)</strong> = muestrea k vecinos aleatoriamente<br>
    <strong>CONCAT</strong> = concatena el vector propio con el agregado de vecinos<br>
    <strong>AGG</strong> = puede ser mean, LSTM, o pool<br>
    Nota: la concatenacion preserva la identidad del nodo + la info del vecindario
  </span>
</div>

<div class="info-box concept">
  <p class="info-title">Analogia: la encuesta rapida</p>
  <p>En vez de preguntarle a TODOS tus contactos (como GCN), GraphSAGE selecciona aleatoriamente un subconjunto de vecinos y les pregunta. Es como hacer una encuesta: no necesitas preguntarle a 10,000 personas, con una muestra representativa alcanza. Ademas, tu opinion propia se concatena (no se pierde en el promedio).</p>
</div>

<p><strong>Ventajas</strong>: Escala a grafos enormes. El sampling reduce la carga computacional. Funciona inductivamente (puede predecir sobre nodos nuevos que no vio durante entrenamiento).<br>
<strong>Desventajas</strong>: El sampling introduce varianza (depende de que vecinos selecciones). La calidad depende de k (cuantos vecinos sampleas).</p>

<div class="info-box success">
  <p class="info-title">Dato de la tesis</p>
  <p>GraphSAGE fue la arquitectura que MEJOR funciono en nuestros experimentos. 7 de los 9 modelos que pasaron el quality gate en Machine B fueron GraphSAGE. Y no es casualidad — la concatenacion de features propias + vecinos y la robustez del sampling la hacen ideal para grafos transaccionales ruidosos.</p>
</div>

<h2>GAT — Graph Attention Network</h2>
<p>GAT introduce el concepto de <strong>atencion</strong> al message passing. La idea: no todos los vecinos son igualmente importantes. Un vecino que es muy relevante para tu clasificacion deberia contribuir MAS al agregado.</p>

<div class="formula-box">
  <span class="formula-label">Formula GAT (simplificada)</span>
  h<sub>v</sub><sup>(l+1)</sup> = &sigma;( &Sigma;<sub>u&isin;N(v)</sub> &alpha;<sub>vu</sub> &middot; W<sup>(l)</sup> h<sub>u</sub><sup>(l)</sup> )
  <span class="formula-annotation">
    <strong>&alpha;<sub>vu</sub></strong> = coeficiente de atencion entre v y u (aprendido!)<br>
    Se calcula con: &alpha;<sub>vu</sub> = softmax( LeakyReLU( a<sup>T</sup> [Wh<sub>v</sub> || Wh<sub>u</sub>] ) )<br>
    <strong>||</strong> = concatenacion<br>
    <strong>a</strong> = vector de atencion aprendible<br>
    Los &alpha; suman 1 (gracias al softmax) — son pesos de importancia
  </span>
</div>

<p>Ademas, GAT usa <strong>multi-head attention</strong>: ejecuta K mecanismos de atencion en paralelo (cada uno con su propio vector a), concatena o promedia los resultados. Es la misma idea que en Transformers.</p>

<div class="info-box concept">
  <p class="info-title">Analogia: el profesor que escucha selectivamente</p>
  <p>Imaginate un profesor en un debate. GCN le daria la misma importancia a cada estudiante. GAT, en cambio, "aprende" a prestarle MAS atencion al estudiante que dice cosas relevantes y MENOS al que habla sin sentido. Los coeficientes de atencion &alpha; se aprenden durante el entrenamiento — la red DESCUBRE a quien prestarle atencion.</p>
</div>

<p><strong>Ventajas</strong>: Adapta la importancia de cada vecino. Multi-head captura multiples patrones. No necesita conocer la estructura global del grafo (atencion local).<br>
<strong>Desventajas</strong>: Mas parametros que GCN. Mas costoso computacionalmente. Puede ser inestable en grafos con mucho ruido.</p>

<h2>TAGCN — Topology Adaptive Graph Convolutional Network</h2>
<p>TAGCN toma un enfoque diferente: en vez de apilar capas para ver mas lejos en el grafo, usa <strong>filtros polinomiales</strong> que miran multiples saltos en una SOLA capa.</p>

<div class="formula-box">
  <span class="formula-label">Formula TAGCN</span>
  H<sup>(l+1)</sup> = &sigma;( &Sigma;<sub>k=0</sub><sup>K</sup> A&#x0302;<sup>k</sup> H<sup>(l)</sup> W<sub>k</sub><sup>(l)</sup> )
  <span class="formula-annotation">
    <strong>A&#x0302;<sup>k</sup></strong> = adyacencia normalizada elevada a la k-esima potencia<br>
    <strong>K</strong> = orden del filtro (cuantos saltos)<br>
    <strong>W<sub>k</sub></strong> = pesos DIFERENTES para cada distancia k<br>
    k=0 es el nodo mismo, k=1 vecinos directos, k=2 vecinos de vecinos, etc.
  </span>
</div>

<div class="info-box concept">
  <p class="info-title">Analogia: el radar con multiples anillos</p>
  <p>Imaginate un radar que detecta objetos a 1km, 2km, 3km, etc. Cada anillo tiene su propia sensibilidad (pesos W<sub>k</sub>). TAGCN puede decir: "los vecinos directos importan mucho, los de 2 saltos un poco, los de 3 casi nada". Todo esto en UNA sola capa, sin necesidad de apilar multiples capas como GCN.</p>
</div>

<p><strong>Ventajas</strong>: Ve multiples saltos sin apilar capas (reduce over-smoothing). Adapta la importancia por distancia. Basado en teoria de señales en grafos.<br>
<strong>Desventajas</strong>: Mas parametros por capa. Computar A<sup>k</sup> puede ser costoso para grafos densos.</p>

<h2>Tabla comparativa</h2>
<table class="data-table">
  <thead>
    <tr>
      <th>Caracteristica</th>
      <th>GCN</th>
      <th>GraphSAGE</th>
      <th>GAT</th>
      <th>TAGCN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Agregacion</strong></td>
      <td>Promedio normalizado</td>
      <td>Sample + concat</td>
      <td>Atencion ponderada</td>
      <td>Polinomios multi-hop</td>
    </tr>
    <tr>
      <td><strong>Trata vecinos igual?</strong></td>
      <td>Si</td>
      <td>Si (dentro del sample)</td>
      <td>No (atencion aprendida)</td>
      <td>Igual por hop, diferente entre hops</td>
    </tr>
    <tr>
      <td><strong>Escalabilidad</strong></td>
      <td>Media</td>
      <td>Alta (sampling)</td>
      <td>Media-baja</td>
      <td>Media</td>
    </tr>
    <tr>
      <td><strong>Complejidad</strong></td>
      <td>Baja</td>
      <td>Media</td>
      <td>Alta</td>
      <td>Media-alta</td>
    </tr>
    <tr>
      <td><strong>Mejor para</strong></td>
      <td>Baseline, grafos chicos</td>
      <td>Grafos grandes, transaccionales</td>
      <td>Cuando importa QUE vecino</td>
      <td>Patrones multi-escala</td>
    </tr>
    <tr>
      <td><strong>En nuestra tesis</strong></td>
      <td><span class="metric-badge bad">Peor</span></td>
      <td><span class="metric-badge good">Mejor</span></td>
      <td><span class="metric-badge warn">Medio</span></td>
      <td><span class="metric-badge warn">Medio</span></td>
    </tr>
  </tbody>
</table>

<div class="info-box concept">
  <p class="info-title">Resumen de la seccion</p>
  <p>Las cuatro arquitecturas comparten la idea de message passing pero difieren en COMO agregan: GCN promedia por igual, GraphSAGE samplea y concatena, GAT aprende pesos de atencion, TAGCN usa filtros polinomiales multi-hop. En nuestros experimentos, GraphSAGE domino, seguida de GAT, TAGCN, y por ultimo GCN.</p>
</div>
`;

  const quiz = [
    {
      q: 'Cual de las cuatro arquitecturas usa mecanismos de ATENCION para ponderar la importancia de cada vecino?',
      options: [
        'GCN',
        'GraphSAGE',
        'GAT',
        'TAGCN'
      ],
      correct: 2,
      explanation: 'GAT (Graph Attention Network) aprende coeficientes de atencion α para cada par de nodos vecinos, determinando cuanto contribuye cada uno.'
    },
    {
      q: 'Que estrategia usa GraphSAGE para ser escalable a grafos enormes?',
      options: [
        'Reduce la dimension de los features',
        'Samplea aleatoriamente un subconjunto de vecinos en vez de usar todos',
        'Comprime la matriz de adyacencia',
        'Usa solo 1 capa de message passing'
      ],
      correct: 1,
      explanation: 'GraphSAGE usa SAMPLE: en vez de agregar TODOS los vecinos (costoso para nodos con miles de conexiones), selecciona k vecinos al azar.'
    },
    {
      q: 'Cual es la arquitectura mas simple de las cuatro?',
      options: [
        'GAT, porque la atencion simplifica el calculo',
        'TAGCN, porque los polinomios son simples',
        'GCN, porque simplemente promedia los vecinos con normalizacion por grado',
        'GraphSAGE, porque samplear es mas facil que agregar'
      ],
      correct: 2,
      explanation: 'GCN es la mas simple: promedio normalizado por grado de los vecinos, multiplicado por pesos aprendibles, con activacion. Es el baseline clasico.'
    },
    {
      q: 'Que hace especial a TAGCN respecto a las demas?',
      options: [
        'Puede procesar grafos dirigidos y no dirigidos simultaneamente',
        'Mira vecinos de multiples distancias (1 salto, 2 saltos, etc.) en una SOLA capa, con pesos distintos por distancia',
        'No necesita matriz de adyacencia',
        'Es la unica que funciona con backpropagation'
      ],
      correct: 1,
      explanation: 'TAGCN usa filtros polinomiales: una capa mira simultáneamente vecinos de distancia 1, 2, ..., K con pesos W_k diferentes para cada distancia.'
    }
  ];

  return { html, quiz };
}
