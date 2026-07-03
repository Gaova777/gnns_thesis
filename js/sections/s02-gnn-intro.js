// Section 2: From Neural Networks to Graph Neural Networks
export function getSection() {
  const html = `
<h1 class="section-title">02 — De redes neuronales a Graph Neural Networks</h1>
<p class="section-subtitle">Ya sabemos que es un MLP. Ahora vamos a entender por que NO sirve para grafos, y como las GNNs resuelven ese problema con una idea brillante: message passing.</p>

<h2>El problema: los grafos no encajan en un MLP</h2>
<p>Un <strong>Multi-Layer Perceptron (MLP)</strong> — la red neuronal clasica — recibe un vector de tamaño fijo como entrada. Por ejemplo, una imagen de 28&times;28 se "aplasta" a un vector de 784 valores, y listo.</p>

<p>Pero con grafos hay dos problemas fundamentales:</p>

<div class="info-box danger">
  <p class="info-title">Problema 1: Tamaño variable</p>
  <p>Un grafo puede tener 5 nodos o 5 millones. Cada nodo puede tener 2 vecinos o 200. No podes definir una capa con un tamaño fijo de entrada si cada nodo tiene diferente cantidad de vecinos.</p>
</div>

<div class="info-box danger">
  <p class="info-title">Problema 2: Sin orden fijo</p>
  <p>En una imagen, el pixel [0,0] siempre es la esquina superior izquierda. En un grafo, no hay "primer nodo" ni "ultimo nodo". Si renumeras los nodos, el grafo es el MISMO, pero la representacion cambia. Un MLP trataria nodos renumerados como datos totalmente diferentes, lo cual es incorrecto.</p>
</div>

<p>Entonces, como procesamos un grafo con una red neuronal? La respuesta es una de las ideas mas elegantes del deep learning moderno:</p>

<h2>La idea central: Message Passing (paso de mensajes)</h2>
<p>La intuicion es simple y poderosa:</p>

<div class="info-box concept">
  <p class="info-title">Message Passing en una frase</p>
  <p>Cada nodo <strong>le pregunta a sus vecinos</strong> que informacion tienen, <strong>combina</strong> esas respuestas, y <strong>actualiza</strong> su propia representacion. Repite esto L veces (una por cada capa de la GNN), y cada nodo termina "sabiendo" sobre su vecindario extendido.</p>
</div>

<p>Es como el juego del telefono, pero bien hecho. En cada ronda:</p>
<ol>
  <li><strong>Recolectar</strong>: cada nodo recibe los vectores de features de sus vecinos.</li>
  <li><strong>Agregar</strong>: combina esos vectores en uno solo (promedio, suma, o maximo).</li>
  <li><strong>Actualizar</strong>: usa el vector agregado para actualizar su propia representacion.</li>
</ol>

<div class="diagram-container">
  <p class="diagram-title">Message Passing — el nodo central recibe mensajes de sus vecinos</p>
  <div class="mp-container">
    <!-- Neighbor nodes -->
    <div class="mp-node neighbor n1">v<sub>1</sub></div>
    <div class="mp-node neighbor n2">v<sub>2</sub></div>
    <div class="mp-node neighbor n3">v<sub>3</sub></div>
    <!-- Center node -->
    <div class="mp-node center animating">v<sub>0</sub></div>
    <!-- Message particles -->
    <div class="msg-particle" style="top:30%; left:46%;"></div>
    <div class="msg-particle" style="top:62%; left:22%;"></div>
    <div class="msg-particle" style="top:62%; right:22%;"></div>
  </div>
  <p style="text-align:center; color:var(--text-muted); font-size:0.85rem; margin-top:8px;">
    Los puntos amarillos representan los "mensajes" (vectores de features) viajando de los vecinos v<sub>1</sub>, v<sub>2</sub>, v<sub>3</sub> hacia el nodo central v<sub>0</sub>. El nodo central los agrega y actualiza su representacion.
  </p>
</div>

<h2>La formula de una GNN (simplificada)</h2>
<p>Si todo esto te suena abstracto, veamos la formula. Prometo que es mas simple de lo que parece:</p>

<div class="formula-box">
  <span class="formula-label">Formula de actualizacion GNN — capa l</span>
  h<sub>v</sub><sup>(l+1)</sup> = &sigma;( W<sup>(l)</sup> &middot; AGG({ h<sub>u</sub><sup>(l)</sup> : u &isin; N(v) }) )
  <span class="formula-annotation">
    <strong>h<sub>v</sub><sup>(l+1)</sup></strong> = nueva representacion del nodo v despues de la capa l<br>
    <strong>&sigma;</strong> = funcion de activacion (ReLU, por ejemplo)<br>
    <strong>W<sup>(l)</sup></strong> = matriz de pesos APRENDIBLES de la capa l<br>
    <strong>AGG</strong> = funcion de agregacion (mean, sum, o max)<br>
    <strong>h<sub>u</sub><sup>(l)</sup></strong> = representacion actual del vecino u<br>
    <strong>N(v)</strong> = conjunto de vecinos del nodo v
  </span>
</div>

<p>Desglosemos paso a paso:</p>

<h3>1. N(v) — Los vecinos</h3>
<p><code>N(v)</code> es simplemente el conjunto de nodos que estan conectados directamente a v. Si v es el nodo B en nuestro grafo de la seccion anterior, N(B) = {A, C, D}. Esto se lee directamente de la matriz de adyacencia.</p>

<h3>2. AGG — La agregacion</h3>
<p>Necesitamos combinar los vectores de TODOS los vecinos en un solo vector. Las opciones mas comunes:</p>
<ul>
  <li><strong>Mean (promedio)</strong>: suma todos los vectores y divide por la cantidad. Funciona bien, pero nodos con muchos vecinos y pocos vecinos contribuyen igual.</li>
  <li><strong>Sum (suma)</strong>: suma directamente. Nodos con mas vecinos generan señales mas fuertes.</li>
  <li><strong>Max</strong>: toma el maximo elemento a elemento. Captura la feature mas "prominente" del vecindario.</li>
</ul>

<h3>3. W — Los pesos aprendibles</h3>
<p>Esto es EXACTAMENTE como en un MLP. La matriz W transforma el vector agregado a un nuevo espacio. Estos pesos se aprenden durante el entrenamiento con backpropagation. La misma W se aplica a TODOS los nodos en la misma capa — esto es lo que hace que la GNN sea invariante al tamaño del grafo.</p>

<h3>4. &sigma; — La activacion</h3>
<p>Tambien igual que en un MLP. Tipicamente ReLU, pero puede ser cualquier funcion no lineal. Sin ella, apilar capas seria equivalente a una sola transformacion lineal.</p>

<div class="info-box concept">
  <p class="info-title">Analogia: el campo receptivo</p>
  <p>Si tenes 1 capa de GNN, cada nodo "ve" a sus vecinos directos (1 salto). Con 2 capas, ve a los vecinos de los vecinos (2 saltos). Con 3 capas, 3 saltos. Es como en las CNNs: cada capa de convolucion expande el "campo receptivo". Pero aca el "campo receptivo" es el vecindario del grafo.</p>
</div>

<h2>Como aprende una GNN? Igual que cualquier red neuronal!</h2>
<p>Este es un punto que mucha gente no capta al principio: <strong>el entrenamiento de una GNN usa exactamente el mismo loop que ya conoces</strong>:</p>

<div class="timeline">
  <div class="timeline-item">
    <div class="timeline-item-title">1. Forward pass</div>
    <div class="timeline-item-desc">Las features de los nodos pasan por L capas de message passing. Cada capa agrega vecinos, transforma con W, aplica activacion. Al final, cada nodo tiene una representacion "enriquecida".</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-item-title">2. Prediccion</div>
    <div class="timeline-item-desc">La representacion final de cada nodo pasa por una capa clasificadora (tipicamente una capa lineal + softmax) que predice la clase: "licita" o "ilicita".</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-item-title">3. Loss</div>
    <div class="timeline-item-desc">Comparamos la prediccion con la etiqueta real. Usamos cross-entropy (u otra funcion de loss). Obtenemos un numero que dice "que tan mal estuvo la prediccion".</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-item-title">4. Backpropagation</div>
    <div class="timeline-item-desc">Los gradientes fluyen HACIA ATRAS a traves de las capas de message passing, exactamente como en un MLP. Actualizamos las matrices W de cada capa con el optimizador (Adam, SGD, etc.).</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-item-title">5. Repetir</div>
    <div class="timeline-item-desc">Epoch tras epoch, los pesos W aprenden a generar representaciones que capturan los patrones estructurales relevantes para la tarea.</div>
  </div>
</div>

<div class="info-box success">
  <p class="info-title">El punto clave</p>
  <p>La GNN NO inventa una nueva forma de entrenar. Usa el mismo backpropagation de siempre. Lo que cambia es la ARQUITECTURA: en vez de capas densas, tiene capas de message passing que son conscientes de la estructura del grafo. Pero loss, gradientes, optimizador — todo igual.</p>
</div>

<h2>GNN vs MLP — la diferencia esencial</h2>

<div class="compare-row">
  <div class="compare-box">
    <h4>MLP</h4>
    <p>Cada nodo procesado <strong>de forma independiente</strong>. El vector [165 features] entra, pasa por capas densas, sale una prediccion. El nodo no sabe NADA de sus vecinos.</p>
    <p style="color:var(--c-error); font-weight:600; margin-top:8px;">Ignora la estructura del grafo.</p>
  </div>
  <div class="compare-box">
    <h4>GNN</h4>
    <p>Cada nodo procesado <strong>en contexto</strong>. El vector [165 features] se enriquece con la informacion de sus vecinos via message passing. Despues de L capas, el nodo "sabe" sobre su vecindario de radio L.</p>
    <p style="color:var(--c-success); font-weight:600; margin-top:8px;">Usa la estructura del grafo como señal.</p>
  </div>
</div>

<h2>Tipos de tareas con GNNs</h2>
<p>Las GNNs pueden resolver distintos tipos de problemas:</p>
<ul>
  <li><strong>Clasificacion de nodos</strong>: predecir la etiqueta de cada nodo. Ejemplo: esta transaccion es licita o ilicita? <em>Este es nuestro caso.</em></li>
  <li><strong>Clasificacion de grafos</strong>: predecir una etiqueta para el grafo completo. Ejemplo: esta molecula es toxica?</li>
  <li><strong>Prediccion de enlaces</strong>: predecir si deberia existir una arista entre dos nodos. Ejemplo: recomendar amigos en una red social.</li>
</ul>

<div class="info-box concept">
  <p class="info-title">Resumen de la seccion</p>
  <p>Las GNNs resuelven el problema de procesar grafos (tamaño variable, sin orden fijo) con <strong>message passing</strong>: cada nodo agrega informacion de sus vecinos, la transforma con pesos aprendibles, y actualiza su representacion. Se entrenan con backpropagation normal. Nosotros hacemos <strong>clasificacion de nodos</strong>: cada transaccion se clasifica como licita o ilicita.</p>
</div>
`;

  const quiz = [
    {
      q: 'Cual es la idea central de las GNNs (message passing)?',
      options: [
        'Cada nodo se procesa independientemente, como en un MLP',
        'Cada nodo recolecta informacion de sus vecinos, la agrega, y actualiza su representacion',
        'El grafo se convierte en una imagen y se usa una CNN',
        'Se calcula la matriz inversa de la adyacencia'
      ],
      correct: 1,
      explanation: 'Message passing = cada nodo pregunta a sus vecinos, agrega la info, y se actualiza. Es la base de TODAS las arquitecturas GNN.'
    },
    {
      q: 'Que hace la funcion AGG en la formula de la GNN?',
      options: [
        'Selecciona el vecino mas importante y descarta el resto',
        'Combina los vectores de todos los vecinos en un solo vector (sum, mean o max)',
        'Elimina los vecinos con features menores a un umbral',
        'Calcula la distancia entre el nodo y sus vecinos'
      ],
      correct: 1,
      explanation: 'AGG (agregacion) combina los vectores de TODOS los vecinos en uno solo. Las opciones comunes son promedio (mean), suma (sum) o maximo (max).'
    },
    {
      q: 'Por que un MLP no puede procesar un grafo directamente?',
      options: [
        'Porque los MLPs son mas lentos que las GNNs',
        'Porque los grafos tienen tamaño variable y sin orden fijo, incompatible con entrada de tamaño fijo',
        'Porque los MLPs no soportan vectores de features',
        'Porque los MLPs no pueden usar backpropagation'
      ],
      correct: 1,
      explanation: 'Los grafos tienen cantidad variable de nodos/vecinos y no tienen un orden canonico. Un MLP necesita entrada de tamaño fijo y ordenada.'
    },
    {
      q: 'Si una GNN tiene 2 capas de message passing, cuantos "saltos" de vecinos puede ver cada nodo?',
      options: [
        '1 salto (solo vecinos directos)',
        '2 saltos (vecinos y vecinos de vecinos)',
        '4 saltos (el doble)',
        'Infinitos saltos'
      ],
      correct: 1,
      explanation: 'Cada capa expande el campo receptivo en 1 salto. 2 capas = 2 saltos. El nodo ve sus vecinos directos Y los vecinos de sus vecinos.'
    },
    {
      q: 'Que significa N(v) en la formula de la GNN?',
      options: [
        'El numero total de nodos del grafo',
        'La norma del vector de features del nodo v',
        'El conjunto de vecinos directos del nodo v',
        'El numero de capas de la red'
      ],
      correct: 2,
      explanation: 'N(v) es el "neighborhood" — el conjunto de nodos que estan conectados directamente a v por una arista.'
    }
  ];

  return { html, quiz };
}
