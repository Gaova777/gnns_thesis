// Section 1: Graphs & Why They Matter
export function getSection() {
  const html = `
<h1 class="section-title">01 — Grafos: la estructura que lo conecta todo</h1>
<p class="section-subtitle">Antes de hablar de redes neuronales sobre grafos, tenemos que entender que es un grafo y por que es TAN poderoso.</p>

<h2>Que es un grafo?</h2>
<p>Un <strong>grafo</strong> es, en esencia, una forma de representar <em>cosas</em> y las <em>conexiones entre esas cosas</em>. Matematicamente, un grafo G se define como G = (V, E), donde:</p>
<ul>
  <li><strong>V</strong> (vertices o nodos): son las "cosas". Pueden ser personas, ciudades, moleculas, transacciones de Bitcoin... lo que sea.</li>
  <li><strong>E</strong> (edges o aristas): son las conexiones entre esas cosas. Una amistad en Facebook, una ruta entre ciudades, un enlace quimico, una transferencia de dinero.</li>
</ul>

<div class="info-box concept">
  <p class="info-title">Analogia: la red social</p>
  <p>Pensa en Instagram. Cada persona es un <strong>nodo</strong>. Cuando alguien sigue a otra persona, eso crea una <strong>arista</strong> (conexion) entre ambos nodos. El grafo completo de Instagram tiene miles de millones de nodos y aristas. Pero la ESTRUCTURA — quien esta conectado con quien — es la informacion clave.</p>
</div>

<p>Veamos un grafo simple con 5 nodos:</p>

<div class="diagram-container">
  <p class="diagram-title">Grafo de ejemplo — 5 nodos, 6 aristas</p>
  <svg class="graph-svg" viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg">
    <!-- Edges -->
    <line class="edge-line" x1="200" y1="50" x2="80" y2="150"/>
    <line class="edge-line" x1="200" y1="50" x2="320" y2="150"/>
    <line class="edge-line" x1="80" y1="150" x2="320" y2="150"/>
    <line class="edge-line" x1="80" y1="150" x2="140" y2="260"/>
    <line class="edge-line" x1="320" y1="150" x2="260" y2="260"/>
    <line class="edge-line" x1="140" y1="260" x2="260" y2="260"/>
    <!-- Nodes -->
    <circle class="node-circle" cx="200" cy="50" r="24" fill="#6366f1"/>
    <text class="node-label" x="200" y="50">A</text>
    <circle class="node-circle" cx="80" cy="150" r="24" fill="#6366f1"/>
    <text class="node-label" x="80" y="150">B</text>
    <circle class="node-circle" cx="320" cy="150" r="24" fill="#6366f1"/>
    <text class="node-label" x="320" y="150">C</text>
    <circle class="node-circle" cx="140" cy="260" r="24" fill="#6366f1"/>
    <text class="node-label" x="140" y="260">D</text>
    <circle class="node-circle" cx="260" cy="260" r="24" fill="#6366f1"/>
    <text class="node-label" x="260" y="260">E</text>
  </svg>
  <p style="text-align:center; color:var(--text-muted); font-size:0.85rem; margin-top:8px;">
    Los circulos son <strong>nodos</strong>. Las lineas son <strong>aristas</strong>. A esta conectado con B y C; B con C y D; etc.
  </p>
</div>

<h2>La matriz de adyacencia</h2>
<p>Hay muchas formas de representar un grafo en la computadora, pero la mas comun es la <strong>matriz de adyacencia</strong>. Es una tabla cuadrada de N&times;N (donde N es el numero de nodos) donde cada celda dice si hay una conexion (1) o no (0).</p>

<p>Para el grafo de arriba, la matriz de adyacencia seria:</p>

<div class="diagram-container" style="text-align:center;">
  <p class="diagram-title">Matriz de adyacencia (5 x 5)</p>
  <div class="adj-matrix" style="grid-template-columns: repeat(6, 40px);">
    <div class="adj-cell header"></div>
    <div class="adj-cell header">A</div>
    <div class="adj-cell header">B</div>
    <div class="adj-cell header">C</div>
    <div class="adj-cell header">D</div>
    <div class="adj-cell header">E</div>

    <div class="adj-cell header">A</div>
    <div class="adj-cell empty">0</div>
    <div class="adj-cell filled">1</div>
    <div class="adj-cell filled">1</div>
    <div class="adj-cell empty">0</div>
    <div class="adj-cell empty">0</div>

    <div class="adj-cell header">B</div>
    <div class="adj-cell filled">1</div>
    <div class="adj-cell empty">0</div>
    <div class="adj-cell filled">1</div>
    <div class="adj-cell filled">1</div>
    <div class="adj-cell empty">0</div>

    <div class="adj-cell header">C</div>
    <div class="adj-cell filled">1</div>
    <div class="adj-cell filled">1</div>
    <div class="adj-cell empty">0</div>
    <div class="adj-cell empty">0</div>
    <div class="adj-cell filled">1</div>

    <div class="adj-cell header">D</div>
    <div class="adj-cell empty">0</div>
    <div class="adj-cell filled">1</div>
    <div class="adj-cell empty">0</div>
    <div class="adj-cell empty">0</div>
    <div class="adj-cell filled">1</div>

    <div class="adj-cell header">E</div>
    <div class="adj-cell empty">0</div>
    <div class="adj-cell empty">0</div>
    <div class="adj-cell filled">1</div>
    <div class="adj-cell filled">1</div>
    <div class="adj-cell empty">0</div>
  </div>
  <p style="color:var(--text-muted); font-size:0.85rem; margin-top:12px;">
    Fila A, columna B = 1 porque A y B estan conectados. Fila A, columna D = 0 porque no lo estan. La diagonal siempre es 0 (un nodo no se conecta consigo mismo).
  </p>
</div>

<div class="info-box important">
  <p class="info-title">Dato clave</p>
  <p>La matriz de adyacencia es <strong>simetrica</strong> en grafos no dirigidos (si A esta conectado con B, entonces B esta conectado con A). En grafos dirigidos (como transacciones de dinero: A le manda plata a B, pero B no necesariamente a A), la matriz NO es simetrica.</p>
</div>

<h2>Features de nodo: cada nodo lleva informacion</h2>
<p>Un grafo no solo tiene estructura (quien esta conectado con quien). Cada nodo puede cargar un <strong>vector de features</strong> — un conjunto de propiedades numéricas que lo describen.</p>

<p>Imaginemos que nuestro grafo representa personas:</p>

<table class="data-table">
  <thead>
    <tr><th>Nodo</th><th>Edad</th><th>Ingreso mensual</th><th>Num. transacciones</th><th>Antiguedad (dias)</th></tr>
  </thead>
  <tbody>
    <tr><td>A</td><td>28</td><td>3200</td><td>145</td><td>730</td></tr>
    <tr><td>B</td><td>45</td><td>8500</td><td>892</td><td>2190</td></tr>
    <tr><td>C</td><td>19</td><td>1100</td><td>23</td><td>60</td></tr>
    <tr><td>D</td><td>62</td><td>12000</td><td>1567</td><td>5475</td></tr>
    <tr><td>E</td><td>33</td><td>4500</td><td>234</td><td>1095</td></tr>
  </tbody>
</table>

<p>El vector de features del nodo A seria <code>[28, 3200, 145, 730]</code>. Cada nodo lleva su propio vector. En nuestro dataset real (Elliptic), cada transaccion de Bitcoin tiene <strong>165 features</strong> — un vector MUCHO mas grande.</p>

<h2>Features de arista (opcional)</h2>
<p>Ademas de features en los nodos, las aristas tambien pueden tener propiedades. Por ejemplo, en una red de pagos: el <em>monto transferido</em>, la <em>fecha de la transaccion</em>, o el <em>tipo de operacion</em>. En nuestra tesis no usamos features de arista, pero es bueno saber que existen.</p>

<h2>Por que los grafos son PERFECTOS para Bitcoin</h2>
<p>Aca es donde todo cobra sentido. Bitcoin es, literalmente, un grafo:</p>
<ul>
  <li>Cada <strong>transaccion</strong> es un nodo.</li>
  <li>Cada <strong>flujo de dinero</strong> entre transacciones es una arista.</li>
  <li>Cada transaccion tiene <strong>propiedades</strong>: monto, fee, cantidad de inputs/outputs, etc.</li>
</ul>

<div class="info-box success">
  <p class="info-title">La clave</p>
  <p>No estas forzando datos tabulares a formato de grafo. Las transacciones de Bitcoin YA SON un grafo por naturaleza. Cada satoshi que se mueve de una transaccion a otra crea una arista real. Es la representacion mas natural posible.</p>
</div>

<p>Y aca viene la parte critica para nuestra tesis: si un grupo de transacciones esta involucrado en <strong>lavado de dinero</strong>, es probable que formen patrones ESTRUCTURALES en el grafo — cadenas rapidas de transacciones, ciclos donde el dinero vuelve al origen, nodos "puente" que conectan clusters sospechosos. Estos patrones son INVISIBLES para un modelo de ML tradicional que mira cada transaccion por separado, pero son evidentes para un modelo que entiende la estructura del grafo.</p>

<div class="compare-row">
  <div class="compare-box">
    <h4>ML Tradicional (MLP)</h4>
    <p>Mira cada transaccion <strong>aislada</strong>. Solo ve el vector de 165 features. No sabe con quien se conecto, no ve patrones de red. Es como juzgar a una persona solo por su CV sin saber quienes son sus amigos.</p>
  </div>
  <div class="compare-box">
    <h4>GNN (Graph Neural Network)</h4>
    <p>Mira cada transaccion <strong>en contexto</strong>. Ve sus features Y las features de sus vecinos. Ve la estructura local del grafo. Es como juzgar a una persona sabiendo su CV, su circulo social y el circulo social de sus amigos.</p>
  </div>
</div>

<div class="info-box concept">
  <p class="info-title">Resumen de la seccion</p>
  <p>Un grafo = nodos + aristas. Cada nodo tiene un vector de features. La matriz de adyacencia codifica las conexiones. Bitcoin es naturalmente un grafo, y los patrones de fraude se esconden en la ESTRUCTURA de las conexiones, no solo en las propiedades individuales.</p>
</div>
`;

  const quiz = [
    {
      q: 'En un grafo, que representa un "nodo"?',
      options: [
        'Una conexion entre dos elementos',
        'Una entidad o elemento del sistema (persona, transaccion, etc.)',
        'El peso de una arista',
        'La dimension del vector de features'
      ],
      correct: 1,
      explanation: 'Un nodo (o vertice) representa una entidad en el grafo. Las conexiones entre nodos se llaman aristas.'
    },
    {
      q: 'Si la celda [B][D] de la matriz de adyacencia es 1, que significa?',
      options: [
        'Que B y D tienen las mismas features',
        'Que B tiene mas conexiones que D',
        'Que existe una arista (conexion) entre B y D',
        'Que B es vecino de segundo grado de D'
      ],
      correct: 2,
      explanation: 'En la matriz de adyacencia, un 1 en la posicion [i][j] significa que hay una arista directa entre el nodo i y el nodo j.'
    },
    {
      q: 'Cual es el vector de features del nodo C en la tabla de ejemplo?',
      options: [
        '[28, 3200, 145, 730]',
        '[19, 1100, 23, 60]',
        '[1, 1, 0, 0, 1]',
        '[33, 4500, 234, 1095]'
      ],
      correct: 1,
      explanation: 'El nodo C tiene edad=19, ingreso=1100, transacciones=23, antiguedad=60. Su vector es [19, 1100, 23, 60].'
    },
    {
      q: 'Por que las transacciones de Bitcoin se modelan naturalmente como un grafo?',
      options: [
        'Porque Bitcoin usa criptografia de grafos',
        'Porque los flujos de dinero entre transacciones ya forman una red de nodos y aristas',
        'Porque los grafos son mas rapidos que las tablas SQL',
        'Porque Bitcoin fue diseñado para usar GNNs'
      ],
      correct: 1,
      explanation: 'Las transacciones de Bitcoin se conectan naturalmente: cada flujo de satoshis de una transaccion a otra es una arista real, formando un grafo.'
    },
    {
      q: 'Que ventaja tiene una GNN sobre un MLP para detectar fraude en Bitcoin?',
      options: [
        'Entrena mas rapido',
        'Usa menos memoria',
        'Puede ver patrones en la estructura de conexiones, no solo features individuales',
        'Siempre tiene mejor accuracy'
      ],
      correct: 2,
      explanation: 'La GNN ve cada transaccion en CONTEXTO: sus features + las features y estructura de sus vecinos. Los patrones de fraude se esconden en la topologia del grafo.'
    }
  ];

  return { html, quiz };
}
