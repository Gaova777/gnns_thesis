// Section 6: Balancing Techniques
export function getSection() {
  const html = `
<h1 class="section-title">06 — Tecnicas de balance: como enseñarle al modelo a NO ignorar el fraude</h1>
<p class="section-subtitle">El desbalance es el problema. Ahora veamos las soluciones. Spoiler: no todas funcionan igual de bien.</p>

<h2>La idea central</h2>
<p>El problema del desbalance se reduce a esto: durante el entrenamiento, la <strong>funcion de loss</strong> recibe MUCHOS mas ejemplos de la clase mayoritaria (licita). Entonces los gradientes estan dominados por esa clase, y el modelo aprende a optimizar para ella.</p>

<p>La solucion: modificar la loss para que los errores en la clase minoritaria (ilicita) <strong>pesen MAS</strong>. Hay varias formas de hacerlo:</p>

<h2>Tecnica 1: None (baseline, sin compensacion)</h2>
<p>La cross-entropy estandar sin ninguna modificacion:</p>

<div class="formula-box">
  <span class="formula-label">Cross-Entropy clasica</span>
  L = -&Sigma;<sub>i</sub> [ y<sub>i</sub> log(p<sub>i</sub>) + (1-y<sub>i</sub>) log(1-p<sub>i</sub>) ]
  <span class="formula-annotation">
    Cada ejemplo contribuye igual a la loss, sin importar su clase.<br>
    Con 98% licitas y 2% ilicitas, el 98% de la loss viene de ejemplos licitos.
  </span>
</div>

<p>Es el baseline: no hacemos nada para compensar el desbalance. Funciona decentemente en 1:1 (ya esta balanceado) pero se degrada rapido con mayor desbalance.</p>

<h2>Tecnica 2: Class Weighting (ponderacion por clase)</h2>
<p>La idea es simple y elegante: asignar un <strong>peso mayor</strong> a la loss de los ejemplos de la clase rara.</p>

<div class="formula-box">
  <span class="formula-label">Cross-Entropy con pesos de clase</span>
  L = -&Sigma;<sub>i</sub> w<sub>y<sub>i</sub></sub> [ y<sub>i</sub> log(p<sub>i</sub>) + (1-y<sub>i</sub>) log(1-p<sub>i</sub>) ]
  <span class="formula-annotation">
    <strong>w<sub>c</sub></strong> = peso de la clase c = N<sub>total</sub> / (N<sub>clases</sub> &times; N<sub>c</sub>)<br>
    Si hay 100 ejemplos y 2 son ilicitos: w<sub>illicit</sub> = 100/(2&times;2) = 25<br>
    Cada error en un ilicito "pesa" 25 veces mas que un error en un licito.
  </span>
</div>

<div class="info-box concept">
  <p class="info-title">Analogia: el profesor con notas</p>
  <p>Imaginate un profesor que corrige examenes. Si tiene 98 alumnos aprobados y 2 desaprobados, un profesor normal le dedica la misma atencion a cada examen. Pero un buen profesor le dedica MUCHA mas atencion a los 2 desaprobados — porque ahí es donde esta el problema. Class weighting es eso: hacer que cada error en la clase rara "duela" mas, forzando al modelo a prestarle atencion.</p>
</div>

<p><strong>Ventajas</strong>:</p>
<ul>
  <li>Se adapta automaticamente a cualquier ratio de desbalance (los pesos se calculan de los datos)</li>
  <li>Cero hiperparametros extra que ajustar</li>
  <li>Simple de implementar (un parametro <code>weight</code> en la loss de PyTorch)</li>
</ul>
<p><strong>Desventajas</strong>:</p>
<ul>
  <li>Puede hacer que el modelo sobrecompense en escenarios extremos</li>
  <li>No distingue entre ejemplos faciles y dificiles dentro de la misma clase</li>
</ul>

<h2>Tecnica 3: Focal Loss (Lin et al., 2017)</h2>
<p>Focal Loss fue inventada para deteccion de objetos (RetinaNet) y tiene una idea genial: no solo ponderar por clase, sino <strong>reducir la loss de los ejemplos FACILES</strong> y concentrarse en los DIFICILES.</p>

<div class="formula-box">
  <span class="formula-label">Focal Loss</span>
  FL(p) = -&alpha; (1-p)<sup>&gamma;</sup> log(p)
  <span class="formula-annotation">
    <strong>&alpha;</strong> = peso de la clase (similar a class weighting)<br>
    <strong>&gamma;</strong> = factor de enfoque. Cuanto mayor, mas se penalizan los ejemplos dificiles<br>
    <strong>(1-p)<sup>&gamma;</sup></strong> = si p es alto (ejemplo facil), este termino es ~0 y la loss casi desaparece<br>
    Si p es bajo (ejemplo dificil), este termino es grande y la loss se mantiene
  </span>
</div>

<div class="diagram-container">
  <p class="diagram-title">Comparacion: Cross-Entropy vs Focal Loss</p>
  <svg class="loss-curve-svg" viewBox="0 0 400 250" xmlns="http://www.w3.org/2000/svg">
    <!-- Axes -->
    <line x1="50" y1="200" x2="370" y2="200" stroke="var(--text-muted)" stroke-width="1"/>
    <line x1="50" y1="200" x2="50" y2="20" stroke="var(--text-muted)" stroke-width="1"/>
    <text x="210" y="235" text-anchor="middle" fill="var(--text-muted)" font-size="11">Probabilidad predicha (p)</text>
    <text x="15" y="110" text-anchor="middle" fill="var(--text-muted)" font-size="11" transform="rotate(-90, 15, 110)">Loss</text>

    <!-- X axis labels -->
    <text x="50" y="215" text-anchor="middle" fill="var(--text-muted)" font-size="10">0</text>
    <text x="210" y="215" text-anchor="middle" fill="var(--text-muted)" font-size="10">0.5</text>
    <text x="370" y="215" text-anchor="middle" fill="var(--text-muted)" font-size="10">1.0</text>

    <!-- Cross-Entropy curve (steeper) -->
    <path d="M50,20 Q100,60 150,100 Q200,140 250,160 Q300,175 350,195" fill="none" stroke="#ef4444" stroke-width="2.5"/>
    <text x="100" y="50" fill="#ef4444" font-size="11" font-weight="600">Cross-Entropy</text>

    <!-- Focal Loss curve (gamma=2, flatter for easy examples) -->
    <path d="M50,40 Q100,90 150,140 Q200,170 250,185 Q300,193 350,198" fill="none" stroke="#6366f1" stroke-width="2.5"/>
    <text x="130" y="125" fill="#6366f1" font-size="11" font-weight="600">Focal Loss (&gamma;=2)</text>

    <!-- Annotation -->
    <line x1="280" y1="160" x2="280" y2="195" stroke="var(--c-warning)" stroke-dasharray="4" stroke-width="1"/>
    <text x="285" y="155" fill="var(--c-warning)" font-size="9">Ejemplos faciles:</text>
    <text x="285" y="167" fill="var(--c-warning)" font-size="9">Focal casi 0</text>
  </svg>
</div>

<div class="info-box danger">
  <p class="info-title">CUIDADO: Focal Loss con &alpha;=0.75 en escenario 1:1 es TOXICO</p>
  <p>Este es uno de los bugs que encontramos y corregimos. El parametro &alpha; en Focal Loss le da mas peso a una clase. Con &alpha;=0.75, la clase ilicita recibe 3x mas peso. En escenario 1:1, donde las clases YA estan balanceadas, esto <strong>desbalancea artificialmente</strong> el entrenamiento — el modelo se sesga hacia la clase ilicita y pierde precision en la licita.</p>
  <p style="margin-top:8px;"><strong>Leccion</strong>: una tecnica de balance aplicada a datos que ya estan balanceados puede ser peor que no hacer nada.</p>
</div>

<h2>Resultados reales: cual funciona mejor?</h2>

<div class="diagram-container">
  <p class="diagram-title">Porcentaje de modelos que pasaron el quality gate (F1 &ge; 0.70, MCC &ge; 0.40)</p>
  <div class="bar-chart" style="height:200px;">
    <div class="bar-item">
      <div class="bar-value">50%</div>
      <div class="bar-fill success" style="height:50%;"></div>
      <div class="bar-label">Class<br>Weighting</div>
    </div>
    <div class="bar-item">
      <div class="bar-value">33%</div>
      <div class="bar-fill warning" style="height:33%;"></div>
      <div class="bar-label">Focal<br>Loss</div>
    </div>
    <div class="bar-item">
      <div class="bar-value">33%</div>
      <div class="bar-fill warning" style="height:33%;"></div>
      <div class="bar-label">None<br>(baseline)</div>
    </div>
  </div>
</div>

<table class="data-table">
  <thead>
    <tr><th>Tecnica</th><th>Tasa de aprobacion</th><th>Mejor escenario</th><th>Peor escenario</th><th>Nota</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Class weighting</strong></td>
      <td><span class="metric-badge good">50%</span></td>
      <td>1:10, 1:50</td>
      <td>1:1 (pero sigue pasando)</td>
      <td>La mas robusta. Funciona en todos los escenarios.</td>
    </tr>
    <tr>
      <td><strong>Focal Loss</strong></td>
      <td><span class="metric-badge warn">33%</span></td>
      <td>1:10</td>
      <td>1:1 (alpha desbalanceado)</td>
      <td>Buena cuando el desbalance es real, toxica en 1:1.</td>
    </tr>
    <tr>
      <td><strong>None</strong></td>
      <td><span class="metric-badge warn">33%</span></td>
      <td>1:1</td>
      <td>1:50, 1:100</td>
      <td>Solo funciona bien cuando los datos ya estan balanceados.</td>
    </tr>
  </tbody>
</table>

<div class="info-box success">
  <p class="info-title">El hallazgo clave</p>
  <p>Class weighting es la tecnica mas robusta. Se adapta automaticamente al ratio de desbalance, no tiene hiperparametros extra que ajustar, y funciona razonablemente bien en TODOS los escenarios. Focal Loss brilla en desbalance moderado (1:10) pero falla en datos balanceados. Y no hacer nada solo sirve cuando ya tenes datos balanceados.</p>
</div>

<div class="info-box concept">
  <p class="info-title">Resumen de la seccion</p>
  <p>Tres tecnicas de balance: None (baseline), Class Weighting (pesos inversamente proporcionales al tamaño de clase), y Focal Loss (reduce loss de ejemplos faciles). Class weighting es la mas robusta (50% pass rate). Focal Loss falla en datos ya balanceados (1:1) porque el alpha sobrecompensa. La eleccion de tecnica de balance interactua con el nivel de desbalance — no hay "mejor tecnica universal".</p>
</div>
`;

  const quiz = [
    {
      q: 'Que hace class weighting en la funcion de loss?',
      options: [
        'Elimina ejemplos de la clase mayoritaria',
        'Duplica los ejemplos de la clase minoritaria',
        'Asigna mayor penalizacion (peso) a los errores en la clase rara',
        'Cambia la arquitectura del modelo para la clase rara'
      ],
      correct: 2,
      explanation: 'Class weighting multiplica la loss de cada ejemplo por un peso inversamente proporcional al tamaño de su clase. Errores en la clase rara "duelen" mas.'
    },
    {
      q: 'Por que Focal Loss con alpha=0.75 falla en el escenario 1:1?',
      options: [
        'Porque Focal Loss no soporta clases binarias',
        'Porque alpha da 3x mas peso a la clase ilicita, desbalanceando datos que ya estaban balanceados',
        'Porque el escenario 1:1 tiene muy pocos datos',
        'Porque gamma=2 es demasiado alto'
      ],
      correct: 1,
      explanation: 'En 1:1, ambas clases tienen la misma cantidad de datos. Alpha=0.75 le da mas peso a la ilicita, CREANDO un desbalance artificial. Es como ponerle muletas a alguien que camina perfecto.'
    },
    {
      q: 'Cual tecnica de balance tiene la mayor tasa de aprobacion en nuestros experimentos?',
      options: [
        'None (baseline) con 50%',
        'Focal Loss con 67%',
        'Class Weighting con 50%',
        'Todas tienen la misma tasa'
      ],
      correct: 2,
      explanation: 'Class Weighting tiene 50% de aprobacion, vs 33% de Focal Loss y 33% de None. Su adaptacion automatica al ratio de desbalance la hace la mas robusta.'
    },
    {
      q: 'Que controla el parametro gamma (&gamma;) en Focal Loss?',
      options: [
        'El peso de cada clase',
        'El learning rate del optimizador',
        'El factor de enfoque: cuanto mayor, mas se penalizan los ejemplos dificiles (donde el modelo falla)',
        'La cantidad de capas de la GNN'
      ],
      correct: 2,
      explanation: 'Gamma controla cuanto se reduce la loss de ejemplos faciles (alta p). Con gamma=0, Focal Loss = Cross-Entropy. Con gamma=2, los ejemplos faciles casi no contribuyen.'
    }
  ];

  return { html, quiz };
}
