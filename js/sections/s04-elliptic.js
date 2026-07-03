// Section 4: The Elliptic Bitcoin Dataset
export function getSection() {
  const html = `
<h1 class="section-title">04 — El dataset Elliptic: Bitcoin real para detectar fraude</h1>
<p class="section-subtitle">No estamos jugando con datos de juguete. Este es un dataset REAL de transacciones de Bitcoin, etiquetado por la empresa Elliptic para deteccion de lavado de dinero.</p>

<h2>Contexto: Anti-Money Laundering (AML)</h2>
<p>El lavado de dinero (<em>money laundering</em>) es el proceso de hacer que dinero obtenido ilegalmente (narcotrafico, fraude, ransomware) parezca legitimo. Bitcoin, por su naturaleza pseudoanonima, es una herramienta atractiva para los criminales.</p>

<p>Las empresas de compliance como <strong>Elliptic</strong> se dedican a analizar el blockchain de Bitcoin y etiquetar transacciones sospechosas. En 2019, Weber et al. publicaron el <strong>Elliptic Dataset</strong> — el primer dataset publico de gran escala de transacciones de Bitcoin etiquetadas para deteccion de fraude.</p>

<h2>Los numeros del dataset</h2>

<div class="welcome-stats">
  <div class="welcome-stat">
    <div class="welcome-stat-num">203,769</div>
    <div class="welcome-stat-label">Transacciones (nodos)</div>
  </div>
  <div class="welcome-stat">
    <div class="welcome-stat-num">234,355</div>
    <div class="welcome-stat-label">Flujos de dinero (aristas)</div>
  </div>
  <div class="welcome-stat">
    <div class="welcome-stat-num">165</div>
    <div class="welcome-stat-label">Features por transaccion</div>
  </div>
  <div class="welcome-stat">
    <div class="welcome-stat-num">49</div>
    <div class="welcome-stat-label">Timesteps temporales</div>
  </div>
</div>

<h2>Las tres clases</h2>
<p>Cada transaccion tiene una etiqueta:</p>

<div class="diagram-container">
  <p class="diagram-title">Distribucion de clases en Elliptic</p>
  <svg class="graph-svg" viewBox="0 0 500 320" xmlns="http://www.w3.org/2000/svg">
    <!-- Licit cluster -->
    <circle cx="130" cy="150" r="90" fill="rgba(16,185,129,0.15)" stroke="#10b981" stroke-width="2"/>
    <text x="130" y="110" text-anchor="middle" fill="#10b981" font-weight="800" font-size="14">LICITA</text>
    <text x="130" y="140" text-anchor="middle" fill="#10b981" font-size="24" font-weight="800">42,019</text>
    <text x="130" y="170" text-anchor="middle" fill="#10b981" font-size="12">(21% del total)</text>
    <text x="130" y="195" text-anchor="middle" fill="#10b981" font-size="11">Exchanges, wallets legales</text>

    <!-- Illicit cluster -->
    <circle cx="300" cy="100" r="50" fill="rgba(239,68,68,0.15)" stroke="#ef4444" stroke-width="2"/>
    <text x="300" y="80" text-anchor="middle" fill="#ef4444" font-weight="800" font-size="14">ILICITA</text>
    <text x="300" y="105" text-anchor="middle" fill="#ef4444" font-size="20" font-weight="800">4,545</text>
    <text x="300" y="130" text-anchor="middle" fill="#ef4444" font-size="11">(2.2% del total)</text>

    <!-- Unknown cluster -->
    <circle cx="350" cy="230" r="70" fill="rgba(148,163,184,0.15)" stroke="#94a3b8" stroke-width="2"/>
    <text x="350" y="210" text-anchor="middle" fill="#94a3b8" font-weight="800" font-size="14">DESCONOCIDA</text>
    <text x="350" y="240" text-anchor="middle" fill="#94a3b8" font-size="20" font-weight="800">157,205</text>
    <text x="350" y="265" text-anchor="middle" fill="#94a3b8" font-size="11">(77% del total)</text>
  </svg>
</div>

<div class="info-box important">
  <p class="info-title">El problema inmediato</p>
  <p>Solo el <strong>2.2%</strong> de las transacciones etiquetadas son ilicitas. Y el 77% NI SIQUIERA tiene etiqueta. Esto hace que el problema sea EXTREMADAMENTE desbalanceado. Un modelo que siempre dice "licita" acierta el 90%+ de las veces. Vamos a hablar de esto en detalle en la seccion 5.</p>
</div>

<h2>Las 165 features</h2>
<p>Cada transaccion (nodo) tiene un vector de 165 features numericas:</p>
<ul>
  <li><strong>Features 1-94</strong> (local transaction features): propiedades directas de la transaccion — monto, fee, numero de inputs/outputs, tamaño, etc. Algunas estan anonimizadas por privacidad.</li>
  <li><strong>Features 95-165</strong> (aggregated features): estadisticas del vecindario inmediato — promedio de montos de transacciones conectadas, desviacion estandar de fees, maximos, minimos, etc.</li>
</ul>

<p>Las features estan anonimizadas (no sabemos exactamente que representa la feature 47, por ejemplo), pero incluyen suficiente informacion para que un modelo aprenda patrones de fraude.</p>

<h2>Estructura temporal: 49 timesteps</h2>
<p>El dataset no es una foto estatica. Esta dividido en <strong>49 ventanas temporales</strong> (timesteps), cada una cubriendo aproximadamente 2 semanas de actividad Bitcoin. Esto es CRUCIAL para nuestro diseño experimental.</p>

<div class="diagram-container">
  <p class="diagram-title">Division temporal del dataset</p>
  <div style="display:flex; align-items:center; gap:4px; flex-wrap:wrap; justify-content:center;">
    ${Array.from({length:49}, (_, i) => {
      let color, label;
      if (i < 34) { color = '#6366f1'; label = 'train'; }
      else if (i < 42) { color = '#f59e0b'; label = 'val'; }
      else { color = '#ef4444'; label = 'test'; }
      return `<div style="width:16px; height:40px; background:${color}; border-radius:2px; opacity:${0.5 + (i/49)*0.5};" title="ts ${i+1} (${label})"></div>`;
    }).join('')}
  </div>
  <div style="display:flex; justify-content:space-between; margin-top:8px; font-size:0.75rem; color:var(--text-muted);">
    <span>ts 1</span>
    <span style="color:#6366f1; font-weight:600;">TRAIN (ts 1-34)</span>
    <span style="color:#f59e0b; font-weight:600;">VAL (ts 35-42)</span>
    <span style="color:#ef4444; font-weight:600;">TEST (ts 43-49)</span>
    <span>ts 49</span>
  </div>
</div>

<p>La division es <strong>estrictamente temporal</strong>: el modelo se entrena con transacciones del PASADO (ts 1-34), se valida con las siguientes (ts 35-42), y se evalua en el FUTURO (ts 43-49). Nunca ve datos del futuro durante entrenamiento. Esto es crucial porque simula un escenario real de deteccion de fraude.</p>

<h2>El problema del "Dark Market Shutdown"</h2>
<p>Aca viene algo que mucha gente no entiende y que es CLAVE para interpretar nuestros resultados:</p>

<div class="info-box danger">
  <p class="info-title">Temporal Covariate Shift</p>
  <p>Los patrones de fraude <strong>CAMBIAN</strong> entre los periodos de entrenamiento y test. Durante el periodo cubierto por el dataset, varios mercados negros de la darknet fueron cerrados por autoridades. Esto significa que:</p>
  <ul>
    <li>Los patrones de fraude en ts 1-34 (train) estan dominados por ciertos mercados negros activos.</li>
    <li>En ts 43-49 (test), esos mercados ya no existen, y aparecen NUEVOS patrones de fraude.</li>
    <li>El modelo ve patrones en test que NUNCA aparecieron en train.</li>
  </ul>
  <p style="margin-top:8px;"><strong>Consecuencia</strong>: el F1 en test es naturalmente mas bajo que en validacion. Esto NO significa que el modelo es malo — significa que el fraude evoluciona.</p>
</div>

<p>Es como entrenar a un detective para reconocer ladrones de bancos, y despues pedirle que detecte estafadores online. Los patrones cambiaron. El detective aprendio bien lo que le enseñaron, pero el crimen evoluciono.</p>

<h2>Por que este dataset importa</h2>

<table class="data-table">
  <thead>
    <tr><th>Caracteristica</th><th>Importancia</th></tr>
  </thead>
  <tbody>
    <tr><td>Datos reales (no sinteticos)</td><td>Los patrones son genuinos, no fabricados</td></tr>
    <tr><td>Escala significativa (~200K nodos)</td><td>Lo suficientemente grande para evaluar escalabilidad</td></tr>
    <tr><td>Benchmark establecido</td><td>Weber et al. 2019, citado 500+ veces. Resultados comparables.</td></tr>
    <tr><td>Desbalance extremo natural</td><td>Refleja la realidad: el fraude es raro</td></tr>
    <tr><td>Estructura temporal</td><td>Permite evaluar generalizacion temporal</td></tr>
    <tr><td>Features anonimizadas</td><td>Publicable sin comprometer privacidad</td></tr>
  </tbody>
</table>

<div class="info-box concept">
  <p class="info-title">Resumen de la seccion</p>
  <p>El dataset Elliptic tiene 203,769 transacciones de Bitcoin con 165 features cada una, conectadas por 234,355 aristas. Solo 2.2% son ilicitas. Esta dividido temporalmente en 49 timesteps, y los patrones de fraude CAMBIAN entre train y test (temporal covariate shift) por el cierre de dark markets. Esto explica por que el F1 en test es mas bajo, y NO es un defecto del modelo.</p>
</div>
`;

  const quiz = [
    {
      q: 'Cuantas transacciones (nodos) tiene el dataset Elliptic?',
      options: [
        '42,019',
        '234,355',
        '203,769',
        '4,545'
      ],
      correct: 2,
      explanation: 'El dataset tiene 203,769 transacciones (nodos). 234,355 son las aristas (conexiones). 42,019 son licitas y 4,545 ilicitas.'
    },
    {
      q: 'Que porcentaje de transacciones ETIQUETADAS son ilicitas?',
      options: [
        '77% (la mayoria)',
        '21% (una de cada cinco)',
        'Aproximadamente 2.2% (menos de 1 en 40)',
        '50% (mitad y mitad)'
      ],
      correct: 2,
      explanation: 'Solo 4,545 de ~46,500 transacciones etiquetadas son ilicitas — aproximadamente 2.2%. El 77% ni siquiera tiene etiqueta.'
    },
    {
      q: 'Por que se usa una division TEMPORAL (train ts 1-34, test ts 43-49) en vez de aleatoria?',
      options: [
        'Porque es mas facil de implementar',
        'Porque simula un escenario real: el modelo debe detectar fraude FUTURO basado en patrones del PASADO',
        'Porque hay mas datos en los primeros timesteps',
        'Porque Bitcoin solo funciona en orden temporal'
      ],
      correct: 1,
      explanation: 'La division temporal simula el mundo real: entrenas con datos historicos y debes detectar fraude nuevo. Una division aleatoria seria irreal porque "filtraria" informacion del futuro.'
    },
    {
      q: 'Que es el "problema del dark market shutdown" y como afecta los resultados?',
      options: [
        'Bitcoin dejo de funcionar en algunos periodos del dataset',
        'Los dark markets generaban datos corruptos que habia que limpiar',
        'Los patrones de fraude cambiaron entre train y test porque cerraron dark markets, causando temporal covariate shift',
        'Los nodos de dark markets se eliminaron del dataset'
      ],
      correct: 2,
      explanation: 'El cierre de dark markets cambio los patrones de fraude. El modelo ve patrones NUEVOS en test que no existian en train. El F1 baja, pero no por fallo del modelo.'
    },
    {
      q: 'Cuantas features tiene cada transaccion en el dataset?',
      options: [
        '49 (una por timestep)',
        '165 (94 locales + 71 agregadas)',
        '234,355 (igual que las aristas)',
        '3 (una por clase)'
      ],
      correct: 1,
      explanation: 'Cada transaccion tiene 165 features: 94 propiedades locales de la transaccion + 71 estadísticas agregadas del vecindario.'
    }
  ];

  return { html, quiz };
}
