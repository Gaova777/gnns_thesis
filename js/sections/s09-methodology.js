// Section 9: Experimental Design & Methodology
export function getSection() {
  const html = `
<h1 class="section-title">09 — Diseño experimental y metodologia</h1>
<p class="section-subtitle">144 configuraciones posibles, 2 maquinas, 3 bugs criticos, y un pipeline que se dividio en dos. Aca esta como se diseño y ejecuto todo.</p>

<h2>La matriz experimental completa</h2>
<p>Nuestra tesis evalua TODAS las combinaciones posibles de:</p>

<table class="data-table">
  <thead>
    <tr><th>Dimension</th><th>Valores</th><th>Cantidad</th></tr>
  </thead>
  <tbody>
    <tr><td><strong>Escenarios de desbalance</strong></td><td>1:1, 1:10, 1:50, 1:100</td><td>4</td></tr>
    <tr><td><strong>Arquitecturas GNN</strong></td><td>GCN, GraphSAGE, GAT, TAGCN</td><td>4</td></tr>
    <tr><td><strong>Tecnicas de balance</strong></td><td>None, Class Weighting, Focal Loss</td><td>3</td></tr>
    <tr><td><strong>Metodos XAI</strong></td><td>GNNExplainer, PGExplainer, GNNShap</td><td>3</td></tr>
  </tbody>
</table>

<div class="formula-box">
  <span class="formula-label">Total de configuraciones</span>
  4 escenarios &times; 4 arquitecturas &times; 3 balanceos &times; 3 explainers = <strong>144 configuraciones</strong>
  <span class="formula-annotation">
    Pero primero hay que entrenar los modelos: 4 &times; 4 &times; 3 = 48 modelos unicos.<br>
    Despues, cada modelo se explica con 3 metodos XAI, cada uno con 5 replicas.
  </span>
</div>

<h2>Distribucion en maquinas</h2>

<div class="compare-row">
  <div class="compare-box">
    <h4>Machine B — RTX 4060 (8GB)</h4>
    <ul>
      <li>Arquitecturas: <strong>GCN, GraphSAGE</strong></li>
      <li>Todos los escenarios (1:1 a 1:100)</li>
      <li>Los 3 balanceos</li>
      <li>= 24 modelos + 72 configs XAI</li>
    </ul>
  </div>
  <div class="compare-box">
    <h4>Machine C — RTX 3050 (4GB)</h4>
    <ul>
      <li>Arquitecturas: <strong>GAT, TAGCN</strong></li>
      <li>Todos los escenarios (1:1 a 1:100)</li>
      <li>Los 3 balanceos</li>
      <li>= 24 modelos + 72 configs XAI</li>
    </ul>
  </div>
</div>

<h2>El Pipeline v3: por que se dividio en dos scripts</h2>
<p>La version original (v1/v2) del pipeline corria todo junto: Optuna, training, explainability. Pero en v2 descubrimos que los modelos tenian F1 de 0.02-0.08 — <strong>BASURA</strong>. La literatura reporta 0.70-0.85. Algo estaba muy mal.</p>

<div class="info-box danger">
  <p class="info-title">Los 3 bugs criticos que encontramos</p>
</div>

<div class="finding-card">
  <span class="finding-card-num">1</span>
  <h4>Focal Loss alpha INVERTIDO</h4>
  <p>El parametro alpha, que deberia dar MAS peso a la clase rara (ilicita), estaba dando mas peso a la clase MAYORITARIA. El efecto: en vez de compensar el desbalance, lo EMPEORABA. F1 se desplomaba porque el modelo aprendia aun mas a ignorar la clase ilicita.</p>
  <p><strong>Archivo</strong>: <code>src/balancing/losses.py</code> | <strong>Fix</strong>: alpha = peso clase rara (0.75)</p>
</div>

<div class="finding-card">
  <span class="finding-card-num">2</span>
  <h4>Early stopping en MCC (ruidoso en imbalance)</h4>
  <p>El early stopping usaba MCC como metrica. MCC es ruidoso cuando hay pocas muestras de la clase rara — fluctua mucho entre epochs. El modelo paraba de entrenar ANTES de converger porque MCC "mejoraba" por ruido y despues "empeoraba" por ruido.</p>
  <p><strong>Archivo</strong>: <code>src/training/trainer.py</code> | <strong>Fix</strong>: cambio a F1 como early stop metric</p>
</div>

<div class="finding-card">
  <span class="finding-card-num">3</span>
  <h4>Optuna sin prior de literatura</h4>
  <p>Optuna buscaba hiperparametros en un espacio demasiado amplio sin guia. Con 50 trials, la probabilidad de encontrar buenas combinaciones era baja. No usaba informacion de papers publicados sobre Elliptic.</p>
  <p><strong>Archivo</strong>: <code>src/training/hyperopt.py</code> | <strong>Fix</strong>: warm-start con priors de arXiv:2602.23599 como trial 0</p>
</div>

<h3>Pipeline dividido en dos etapas</h3>
<p>La solucion fue separar entrenamiento de explicabilidad:</p>

<div class="timeline">
  <div class="timeline-item">
    <div class="timeline-item-title">Script 1: train_matrix.py</div>
    <div class="timeline-item-desc">Corre Optuna con warm-start (trial 0 = hiperparametros de la literatura). 50 trials de busqueda + entrenamiento final con 600 epochs, patience 50, early stop por F1. Quality gate: F1 &ge; 0.70, MCC &ge; 0.40. Guarda modelo + metadata JSON con metricas.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-item-title">Quality Gate (filtro)</div>
    <div class="timeline-item-desc">Solo modelos que pasan el quality gate avanzan. Si un modelo tiene F1 = 0.15, no tiene sentido explicar sus decisiones — no aprendio nada. Ahorramos tiempo y evitamos generar explicaciones de basura.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-item-title">Script 2: explain_matrix.py</div>
    <div class="timeline-item-desc">Lee metadata de modelos entrenados, filtra por quality_passed=True. Corre GNNExplainer + PGExplainer + GNNShap con 5 replicas estocasticas. Calcula Jaccard y Spearman. Guarda resultados en CSV + MLflow.</div>
  </div>
</div>

<h2>Por que el quality gate usa metricas de VALIDACION, no de TEST?</h2>
<p>Esta es una pregunta clave que cualquier evaluador de tesis haria:</p>

<div class="info-box important">
  <p class="info-title">La respuesta tiene que ver con el temporal covariate shift</p>
  <p>Ya vimos que los patrones de fraude CAMBIAN entre train y test (dark market shutdowns). El F1 en test es naturalmente mas bajo porque el modelo ve fraude "nuevo" que no existia en train. Si usaramos test F1 para el quality gate, RECHAZARIAMOS modelos que SI aprendieron buenos patrones pero sufren el shift temporal.</p>
  <p style="margin-top:8px;">El F1 en validacion (ts 35-42) es mas cercano temporalmente al train (ts 1-34), asi que refleja mejor si el modelo aprendio los patrones disponibles. Ademas, NUNCA usamos el test set para tomar decisiones de diseño — solo para reportar resultados finales. Esto es buena practica de ML.</p>
</div>

<h2>Warm-start de Optuna: la ventaja de leer papers</h2>
<p>El warm-start funciona asi:</p>
<ol>
  <li>Leemos de la literatura que hiperparametros funcionan bien para GNNs en Elliptic (arXiv:2602.23599)</li>
  <li>Estos hiperparametros se inyectan como <strong>trial 0</strong> en Optuna</li>
  <li>Optuna empieza desde un "buen vecindario" en el espacio de busqueda, en vez de explorar desde cero</li>
  <li>Los 49 trials restantes exploran alrededor y mas alla de ese punto inicial</li>
</ol>

<p>Es como darle a un explorador un mapa con marcas de "otros exploradores encontraron agua aqui". No garantiza que vaya directo, pero le ahorra tiempo descartando desiertos.</p>

<h2>Threshold calibration</h2>
<p>En clasificacion binaria, el modelo produce una probabilidad (0 a 1) y un umbral decide cuando la transaccion es "ilicita". Por defecto el umbral es 0.5, pero esto no es optimo cuando las clases estan desbalanceadas.</p>

<p>Usamos <strong>prevalence matching</strong>: ajustamos el umbral para que la proporcion de predicciones positivas en validacion sea similar a la proporcion real de la clase ilicita. Esto calibra mejor el trade-off precision/recall.</p>

<h2>Resumen del flujo experimental</h2>

<div class="diagram-container">
  <p class="diagram-title">Flujo completo: de configuracion a resultado</p>
  <div style="display:flex; flex-wrap:wrap; gap:8px; justify-content:center; align-items:center;">
    <div style="background:var(--c-primary); color:white; padding:8px 16px; border-radius:8px; font-size:0.85rem; font-weight:600;">Config YAML</div>
    <span style="color:var(--text-muted);">&rarr;</span>
    <div style="background:var(--c-primary); color:white; padding:8px 16px; border-radius:8px; font-size:0.85rem; font-weight:600;">Optuna (50 trials)</div>
    <span style="color:var(--text-muted);">&rarr;</span>
    <div style="background:var(--c-primary); color:white; padding:8px 16px; border-radius:8px; font-size:0.85rem; font-weight:600;">Train (600 ep)</div>
    <span style="color:var(--text-muted);">&rarr;</span>
    <div style="background:var(--c-warning); color:white; padding:8px 16px; border-radius:8px; font-size:0.85rem; font-weight:600;">Quality Gate</div>
    <span style="color:var(--text-muted);">&rarr;</span>
    <div style="background:var(--c-success); color:white; padding:8px 16px; border-radius:8px; font-size:0.85rem; font-weight:600;">3 Explainers &times; 5 replicas</div>
    <span style="color:var(--text-muted);">&rarr;</span>
    <div style="background:var(--c-success); color:white; padding:8px 16px; border-radius:8px; font-size:0.85rem; font-weight:600;">Jaccard + Spearman</div>
  </div>
</div>

<div class="info-box concept">
  <p class="info-title">Resumen de la seccion</p>
  <p>144 configuraciones totales, ejecutadas en 2 maquinas (B: GCN/GraphSAGE, C: GAT/TAGCN). El pipeline se dividio en dos scripts despues de encontrar 3 bugs criticos (alpha invertido, MCC ruidoso, Optuna sin priors). El quality gate usa metricas de validacion (no test) por el temporal covariate shift. Warm-start de Optuna inyecta conocimiento de la literatura. Solo modelos que pasan el gate se explican.</p>
</div>
`;

  const quiz = [
    {
      q: 'Por que el pipeline se dividio en dos scripts separados (train + explain)?',
      options: [
        'Para correr en paralelo en 2 GPUs',
        'Para poder iterar sobre la calidad del modelo independientemente de los explainers, y no explicar modelos que no aprendieron',
        'Porque PyTorch no soporta training y XAI en el mismo proceso',
        'Porque los explainers necesitan Python 2 y el training Python 3'
      ],
      correct: 1,
      explanation: 'Separar permite: (1) diagnosticar y arreglar problemas de training sin correr explainers, (2) filtrar modelos malos con quality gate, (3) iterar independientemente.'
    },
    {
      q: 'Por que el quality gate usa metricas de VALIDACION en vez de TEST?',
      options: [
        'Porque el test set es mas chico',
        'Porque nunca debemos usar test para tomar decisiones de diseño, y el temporal shift hace que test F1 sea naturalmente mas bajo',
        'Porque las metricas de validacion son mas precisas',
        'Porque el test set no tiene etiquetas'
      ],
      correct: 1,
      explanation: 'El F1 en test es bajo por temporal covariate shift (dark market shutdowns). El val F1 refleja mejor si el modelo aprendio. Ademas, usar test para decisions viola buena practica de ML.'
    },
    {
      q: 'Que es el "warm-start" de Optuna?',
      options: [
        'Precalentar la GPU antes de correr Optuna',
        'Inyectar hiperparametros de la LITERATURA como trial 0, para que Optuna explore desde un buen punto de partida',
        'Correr Optuna dos veces seguidas',
        'Usar los resultados de Machine B como inicio para Machine C'
      ],
      correct: 1,
      explanation: 'Warm-start = usar conocimiento previo (paper arXiv:2602.23599) como punto de partida. Trial 0 de Optuna usa esos HPs, y los 49 restantes exploran alrededor.'
    },
    {
      q: 'Cuantas configuraciones en total tiene el diseño experimental completo?',
      options: [
        '48 (4 x 4 x 3)',
        '12 (4 escenarios x 3 explainers)',
        '144 (4 escenarios x 4 arquitecturas x 3 balanceos x 3 explainers)',
        '576 (144 x 4 metricas)'
      ],
      correct: 2,
      explanation: '4 escenarios x 4 arquitecturas x 3 tecnicas de balance x 3 metodos XAI = 144. Cada una incluye 5 replicas estocasticas para medir estabilidad.'
    }
  ];

  return { html, quiz };
}
