// Section 5: Class Imbalance Problem
export function getSection() {
  const html = `
<h1 class="section-title">05 — El desbalance de clases: el enemigo silencioso</h1>
<p class="section-subtitle">Un modelo con 98% de accuracy puede ser COMPLETAMENTE inutil. Aca vas a entender por que, y como medimos de verdad si un modelo funciona.</p>

<h2>Que es el desbalance de clases?</h2>
<p>Imaginate que tenes 100 transacciones de Bitcoin. De esas 100:</p>
<ul>
  <li><span style="color:var(--c-success); font-weight:700;">98 son licitas</span> (transacciones normales)</li>
  <li><span style="color:var(--c-error); font-weight:700;">2 son ilicitas</span> (lavado de dinero)</li>
</ul>

<div class="diagram-container">
  <p class="diagram-title">Visualizacion de desbalance 1:50 (100 transacciones)</p>
  <div class="imbalance-viz">
    ${Array.from({length: 98}, (_, i) => `<div class="imbalance-ball licit" style="animation-delay:${i*0.02}s"></div>`).join('')}
    ${Array.from({length: 2}, (_, i) => `<div class="imbalance-ball illicit" style="animation-delay:${(98+i)*0.02}s"></div>`).join('')}
  </div>
  <p style="text-align:center; color:var(--text-muted); font-size:0.85rem; margin-top:8px;">
    <span style="color:var(--c-success);">Verde</span> = licita | <span style="color:var(--c-error);">Rojo</span> = ilicita. Encontra los dos puntos rojos.
  </p>
</div>

<p>Ahora imaginate un "modelo" que simplemente dice <strong>"licita"</strong> para TODA transaccion, sin importar nada. Cual seria su accuracy?</p>

<div class="formula-box">
  <span class="formula-label">El modelo "siempre licita"</span>
  Accuracy = 98 / 100 = <strong>98%</strong>
  <span class="formula-annotation">
    Parece impresionante, no? Pero este modelo no detecto NI UNA SOLA transaccion ilicita.<br>
    Es el equivalente a un guardia de seguridad que dice "todo bien" sin siquiera mirar.
  </span>
</div>

<div class="info-box danger">
  <p class="info-title">La trampa de la accuracy</p>
  <p>En problemas desbalanceados, la <strong>accuracy es MENTIROSA</strong>. Un modelo con 98% de accuracy puede tener 0% de deteccion de fraude. Es la metrica mas engañosa en machine learning aplicado a fraude, diagnostico medico, ciberseguridad, y cualquier problema donde la clase rara es la que importa.</p>
</div>

<h2>Metricas que SI sirven</h2>
<p>Necesitamos metricas que no se dejen engañar por el desbalance:</p>

<h3>F1-Score: la media armonica de Precision y Recall</h3>
<p>Primero, definamos Precision y Recall para la clase ilicita:</p>
<ul>
  <li><strong>Precision</strong>: de todos los que etiquete como "ilicita", cuantos REALMENTE lo eran? <code>TP / (TP + FP)</code></li>
  <li><strong>Recall</strong>: de todos los REALMENTE ilicitos, cuantos detecte? <code>TP / (TP + FN)</code></li>
</ul>

<div class="formula-box">
  <span class="formula-label">F1-Score</span>
  F1 = 2 &times; (Precision &times; Recall) / (Precision + Recall)
  <span class="formula-annotation">
    F1 = 0 si precision O recall es 0 (nuestro modelo "siempre licita" tiene F1 = 0).<br>
    F1 = 1 solo si precision Y recall son perfectas.<br>
    Es la metrica estandar para problemas desbalanceados.
  </span>
</div>

<h3>MCC: Matthews Correlation Coefficient</h3>
<p>El MCC es como la correlacion de Pearson entre las predicciones y las etiquetas reales:</p>
<ul>
  <li><strong>MCC = +1</strong>: prediccion perfecta</li>
  <li><strong>MCC = 0</strong>: prediccion aleatoria (no mejor que tirar una moneda)</li>
  <li><strong>MCC = -1</strong>: prediccion perfectamente invertida</li>
</ul>
<p>La ventaja del MCC es que usa las 4 celdas de la matriz de confusion (TP, TN, FP, FN), asi que es dificil de "hackear" con un modelo trivial.</p>

<h3>PR-AUC: Area bajo la curva Precision-Recall</h3>
<p>PR-AUC resume como se comportan precision y recall a distintos umbrales de decision. A diferencia del ROC-AUC (que puede ser engañosamente alto con clases desbalanceadas), PR-AUC se enfoca en la clase POSITIVA (ilicita) y es mas informativa.</p>

<div class="info-box success">
  <p class="info-title">En nuestra tesis</p>
  <p>Usamos <strong>F1</strong> como metrica principal para el quality gate (F1 &ge; 0.70), <strong>MCC</strong> como metrica secundaria (MCC &ge; 0.40), y <strong>PR-AUC</strong> como metrica de optimizacion para Optuna. La accuracy NUNCA se usa para tomar decisiones.</p>
</div>

<h2>Los 4 escenarios experimentales</h2>
<p>Para estudiar como el desbalance afecta la estabilidad de las explicaciones, creamos 4 niveles de imbalance artificial:</p>

<table class="data-table">
  <thead>
    <tr><th>Escenario</th><th>Ratio licit:illicit</th><th>Nodos licitos</th><th>Nodos ilicitos</th><th>Total nodos</th><th>Como se crea</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>1:1</strong></td>
      <td>Balanceado</td>
      <td>~3,462</td>
      <td>~3,462</td>
      <td>~6,924</td>
      <td>Subsamplear licitos al nivel de ilicitos</td>
    </tr>
    <tr>
      <td><strong>1:10</strong></td>
      <td>Moderado</td>
      <td>~34,620</td>
      <td>~3,462</td>
      <td>~38,082</td>
      <td>Subsamplear licitos a 10x ilicitos</td>
    </tr>
    <tr>
      <td><strong>1:50</strong></td>
      <td>Severo</td>
      <td>~42,019</td>
      <td>~840</td>
      <td>~42,859</td>
      <td>Subsamplear ilicitos a 1/50 de licitos</td>
    </tr>
    <tr>
      <td><strong>1:100</strong></td>
      <td>Extremo</td>
      <td>~42,019</td>
      <td>~420</td>
      <td>~42,439</td>
      <td>Subsamplear ilicitos a 1/100 de licitos</td>
    </tr>
  </tbody>
</table>

<div class="diagram-container">
  <p class="diagram-title">Proporcion visual de clases por escenario</p>
  <div class="bar-chart" style="height:180px;">
    <div class="bar-item">
      <div class="bar-value">50%</div>
      <div class="bar-fill success" style="height:50%;"></div>
      <div class="bar-value" style="color:var(--c-error);">50%</div>
      <div class="bar-fill error" style="height:50%;"></div>
      <div class="bar-label">1:1</div>
    </div>
    <div class="bar-item">
      <div class="bar-value">91%</div>
      <div class="bar-fill success" style="height:91%;"></div>
      <div class="bar-value" style="color:var(--c-error);">9%</div>
      <div class="bar-fill error" style="height:9%;"></div>
      <div class="bar-label">1:10</div>
    </div>
    <div class="bar-item">
      <div class="bar-value">98%</div>
      <div class="bar-fill success" style="height:98%;"></div>
      <div class="bar-value" style="color:var(--c-error);">2%</div>
      <div class="bar-fill error" style="height:2%;"></div>
      <div class="bar-label">1:50</div>
    </div>
    <div class="bar-item">
      <div class="bar-value">99%</div>
      <div class="bar-fill success" style="height:99%;"></div>
      <div class="bar-value" style="color:var(--c-error);">1%</div>
      <div class="bar-fill error" style="height:1%;"></div>
      <div class="bar-label">1:100</div>
    </div>
  </div>
</div>

<h2>El trade-off fundamental</h2>

<div class="compare-row">
  <div class="compare-box">
    <h4>Escenario balanceado (1:1)</h4>
    <p><strong>Pro</strong>: el modelo ve igual cantidad de ambas clases. No tiene incentivo a ignorar la clase rara.</p>
    <p><strong>Contra</strong>: tiramos el 92% de los datos licitos! Perdemos informacion valiosa sobre patrones normales. El modelo entrena con MUCHO menos datos.</p>
  </div>
  <div class="compare-box">
    <h4>Escenario extremo (1:100)</h4>
    <p><strong>Pro</strong>: usamos todos los datos disponibles. El modelo ve la distribucion "real" del mundo.</p>
    <p><strong>Contra</strong>: el modelo puede aprender a decir "licita" siempre y obtener 99% accuracy. La señal de la clase ilicita se pierde en el ruido.</p>
  </div>
</div>

<div class="info-box concept">
  <p class="info-title">Resumen de la seccion</p>
  <p>El desbalance de clases hace que modelos con alta accuracy sean inutiles. Usamos F1, MCC y PR-AUC en vez de accuracy. Creamos 4 escenarios (1:1 a 1:100) para estudiar como el nivel de desbalance afecta tanto la calidad del modelo como la estabilidad de las explicaciones. El "sweet spot" resulto ser 1:10 — suficiente señal de ambas clases sin desperdiciar datos.</p>
</div>
`;

  const quiz = [
    {
      q: 'Por que la accuracy es una metrica MALA para detectar fraude?',
      options: [
        'Porque es dificil de calcular',
        'Porque no funciona con redes neuronales',
        'Porque un modelo que siempre dice "no fraude" puede tener accuracy altisima sin detectar nada',
        'Porque la accuracy solo sirve para imagenes'
      ],
      correct: 2,
      explanation: 'Con 2% de fraude, decir "no fraude" siempre da 98% accuracy. La accuracy premia ignorar la clase rara, que es justamente la que nos importa.'
    },
    {
      q: 'Que mide el F1-Score?',
      options: [
        'La velocidad del modelo',
        'La media armonica entre Precision y Recall — balancea ambas',
        'El porcentaje de datos correctamente clasificados',
        'La cantidad de parametros del modelo'
      ],
      correct: 1,
      explanation: 'F1 = 2 * (Precision * Recall) / (Precision + Recall). Es 0 si cualquiera de las dos es 0, forzando al modelo a ser bueno en AMBAS.'
    },
    {
      q: 'En el escenario 1:100, que le pasa al modelo si no se toman medidas?',
      options: [
        'Aprende mas rapido porque tiene mas datos',
        'Aprende a decir "licita" casi siempre porque la penalizacion por ignorar la clase rara es minima',
        'No puede entrenar porque los datos no caben en memoria',
        'Overfittea a la clase ilicita'
      ],
      correct: 1,
      explanation: 'Con 99% licitas, el gradiente de la loss empuja al modelo a predecir "licita" siempre. La señal de los pocos ejemplos ilicitos se pierde.'
    },
    {
      q: 'Por que crear el escenario 1:1 (balanceado) tiene un costo?',
      options: [
        'Porque se necesita mas GPU',
        'Porque hay que eliminar el 92% de los datos licitos, perdiendo informacion',
        'Porque las metricas no funcionan con datos balanceados',
        'Porque 1:1 es imposible de crear con datos reales'
      ],
      correct: 1,
      explanation: 'Para igualar las clases hay que tirar la gran mayoria de datos licitos. Menos datos = menos patrones normales aprendidos = potencial underfitting.'
    }
  ];

  return { html, quiz };
}
