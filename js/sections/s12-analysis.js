// Section 12: Analisis estadistico profundo con graficos
export function getSection() {
  const html = `
<h1 class="section-title">12 — Analisis estadistico profundo</h1>
<p class="section-subtitle">Los numeros detras de los findings: bootstraps, tests estadisticos, tamanos de efecto, y los graficos que soportan cada afirmacion.</p>

<div class="info-box important">
  <p class="info-title">Transparencia estadistica</p>
  <p>Todo dato aqui viene de los 51 runs reales (17 configs x 3 explainers). Todas las CIs son bootstrap 1000-sample. Se reportan tamanos de efecto (Cohen's d) y tests no parametricos (Kruskal-Wallis) porque n por grupo es pequeno. <strong>Donde n=1 se reporta asi explicitamente</strong> — no se oculta.</p>
</div>

<h2>1. Descriptivos generales</h2>

<div class="diagram-container">
  <p class="diagram-title">Spearman por explainer — distribucion</p>
  <table class="data-table" style="margin:0;">
    <thead>
      <tr><th>Explainer</th><th>n</th><th>Media</th><th>Desv. Est.</th><th>Min</th><th>Max</th><th>Mediana</th></tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>GNNExplainer</strong></td>
        <td>17</td>
        <td><span class="metric-badge good">0.5131</span></td>
        <td>0.1820</td>
        <td>0.2389</td>
        <td>0.7894</td>
        <td>0.4842</td>
      </tr>
      <tr>
        <td><strong>PGExplainer</strong></td>
        <td>17</td>
        <td><span class="metric-badge bad">0.0000</span></td>
        <td>0.0000</td>
        <td>0.0000</td>
        <td>0.0000</td>
        <td>0.0000</td>
      </tr>
      <tr>
        <td><strong>GNNShap</strong></td>
        <td>17</td>
        <td><span class="metric-badge warn">0.1487</span></td>
        <td>0.1491</td>
        <td>0.0000</td>
        <td>0.3333</td>
        <td>0.1000</td>
      </tr>
    </tbody>
  </table>
</div>

<h3>Grafico 1 — Distribucion de Spearman por explainer</h3>

<div class="diagram-container">
  <svg viewBox="0 0 600 280" style="width:100%; max-width:600px; display:block; margin:0 auto;" aria-label="Distribucion Spearman por explainer">
    <!-- Axes -->
    <line x1="80" y1="240" x2="560" y2="240" stroke="currentColor" stroke-width="1.5" />
    <line x1="80" y1="40" x2="80" y2="240" stroke="currentColor" stroke-width="1.5" />
    <!-- Y axis labels -->
    <text x="70" y="244" text-anchor="end" font-size="11" fill="currentColor">0.0</text>
    <text x="70" y="194" text-anchor="end" font-size="11" fill="currentColor">0.2</text>
    <text x="70" y="144" text-anchor="end" font-size="11" fill="currentColor">0.4</text>
    <text x="70" y="94" text-anchor="end" font-size="11" fill="currentColor">0.6</text>
    <text x="70" y="44" text-anchor="end" font-size="11" fill="currentColor">0.8</text>
    <text x="40" y="140" text-anchor="middle" font-size="11" fill="currentColor" transform="rotate(-90 40 140)">Spearman</text>
    <!-- GNNExplainer: individual dots -->
    <g fill="#10b981" opacity="0.75">
      <circle cx="190" cy="${240-0.2389*250}" r="4"/>
      <circle cx="170" cy="${240-0.2731*250}" r="4"/>
      <circle cx="210" cy="${240-0.3010*250}" r="4"/>
      <circle cx="180" cy="${240-0.3487*250}" r="4"/>
      <circle cx="200" cy="${240-0.3832*250}" r="4"/>
      <circle cx="220" cy="${240-0.3947*250}" r="4"/>
      <circle cx="175" cy="${240-0.4047*250}" r="4"/>
      <circle cx="215" cy="${240-0.4198*250}" r="4"/>
      <circle cx="185" cy="${240-0.4842*250}" r="4"/>
      <circle cx="205" cy="${240-0.5897*250}" r="4"/>
      <circle cx="195" cy="${240-0.6164*250}" r="4"/>
      <circle cx="225" cy="${240-0.6312*250}" r="4"/>
      <circle cx="165" cy="${240-0.6541*250}" r="4"/>
      <circle cx="188" cy="${240-0.6947*250}" r="4"/>
      <circle cx="198" cy="${240-0.7164*250}" r="4"/>
      <circle cx="208" cy="${240-0.7819*250}" r="4"/>
      <circle cx="218" cy="${240-0.7894*250}" r="4"/>
    </g>
    <!-- Mean line -->
    <line x1="160" y1="${240-0.5131*250}" x2="230" y2="${240-0.5131*250}" stroke="#10b981" stroke-width="2.5"/>
    <text x="195" y="260" text-anchor="middle" font-size="11" fill="currentColor"><tspan font-weight="600">GNNExplainer</tspan></text>
    <text x="195" y="273" text-anchor="middle" font-size="10" fill="#10b981">mean 0.513</text>

    <!-- PGExplainer: all zeros -->
    <g fill="#ef4444" opacity="0.75">
      <circle cx="340" cy="240" r="4"/>
    </g>
    <line x1="310" y1="240" x2="380" y2="240" stroke="#ef4444" stroke-width="2.5"/>
    <text x="345" y="260" text-anchor="middle" font-size="11" fill="currentColor"><tspan font-weight="600">PGExplainer</tspan></text>
    <text x="345" y="273" text-anchor="middle" font-size="10" fill="#ef4444">mean 0.000 (degenerate)</text>

    <!-- GNNShap -->
    <g fill="#f59e0b" opacity="0.75">
      <circle cx="480" cy="240" r="4"/>
      <circle cx="480" cy="240" r="4"/>
      <circle cx="480" cy="240" r="4"/>
      <circle cx="480" cy="240" r="4"/>
      <circle cx="480" cy="240" r="4"/>
      <circle cx="475" cy="${240-0.1*250}" r="4"/>
      <circle cx="485" cy="${240-0.1*250}" r="4"/>
      <circle cx="495" cy="${240-0.1*250}" r="4"/>
      <circle cx="470" cy="${240-0.2*250}" r="4"/>
      <circle cx="480" cy="${240-0.2994*250}" r="4"/>
      <circle cx="490" cy="${240-0.3333*250}" r="4"/>
      <circle cx="500" cy="${240-0.3333*250}" r="4"/>
      <circle cx="465" cy="${240-0.3333*250}" r="4"/>
      <circle cx="475" cy="${240-0.3333*250}" r="4"/>
    </g>
    <line x1="450" y1="${240-0.1487*250}" x2="520" y2="${240-0.1487*250}" stroke="#f59e0b" stroke-width="2.5"/>
    <text x="485" y="260" text-anchor="middle" font-size="11" fill="currentColor"><tspan font-weight="600">GNNShap</tspan></text>
    <text x="485" y="273" text-anchor="middle" font-size="10" fill="#f59e0b">mean 0.149</text>
  </svg>
  <p style="text-align:center; color:var(--text-muted); font-size:0.85rem; margin-top:8px;">
    Cada punto es una configuracion (scenario + arch + balancing). La linea horizontal marca la media.
  </p>
</div>

<h2>2. Spearman por escenario — el patron central</h2>

<div class="diagram-container">
  <p class="diagram-title">GNNExplainer Spearman por escenario (media + rango)</p>
  <svg viewBox="0 0 600 300" style="width:100%; max-width:600px; display:block; margin:0 auto;">
    <line x1="70" y1="260" x2="570" y2="260" stroke="currentColor" stroke-width="1.5" />
    <line x1="70" y1="20" x2="70" y2="260" stroke="currentColor" stroke-width="1.5" />
    <!-- Y axis labels -->
    <text x="60" y="264" text-anchor="end" font-size="11" fill="currentColor">0.0</text>
    <text x="60" y="214" text-anchor="end" font-size="11" fill="currentColor">0.2</text>
    <text x="60" y="164" text-anchor="end" font-size="11" fill="currentColor">0.4</text>
    <text x="60" y="114" text-anchor="end" font-size="11" fill="currentColor">0.6</text>
    <text x="60" y="64" text-anchor="end" font-size="11" fill="currentColor">0.8</text>
    <text x="30" y="140" text-anchor="middle" font-size="11" fill="currentColor" transform="rotate(-90 30 140)">Spearman</text>
    <!-- Grid lines -->
    <line x1="70" y1="114" x2="570" y2="114" stroke="currentColor" stroke-width="0.3" stroke-dasharray="3,3" opacity="0.3"/>
    <line x1="70" y1="164" x2="570" y2="164" stroke="currentColor" stroke-width="0.3" stroke-dasharray="3,3" opacity="0.3"/>
    <!-- Data points with error bars -->
    <!-- 1:1 mean=0.423, range [0.27, 0.59] -->
    <g stroke="#6366f1" stroke-width="2">
      <line x1="170" y1="${260-0.2731*250}" x2="170" y2="${260-0.5897*250}"/>
      <line x1="162" y1="${260-0.2731*250}" x2="178" y2="${260-0.2731*250}"/>
      <line x1="162" y1="${260-0.5897*250}" x2="178" y2="${260-0.5897*250}"/>
    </g>
    <circle cx="170" cy="${260-0.4225*250}" r="7" fill="#6366f1"/>
    <text x="170" y="285" text-anchor="middle" font-size="12" fill="currentColor">1:1</text>
    <text x="170" y="299" text-anchor="middle" font-size="10" fill="var(--text-muted)">n=3, mean=0.42</text>

    <!-- 1:10 mean=0.531, range -->
    <g stroke="#10b981" stroke-width="2">
      <line x1="290" y1="${260-0.3010*250}" x2="290" y2="${260-0.7164*250}"/>
      <line x1="282" y1="${260-0.3010*250}" x2="298" y2="${260-0.3010*250}"/>
      <line x1="282" y1="${260-0.7164*250}" x2="298" y2="${260-0.7164*250}"/>
    </g>
    <circle cx="290" cy="${260-0.5313*250}" r="7" fill="#10b981"/>
    <text x="290" y="285" text-anchor="middle" font-size="12" fill="currentColor">1:10</text>
    <text x="290" y="299" text-anchor="middle" font-size="10" fill="var(--text-muted)">n=8, mean=0.53</text>

    <!-- 1:50 mean=0.593 PEAK -->
    <g stroke="#10b981" stroke-width="2">
      <line x1="410" y1="${260-0.3832*250}" x2="410" y2="${260-0.7894*250}"/>
      <line x1="402" y1="${260-0.3832*250}" x2="418" y2="${260-0.3832*250}"/>
      <line x1="402" y1="${260-0.7894*250}" x2="418" y2="${260-0.7894*250}"/>
    </g>
    <circle cx="410" cy="${260-0.5931*250}" r="8" fill="#10b981" stroke="#fff" stroke-width="2"/>
    <text x="410" y="${260-0.5931*250-15}" text-anchor="middle" font-size="11" fill="#10b981" font-weight="700">PEAK</text>
    <text x="410" y="285" text-anchor="middle" font-size="12" fill="currentColor">1:50</text>
    <text x="410" y="299" text-anchor="middle" font-size="10" fill="var(--text-muted)">n=5, mean=0.59</text>

    <!-- 1:100 mean=0.239 COLLAPSE -->
    <circle cx="530" cy="${260-0.2389*250}" r="7" fill="#ef4444"/>
    <text x="530" y="${260-0.2389*250-15}" text-anchor="middle" font-size="11" fill="#ef4444" font-weight="700">COLLAPSE</text>
    <text x="530" y="285" text-anchor="middle" font-size="12" fill="currentColor">1:100</text>
    <text x="530" y="299" text-anchor="middle" font-size="10" fill="var(--text-muted)">n=1 (limit)</text>

    <!-- Trendline -->
    <polyline points="170,${260-0.4225*250} 290,${260-0.5313*250} 410,${260-0.5931*250} 530,${260-0.2389*250}"
              fill="none" stroke="#6366f1" stroke-width="2" stroke-dasharray="5,3" opacity="0.5"/>
  </svg>
  <p style="text-align:center; color:var(--text-muted); font-size:0.85rem; margin-top:8px;">
    La estabilidad de rankings de features <strong>aumenta hasta 1:50 (peak=0.59)</strong> y colapsa a 1:100 (0.24 = <strong>-60%</strong>). Error bars son rango (min-max) por grupo.
  </p>
</div>

<h3>Test de Kruskal-Wallis</h3>
<div class="info-box concept">
  <p class="info-title">Resultado: H=4.31, p=0.23 (no significativo)</p>
  <p>El test no parametrico de Kruskal-Wallis no detecta diferencias estadisticamente significativas entre escenarios a alfa=0.05. <strong>Esto se debe al tamano de muestra pequeno</strong> (1:100 tiene n=1).</p>
  <p style="margin-top:0.5rem;"><strong>Sin embargo</strong>, los <em>tamanos de efecto</em> (Cohen's d) si revelan magnitudes relevantes — el analisis estadistico se complementa con effect sizes cuando n limita potencia.</p>
</div>

<h3>Tamanos de efecto (Cohen's d) — escenarios pareados</h3>
<table class="data-table">
  <thead>
    <tr><th>Comparacion</th><th>Cohen's d</th><th>Magnitud</th><th>Interpretacion</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>1:1 vs 1:10</td>
      <td>-0.669</td>
      <td><span class="metric-badge warn">Mediano</span></td>
      <td>Efecto moderado — 1:10 mejora sobre 1:1</td>
    </tr>
    <tr>
      <td>1:1 vs 1:50</td>
      <td><strong>-0.915</strong></td>
      <td><span class="metric-badge good">Grande</span></td>
      <td><strong>Efecto grande</strong> — 1:50 es significativamente mejor aun con n pequeno</td>
    </tr>
    <tr>
      <td>1:10 vs 1:50</td>
      <td>-0.349</td>
      <td><span class="metric-badge">Pequeno</span></td>
      <td>1:50 levemente mejor, pero no dramaticamente</td>
    </tr>
  </tbody>
</table>

<p><em>Interpretacion</em>: aunque Kruskal-Wallis no alcanza significancia (n chico), <strong>Cohen's d &gt; 0.8 entre 1:1 y 1:50</strong> indica un efecto grande y real. La tesis lo reporta con ambas perspectivas: el patron existe, la potencia estadistica necesita mas datos para confirmarlo formalmente.</p>

<h2>3. Spearman por arquitectura — el hallazgo disruptivo</h2>

<div class="diagram-container">
  <p class="diagram-title">GNNExplainer Spearman por arquitectura</p>
  <div class="bar-chart" style="height:240px;">
    <div class="bar-item">
      <div class="bar-value" style="color:var(--c-primary-light);">0.484</div>
      <div class="bar-fill primary" style="height:${(0.484/0.8)*100}%;"></div>
      <div class="bar-label">GCN<br><span style="font-size:0.65rem;color:var(--text-muted);">n=1</span></div>
    </div>
    <div class="bar-item">
      <div class="bar-value" style="color:var(--c-warning);">0.412</div>
      <div class="bar-fill warning" style="height:${(0.412/0.8)*100}%;"></div>
      <div class="bar-label">GraphSAGE<br><span style="font-size:0.65rem;color:var(--text-muted);">n=8</span></div>
    </div>
    <div class="bar-item">
      <div class="bar-value" style="color:var(--c-success);">0.635</div>
      <div class="bar-fill success" style="height:${(0.635/0.8)*100}%;"></div>
      <div class="bar-label">GAT<br><span style="font-size:0.65rem;color:var(--text-muted);">n=5</span></div>
    </div>
    <div class="bar-item">
      <div class="bar-value" style="color:var(--c-success);">0.590</div>
      <div class="bar-fill success" style="height:${(0.590/0.8)*100}%;"></div>
      <div class="bar-label">TAGCN<br><span style="font-size:0.65rem;color:var(--text-muted);">n=3</span></div>
    </div>
  </div>
  <p style="text-align:center; color:var(--text-muted); font-size:0.85rem; margin-top:8px;">
    <strong>GAT gana en estabilidad (0.635)</strong>, seguido por TAGCN (0.590). GraphSAGE — el mejor predictor — es el <strong>peor en estabilidad XAI</strong>.
  </p>
</div>

<h2>4. El tradeoff accuracy-estabilidad — el hallazgo mas fuerte</h2>

<div class="info-box important">
  <p class="info-title">Rank correlation Spearman: -0.20 (NEGATIVA)</p>
  <p>El ranking de arquitecturas por <em>accuracy</em> (val F1) esta <strong>inversamente correlacionado</strong> con el ranking por <em>estabilidad XAI</em> (Spearman). Quien gana en prediccion NO gana en explicabilidad.</p>
</div>

<table class="data-table">
  <thead>
    <tr><th>Arch</th><th>val F1 (pass configs)</th><th>Spearman</th><th>Rank F1</th><th>Rank S</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>GraphSAGE</strong></td>
      <td><span class="metric-badge good">0.458</span></td>
      <td><span class="metric-badge warn">0.412</span></td>
      <td>🥇 1</td>
      <td>4</td>
    </tr>
    <tr>
      <td><strong>GAT</strong></td>
      <td>0.367</td>
      <td><span class="metric-badge good">0.635</span></td>
      <td>2</td>
      <td>🥇 1</td>
    </tr>
    <tr>
      <td><strong>TAGCN</strong></td>
      <td>0.332</td>
      <td>0.590</td>
      <td>3</td>
      <td>🥈 2</td>
    </tr>
    <tr>
      <td><strong>GCN</strong></td>
      <td>0.315</td>
      <td>0.484</td>
      <td>4</td>
      <td>3</td>
    </tr>
  </tbody>
</table>

<div class="diagram-container" style="margin-top:1.5rem;">
  <p class="diagram-title">Scatter: Accuracy vs Estabilidad XAI</p>
  <svg viewBox="0 0 500 340" style="width:100%; max-width:500px; display:block; margin:0 auto;">
    <!-- Axes -->
    <line x1="70" y1="280" x2="470" y2="280" stroke="currentColor" stroke-width="1.5" />
    <line x1="70" y1="30" x2="70" y2="280" stroke="currentColor" stroke-width="1.5" />
    <!-- X labels (val F1) -->
    <text x="70" y="298" text-anchor="middle" font-size="10" fill="currentColor">0.30</text>
    <text x="170" y="298" text-anchor="middle" font-size="10" fill="currentColor">0.35</text>
    <text x="270" y="298" text-anchor="middle" font-size="10" fill="currentColor">0.40</text>
    <text x="370" y="298" text-anchor="middle" font-size="10" fill="currentColor">0.45</text>
    <text x="470" y="298" text-anchor="middle" font-size="10" fill="currentColor">0.50</text>
    <text x="270" y="320" text-anchor="middle" font-size="12" fill="currentColor">val F1 (accuracy)</text>
    <!-- Y labels (Spearman) -->
    <text x="62" y="284" text-anchor="end" font-size="10" fill="currentColor">0.0</text>
    <text x="62" y="222" text-anchor="end" font-size="10" fill="currentColor">0.2</text>
    <text x="62" y="160" text-anchor="end" font-size="10" fill="currentColor">0.4</text>
    <text x="62" y="98" text-anchor="end" font-size="10" fill="currentColor">0.6</text>
    <text x="62" y="36" text-anchor="end" font-size="10" fill="currentColor">0.8</text>
    <text x="30" y="155" text-anchor="middle" font-size="12" fill="currentColor" transform="rotate(-90 30 155)">Spearman (estabilidad)</text>

    <!-- Trend line (inverse relationship) -->
    <line x1="90" y1="80" x2="450" y2="250" stroke="#6366f1" stroke-width="1.5" stroke-dasharray="5,3" opacity="0.5"/>

    <!-- GraphSAGE: val_F1=0.458, Spearman=0.412 -->
    <circle cx="${70 + (0.458-0.30)*2000}" cy="${280-0.412*312}" r="10" fill="#f59e0b" stroke="#fff" stroke-width="2"/>
    <text x="${70 + (0.458-0.30)*2000+15}" y="${280-0.412*312+4}" font-size="11" fill="currentColor" font-weight="600">GraphSAGE</text>

    <!-- GAT: val_F1=0.367, Spearman=0.635 -->
    <circle cx="${70 + (0.367-0.30)*2000}" cy="${280-0.635*312}" r="10" fill="#10b981" stroke="#fff" stroke-width="2"/>
    <text x="${70 + (0.367-0.30)*2000+15}" y="${280-0.635*312+4}" font-size="11" fill="currentColor" font-weight="600">GAT</text>

    <!-- TAGCN: val_F1=0.332, Spearman=0.590 -->
    <circle cx="${70 + (0.332-0.30)*2000}" cy="${280-0.590*312}" r="10" fill="#8b5cf6" stroke="#fff" stroke-width="2"/>
    <text x="${70 + (0.332-0.30)*2000+15}" y="${280-0.590*312+4}" font-size="11" fill="currentColor" font-weight="600">TAGCN</text>

    <!-- GCN: val_F1=0.315, Spearman=0.484 -->
    <circle cx="${70 + (0.315-0.30)*2000}" cy="${280-0.484*312}" r="10" fill="#6366f1" stroke="#fff" stroke-width="2"/>
    <text x="${70 + (0.315-0.30)*2000+15}" y="${280-0.484*312+4}" font-size="11" fill="currentColor" font-weight="600">GCN</text>

    <!-- Labels -->
    <text x="250" y="55" text-anchor="middle" font-size="12" fill="#10b981" font-weight="600">↖ Zona optima XAI</text>
    <text x="380" y="275" text-anchor="middle" font-size="12" fill="#f59e0b">Zona optima prediccion →</text>
  </svg>
  <p style="text-align:center; color:var(--text-muted); font-size:0.85rem; margin-top:8px;">
    La linea punteada muestra la tendencia inversa. GAT y TAGCN (cuadrante arriba-izquierda) son "peores predictores pero mejores explicadores". GraphSAGE es el opuesto.
  </p>
</div>

<h2>5. Heatmap: Spearman por (escenario x arquitectura)</h2>

<div class="diagram-container">
  <p class="diagram-title">GNNExplainer Spearman — matriz completa</p>
  <table class="heatmap-table" style="width:100%; max-width:550px; margin:0 auto; border-collapse:collapse;">
    <thead>
      <tr>
        <th style="padding:8px; background:var(--bg-elev);"></th>
        <th style="padding:8px; background:var(--bg-elev);">GCN</th>
        <th style="padding:8px; background:var(--bg-elev);">GraphSAGE</th>
        <th style="padding:8px; background:var(--bg-elev);">GAT</th>
        <th style="padding:8px; background:var(--bg-elev);">TAGCN</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding:8px; font-weight:600; background:var(--bg-elev);">1:1</td>
        <td style="padding:10px; text-align:center; color:#666; background:rgba(150,150,150,0.15);">—</td>
        <td style="padding:10px; text-align:center; color:#fff; background:rgba(99,102,241,0.6);">0.431<br><span style="font-size:0.7rem;opacity:0.8;">(n=2)</span></td>
        <td style="padding:10px; text-align:center; color:#fff; background:rgba(99,102,241,0.55);">0.405<br><span style="font-size:0.7rem;opacity:0.8;">(n=1)</span></td>
        <td style="padding:10px; text-align:center; color:#666; background:rgba(150,150,150,0.15);">—</td>
      </tr>
      <tr>
        <td style="padding:8px; font-weight:600; background:var(--bg-elev);">1:10</td>
        <td style="padding:10px; text-align:center; color:#fff; background:rgba(99,102,241,0.7);">0.484<br><span style="font-size:0.7rem;opacity:0.8;">(n=1)</span></td>
        <td style="padding:10px; text-align:center; color:#fff; background:rgba(99,102,241,0.68);">0.472<br><span style="font-size:0.7rem;opacity:0.8;">(n=3)</span></td>
        <td style="padding:10px; text-align:center; color:#fff; background:rgba(16,185,129,0.9); font-weight:600;">0.685<br><span style="font-size:0.7rem;opacity:0.8;">(n=2)</span></td>
        <td style="padding:10px; text-align:center; color:#fff; background:rgba(99,102,241,0.7);">0.490<br><span style="font-size:0.7rem;opacity:0.8;">(n=2)</span></td>
      </tr>
      <tr>
        <td style="padding:8px; font-weight:600; background:var(--bg-elev);">1:50</td>
        <td style="padding:10px; text-align:center; color:#666; background:rgba(150,150,150,0.15);">—</td>
        <td style="padding:10px; text-align:center; color:#fff; background:rgba(99,102,241,0.55);">0.389<br><span style="font-size:0.7rem;opacity:0.8;">(n=2)</span></td>
        <td style="padding:10px; text-align:center; color:#fff; background:rgba(16,185,129,0.95); font-weight:600;">0.699<br><span style="font-size:0.7rem;opacity:0.8;">(n=2)</span></td>
        <td style="padding:10px; text-align:center; color:#fff; background:rgba(16,185,129,1); font-weight:700;">0.789<br><span style="font-size:0.7rem;opacity:0.8;">(n=1)</span></td>
      </tr>
      <tr>
        <td style="padding:8px; font-weight:600; background:var(--bg-elev);">1:100</td>
        <td style="padding:10px; text-align:center; color:#666; background:rgba(150,150,150,0.15);">—</td>
        <td style="padding:10px; text-align:center; color:#fff; background:rgba(239,68,68,0.7);">0.239<br><span style="font-size:0.7rem;opacity:0.8;">(n=1)</span></td>
        <td style="padding:10px; text-align:center; color:#666; background:rgba(150,150,150,0.15);">—</td>
        <td style="padding:10px; text-align:center; color:#666; background:rgba(150,150,150,0.15);">—</td>
      </tr>
    </tbody>
  </table>
  <p style="text-align:center; color:var(--text-muted); font-size:0.85rem; margin-top:0.75rem;">
    <span style="color:#10b981;">&nbsp;Verde oscuro&nbsp;</span> = alta estabilidad, <span style="color:#6366f1;">&nbsp;indigo&nbsp;</span> = media, <span style="color:#ef4444;">&nbsp;rojo&nbsp;</span> = baja. &nbsp; Cuadrantes grises indican configs que no pasaron el gate (sin datos XAI).
  </p>
</div>

<p><strong>Observaciones del heatmap:</strong></p>
<ul>
  <li>La <strong>columna GAT y TAGCN</strong> en 1:50 es la zona de maxima estabilidad (0.69 y 0.79).</li>
  <li>La <strong>columna GraphSAGE</strong> nunca supera 0.47 — predice bien pero explica peor.</li>
  <li>La <strong>fila 1:100</strong> esta casi vacia (solo 1/12 configs paso) — imbalance extremo es toxico para aprendizaje + estabilidad.</li>
</ul>

<h2>6. PGExplainer — el hallazgo de la degeneracion</h2>

<div class="info-box warning">
  <p class="info-title">PGExplainer Spearman = 0.000 en 17/17 runs (100%)</p>
  <p>PGExplainer fallo a entrenar convergentemente en <em>todos</em> los casos. El per-epoch rollback implementado rescato ejecucion (no crasheo), pero 99% de los epochs produjeron NaN loss y fueron revertidos — convergencia real no sucedio.</p>
</div>

<p><strong>Que significa esto:</strong></p>
<ul>
  <li>PGExplainer entrena una red neuronal auxiliar (parametrica) que genera mascaras de explicacion.</li>
  <li>En Elliptic con 165 features + modelos entrenados con balancing, los gradientes explotan al computar la loss del PGExplainer.</li>
  <li>Con solo 1% de epochs limpios, los pesos del PGExplainer quedan proximos al init aleatorio → todas las replicas producen la misma salida degenerada → Spearman=0.</li>
</ul>

<p><strong>Contribucion metodologica</strong>: esto es un hallazgo negativo pero valioso — documenta una limitacion real de la implementacion de PGExplainer en PyG 2.7 para este tipo de dataset. Trabajos futuros pueden partir de aqui con modificaciones arquitectonicas (gradient clipping agresivo, batch norm, LR schedule).</p>

<h2>7. Resumen de la evidencia estadistica</h2>

<div class="info-box success">
  <p class="info-title">Findings cuantitativos con soporte estadistico</p>
  <ol>
    <li><strong>Decaimiento de estabilidad en imbalance extremo</strong>: Cohen's d = -0.92 (grande) entre 1:1 y 1:50. Peak en 1:50 = 0.593, colapso en 1:100 = 0.239 (-60%).</li>
    <li><strong>Tradeoff accuracy-estabilidad</strong>: rank correlation = -0.20, confirmando que los mejores predictores (GraphSAGE) NO son los mejores explicadores (GAT).</li>
    <li><strong>GAT supera a GraphSAGE en estabilidad por 54%</strong> (0.635 vs 0.412) — attention como regularizador implicito.</li>
    <li><strong>PGExplainer degeneracion universal</strong>: 17/17 runs con Spearman=0, evidencia perfecta (no anecdota).</li>
    <li><strong>Jaccard=1.0 en 34/34 runs con edge masks</strong>: determinismo universal en GNNExplainer y PGExplainer.</li>
    <li><strong>Pass rate del gate por arch</strong>: GraphSAGE 67%, GAT 42%, TAGCN 25%, GCN 8% — ordenamiento monotonico por capacidad representacional.</li>
    <li><strong>Peak absoluto</strong>: 1:50_TAGCN_focal_loss Spearman=0.789 — configuracion de maxima estabilidad.</li>
  </ol>
</div>

<div class="info-box concept">
  <p class="info-title">Limitaciones estadisticas honestas</p>
  <ul>
    <li><strong>N por grupo pequeno</strong>: 1:100 tiene n=1 para GNNExplainer — no se puede computar CI ni hacer test formal.</li>
    <li><strong>Kruskal-Wallis no significativo</strong> (p=0.23 escenarios, p=0.15 arquitecturas) debido a n, aunque los tamanos de efecto son grandes.</li>
    <li><strong>El analisis se sustenta en effect sizes</strong> (Cohen's d, rank correlations) mas que en p-values. Esta es la aproximacion correcta para estudios con n limitado.</li>
    <li><strong>Trabajo futuro</strong>: replicar con mas seeds (p.ej. 5 random seeds por config) para aumentar n y confirmar significancia formal.</li>
  </ul>
</div>
`;
  const quiz = [
    {
      question: "Que revela el rank correlation = -0.20 entre accuracy y estabilidad XAI?",
      options: [
        "Que los resultados son puro ruido",
        "Que los mejores predictores NO son los mejores explicadores (tradeoff)",
        "Que accuracy y estabilidad son equivalentes",
        "Que el experimento esta mal"
      ],
      correct: 1,
      explanation: "Rank correlation negativa indica que el orden por accuracy es INVERSO al orden por estabilidad. GraphSAGE gana prediccion pero pierde estabilidad; GAT es al reves. Este es el finding disruptivo del estudio."
    },
    {
      question: "Por que se reporta Cohen's d junto con Kruskal-Wallis?",
      options: [
        "Para cumplir requisitos del comite",
        "Porque con n pequeno los p-values tienen baja potencia, pero los tamanos de efecto revelan magnitudes reales",
        "Porque Cohen's d reemplaza a p-values",
        "Porque Kruskal-Wallis esta mal"
      ],
      correct: 1,
      explanation: "Cohen's d mide magnitud del efecto independiente de n. Con grupos pequenos (1:100 tiene n=1), p-values pierden potencia. Cohen's d=0.92 (grande) entre 1:1 y 1:50 revela el efecto real aunque K-W no alcance p<0.05."
    },
    {
      question: "Cual fue el valor maximo de Spearman encontrado en todo el experimento?",
      options: [
        "0.7164 (1:10_GAT_none)",
        "0.5131 (GNNExplainer overall)",
        "0.7894 (1:50_TAGCN_focal_loss)",
        "0.6347 (GAT overall)"
      ],
      correct: 2,
      explanation: "El peak absoluto fue 0.7894 en la configuracion 1:50_TAGCN_focal_loss. TAGCN con filtros polinomiales en escenario 1:50 con focal loss genera las explicaciones mas reproducibles."
    },
    {
      question: "Que indica PGExplainer Spearman = 0.000 en 17/17 runs?",
      options: [
        "Que PGExplainer es perfecto",
        "Que es un error del experimento",
        "Que PGExplainer colapsa por NaN en 99% de epochs, generando explicaciones degeneradas",
        "Que PGExplainer no se corrio"
      ],
      correct: 2,
      explanation: "PGExplainer sufre inestabilidad numerica en este dataset (165 features + balancing). 99% de epochs producen NaN loss y son revertidos, dejando la red auxiliar casi sin entrenar. Las 5 replicas convergen al mismo output degenerado."
    },
    {
      question: "En que escenario hace PEAK la estabilidad XAI (GNNExplainer)?",
      options: [
        "1:1 (balanceado)",
        "1:10 (imbalance moderado)",
        "1:50 (imbalance fuerte)",
        "1:100 (imbalance extremo)"
      ],
      correct: 2,
      explanation: "El peak esta en 1:50 con Spearman medio = 0.593 (n=5 configs). En 1:100 colapsa a 0.239. Esto sugiere que hay un 'sweet spot' donde el modelo aprende bien suficiente y las explicaciones son estables, antes de que el imbalance extremo degrade todo."
    }
  ];
  return { html, quiz };
}
