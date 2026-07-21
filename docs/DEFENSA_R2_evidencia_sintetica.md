# Guion de defensa — R2: "La evidencia fuerte viene del dataset que ustedes construyeron"

> Material de estudio interno para la sustentación (Alejandro Gómez · Juan Diego Garzón).
> Objetivo: responder con firmeza, estructura y honestidad la objeción más difícil de la defensa.
> **No es para publicar.** Ensayar en voz alta hasta que salga natural.

---

## 0. La objeción, en palabras del jurado (variantes a anticipar)

- *"Su hallazgo con más fuerza estadística (p ≈ 2,6×10⁻³⁵) sale de un grafo que ustedes mismos construyeron. ¿No es circular?"*
- *"¿No midieron lo que fabricaron para poder medirlo?"*
- *"¿Cómo sabemos que no diseñaron los datos para obtener la respuesta que buscaban?"*
- *"Si el resultado importante depende del dataset sintético, ¿qué aporta realmente el estudio sobre datos reales de lavado?"*
- *"El grafo sintético es demasiado simple / no se parece a Bitcoin real."*

Todas apuntan a lo mismo: **validez y no-circularidad de la evidencia sintética.**

---

## 1. La respuesta en una frase (memorizar y usar para abrir)

> **"El eje sintético no es un dataset a conveniencia: es el único régimen donde la plausibilidad y
> la fidelidad SON medibles —porque exigen un ground-truth que Elliptic no tiene—, es la práctica
> estándar en la evaluación de XAI sobre grafos, y la prueba de que no lo construimos para
> confirmar nuestra tesis es que refutó nuestra propia hipótesis central."**

Dicho esto con calma, ya se ganó el 70% del intercambio. Los pilares 1-5 lo respaldan.

---

## 2. Los cinco pilares (qué decir + respaldo)

### Pilar 1 — Necesidad metodológica: plausibilidad/fidelidad NO son medibles en Elliptic

**Qué decir:** *"Elliptic solo etiqueta cada transacción como lícita o ilícita — no dice CUÁL subgrafo
constituye el patrón de lavado. Para medir plausibilidad necesitamos saber qué aristas SON el fraude,
y ese ground-truth no existe en Elliptic. Además, su campo receptivo es degenerado: mediana de ~2
nodos y 1-2 aristas, con lo que la plausibilidad de subgrafo y el Jaccard son triviales. No es que
prefiriéramos el sintético; es que en Elliptic esas preguntas son literalmente incomputables."*

**Respaldo:**
- Weber (2019) etiqueta nodos ilícito/lícito, sin ground-truth de tipología ni de aristas.
- Campo receptivo de Elliptic: mediana ≈ 2 nodos (documentado en Cap. 4 y en `stab_subgraph_n_nodes`).
- Por eso en Elliptic **solo** se puede medir estabilidad (Spearman de features); plausibilidad y
  fidelidad requieren el eje sintético.

### Pilar 2 — Es la práctica estándar en XAI sobre grafos, no un atajo

**Qué decir:** *"Usar grafos sintéticos con motivos plantados es exactamente cómo se evalúa la
explicabilidad en grafos. El propio paper de GNNExplainer se validó sobre BA-Shapes y Tree-Cycles;
GraphFramEx, GNNX-BENCH y Agarwal et al. — los benchmarks que citamos en el marco teórico — construyen
grafos con ground-truth conocido por la misma razón: los datos reales no traen la explicación
verdadera. Estamos siguiendo la metodología establecida, no inventando un dataset conveniente."*

**Respaldo:**
- La evaluación de fidelidad/plausibilidad de explicadores requiere ground-truth ⇒ el campo usa
  datasets sintéticos con motivos (BA-Shapes, Tree-Grid, etc.).
- Estos benchmarks ya están en el marco teórico de la tesis (GraphFramEx, GNNX-BENCH, Agarwal).

### Pilar 3 — Anti-conveniencia: el sintético REFUTÓ nuestra hipótesis (el pilar más fuerte)

**Qué decir:** *"Si hubiéramos construido el grafo para confirmar la tesis, habríamos fracasado.
Nuestra hipótesis central era que una explicación más estable sería más plausible — un 'puente'
estabilidad→plausibilidad. El grafo sintético lo REFUTÓ: la correlación salió nula, r ≈ −0,01, con
intervalo de confianza que incluye el cero. Nadie diseña un dataset para matar su propia hipótesis.
La disociación entre plausibilidad y fidelidad tampoco era el resultado que buscábamos. Son hallazgos
incómodos, no confirmatorios."*

**Respaldo:**
- Puente estabilidad→plausibilidad: GNNExplainer r = −0,014, IC bootstrap incluye 0 → **nulo**.
- Disociación: PGExplainer plaus. aristas 0,80 (la mejor) pero fidelidad 0,11 (la peor); GNNExplainer
  fidelidad 0,56 pero plausibilidad 0,50. El "mejor" explicador depende de qué dimensión mires.
- Ambos van CONTRA la narrativa original del anteproyecto (que esperaba a TAGCN dominando y un puente
  positivo).

> **Este es el argumento que cierra el debate.** La circularidad requiere que el resultado favorezca
> a quien construyó el dato; aquí el dato mordió la mano que lo construyó.

### Pilar 4 — La evidencia sintética es robusta (no es un grafo con suerte)

**Qué decir:** *"El hallazgo no depende de una instancia. Lo replicamos sobre tres grafos generados
con semillas distintas y tres semillas de entrenamiento por celda, y se mantiene, con dispersión
entre semillas de σ ≈ 0,007. Las tipologías que plantamos son las canónicas del lavado —structuring,
layering, fan-in y fan-out—, no patrones exóticos elegidos para favorecer a un explicador."*

**Respaldo:**
- 3 grafos (g0, g1, g2) × 3 semillas de modelo {42,43,44} → 540 filas (`results_robust.csv`).
- Wilcoxon PGExplainer > GNNExplainer en plausibilidad de aristas: p ≈ 2,6×10⁻³⁵, positivo en 91,5%
  de los pares. σ entre semillas ≈ 0,007.
- Grafo: ≈9.500 nodos, ≈1.530 ilícitos, ≈31.000 aristas, 4 tipologías estándar.

### Pilar 5 — La construcción es neutral al explicador

**Qué decir:** *"El ground-truth lo fija el generador, que es completamente ciego a qué explicador se
va a evaluar después. Que PGExplainer gane en plausibilidad no está cableado en ninguna parte. Es
más: introdujimos aristas distractoras a propósito para que la métrica discrimine —sin ellas
cualquier selección top-k acertaría— y atenuamos la señal de las features para que el problema no
fuera trivialmente separable. Son decisiones que ENDURECEN la prueba, no que la inflen."*

**Respaldo:**
- El generador (`phase1/synthetic_aml_generator.py`) define `typology_edge`/`typology_node` sin
  referencia a ningún explicador.
- Aristas distractoras (`n_distractors=3`) para que la plausibilidad discrimine (comentado en el
  código como requisito para que no todo top-k acierte).
- Firma de features **atenuada** de +4 a +1,5 para evitar separabilidad trivial (re-corrida B).

---

## 3. Preguntas de seguimiento ("si insisten…")

**"El grafo es demasiado simple / no parece Bitcoin real."**
→ *"Es un compromiso deliberado entre control y realismo. Simetrizamos las aristas para lograr campos
receptivos realistas (mediana ~29 nodos, no los ~2 de Elliptic dirigido), agregamos ruido de fondo y
distractores, y atenuamos la firma. No buscábamos replicar Bitcoin, sino aislar la pregunta
mecanística con ground-truth. La validez externa la aporta el eje Elliptic; la interna, el sintético.
Son complementarios."*

**"Entonces Elliptic no sirvió de nada."**
→ *"Al contrario. Elliptic aporta la validez externa de la estabilidad y, además, un hallazgo negativo
importante: en datos reales dispersos, la explicación de subgrafo es degenerada (campo receptivo ~2) y
PGExplainer colapsa (mode collapse en ~90% de las épocas). Eso es evidencia directa sobre la
(in)viabilidad de estas técnicas en producción — un resultado, no un vacío."*

**"¿Por qué no usaron AMLSim, el estándar de la industria?"**
→ *"Lo intentamos. AMLSim depende de MASON v20, que no existe como binario distribuible (Maven Central
solo llega a v14-18 y el release v20 no trae assets), y sus muestras pre-generadas solo traen
`isFraud` por nodo, sin ground-truth por arista ni por tipología —justo lo que necesitábamos. Lo
documentamos y lo dejamos como trabajo futuro. El generador propio nos dio el ground-truth por arista
que ningún dataset disponible ofrecía."*

**"¿No es circular medir plausibilidad contra un ground-truth que ustedes definieron?"**
→ *"El ground-truth define qué ES el patrón de lavado, no qué explicador debe ganar. La circularidad
existiría si hubiéramos ajustado el ground-truth mirando las explicaciones; hicimos lo contrario: el
grafo se fija primero, los explicadores se evalúan después, ciegos entre sí. Y si el diseño tuviera
sesgo confirmatorio, no habría refutado nuestra hipótesis del puente."* (enlazar con Pilar 3)

---

## 4. La concesión honesta (qué admitir — desarma el ataque)

Decir esto **proactivamente**, antes de que lo digan ellos:

> *"Somos claros con el alcance: la validez externa de los hallazgos de plausibilidad y fidelidad se
> limita al régimen controlado. No afirmamos que en producción AML estas tres dimensiones se disocien
> universalmente; afirmamos que, en un entorno con ground-truth y bajo replicación, se disocian de
> forma robusta. Generalizarlo a grafos reales a escala —Elliptic2, o AMLSim cuando se desbloquee— es
> trabajo futuro explícito en el Capítulo 8."*

Conceder el límite correcto **fortalece** la credibilidad y cierra la puerta al ataque de sobre-afirmación.

---

## 5. Qué NO decir (trampas a evitar)

- ❌ **No ponerse a la defensiva ni pedir disculpas por el sintético.** Es una decisión metodológica
  sólida; preséntenla con seguridad, no como un mal menor.
- ❌ **No afirmar que el sintético prueba comportamiento en producción.** Rompe la concesión del §4 y
  los expone.
- ❌ **No menospreciar Elliptic** ("no servía"). Elliptic aporta validez externa y el hallazgo negativo.
- ❌ **No entrar en detalles de implementación del generador** salvo que lo pidan; quédense en el
  argumento metodológico (necesidad de ground-truth) y usen el Pilar 3 como ancla.
- ❌ **No improvisar números.** Los del §2 están verificados; si dudan de una cifra, digan "está en la
  tabla X del anexo" en lugar de arriesgar un valor.

---

## 6. Reparto a dos voces (sugerencia)

- **Quien lidere el eje sintético / experimentación** abre con el *one-liner* (§1) y los Pilares 1-2
  (necesidad metodológica + práctica estándar).
- **Quien lidere el análisis / redacción** remata con el Pilar 3 (anti-conveniencia — el puente nulo)
  y hace la concesión honesta del §4.
- Si el jurado insiste, quien tenga más fresco el código del generador responde el Pilar 5 y la
  pregunta de circularidad; el otro apoya con los números de robustez (Pilar 4).
- **Regla de oro a dos voces:** una sola persona responde cada pregunta hasta el final; el otro
  complementa solo si aporta algo nuevo. No pisarse ni contradecirse en cifras.

---

## Resumen de bolsillo (para la última repasada antes de entrar)

1. Plausibilidad/fidelidad **exigen ground-truth** → Elliptic no lo tiene → sintético es la única vía.
2. Es la **práctica estándar** en XAI de grafos (GNNExplainer, GraphFramEx, GNNX-BENCH).
3. **Refutó nuestra propia hipótesis** (puente nulo r≈−0,01) → imposible que sea confirmatorio.
4. **Robusto:** 3 grafos × 3 semillas, p≈2,6×10⁻³⁵, σ≈0,007.
5. **Neutral:** ground-truth ciego al explicador; distractores y firma atenuada endurecen la prueba.
6. **Concesión:** validez externa limitada al régimen controlado → generalización = trabajo futuro.
