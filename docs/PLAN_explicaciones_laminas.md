# PLAN — Explicaciones lámina por lámina de la defensa (brief para Sonnet)

> **Para quién es esto:** es un encargo para que un modelo (Sonnet) produzca, de forma
> autónoma, un material de estudio que explique **cada lámina** de la presentación de defensa,
> con sus conceptos y su discurso, en profundidad y **sin tecnicismos**. Este documento NO es el
> material final: es la especificación de cómo elaborarlo. Léelo entero antes de empezar.

---

## 0. Objetivo del encargo

Para **cada lámina** del deck de defensa, escribir una explicación de estudio que:

1. Diga el **objetivo principal** de la lámina (qué busca lograr, por qué está ahí).
2. Explique **item por item** lo que dice la lámina (cada viñeta, tabla o figura).
3. Cada vez que aparezca un **concepto o técnica** (p. ej. Spearman, Kruskal-Wallis, η²,
   paso de mensajes, PGExplainer, PR-AUC…), lo explique **con profundidad y sin tecnicismos**,
   como si se lo contaras a alguien inteligente pero sin formación en el tema, usando analogías
   concretas.
4. Incluya el **discurso hablado** de esa lámina (tomado del guion) y una **versión ampliada y
   explicada** de ese discurso.
5. Anticipe **preguntas de jurado** probables sobre la lámina, con su respuesta corta.

El tono y la profundidad deben imitar el trabajo de tutoría que ya se venía haciendo con el autor
(ver §4 y el ejemplo resuelto en §9).

---

## 1. Qué producir (entregable y estructura de archivos)

Crear la carpeta **`docs/explicaciones/`** con:

- **`README.md`** — índice de todas las láminas (enlaces), + el **glosario** (§6) + el **banco de
  analogías** (§7) + un recordatorio de los **guardarraíles** (§5). Es la puerta de entrada.
- **Un archivo por lámina de contenido**, nombrado `lamina-NN-<slug>.md`, donde `NN` es el número
  de **fracción** que muestra la lámina (p. ej. `lamina-26-desbalance.md`). Ver el mapa en §8.
- Al final, los **respaldos** en `respaldo-<slug>.md` (5 archivos, ver §8).

Cada archivo de lámina debe ser **autocontenido**: alguien tiene que poder estudiarlo solo, sin
haber leído los demás. Si un concepto ya se explicó en otra lámina, se puede resumir y **enlazar**
al glosario (`../explicaciones/README.md#<ancla>`), pero siempre dando al menos la noción básica en
el propio archivo.

---

## 2. Fuentes de verdad (leer en este orden, ANTES de escribir nada)

1. **`CLAUDE.md`** (raíz) — contiene las **reglas duras** y la **narrativa retractada**. Es la
   fuente de exactitud más importante. Todo lo de §5 sale de aquí; si hay conflicto entre este plan
   y `CLAUDE.md`, manda `CLAUDE.md`.
2. **`presentacion_latex/beamer_defensa_v3.tex`** — el deck. Es la fuente de **qué dice cada
   lámina** (viñetas, tablas, figuras, pies). Trabaja siempre sobre el `.tex`, no sobre capturas.
3. **`docs/DISCURSO_defensa_dos_voces.md`** — el guion hablado, ya sincronizado con el deck. De
   aquí sale el **discurso** de cada lámina y el mapa lámina→página.
4. **`tesis_latex/`** (manuscrito, `main.pdf` y capítulos) — la **fuente de verdad de los números
   y hallazgos**. Úsalo para verificar cualquier cifra antes de escribirla. Nunca inventes un
   número: si no lo encuentras, dilo.
5. Material de apoyo: `docs/GUION_defensa_por_capitulo.md` (mapa lámina→capítulo + preguntas de
   jurado) y `docs/DEFENSA_R2_evidencia_sintetica.md` (defensa de la validez del eje sintético).

**Regla de oro:** el guion y el deck deben estar **de la mano**. Antes de explicar una lámina,
compara su discurso (en el DISCURSO) con sus viñetas (en el `.tex`) y confirma que hablan de lo
mismo. Si detectas un desajuste, anótalo en el archivo bajo un encabezado `> ⚠️ Posible desajuste`
en lugar de inventar una reconciliación.

---

## 3. Plantilla obligatoria por lámina

Cada `lamina-NN-<slug>.md` debe seguir **exactamente** esta estructura:

```markdown
# Lámina NN — <Título exacto de la lámina>

> Bloque: <Problema y motivación | Marco conceptual | Metodología | Resultados | Conclusiones>
> Voz: <Alejandro | Juan Diego> · Página PDF <N> · Tiempo objetivo <segundos>

## 🎯 Objetivo de la lámina
<1–3 frases: qué busca lograr esta lámina y por qué está aquí, en el hilo de la defensa.>

## 📋 Qué dice, item por item
<Recorre CADA viñeta / celda / elemento de la lámina y explícalo en lenguaje llano.
Un subtítulo o negrita por item. No dejes ningún item sin explicar.>

## 🔑 Conceptos y técnicas que aparecen
<Por cada concepto/técnica mencionado en la lámina: qué es, sin tecnicismos, con una analogía.
Si el concepto ya está en el glosario, da la noción breve y enlaza. Si es nuevo, explícalo a fondo
aquí y proponlo para el glosario.>

## 📈 Cómo leer la figura / tabla  (solo si la lámina tiene una)
<Explica ejes, qué mirar primero, qué significa la forma/valores, y la línea de azar si aplica.>

## 🎤 El discurso (como se dice en voz alta)
<Pega el discurso de esta lámina tal cual está en DISCURSO_defensa_dos_voces.md.>

### Versión ampliada y explicada
<Reescribe el discurso "desplegado": cada frase del guion seguida de la explicación de por qué se
dice y qué concepto encierra. Esta es la parte de estudio profundo.>

## 🧑‍⚖️ Preguntas de jurado probables
<2–4 preguntas que un jurado estricto podría hacer sobre ESTA lámina, cada una con una respuesta
corta y sólida (1–3 frases). Prioriza las que ataquen un número o una decisión de la lámina.>

## 🧠 En una frase
<El mensaje de la lámina condensado en una sola oración memorizable.>
```

No añadas secciones nuevas ni quites ninguna. Si una sección no aplica (p. ej. no hay figura),
escribe "No aplica" y sigue.

---

## 4. Estilo de redacción (cómo venimos trabajando)

- **Sin tecnicismos.** Explica como a una persona lista de 15 años: cero jerga sin traducir. Si es
  inevitable nombrar un término técnico, defínelo en la misma frase con palabras simples.
- **Analogías concretas y cotidianas.** Reusa las del banco (§7) para dar coherencia; crea nuevas
  solo si hacen falta y son sencillas.
- **Empieza por el objetivo**, no por el detalle. Primero "para qué sirve esto", luego el cómo.
- **Números con coma decimal** (formato español: 0,80) y siempre acompañados de su significado
  ("0,80, es decir, el doble del azar").
- **Español de Colombia, cálido y directo.** Frases cortas. Evita el relleno.
- **Honestidad como valor.** Cuando la lámina reporta un no-resultado o una hipótesis refutada,
  subráyalo como fortaleza (compromiso previo + reporte honesto), no como debilidad.
- **Longitud:** cada archivo de lámina, entre ~400 y ~900 palabras. Profundo pero no interminable.
- Prohibido inventar cifras, citas o secciones del manuscrito. Ante la duda, verifica en
  `tesis_latex/` o marca `> ⚠️ Verificar`.

---

## 5. Guardarraíles de exactitud (NO reintroducir errores)

Estos hallazgos fueron **corregidos**; escribir la versión vieja sería un error grave. (Fuente:
`CLAUDE.md` → "Narrativa retractada" y "Reglas duras".)

1. **Es una partición en DOS GRUPOS, no un ranking de cuatro.** Grupo alto: GAT y GCN. Grupo bajo:
   GraphSAGE y TAGCN. Diferencias significativas **entre** grupos, no **dentro**. **Nunca** escribir
   "GAT es la más estable": GAT y GCN se permutan entre semillas.
2. **Desbalance (Resultado 4):** NO decir "perfil plano de tres centésimas". Lo correcto: el rango
   va de 0,696 (1:1) a 0,765 (1:50), **siete centésimas**; Kruskal-Wallis sobre el escenario da
   **p = 0,18 (no significativo)**, η² = 0,05; el mínimo está en 1:1 (el más equilibrado), o sea que
   lo poco que varía va **en contra** de H1. Cifras a 3 semillas.
3. **Tamaños de efecto = η² de ANOVA**, no ε² de Kruskal-Wallis (aunque el código llame `eps2` a la
   variable). El estadístico H sí es de Kruskal-Wallis.
4. **GNNShap:** su plausibilidad es de **features**, con línea base de azar **0,075** (no 0,40).
   No produce máscara de aristas por diseño. **Nunca** ponerla en la misma figura/columna que la
   plausibilidad de **aristas** de GNNExplainer/PGExplainer.
5. **PGExplainer en Elliptic degenera** (Spearman nulo) por la **dispersión del grafo**, no por ser
   mal método: en el grafo **denso** sintético es **el mejor** en plausibilidad de aristas (0,80).
   Ese contraste es un aporte, no un fallo.
6. **Retractado, no reintroducir:** "inversión por densidad" y "GraphSAGE lidera en estabilidad".
   Ambos eran artefactos de medición. La concordancia entre regímenes es **+0,80**.
7. **La estabilidad se mide sobre los verdaderos positivos de VALIDACIÓN**, no de test: el modelo
   colapsa en test por el *shift* temporal, y explicar predicciones erradas no informa. Está
   declarado abiertamente (§4.4 del manuscrito).
8. **Métricas primarias: PR-AUC y precisión@k**, no F1 ni ROC-AUC. El ROC-AUC se ve engañosamente
   alto bajo desbalance (0,88 validación vs PR-AUC 0,37).
9. **Dos artefactos de medición** (contribución central): (a) OOM silencioso de GAT sobre el grafo
   completo → se corrige calculando sobre el **subgrafo receptivo**; (b) **truncamiento de Spearman**
   (fix R1): dimensionaba los rangos por `top_k` en vez de por el nº real de features. Cada uno,
   por sí solo, bastaba para invertir la conclusión.
10. **Reproducibilidad:** el pipeline **no** es determinista a nivel de pesos (sumas atómicas en
    GPU); reproduce **conclusiones**, no decimales.

Si alguna cifra del deck/guion contradice a `CLAUDE.md` o al manuscrito, **no la "corrijas" en
silencio**: márcala con `> ⚠️ Revisar con los autores` y sigue.

---

## 6. Glosario de conceptos recurrentes (explicar una vez en el README, enlazar desde las láminas)

Definir en `docs/explicaciones/README.md`, cada uno sin tecnicismos + su analogía. Lista mínima:

- **Grafo / red de transacciones**, nodo, arista, vecindario, campo receptivo.
- **GNN y paso de mensajes.** Las 4 arquitecturas (GCN, GraphSAGE, GAT, TAGCN).
- **Explicador post-hoc.** Máscara de aristas y ranking de features. GNNExplainer, PGExplainer,
  GNNShap (y en qué se diferencian).
- **Las tres propiedades:** estabilidad, plausibilidad, fidelidad (y que son independientes).
- **Desbalance de clases** y **escenarios** (1:1 … 1:100, nativo). **Balanceo** (la técnica) vs
  **desbalance** (el problema) — no confundirlos.
- **Spearman** (correlación de rangos). **Kruskal-Wallis**. **Mann-Whitney**. **Wilcoxon pareado**.
  **Bootstrap / intervalo de confianza**. **Corrección de Holm**. **p-valor**. **Tamaño de efecto
  (η²)**. **No paramétrico**.
- **PR-AUC, ROC-AUC, precisión@k** (y por qué el ROC-AUC engaña bajo desbalance).
- **Ground-truth**, **validez interna vs externa**, **eje real vs eje sintético**.
- **Shift temporal / colapso validación→test.**

---

## 7. Banco de analogías ya establecidas (reusar para dar coherencia)

- **Reloj parado** → estabilidad ≠ acierto (una explicación puede ser muy estable y siempre
  equivocada). Úsalo en "puente nulo" y en "tres propiedades".
- **Perilla de volumen que no está conectada** → el desbalance no gobierna la estabilidad.
- **Guardia de un pueblo que dice "aquí no hay ladrones"** (acierta el 99,9% pero atrapa cero) →
  por qué el ROC-AUC engaña bajo desbalance.
- **Corredor de 100 m vs nadador** → GNNShap "no corre la misma carrera" (mide features, no
  aristas): no se le puede comparar en el eje de aristas.
- **Filtro de spam entrenado en 2015 y probado en 2024** → shift temporal / colapso val→test.
- **Un médico al que evalúas en los casos que acertó, no donde está perdido** → por qué la
  estabilidad se mide sobre los verdaderos positivos de validación.
- **Notas de un examen y "¿cuánto explica el profesor?"** → tamaño de efecto η² (qué tajada de la
  variación explica un factor). η² pequeño = el factor casi no mueve la aguja.
- **Detective en escena vacía** → PGExplainer en Elliptic: sin aristas que señalar, "dice que todo
  es igual de importante" (colapso de modo) → Spearman nulo.

---

## 8. Mapa de láminas + qué conceptos toca cada una

Deck vigente: `beamer_defensa_v3.pdf`, 47 páginas; **33 láminas de contenido numeradas** (fracción
`N/40`). Formato de fila: **Fracc. (pág. PDF) — título — voz — conceptos a explicar a fondo**.

| Fracc. | PDF | Lámina | Voz | Conceptos clave |
|---|---|---|---|---|
| 1 | 2 | Contenido | A | (índice, ligero) |
| 2 | 4 | El lavado es un problema de red | A | fases del lavado; tipologías (structuring, layering, fan-in/out); por qué grafo |
| 3 | 5 | El problema: detectar sin caja negra | A | %PIB; falsos positivos; reglas vs GNN; caja negra |
| 4 | 6 | Por qué la caja negra es inaceptable | A | entorno regulado; reproducibilidad; explicabilidad como requisito; estabilidad |
| 5 | 7 | Pregunta e objetivos | A | la pregunta; O1–O4 |
| 6 | 8 | Tres hipótesis falsables | A | H1, H2, H3; qué es falsabilidad |
| 7 | 9 | La brecha en el estado del arte | A | estado del arte; ausencia de ground-truth en Elliptic |
| 8 | 11 | Cómo funciona una GNN: paso de mensajes | A | paso de mensajes; capas; campo receptivo (L saltos) |
| 9 | 12 | Cuatro arquitecturas GNN | A | GCN, GraphSAGE, GAT, TAGCN |
| 10 | 13 | Tres explicadores post-hoc | A | post-hoc; máscara+ranking; GNNExplainer, PGExplainer, GNNShap |
| 11 | 14 | Tres propiedades que no son lo mismo | A | estabilidad, plausibilidad, fidelidad; independencia (reloj parado) |
| 12 | 16 | Qué produce un explicador | A | máscara de aristas; ranking de features; Spearman; vecindario ~2 nodos |
| 13 | 17 | Metodología: dos ejes | A | validez interna/externa; Elliptic vs sintético |
| 14 | 18 | Diseño factorial | A | matriz 4×3×3×5; réplicas; KW/Wilcoxon/bootstrap |
| 15 | 19 | Eje 1: Elliptic | A | tamaño; 2,2% ilícito; partición temporal causal; dispersión (~2 nodos) |
| 16 | 20 | Eje 2: grafo sintético | A | por qué se construye; tipologías; decisiones de diseño |
| 17 | 21 | Cómo se construyó el sintético | A | simetrizar; aristas distractoras; atenuar firma; ground-truth ciego al explicador |
| 18 | 22 | Métricas y protocolo estadístico | A | Spearman; plausibilidad; fidelidad; PR-AUC; KW/Wilcoxon/bootstrap |
| 19 | 24 | Contribución: dos artefactos | JD | OOM/subgrafo receptivo; truncamiento de Spearman |
| 20 | 25 | El segundo artefacto: truncamiento | JD | top_k; asimetría del sesgo; el "liderazgo" de GraphSAGE como artefacto |
| 21 | 26 | Resultado 1: dos grupos, no cuatro | JD | partición en dos grupos; KW global; Mann-Whitney |
| 22 | 27 | Robustez: qué sostiene la partición | JD | 180 modelos; KW dentro/entre; Holm; bootstrap; reproducibilidad pesos vs conclusiones |
| 23 | 28 | Resultado 1b: concordancia | JD | "inversión por densidad" retractada; concordancia +0,80 |
| 24 | 29 | Resultado 2: disociación | JD | plausibilidad vs fidelidad; PG vs GNNExplainer; azar 0,40; Wilcoxon; GNNShap (features) |
| 25 | 30 | Resultado 3: el puente que no existe | JD | correlación r=−0,01; IC incluye 0; no-resultado (reloj parado) |
| 26 | 31 | Resultado 4: el desbalance no gobierna | JD | 7 centésimas; KW p=0,18; η²=0,05; mínimo en 1:1; balanceo η²≤0,01 |
| 27 | 32 | Rendimiento y colapso validación→test | JD | PR-AUC, ROC-AUC, precisión@50; shift temporal |
| 28 | 33 | Rigor métrico: PR-AUC y no ROC-AUC | JD | por qué el ROC-AUC engaña bajo desbalance |
| 29 | 34 | Las tres hipótesis y su veredicto | JD | H1/H2/H3 refutadas; valor de reportarlo |
| 30 | 36 | Conclusiones del proyecto | JD | los 6 mensajes |
| 31 | 37 | Los cuatro objetivos, respondidos | JD | O1–O4 |
| 32 | 38 | Matriz de recomendación | JD | recomendación condicional al propósito |
| 33 | 39 | Contribuciones y limitaciones | JD | aportes; límites honestos |
| — | 40 | Gracias / Preguntas (cierre) | JD | respuesta a la pregunta de investigación; trabajo futuro |

**Respaldos (págs. 43–47, "solo si preguntan"):** disociación con valores exactos; estabilidad por
semilla de modelo; detalle estadístico de la partición; por qué GNNExplainer para el ranking; curvas
ROC y precisión-recall. Un archivo `respaldo-<slug>.md` por cada uno, con la misma plantilla (sin la
sección de discurso si no tienen entrada en el guion; en su lugar, "Cuándo abrir esta lámina").

> Antes de empezar, **regenera este mapa** desde el `.tex` y el DISCURSO por si el deck cambió:
> los números de fracción/página son la fuente. Si no coinciden con esta tabla, manda el PDF actual.

---

## 9. Ejemplo resuelto (una lámina modelo — calibra aquí la profundidad esperada)

> Esto es un **modelo de referencia** de cómo debe verse un archivo terminado (abreviado). Iguala
> este nivel de detalle y este tono.

### `lamina-25-puente-nulo.md` (extracto)

**🎯 Objetivo:** cerrar la tercera hipótesis mostrando que una creencia muy lógica es falsa: que un
explicador más *consistente* también sea más *acertado*. Se reporta el no-resultado con honestidad.

**📋 Qué dice, item por item:**
- *"Correlación estabilidad–plausibilidad r = −0,01, IC 95% de −0,038 a +0,011."* → medimos si esas
  dos cosas suben juntas; el número salió prácticamente cero.
- *"El intervalo incluye el cero: no hay relación."* → como el margen va de negativo a positivo
  pasando por cero, la lectura honesta es que no hay relación.

**🔑 Conceptos:**
- *Correlación:* mide si dos cosas se mueven juntas (+1 de la mano, −1 opuestas, 0 nada). Aquí ≈ 0.
- *IC que incluye el cero:* ni siquiera estamos seguros del signo → no hay efecto.

**🧠 Analogía (reloj parado):** un reloj parado es 100% estable y siempre está mal. Ser consistente
no te hace tener razón; estabilidad y acierto son independientes.

**🧑‍⚖️ Pregunta de jurado:** *"¿No será que su métrica no captó la relación?"* → "El intervalo de
confianza cruza el cero y el efecto es nulo en las dos métricas; además lo reportamos como
no-resultado, no lo escondimos."

**🧠 En una frase:** esperábamos un puente entre estabilidad y acierto; no existe, y decirlo es parte
del aporte.

*(El archivo real desarrolla cada punto más y añade el discurso completo + su versión ampliada.)*

---

## 10. Checklist de control de calidad (aplicar a cada archivo antes de darlo por hecho)

- [ ] Tiene las 7 secciones de la plantilla (§3), en orden.
- [ ] Explica **todos** los items de la lámina (ninguno sin traducir a lenguaje llano).
- [ ] Cada concepto/técnica tiene su explicación sin tecnicismos + analogía.
- [ ] Ninguna cifra contradice §5 ni el manuscrito; las cifras se verificaron en `tesis_latex/`.
- [ ] El discurso pegado coincide con el de `DISCURSO_defensa_dos_voces.md` (misma página).
- [ ] No reintroduce narrativa retractada (§5.1, §5.2, §5.6).
- [ ] Español con coma decimal, tono cálido, sin relleno, 400–900 palabras.
- [ ] Enlaces al glosario funcionan (anclas correctas).

---

## 11. Orden de trabajo sugerido

1. Leer §2 (fuentes) y §5 (guardarraíles). Regenerar el mapa (§8) desde el deck actual.
2. Escribir primero el **`README.md`** con glosario (§6) y banco de analogías (§7): así los
   archivos de lámina pueden enlazar desde el inicio.
3. Escribir las láminas **en orden de fracción** (1 → 33), luego los respaldos.
4. Al terminar cada bloque (Problema, Marco, Metodología, Resultados, Conclusiones), releer los
   archivos de ese bloque juntos para que no se repitan ni se contradigan.
5. Pasar el checklist (§10) a cada archivo.
6. No hacer `commit` ni `push` salvo que los autores lo pidan; entregar los archivos y avisar.

---

*Documento de planeación. Autor del plan: sesión de tutoría de defensa (Claude). Fecha base del
deck: commit `9399562` (deck v3, 47 páginas).*
