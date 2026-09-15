# Respaldo · Disociación: valores exactos

> Bloque: Respaldo (solo si el jurado pregunta)
> Página PDF 43 · Sin discurso propio en el guion (se abre solo si se pide más detalle)

## 🎯 Objetivo de la lámina

Dar el detalle numérico completo, explícito y correctamente etiquetado, de la plausibilidad y la
fidelidad de los tres explicadores en el grafo sintético — incluyendo a **GNNShap**, que en la lámina
principal de disociación ([lámina 24](lamina-24-resultado2-disociacion.md)) se excluye de la figura
por medirse en otra escala. Esta lámina es la referencia exacta para responder con precisión si el
jurado pide esos números.

## 📋 Qué dice, item por item

- **Tabla 6 — Plausibilidad y Fidelity+ por explicador (eje sintético, g0, 3 semillas):**
  - **GNNExplainer:** plausibilidad de aristas 0,50; plausibilidad de features **0,45**; Fidelity+
    **0,56**.
  - **PGExplainer:** plausibilidad de aristas **0,80**; plausibilidad de features n/d (no aplica: no
    produce ranking de features); Fidelity+ 0,11.
  - **GNNShap:** plausibilidad de aristas n/d (no aplica: no produce máscara de aristas); plausibilidad
    de features 0,15; Fidelity+ 0,46.
  - **Azar (línea base):** aristas 0,40; features 0,08; no aplica para fidelidad.
- **Nota:** cada explicador se compara contra la línea base de azar de **su propia métrica**, no
  contra una sola línea base compartida.
- **Texto de apoyo:** las celdas "n/d" corresponden a métricas que el explicador no produce **por
  diseño** (PGExplainer no ordena features; GNNShap no genera máscara de aristas). Contra su propia
  línea base, PGExplainer duplica el azar en aristas (0,80 vs. 0,40) y GNNExplainer lo sextuplica en
  features (0,45 vs. 0,08 — en realidad más de 5,6 veces). PGExplainer supera a GNNExplainer en
  plausibilidad de aristas (Wilcoxon pareado p≈2,6×10⁻³⁵, n=225 comparaciones). GNNShap es el más
  estable internamente (0,98).

## 🔑 Conceptos y técnicas que aparecen

- **n/d (no disponible/no aplica):** una celda marcada así no significa "no se midió por error"; significa
  que esa métrica específica **no existe** para ese explicador por cómo está diseñado. PGExplainer no
  produce un ranking de atributos, así que no tiene sentido calcular su plausibilidad de features.
  GNNShap no produce una máscara de aristas, así que no tiene sentido calcular su plausibilidad de
  aristas.
- **Cada métrica con su propia línea base de azar:** es un principio metodológico central en esta
  lámina. La plausibilidad de aristas y la de features son dos mediciones distintas, con escalas
  distintas (por ejemplo, hay muchas más features posibles que aristas relevantes en el subgrafo, lo
  que hace que el azar de features sea mucho más bajo, 0,08, que el de aristas, 0,40). Mezclarlas en
  una sola comparación sería engañoso.
- **Estabilidad interna de GNNShap (0,98):** aquí se confirma con un número exacto lo que se menciona
  de pasada en la lámina 24: GNNShap, aunque no lidere ni plausibilidad ni fidelidad, es el más
  consistente entre ejecuciones repetidas de los tres explicadores.

## 📈 Cómo leer la figura / tabla

La Tabla 6 tiene tres columnas de métricas (plausibilidad de aristas, plausibilidad de features,
Fidelity+) y cuatro filas (los tres explicadores más la línea de azar). La forma correcta de leerla es
por columna, no por fila: dentro de cada columna, comparar el valor de un explicador contra la línea
de azar **de esa misma columna**. Por ejemplo, el 0,15 de GNNShap en plausibilidad de features se
compara contra 0,08 (el azar de esa columna), no contra 0,40 (el azar de la columna de aristas), lo
que muestra que GNNShap **casi duplica** su propio azar, en vez de parecer "peor que el azar" si se
comparara incorrectamente contra 0,40.

## 🕐 Cuándo abrir esta lámina

Abrir esta lámina si el jurado pregunta específicamente: "¿por qué GNNShap no aparece en la figura de
disociación?", "¿cuáles son los valores exactos de plausibilidad de features?", o "¿qué tan estable es
GNNShap comparado con los otros dos explicadores?". Es la respuesta ensayada exacta a la pregunta de
jurado documentada en el guion: *"Porque no produce máscara de aristas por diseño. Su plausibilidad se
mide sobre features, que es otra métrica, con otra escala y otra línea base de azar: 0,075 [nota: el
valor preciso reportado en el manuscrito es 0,075; esta tabla del deck redondea a 0,08], frente a 0,40
de aristas. Si pusiéramos su 0,15 junto a las barras de aristas, la figura sugeriría que está por
debajo del azar, cuando en realidad lo duplica. Sus tres valores están en la lámina de respaldo, cada
uno contra la línea base que le corresponde."*

## 🧑‍⚖️ Preguntas de jurado probables

- **"¿Por qué la línea base de azar de features (0,08) es tan distinta de la de aristas (0,40)?"** →
  Porque dependen de cuántos elementos hay para elegir al azar en cada caso: el subgrafo de aristas
  relevantes es proporcionalmente más grande respecto al total de aristas posibles, que el conjunto de
  features realmente importantes respecto al total de 166 disponibles, así que el azar "acierta" con
  más frecuencia en aristas que en features.
- **"Si GNNShap solo llega a 0,15 en features, ¿es un mal explicador?"** → No: 0,15 duplica su propio
  azar (0,08), lo que indica que sí captura señal real, aunque menos que GNNExplainer (0,45); además,
  GNNShap tiene su propia fortaleza distintiva en la consistencia interna (0,98).

## 🧠 En una frase

Cada explicador se compara contra la línea base de azar de su propia métrica, no contra una compartida:
así se ve que GNNShap duplica su azar en features, en vez de parecer "peor que el azar" por error de
comparación.
