# Respaldo · ¿Por qué GNNExplainer para el ranking?

> Bloque: Respaldo (solo si el jurado pregunta)
> Página PDF 46 · Sin discurso propio en el guion (se abre solo si se pide más detalle)

## 🎯 Objetivo de la lámina

Justificar, con los tres explicadores lado a lado, por qué el ranking de estabilidad por arquitectura
(láminas 21-23) se calcula usando exclusivamente **GNNExplainer**, y no un promedio o comparación entre
los tres explicadores disponibles.

## 📋 Qué dice, item por item

- **Tabla 9 — Estabilidad (Spearman) por explicador y arquitectura en Elliptic:**
  - **GNNExplainer:** GAT 0,78; GCN 0,76; GraphSAGE 0,74; TAGCN 0,67.
  - **PGExplainer:** GAT 0,00; GCN 0,00; GraphSAGE 0,00; TAGCN 0,00.
  - **GNNShap:** GAT 0,97\*; GCN 0,97; GraphSAGE 0,95; TAGCN 0,97\*.
- **Texto de apoyo:** GNNExplainer es el único que discrimina entre arquitecturas. PGExplainer
  degenera (Spearman nulo en las cuatro). GNNShap se satura cerca de 0,96, indistinguible entre
  arquitecturas. Además, GNNExplainer es el explicador común a los dos ejes, lo que hace comparable la
  partición entre Elliptic y el sintético.
- **Nota al pie:** GNNExplainer está calculado a 3 semillas en las cuatro arquitecturas. GNNShap está a
  3 semillas en GCN y GraphSAGE; las marcadas con asterisco (GAT y TAGCN) solo están en la semilla 42,
  por una razón operativa (no por una limitación de diseño de esas arquitecturas).

## 🔑 Conceptos y técnicas que aparecen

- **PGExplainer degenerado (Spearman = 0,00 en las cuatro):** este cero **no es un error de cálculo**.
  En un grafo tan disperso como Elliptic (vecindarios de ~2 nodos, ver lámina 15), PGExplainer entra en
  un "colapso de modo": asigna prácticamente la misma importancia a todas las aristas disponibles. Si
  todo vale igual, no hay un orden real que comparar entre repeticiones, y la correlación de Spearman
  sale exactamente nula. La prueba de que esto es un problema del **dato** (dispersión extrema) y no
  del **método**: el mismo PGExplainer, sobre el grafo sintético denso, es el explicador **más
  plausible** de los tres (0,80, ver lámina 24). Ver la analogía del
  [detective en una escena vacía](README.md#banco-de-analogías-para-dar-coherencia-entre-láminas).
- **GNNShap saturado (~0,96, indistinguible entre arquitecturas):** GNNShap es tan internamente
  estable (por su método basado en valores de Shapley, ver lámina 10) que da valores muy altos y muy
  parecidos entre sí para las cuatro arquitecturas. Esto es una fortaleza para medir su propia
  consistencia interna, pero significa que **no sirve** para distinguir si una arquitectura es más
  estable que otra: todas se ven casi igual de bien con esta métrica.
- **Por qué GNNExplainer sí discrimina:** da valores distintos y ordenados (0,78 / 0,76 / 0,74 / 0,67),
  con suficiente separación entre ellos como para que las pruebas estadísticas (Kruskal-Wallis,
  Mann-Whitney) puedan detectar diferencias reales entre grupos.
- **Explicador común a los dos ejes:** que el mismo método (GNNExplainer) se use tanto en Elliptic
  como en el grafo sintético es lo que permite afirmar que la concordancia entre regímenes (lámina 23)
  es una comparación válida y no una comparación entre manzanas y naranjas.

## 📈 Cómo leer la figura / tabla

La Tabla 9 tiene tres filas (una por explicador) y cuatro columnas (una por arquitectura). La lectura
correcta es fijarse en **cuánta variación hay dentro de cada fila**: la fila de GNNExplainer muestra
valores claramente distintos entre columnas (0,78 a 0,67, un rango amplio); la fila de PGExplainer
muestra el mismo valor (0,00) en las cuatro columnas (ninguna variación, porque colapsa siempre); la
fila de GNNShap muestra valores muy juntos entre sí (0,95 a 0,97, casi sin variación). Solo la primera
fila tiene la variación necesaria para servir de base a una comparación entre arquitecturas.

## 🕐 Cuándo abrir esta lámina

Abrir esta lámina si el jurado pregunta: "¿por qué el ranking por arquitectura solo usa
GNNExplainer?", "¿qué pasa con los otros dos explicadores en Elliptic?", o "¿ese 0,00 de PGExplainer
es un error?".

## 🧑‍⚖️ Preguntas de jurado probables

- **"¿Por qué PGExplainer da exactamente 0,00 en las cuatro arquitecturas?"** → Porque en un grafo tan
  disperso como Elliptic (mediana de ~2 nodos por vecindario), PGExplainer no tiene suficiente
  estructura que analizar y colapsa asignando la misma importancia a todo (mode collapse); sin un orden
  real que comparar, la correlación de Spearman entre repeticiones sale nula. No es un bug: es un
  hallazgo documentado sobre el comportamiento de este explicador bajo dispersión extrema.
- **"Si PGExplainer da 0,00 en Elliptic, ¿es un mal método?"** → No en general: el mismo PGExplainer,
  sobre el grafo sintético denso, resulta ser el explicador más plausible de los tres (ver lámina 24).
  El contraste entre los dos regímenes demuestra que su degeneración en Elliptic es un efecto de la
  dispersión del grafo, no una limitación intrínseca del método.
- **"¿Por qué no promediar los tres explicadores para el ranking, en vez de usar solo uno?"** → Porque
  dos de los tres (PGExplainer y GNNShap) no aportan señal útil para distinguir arquitecturas en este
  eje específico (uno colapsa, el otro se satura); promediarlos con GNNExplainer solo añadiría ruido
  sin aportar información real.

## 🧠 En una frase

De los tres explicadores, solo GNNExplainer produce valores distintos y ordenados entre arquitecturas
en Elliptic; PGExplainer colapsa por la dispersión del grafo (no por ser un mal método) y GNNShap se
satura por ser demasiado consistente internamente.
