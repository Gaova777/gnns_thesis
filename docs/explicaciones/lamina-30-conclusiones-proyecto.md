# Lámina 30 — Conclusiones del proyecto

> Bloque: Conclusiones
> Voz: Juan Diego · Página PDF 36 (30/40) · Tiempo objetivo 75 s

## 🎯 Objetivo de la lámina

Reunir, en seis mensajes concisos, todo lo que sostiene el trabajo completo, antes de pasar a
responder objetivo por objetivo (siguiente lámina). Es el resumen de más alto nivel de toda la tesis.

## 📋 Qué dice, item por item

1. **"Tres dimensiones independientes"** (estabilidad, plausibilidad, fidelidad): no existe una única
   "buena explicación"; el mejor explicador depende del objetivo de la auditoría.
2. **"La estabilidad por arquitectura son dos grupos"** (alto: GAT y GCN; bajo: GraphSAGE y TAGCN), y
   la partición **concuerda** entre datos reales y sintéticos.
3. **"El explicador es la palanca dominante"** (η² de 0,36 a 0,64); el balanceo es despreciable
   (η² ≤ 0,01) y el escenario de desbalance, pequeño y no significativo (η² ≤ 0,05).
4. **"El más plausible no es el más fiel"** (PGExplainer vs. GNNExplainer): elegir según se quiera
   reconocer el patrón o auditar el modelo.
5. **"Aporte metodológico":** dos artefactos de medición corregidos, más distinguir reproducibilidad
   de pesos vs. de conclusiones.
6. **"Las tres hipótesis se refutaron"** y se reportan tal cual, sin ajustar el análisis para
   salvarlas.

## 🔑 Conceptos y técnicas que aparecen

Esta lámina no introduce conceptos nuevos: es una **síntesis** de todo lo ya explicado. Cada mensaje
enlaza con la lámina donde se desarrolla en detalle:

- Mensaje 1 → [lámina 11](lamina-11-tres-propiedades.md) (tres propiedades independientes).
- Mensaje 2 → [lámina 21](lamina-21-resultado1-dos-grupos.md) y
  [lámina 23](lamina-23-resultado1b-concordancia.md) (dos grupos y concordancia entre regímenes).
- Mensaje 3 → [lámina 26](lamina-26-resultado4-desbalance.md) (el explicador domina sobre desbalance y
  balanceo). El rango de η² de 0,36 a 0,64 para el explicador es un dato nuevo aquí, que se puede
  contrastar con el η² ≤ 0,05 del escenario y η² ≤ 0,01 del balanceo, para dimensionar cuánto más pesa
  el explicador.
- Mensaje 4 → [lámina 24](lamina-24-resultado2-disociacion.md) (disociación plausibilidad/fidelidad).
- Mensaje 5 → [lámina 19](lamina-19-dos-artefactos.md), [lámina 20](lamina-20-segundo-artefacto-truncamiento.md)
  y [lámina 22](lamina-22-robustez-particion.md) (los dos artefactos y la reproducibilidad de pesos vs.
  conclusiones).
- Mensaje 6 → [lámina 29](lamina-29-tres-hipotesis-veredicto.md) (las tres hipótesis refutadas).

## 📈 Cómo leer la figura / tabla

No aplica: es una lista enumerada de seis conclusiones, sin figura ni tabla numérica.

## 🎤 El discurso (como se dice en voz alta)

> Antes de responder objetivo por objetivo, permitanme dejar las seis conclusiones que sostienen todo el
> trabajo. *(pausa)* Primera: estabilidad, plausibilidad y fidelidad son tres dimensiones independientes.
> No existe una unica buena explicacion. El mejor explicador depende del objetivo de la auditoria.
> Segunda: la estabilidad por arquitectura no es un ranking de cuatro puestos sino una particion en dos
> grupos, y esa particion concuerda entre los datos reales y los sinteticos, que es la evidencia mas
> fuerte que tenemos de que no es un artefacto de un dataset. *(pausa)* Tercera: el explicador es la
> palanca dominante, con tamanos de efecto de cero coma treinta y seis a cero coma sesenta y cuatro,
> mientras que el balanceo es despreciable y el escenario de desbalance es pequeno y no significativo.
> Cuarta: el mas plausible no es el mas fiel, y por eso la eleccion se hace segun se quiera reconocer el
> patron o auditar el modelo. *(pausa)* Quinta, y es la que mas nos importa como aporte metodologico:
> corregimos dos artefactos de medicion, y aprendimos a distinguir la reproducibilidad de los pesos de la
> reproducibilidad de las conclusiones. Y sexta: las tres hipotesis que planteamos se refutaron, y lo
> reportamos tal cual, sin ajustar el analisis para salvarlas.

### Versión ampliada y explicada

*"Antes de responder objetivo por objetivo, permitanme dejar las seis conclusiones que sostienen todo
el trabajo."* — Explica la posición de esta lámina: antes de la respuesta formal a cada objetivo
(siguiente lámina), se da una síntesis de alto nivel que funciona como resumen ejecutivo de toda la
tesis.

*"Primera: estabilidad, plausibilidad y fidelidad son tres dimensiones independientes. No existe una
unica buena explicacion. El mejor explicador depende del objetivo de la auditoria."* — Repite, en su
forma más condensada, la tesis central presentada por primera vez en la lámina 11 y demostrada
empíricamente en los resultados.

*"Segunda: la estabilidad por arquitectura no es un ranking de cuatro puestos sino una particion en
dos grupos, y esa particion concuerda entre los datos reales y los sinteticos, que es la evidencia mas
fuerte que tenemos de que no es un artefacto de un dataset."* — Aquí se añade una precisión importante:
se llama explícitamente a la concordancia entre los dos ejes "la evidencia más fuerte" de que el
hallazgo es real y no una casualidad de un dataset en particular.

*"Tercera: el explicador es la palanca dominante, con tamanos de efecto de cero coma treinta y seis a
cero coma sesenta y cuatro, mientras que el balanceo es despreciable y el escenario de desbalance es
pequeno y no significativo."* — Aquí aparece un dato nuevo y muy útil para dimensionar la magnitud del
efecto del explicador: entre 0,36 y 0,64, muy por encima de los efectos de escenario (≤0,05) y
balanceo (≤0,01). Esta comparación de tres números en una sola frase es la evidencia cuantitativa más
directa de cuál es "la palanca que sí importa".

*"Cuarta: el mas plausible no es el mas fiel, y por eso la eleccion se hace segun se quiera reconocer
el patron o auditar el modelo."* — Resume la disociación (lámina 24) y ya apunta hacia su consecuencia
práctica (la matriz de recomendación, lámina 32).

*"Quinta, y es la que mas nos importa como aporte metodologico: corregimos dos artefactos de medicion,
y aprendimos a distinguir la reproducibilidad de los pesos de la reproducibilidad de las
conclusiones."* — El énfasis en "la que más nos importa" señala que, para los propios autores, el
aporte metodológico (no solo los hallazgos sustantivos) es una parte central del valor de su trabajo.

*"Y sexta: las tres hipotesis que planteamos se refutaron, y lo reportamos tal cual, sin ajustar el
analisis para salvarlas."* — Cierra reafirmando, una vez más, el compromiso de honestidad científica
que atraviesa toda la tesis.

## 🧑‍⚖️ Preguntas de jurado probables

- **"De estas seis conclusiones, ¿cuál consideran el aporte más importante?"** → Depende de la
  perspectiva: sustantivamente, que las tres dimensiones son independientes y que el explicador (no el
  desbalance ni el balanceo) es la palanca dominante; metodológicamente, haber corregido dos
  artefactos de medición y distinguido la reproducibilidad de pesos de la de conclusiones.
- **"¿Estas conclusiones son generalizables a otros dominios más allá del lavado de dinero?"** → La
  independencia de las tres dimensiones y la disociación plausibilidad/fidelidad son fenómenos
  plausiblemente generales en XAI para grafos, pero su magnitud exacta (los números concretos) se
  reporta específicamente para este dominio; generalizar a otros dominios es trabajo futuro.

## 🧠 En una frase

Seis mensajes resumen todo el trabajo: las tres dimensiones son independientes, hay dos grupos de
arquitecturas que concuerdan entre ejes, el explicador manda sobre desbalance y balanceo, el más
plausible no es el más fiel, se corrigieron dos artefactos de medición, y las tres hipótesis se
refutaron con honestidad.
