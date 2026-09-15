# Lámina 33 — Contribuciones y limitaciones

> Bloque: Conclusiones
> Voz: Juan Diego · Página PDF 39 (33/40) · Tiempo objetivo 70 s

## 🎯 Objetivo de la lámina

Recapitular, con la misma honestidad que ha marcado toda la defensa, qué aporta realmente esta tesis y
hasta dónde llega esa evidencia. Se dicen las limitaciones **antes** de que el jurado las encuentre,
como estrategia deliberada de credibilidad.

## 📋 Qué dice, item por item

- **Contribuciones (bloque izquierdo):**
  - Tres dimensiones independientes.
  - Dos artefactos de medición corregidos + dos *bugs* reportados en PGExplainer.
  - Generador sintético con ground-truth por arista.
  - Reproducibilidad de pesos vs. de conclusiones.
  - Matriz de recomendación por objetivo.
- **Limitaciones (bloque derecho, en un recuadro de alerta):**
  - La inferencia más fuerte proviene del eje sintético.
  - El clasificador colapsa en test (shift temporal).
  - Un solo dataset real, y anonimizado.
  - Con 3 semillas se sostiene el grupo, pero no el orden fino dentro de cada grupo.

## 🔑 Conceptos y técnicas que aparecen

- **Declarar limitaciones proactivamente:** una práctica de rigor científico donde el propio equipo
  identifica y comunica los límites de su evidencia, en vez de esperar a que un evaluador externo los
  señale. Esto se ha visto repetidamente en toda la defensa (por ejemplo, en la lámina 22 sobre la
  cobertura desigual de la replicación con tres semillas).
- **Por qué la inferencia más fuerte viene del eje sintético:** porque es el único régimen donde
  **todas** las propiedades (estabilidad, plausibilidad, fidelidad) son medibles a la vez, con
  ground-truth y replicación completa; el eje real (Elliptic) solo permite medir estabilidad.
- **Dataset anonimizado:** Elliptic no revela la identidad real de las cuentas ni el significado
  concreto de sus atributos (por razones de privacidad y seguridad), lo que limita la posibilidad de
  interpretar cualitativamente por qué ciertos atributos resultan importantes.
- **Sostener el grupo pero no el orden fino:** con tres semillas de modelo, hay suficiente evidencia
  para afirmar que existen dos grupos bien diferenciados, pero no suficiente para afirmar con certeza
  cuál de las dos arquitecturas dentro de un mismo grupo es "mejor" que la otra.

## 📈 Cómo leer la figura / tabla

No aplica: son dos bloques de texto en columnas paralelas (contribuciones a la izquierda,
limitaciones a la derecha), sin figura ni tabla numérica.

## 🎤 El discurso (como se dice en voz alta)

> Recapitulo aportes y limites. *(pausa)* Contribuciones: mostramos que
> estabilidad, plausibilidad y fidelidad son tres dimensiones independientes en este dominio. Corregimos
> dos artefactos de evaluacion y reportamos dos bugs de PGExplainer. Construimos un generador sintetico
> con patron verdadero por arista, y entregamos la matriz de recomendacion. *(pausa)* Limitaciones, y las
> decimos nosotros antes de que el jurado las encuentre: la evidencia inferencial mas fuerte proviene del eje
> sintetico, que es el unico donde plausibilidad y fidelidad son medibles. El clasificador colapsa en test
> por el desplazamiento temporal, y aunque replicamos el entrenamiento con tres semillas de modelo en
> ambos ejes, el numero de entrenamientos por celda sigue siendo modesto, suficiente para separar los dos
> grupos de arquitecturas pero no para ordenar dentro de cada grupo, algo especialmente cierto en TAGCN
> sobre Elliptic, cuya dispersion entre semillas es la mayor de las cuatro. Ninguna de estas invalida los
> hallazgos, pero marcan hasta donde llegan.

### Versión ampliada y explicada

*"Recapitulo aportes y limites."* — Anuncia el propósito dual de la lámina: no es solo un resumen de
logros, es también un ejercicio de honestidad sobre lo que la evidencia no permite afirmar.

*"Contribuciones: mostramos que estabilidad, plausibilidad y fidelidad son tres dimensiones
independientes en este dominio."* — Repite, por tercera vez en la presentación, el hallazgo central,
porque es tan importante que merece reafirmarse en cada resumen.

*"Corregimos dos artefactos de evaluacion y reportamos dos bugs de PGExplainer. Construimos un
generador sintetico con patron verdadero por arista, y entregamos la matriz de recomendacion."* —
Enumera de forma compacta las cuatro contribuciones restantes, cada una ya desarrollada a fondo en
láminas anteriores.

*"Limitaciones, y las decimos nosotros antes de que el jurado las encuentre..."* — Esta frase, dicha
en voz alta, es una declaración explícita de estrategia: mostrar autoconciencia sobre los límites del
propio trabajo genera más confianza que ocultarlos y esperar a que se descubran en la ronda de
preguntas.

*"...la evidencia inferencial mas fuerte proviene del eje sintetico, que es el unico donde
plausibilidad y fidelidad son medibles."* — Es una limitación importante y honesta: los resultados más
"fuertes" estadísticamente (la disociación, el puente nulo) dependen del entorno controlado, no del
dato real, aunque ese entorno se haya diseñado con mucho cuidado para no ser sesgado (ver
`DEFENSA_R2_evidencia_sintetica.md`).

*"El clasificador colapsa en test por el desplazamiento temporal, y aunque replicamos el entrenamiento
con tres semillas de modelo en ambos ejes, el numero de entrenamientos por celda sigue siendo modesto,
suficiente para separar los dos grupos de arquitecturas pero no para ordenar dentro de cada grupo,
algo especialmente cierto en TAGCN sobre Elliptic, cuya dispersion entre semillas es la mayor de las
cuatro."* — Reconoce el límite estadístico exacto de la replicación: alcanza para lo que se afirma (la
partición en dos grupos), pero no alcanza para afirmaciones más finas (un orden dentro de cada grupo),
y se señala específicamente a TAGCN como el caso con más variabilidad entre semillas.

*"Ninguna de estas invalida los hallazgos, pero marcan hasta donde llegan."* — Cierra con la frase que
resume la actitud de toda la tesis frente a sus propios límites: reconocerlos no anula el trabajo,
solo delimita con precisión su alcance.

## 🧑‍⚖️ Preguntas de jurado probables

- **"Si el eje sintético es donde está la evidencia más fuerte, ¿qué tan preocupante es que dependa
  de un dataset construido por ustedes?"** → Se aborda directamente con los cinco pilares del material
  de respaldo `DEFENSA_R2_evidencia_sintetica.md`: es necesidad metodológica (Elliptic no permite
  medir plausibilidad/fidelidad), es práctica estándar en la literatura de XAI para grafos, el
  sintético refutó la propia hipótesis central del equipo (prueba de que no fue diseñado a modo), es
  robusto (replicado en tres grafos y tres semillas), y su construcción es neutral al explicador.
- **"¿Por qué usar un solo dataset real y no varios?"** → Por disponibilidad: Elliptic es el dataset de
  referencia más usado y documentado en este dominio; usar un dataset real adicional (como AMLSim)
  se exploró pero quedó bloqueado por limitaciones técnicas, y se deja como trabajo futuro explícito.

## 🧠 En una frase

Se reconocen abiertamente los límites de la evidencia —dependencia del eje sintético para los
resultados más fuertes, el colapso en test, y una replicación suficiente para el grupo pero no para el
orden fino dentro de él— porque admitirlos fortalece la credibilidad del trabajo.
