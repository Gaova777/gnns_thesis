# Respaldo · Curvas ROC y precisión-recall

> Bloque: Respaldo (solo si el jurado pregunta)
> Página PDF 47 · Sin discurso propio en el guion (se abre solo si se pide más detalle)

## 🎯 Objetivo de la lámina

Mostrar, de forma visual y completa, las curvas ROC y de precisión-recall (no solo el número de área
bajo la curva), para que quede completamente claro **por qué** el ROC-AUC se ve más favorable de lo
que la tarea realmente es, complementando la tabla numérica de la
[lámina 28](lamina-28-rigor-metrico-prauc-rocauc.md).

## 📋 Qué dice, item por item

- **Figura 11 (dos gráficos lado a lado):**
  - **Izquierda — Curva ROC:** validación (AUC=0,884) frente a test (AUC=0,653), más una línea de
    referencia diagonal que representa el azar.
  - **Derecha — Curva de precisión y exhaustividad (PR):** validación (AUC=0,367) frente a test
    (AUC=0,017).
- **Pie de figura:** el ROC cae de 0,88 a 0,65, pero la precisión-recall se desploma casi a la tasa
  base de transacciones ilícitas; por eso la métrica primaria es PR-AUC, no ROC-AUC.

## 🔑 Conceptos y técnicas que aparecen

- **Curva ROC:** un gráfico que muestra, para cada posible umbral de decisión, la relación entre la
  tasa de verdaderos positivos (cuántos fraudes reales se detectan) y la tasa de falsos positivos
  (cuántas transacciones normales se marcan por error). Una curva que se pega a la esquina superior
  izquierda es "buena"; una diagonal representa el comportamiento de una elección al azar.
- **Curva de precisión-recall (PR):** un gráfico que muestra, para cada umbral, la relación entre
  precisión (de lo que el modelo marcó como fraude, ¿cuánto era real?) y recall/exhaustividad (de todo
  el fraude real, ¿cuánto se detectó?). Bajo desbalance extremo, esta curva es mucho más informativa
  que la ROC porque no se deja "inflar" por la enorme mayoría de casos normales.
- **Por qué la curva ROC de test "se ve razonable" a pesar del colapso real:** al comparar visualmente
  las dos curvas ROC (validación y test), la de test se aleja de la diagonal (azar) de forma
  moderada, lo que visualmente sugiere "todavía funciona algo". La curva PR de test, en cambio, se
  desploma dramáticamente hacia niveles cercanos a la tasa base de fraude, revelando la magnitud real
  del colapso.
- **Tasa base de transacciones ilícitas:** el porcentaje de fraude real en el conjunto de datos (en
  Elliptic, aproximadamente 2,2%, ver lámina 15). Cuando la curva PR de test cae "casi a la tasa base",
  significa que el modelo, en test, ya casi no aporta ninguna capacidad de discriminación por encima de
  simplemente adivinar al azar según esa proporción base.

## 📈 Cómo leer la figura / tabla

La Figura 11 se lee comparando el gráfico izquierdo (ROC) con el derecho (PR) para el mismo par de
curvas (validación y test). La observación clave: en el gráfico ROC, la curva de test todavía se separa
visiblemente de la diagonal de azar; en el gráfico PR, la curva de test se pega mucho más cerca de una
línea plana y baja, cercana a la tasa base de fraude. Esa diferencia visual, entre "todavía se distingue
algo" (ROC) y "prácticamente no hay señal" (PR), es la evidencia gráfica de por qué el ROC-AUC engaña
bajo desbalance extremo.

## 🕐 Cuándo abrir esta lámina

Abrir esta lámina si el jurado pide "ver las curvas, no solo los números de área", pregunta cómo se ve
exactamente la disociación entre ROC y PR de forma visual, o pide más evidencia gráfica sobre el
colapso de validación a test.

## 🧑‍⚖️ Preguntas de jurado probables

- **"¿Estas curvas usan los mismos modelos que la tabla de la lámina 28?"** → Sí, son exactamente los
  mismos modelos evaluados en ambas particiones (validación y test); la figura solo añade la
  representación gráfica completa de la curva, en vez de resumirla en un único número de área.
- **"¿Por qué la curva ROC de test parece 'razonable' mientras la curva PR se desploma?"** → Porque la
  tasa de falsos positivos del ROC se calcula sobre el enorme conjunto de casos negativos reales, que
  domina el cálculo y mantiene esa tasa baja aunque el modelo falle en detectar los pocos casos
  positivos; la precisión, en cambio, se calcula sobre las predicciones positivas del modelo, así que
  refleja de inmediato cuando esas predicciones dejan de ser confiables.

## 🧠 En una frase

Vistas como curvas completas, no solo como un número de área, la disociación entre ROC y PR bajo
desbalance extremo se ve con total claridad: el ROC "aguanta" visualmente mientras el PR se desploma.
