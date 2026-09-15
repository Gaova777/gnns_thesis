# Respaldo · Detalle estadístico de la partición

> Bloque: Respaldo (solo si el jurado pregunta)
> Página PDF 45 · Sin discurso propio en el guion (se abre solo si se pide más detalle)

## 🎯 Objetivo de la lámina

Mostrar, comparación por comparación, el detalle del test de Wilcoxon pareado entre cada par de
arquitecturas, tanto en Elliptic como en el eje sintético, para poder responder con precisión exacta
si el jurado pide el desglose completo detrás de la partición en dos grupos.

## 📋 Qué dice, item por item

- **Tabla 8 — Wilcoxon pareado por pares de arquitecturas (Elliptic y sintético):**
  - **Dentro de grupo alto — GAT vs. GCN:** Elliptic p=0,165 (no significativo); sintético p=0,0013
    (significativo).
  - **Dentro de grupo bajo — GraphSAGE vs. TAGCN:** Elliptic p=0,096 (no significativo); sintético
    p=0,519 (no significativo).
  - **Entre grupos (las seis comparaciones cruzadas restantes):** todas con p muy pequeños, desde
    0,0033 hasta valores menores de 10⁻⁴, tanto en Elliptic como en el sintético.
- **Nota:** la única diferencia intragrupo significativa es GCN sobre GAT en el sintético, por un
  margen de solo **0,006** sin relevancia práctica, detectable solo por el mayor número de muestras
  (n) de ese eje.

## 🔑 Conceptos y técnicas que aparecen

- **Significativo vs. no significativo (con este p):** un p-valor por debajo de un umbral convencional
  (habitualmente 0,05) se considera "significativo" (evidencia de diferencia real); por encima, "no
  significativo" (compatible con azar). Aquí, casi todas las comparaciones **dentro** de un mismo
  grupo dan p altos (no significativos), y casi todas las comparaciones **entre** grupos dan p
  bajísimos (significativos), lo que confirma visualmente, fila por fila, la estructura de dos grupos.
- **Significativo pero sin relevancia práctica:** el caso de GCN vs. GAT en el sintético (p=0,0013,
  técnicamente significativo) es un ejemplo importante de por qué el p-valor solo no basta: la
  diferencia real detectada es de apenas 0,006 (prácticamente nada en términos prácticos), y solo se
  detecta como "significativa" porque el eje sintético tiene muchas más muestras, lo que le da más
  poder estadístico para detectar diferencias diminutas. Esto es un ejemplo de por qué siempre conviene
  mirar también el tamaño de efecto, no solo el p-valor.
- **Comparaciones "entre grupos" (cruzadas):** por ejemplo, GAT vs. GraphSAGE, GAT vs. TAGCN, GCN vs.
  GraphSAGE, GCN vs. TAGCN — todas las combinaciones que mezclan una arquitectura del grupo alto con
  una del grupo bajo, y que consistentemente muestran diferencias muy significativas.

## 📈 Cómo leer la figura / tabla

La Tabla 8 se lee por bloques: las dos primeras filas (comparaciones "dentro de grupo") deberían
mostrar p-valores altos si la partición en dos grupos es correcta, y las cuatro filas siguientes
(comparaciones "entre grupos") deberían mostrar p-valores bajos. Ese es exactamente el patrón que
aparece, con la única excepción parcial de GAT vs. GCN en el sintético (significativo, pero con una
diferencia mínima sin relevancia práctica, como se explica en la nota).

## 🕐 Cuándo abrir esta lámina

Abrir esta lámina si el jurado pide "el detalle estadístico completo" de la partición, pregunta por
qué GCN vs. GAT da un resultado técnicamente significativo en el sintético, o pide ver los seis pares
de comparación uno por uno en vez de solo el resumen de la lámina 22.

## 🧑‍⚖️ Preguntas de jurado probables

- **"Si GCN vs. GAT da p=0,0013 en el sintético, ¿no contradice eso la afirmación de que no hay
  diferencia dentro del grupo alto?"** → No, porque la magnitud de esa diferencia es de apenas 0,006,
  sin relevancia práctica; el resultado "significativo" aparece solo por el mayor número de muestras
  del eje sintético, que le da poder estadístico para detectar hasta diferencias diminutas. Es un
  ejemplo de por qué la tesis siempre reporta también el tamaño de efecto, no solo el p-valor.
- **"¿Por qué las comparaciones entre grupos dan p tan extremadamente pequeños (menores de
  10⁻⁴)?"** → Porque la diferencia entre el grupo alto y el bajo es consistente y de magnitud
  considerable en ambos ejes, lo que produce evidencia estadística muy fuerte en cualquier
  comparación cruzada entre un miembro de cada grupo.

## 🧠 En una frase

El desglose completo por pares confirma la partición: casi todas las comparaciones dentro de grupo dan
p altos (sin diferencia real) y todas las comparaciones entre grupos dan p bajísimos, con una única
excepción de magnitud tan pequeña que no tiene relevancia práctica.
