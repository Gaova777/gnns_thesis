# Lámina 07 — La brecha en el estado del arte

> Bloque: Problema y motivación
> Voz: Alejandro · Página PDF 9 (7/40) · Tiempo objetivo 60 s

## 🎯 Objetivo de la lámina

Mostrar que la tesis no repite lo que ya se sabía, sino que llena un vacío concreto: nadie había
evaluado sistemáticamente la estabilidad de las explicaciones sobre grafos financieros, y el dataset
de referencia (Elliptic) ni siquiera permite medir si una explicación es plausible, porque no trae la
respuesta correcta de antemano.

## 📋 Qué dice, item por item

- **"Lo que ya se sabía" (bloque izquierdo):**
  - Las GNN superan a los métodos tabulares en este dominio (referencia: Weber y colaboradores, 2019).
  - TAGCN combinado con SHAP logra alto rendimiento (referencia: He y colaboradores, 2026).
  - La predicción (¿es fraude o no?) y la explicabilidad (¿por qué?) se habían estudiado por
    separado, no de forma conjunta.
- **"Lo que faltaba" (bloque derecho, destacado):**
  - La **estabilidad** de las explicaciones sobre grafos de lavado de dinero no se había evaluado de
    forma sistemática.
  - Elliptic (el dataset real de referencia) **no tiene ground-truth de tipología**: no dice cuál es
    el patrón de lavado verdadero detrás de cada caso, así que la **plausibilidad** de una explicación
    (si señala o no el patrón correcto) simplemente **no era medible** con ese dataset.

## 🔑 Conceptos y técnicas que aparecen

- **Estado del arte:** lo que ya está publicado y aceptado en la literatura científica sobre un tema,
  antes de que empiece un nuevo trabajo.
- **Ground-truth de tipología:** la respuesta correcta conocida de antemano sobre cuál es el patrón de
  lavado detrás de un grupo de transacciones (por ejemplo, "estas diez transacciones forman un caso
  de fan-in"). Elliptic no incluye esa información: solo dice si cada transacción, individualmente, es
  lícita o ilícita. Ver [glosario](README.md#ground-truth-y-validez-interna-vs-externa).
- **Elliptic:** el dataset real de transacciones de Bitcoin que se usa en este estudio como el "eje
  real". Se describe en detalle en la [lámina 15](lamina-15-eje1-elliptic.md).

## 📈 Cómo leer la figura / tabla

No aplica: es un contraste en dos columnas ("lo que ya se sabía" vs. "lo que faltaba"), sin figura ni
tabla numérica.

## 🎤 El discurso (como se dice en voz alta)

> Que se sabia ya. Se sabia que las redes de grafos superan a los metodos tabulares en este dominio, y
> se habian estudiado la prediccion y la explicabilidad, pero por separado. *(pausa)* Que faltaba. Nadie
> habia evaluado de forma sistematica la estabilidad de las explicaciones sobre grafos financieros, es
> decir, si una explicacion se sostiene o cambia cuando se vuelve a calcular. Y habia un obstaculo de
> fondo: el dataset de referencia, Elliptic, no trae un patron verdadero de tipologia de lavado, de modo
> que la plausibilidad de una explicacion, si senala o no el patron correcto, simplemente no era medible.
> Esa doble brecha, la estabilidad no evaluada y la plausibilidad no medible, es la que esta tesis viene
> a cerrar.

### Versión ampliada y explicada

*"Que se sabia ya."* — Empieza reconociendo el terreno ya recorrido por otros: no se parte de cero,
sino que se construye sobre trabajo previo publicado.

*"Se sabia que las redes de grafos superan a los metodos tabulares en este dominio, y se habian
estudiado la prediccion y la explicabilidad, pero por separado."* — Dos cosas ya establecidas: (1)
las GNN funcionan mejor que los métodos tradicionales, y (2) hay estudios de predicción y estudios de
explicabilidad, pero nadie había cruzado ambos temas de forma sistemática.

*"Que faltaba. Nadie habia evaluado de forma sistematica la estabilidad de las explicaciones sobre
grafos financieros, es decir, si una explicacion se sostiene o cambia cuando se vuelve a calcular."*
— Aquí se nombra explícitamente el vacío #1: la estabilidad, tal como se definió en las láminas
anteriores, simplemente no había sido puesta a prueba de forma rigurosa en este dominio.

*"Y habia un obstaculo de fondo: el dataset de referencia, Elliptic, no trae un patron verdadero de
tipologia de lavado, de modo que la plausibilidad de una explicacion... simplemente no era medible."*
— Aquí se nombra el vacío #2, y además se explica **por qué** existía ese vacío: no era solo que
nadie lo hubiera intentado, es que el dataset más usado en el campo no daba las herramientas
necesarias para medirlo. Esto anticipa la necesidad del grafo sintético (lámina 16).

*"Esa doble brecha, la estabilidad no evaluada y la plausibilidad no medible, es la que esta tesis
viene a cerrar."* — Cierra la lámina conectando directamente con la contribución de la tesis: no es
una casualidad que se estudien estas dos brechas, es precisamente lo que el trabajo se propuso llenar.

## 🧑‍⚖️ Preguntas de jurado probables

- **"¿Por qué nadie había medido la estabilidad antes si es algo tan importante?"** → Porque requiere
  un protocolo de réplicas y pruebas estadísticas específico que no era estándar en los estudios de
  XAI para GNNs en este dominio; esta tesis construye y aplica ese protocolo (ver lámina 14, diseño
  factorial).
- **"¿Por qué Elliptic, siendo el dataset de referencia, no tiene ground-truth de tipología?"** →
  Porque fue etiquetado originalmente solo para la tarea de clasificación (lícito/ilícito), no para
  evaluar explicabilidad; documentar y aportar esa brecha específica es parte del valor de esta tesis.

## 🧠 En una frase

Faltaba evaluar sistemáticamente si las explicaciones sobre grafos financieros son estables, y
faltaba una forma de medir si son plausibles, porque el dataset de referencia no trae la respuesta
correcta de antemano.
