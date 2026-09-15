# Lámina 13 — Metodología: un diseño de dos ejes

> Bloque: Metodología
> Voz: Alejandro · Página PDF 17 (13/40) · Tiempo objetivo 70 s

## 🎯 Objetivo de la lámina

Presentar la decisión metodológica más importante de toda la tesis: en lugar de usar un solo dataset,
se usan **dos**, cada uno con fortalezas y límites distintos, que se complementan para poder medir
todo lo que se necesita medir.

## 📋 Qué dice, item por item

- **Eje 1 · Elliptic (real):**
  - Validez **externa**: son transacciones reales.
  - Sin ground-truth: solo permite medir estabilidad.
  - Campo receptivo minúsculo (~2 nodos).
- **Eje 2 · Sintético (propio):**
  - Validez **interna**: ground-truth por nodo y por arista.
  - Permite medir plausibilidad y fidelidad, algo imposible en Elliptic.
  - 4 tipologías estándar y 3 realizaciones del grafo (tres versiones distintas generadas de forma
    independiente).

## 🔑 Conceptos y técnicas que aparecen

- **Validez externa:** qué tanto se puede confiar en que un hallazgo se sostiene en el mundo real,
  fuera del laboratorio. Se logra usando datos reales, con todo su ruido y sus imperfecciones.
- **Validez interna:** qué tanto se puede confiar en que el hallazgo realmente mide lo que dice medir,
  controlando todas las demás variables. Se logra construyendo un entorno donde se conoce la
  respuesta correcta de antemano.
- **Ground-truth por nodo y por arista:** en el grafo sintético, se sabe de antemano exactamente qué
  nodos y qué aristas forman parte de un patrón de lavado real (porque el propio equipo lo construyó
  así), a diferencia de Elliptic, donde solo se sabe si una transacción, individualmente, es lícita
  o no. Ver [glosario](README.md#ground-truth-y-validez-interna-vs-externa).
- **Realización del grafo:** una versión concreta del grafo sintético, generada con una semilla de
  azar distinta. Usar tres realizaciones (en vez de solo una) permite comprobar que los hallazgos no
  dependen de una casualidad de un único grafo particular.

## 📈 Cómo leer la figura / tabla

No aplica: son dos bloques (recuadros) en paralelo, uno por eje, sin figura ni tabla numérica.

## 🎤 El discurso (como se dice en voz alta)

> Como se prueba algo asi. Con un diseno de dos ejes, que es la decision metodologica mas importante de
> la tesis. *(pausa)* El primer eje es Elliptic, el dataset real de transacciones de Bitcoin. Nos da
> validez externa, porque son datos reales con todo su ruido y su desbalance, pero tiene dos limites:
> no trae patron verdadero, asi que solo permite medir estabilidad, y sus vecindarios son minusculos,
> de unos dos nodos. El segundo eje es un grafo sintetico que construimos nosotros, con patron verdadero
> por cada nodo y por cada arista. Nos da validez interna: como sabemos cual es el patron correcto,
> podemos medir plausibilidad y fidelidad, cosa imposible en Elliptic. *(pausa)* La clave es que los dos
> ejes se complementan. Ninguno solo alcanza. Juntos permiten afirmar cosas que ninguno probaria por su
> cuenta. Y aqui adelanto un punto que mi companero va a demostrar: los dos ejes, bien medidos, cuentan
> la misma historia.

### Versión ampliada y explicada

*"Como se prueba algo asi. Con un diseno de dos ejes, que es la decision metodologica mas importante
de la tesis."* — Abre destacando la importancia de esta decisión por encima de cualquier otro detalle
técnico: es la columna vertebral de todo el diseño experimental.

*"El primer eje es Elliptic, el dataset real de transacciones de Bitcoin. Nos da validez externa,
porque son datos reales con todo su ruido y su desbalance, pero tiene dos limites: no trae patron
verdadero, asi que solo permite medir estabilidad, y sus vecindarios son minusculos, de unos dos
nodos."* — Presenta a Elliptic con honestidad: su fortaleza (realismo) viene junto con dos límites
concretos y bien delimitados, no escondidos.

*"El segundo eje es un grafo sintetico que construimos nosotros, con patron verdadero por cada nodo y
por cada arista. Nos da validez interna: como sabemos cual es el patron correcto, podemos medir
plausibilidad y fidelidad, cosa imposible en Elliptic."* — Presenta al grafo sintético como la pieza
que llena exactamente el vacío que Elliptic deja: donde Elliptic no puede (plausibilidad y fidelidad),
el sintético sí puede, porque se conoce la respuesta correcta.

*"La clave es que los dos ejes se complementan. Ninguno solo alcanza. Juntos permiten afirmar cosas
que ninguno probaria por su cuenta."* — Es la idea central de la lámina: no se trata de elegir el
"mejor" dataset, sino de usar dos que se cubren mutuamente las espaldas.

*"Y aqui adelanto un punto que mi companero va a demostrar: los dos ejes, bien medidos, cuentan la
misma historia."* — Un adelanto hacia uno de los resultados más importantes: la concordancia entre
regímenes (ver [lámina 23](lamina-23-resultado1b-concordancia.md)), que es la prueba de que los dos
ejes, lejos de contradecirse, se refuerzan mutuamente.

## 🧑‍⚖️ Preguntas de jurado probables

- **"¿Por qué no usar solo el grafo sintético, si ahí sí se puede medir todo?"** → Porque el sintético,
  por más cuidado que se ponga en su diseño, sigue siendo un entorno controlado; sin el eje real
  (Elliptic) no habría evidencia de que los hallazgos se sostienen fuera del laboratorio. Ver también
  la defensa completa de esta objeción en el material de respaldo `DEFENSA_R2_evidencia_sintetica.md`.
- **"¿Por qué no usar un dataset real distinto que sí tenga ground-truth de tipología?"** → No existe
  uno disponible que lo ofrezca con el detalle necesario (se exploró AMLSim como alternativa y quedó
  descartado por limitaciones técnicas y de datos, documentado como trabajo futuro).

## 🧠 En una frase

Ningún dataset por sí solo bastaba: Elliptic aporta realismo (validez externa) y el grafo sintético
aporta control con respuesta conocida (validez interna), y juntos permiten medir las tres propiedades.
