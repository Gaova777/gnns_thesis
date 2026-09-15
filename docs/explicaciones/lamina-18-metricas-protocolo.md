# Lámina 18 — Métricas y protocolo estadístico

> Bloque: Metodología
> Voz: Alejandro · Página PDF 22 (18/40) · Tiempo objetivo 55 s

## 🎯 Objetivo de la lámina

Cerrar el bloque de metodología resumiendo, de forma compacta, cómo se mide cada una de las tres
propiedades (y el rendimiento del clasificador), y qué herramientas estadísticas respaldan esas
mediciones. Es la última lámina de Alejandro antes de ceder la palabra a Juan Diego para los
resultados.

## 📋 Qué dice, item por item

- **Estabilidad — "correlación de Spearman entre rankings de features (primaria)."** — La métrica
  principal para medir si una explicación se reproduce: comparar dos rankings de atributos con
  Spearman.
- **Plausibilidad — "coincidencia con el ground-truth de la tipología."** — Se mide comparando la
  explicación contra el patrón verdadero conocido de antemano (en el grafo sintético).
- **Fidelidad — "caída de la predicción al retirar lo importante (Fidelity+)."** — Se mide quitando
  del modelo lo que el explicador marcó como importante, y viendo cuánto cambia (cae) la predicción;
  si de verdad era importante, la predicción debería cambiar mucho.
- **Rendimiento — "PR-AUC como métrica primaria (el F1 en umbral es degenerado bajo desbalance)."** —
  Para medir qué tan bien predice el modelo (no la explicación, sino la clasificación en sí), se usa
  PR-AUC en vez de F1, porque F1 con un umbral fijo deja de ser útil cuando el desbalance es extremo.
- **Recuadro final:** toda esta medición se acompaña de estadística: Kruskal-Wallis para comparar
  factores, Wilcoxon para comparaciones pareadas, e intervalos de confianza por bootstrap (pruebas no
  paramétricas: no asumen normalidad).

## 🔑 Conceptos y técnicas que aparecen

- **Fidelity+ (fidelidad positiva):** el nombre técnico de la métrica de fidelidad usada. "Positiva"
  se refiere a que se mide retirando específicamente lo que el explicador marcó como **importante**
  (a diferencia de una variante que retiraría lo marcado como poco importante).
- **PR-AUC vs. F1:** F1 necesita fijar un umbral de decisión (por ejemplo, "si la probabilidad supera
  0,5, se marca como fraude"), y bajo desbalance extremo ese umbral fijo se vuelve poco informativo
  (casi cualquier umbral da resultados degenerados). PR-AUC resume el comportamiento del modelo a
  través de todos los umbrales posibles, sin depender de fijar uno solo. Ver
  [glosario](README.md#pr-auc-roc-auc-y-precisiónk).
- **No paramétrico:** ver [glosario](README.md#qué-significa-no-paramétrico). Se elige este enfoque
  estadístico porque no hay garantía de que los datos de estabilidad o plausibilidad sigan una
  distribución normal (campana de Gauss).

## 📈 Cómo leer la figura / tabla

No aplica: es una lista de cuatro métricas seguida de un bloque sobre el aparato estadístico, sin
figura ni tabla numérica.

## 🎤 El discurso (como se dice en voz alta)

> Cierro mi bloque con las metricas. La estabilidad la medimos con la correlacion de Spearman entre los
> rankings de atributos, que es nuestra metrica primaria. La plausibilidad, como coincidencia con el
> patron verdadero de la tipologia. La fidelidad, como cuanto cae la prediccion cuando quitamos lo que el
> explicador marco como importante. Y para el rendimiento del clasificador usamos PR-AUC como metrica
> primaria, porque el F1 con umbral fijo se degrada bajo desbalance extremo. *(pausa)* Todo esto se
> acompana de estadistica: Kruskal-Wallis para comparar factores, Wilcoxon para comparaciones pareadas, e
> intervalos de confianza por bootstrap. *(pausa, gira hacia Juan Diego)* Con la metodologia sobre la
> mesa, le paso la palabra a Juan Diego para los resultados.

### Versión ampliada y explicada

*"Cierro mi bloque con las metricas."* — Marca explícitamente el final del bloque de contexto y
metodología, preparando el relevo hacia Juan Diego.

*"La estabilidad la medimos con la correlacion de Spearman entre los rankings de atributos, que es
nuestra metrica primaria."* — Reafirma, por última vez antes de los resultados, cuál es la métrica
central de todo el eje real (Elliptic): Spearman sobre atributos, ya justificada en las láminas 12 y
15 por la dispersión del grafo.

*"La plausibilidad, como coincidencia con el patron verdadero de la tipologia. La fidelidad, como
cuanto cae la prediccion cuando quitamos lo que el explicador marco como importante."* — Resume, en
dos frases muy cortas, las definiciones operativas de plausibilidad y fidelidad, ya presentadas
conceptualmente en la lámina 11.

*"Y para el rendimiento del clasificador usamos PR-AUC como metrica primaria, porque el F1 con umbral
fijo se degrada bajo desbalance extremo."* — Introduce, de pasada, un tema que se va a desarrollar
extensamente en los resultados (láminas 27 y 28): por qué se elige PR-AUC en vez de las métricas más
tradicionales (F1, ROC-AUC).

*"Todo esto se acompana de estadistica: Kruskal-Wallis para comparar factores, Wilcoxon para
comparaciones pareadas, e intervalos de confianza por bootstrap."* — Cierra recordando el aparato
estadístico que va a aparecer una y otra vez en los resultados, para que el jurado ya tenga el
vocabulario a mano cuando aparezcan esos términos en las siguientes láminas.

*"Con la metodologia sobre la mesa, le paso la palabra a Juan Diego para los resultados."* — El
relevo formal: aquí termina el bloque de Alejandro y comienza el de Juan Diego (el "[RELEVO]" del
guion, que ocurre en la página siguiente, el separador de sección "Resultados").

## 🧑‍⚖️ Preguntas de jurado probables

- **"¿Por qué Fidelity+ y no otra variante de fidelidad?"** → Porque mide directamente lo que interesa
  para auditar: si lo que el explicador señala como importante realmente es indispensable para la
  predicción del modelo, verificado al quitarlo y observar el cambio.
- **"¿Por qué elegir pruebas no paramétricas en vez de, por ejemplo, un ANOVA clásico?"** → Porque no
  hay garantía de que los datos de estabilidad, plausibilidad o fidelidad sigan una distribución
  normal, y las pruebas no paramétricas (Kruskal-Wallis, Wilcoxon) son más prudentes en ese escenario,
  sin perder poder para detectar diferencias reales.

## 🧠 En una frase

Cada propiedad se mide con una métrica concreta y verificable —Spearman, coincidencia con el
ground-truth, y caída de la predicción—, respaldadas todas por pruebas estadísticas no paramétricas.
