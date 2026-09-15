# Lámina 24 — Resultado 2: disociación plausibilidad / fidelidad

> Bloque: Resultados
> Voz: Juan Diego · Página PDF 29 (24/40) · Tiempo objetivo 85 s

## 🎯 Objetivo de la lámina

Presentar el hallazgo central sobre los explicadores: PGExplainer y GNNExplainer se reparten las dos
propiedades de forma opuesta. El explicador que mejor **recupera el patrón real** (plausibilidad) no
es el que mejor **refleja el mecanismo del modelo** (fidelidad). Esto demuestra, con datos concretos,
la independencia que la lámina 11 ya anticipaba conceptualmente.

## 📋 Qué dice, item por item

- **"Por un lado, PGExplainer domina la plausibilidad: 0,80 vs. 0,50, con 0,40 de azar (Wilcoxon
  p≈2,6×10⁻³⁵)."** — PGExplainer alcanza 0,80 de plausibilidad de aristas, muy por encima del 0,50 de
  GNNExplainer, y ambos se comparan contra un azar de 0,40 (lo que obtendría un explicador que
  eligiera aristas al azar con el mismo protocolo).
- **"Por otro, colapsa en fidelidad: 0,11 vs. 0,56 de GNNExplainer."** — El mismo PGExplainer que
  domina en plausibilidad se desploma en fidelidad, muy por debajo de GNNExplainer.
- **"El explicador más 'plausible' no es el más fiel."** — La conclusión central, formulada de forma
  directa.
- **Nota al pie:** la figura compara solo los dos explicadores que producen máscara de aristas
  (GNNExplainer y PGExplainer). GNNShap no la produce por diseño: su plausibilidad es de features, con
  otra escala y otra línea base, y va en la lámina de respaldo.
- **Figura 7:** un gráfico que muestra la plausibilidad de aristas frente a Fidelity+, con intervalos
  de confianza al 95% y la línea de azar de plausibilidad marcada en 0,40.

## 🔑 Conceptos y técnicas que aparecen

- **Disociación:** que dos cosas que uno esperaría que fueran de la mano (aquí, plausibilidad y
  fidelidad) resultan, en los datos, estar en tensión: mejorar una no mejora la otra, y en este caso
  incluso van en direcciones opuestas.
- **Línea base de azar:** el valor que se obtendría si el explicador eligiera aristas completamente al
  azar, siguiendo el mismo protocolo de medición. Sirve como punto de referencia para saber si un
  resultado es realmente bueno o simplemente "no tan malo". Aquí, 0,40 para plausibilidad de aristas
  (calculado sobre el mismo subgrafo de dos saltos y el mismo top-k balanceado que usan los
  explicadores reales).
- **Wilcoxon pareado, p≈2,6×10⁻³⁵:** un p-valor extraordinariamente pequeño, que confirma que la
  ventaja de PGExplainer sobre GNNExplainer en plausibilidad no es casualidad: en el guion se explica
  que PGExplainer gana en 92 de cada 100 comparaciones directas y emparejadas, algo que sería
  extremadamente improbable si los dos explicadores fueran igual de buenos en esta métrica.
- **Por qué GNNShap no aparece en esta figura:** GNNShap no genera una máscara de aristas por diseño
  (mide importancia de atributos, no de conexiones). Su plausibilidad se mide sobre features, con su
  propia línea base de azar (0,075, no 0,40). Ponerlo en la misma figura, contra la línea de 0,40,
  haría que pareciera peor que el azar cuando en realidad lo duplica en su propia escala. Ver
  [respaldo de disociación](respaldo-disociacion-valores-exactos.md).

## 📈 Cómo leer la figura / tabla

La Figura 7 muestra, para GNNExplainer y PGExplainer, dos valores por cada uno: su plausibilidad de
aristas (con intervalo de confianza) y su Fidelity+ (con intervalo de confianza), además de una línea
horizontal marcando el azar de plausibilidad (0,40). La lectura clave es el cruce: PGExplainer está
más arriba que GNNExplainer en plausibilidad, pero **más abajo** en fidelidad. Ese cruce visual es la
disociación en sí misma: no hay un explicador que domine ambas métricas a la vez.

## 🎤 El discurso (como se dice en voz alta)

> Ahora el hallazgo central sobre los explicadores, y es un resultado con dos caras. *(pausa)* Por un
> lado, PGExplainer es claramente el que mejor recupera el patron real: su plausibilidad de aristas es de
> cero coma ochenta, frente a cero coma cincuenta de GNNExplainer, y la ventaja es sistematica: gana en
> noventa y dos de cada cien comparaciones pareadas, algo que no ocurriria en la practica si los dos
> explicadores fueran igual de buenos. Y
> para que estas cifras signifiquen algo, las contrastamos con el azar: un explicador que eligiera las
> aristas al azar, con el mismo protocolo, obtiene cero coma cuarenta. PGExplainer duplica ese nivel,
> mientras que GNNExplainer apenas lo supera, lo que ya anticipa la disociacion que viene: el fuerte de
> GNNExplainer no es recuperar el patron, sino la fidelidad al modelo. Es
> decir, si el objetivo es senalar el patron de lavado, PGExplainer gana sin discusion. *(pausa)* Pero
> por otro lado, ese mismo PGExplainer colapsa en fidelidad: cuando medimos cuanto depende la prediccion
> del modelo de las aristas que PGExplainer marca, el valor cae a cero coma once, frente a cero coma
> cincuenta y seis de GNNExplainer. La lectura va contra la intuicion: el explicador mas plausible
> no es el mas fiel. PGExplainer recupera las aristas que definen el patron que un humano reconoce, pero
> GNNExplainer recupera las aristas que el modelo realmente usa, y esos dos conjuntos no coinciden.
> *(pausa)* Una precision sobre la figura: compara los dos explicadores que producen mascara de aristas,
> que son los unicos comparables contra esa linea base de cero coma cuarenta. GNNShap no produce mascara
> de aristas por diseno, su plausibilidad es de features y vive en otra escala, con su propia linea base
> de cero coma cero siete cinco. Sus valores estan en la lamina de respaldo con la metrica etiquetada. Lo
> que si cabe decir de GNNShap aqui es que es el mas estable internamente de los tres, el mas consistente
> entre ejecuciones, aunque no lidere ni plausibilidad ni fidelidad. Cada explicador, entonces, tiene su
> fortaleza en una dimension distinta. Esta disociacion solo se puede exhibir cuando tienes un patron
> verdadero contra el cual medir, y por eso el eje sintetico era indispensable.

### Versión ampliada y explicada

*"Ahora el hallazgo central sobre los explicadores, y es un resultado con dos caras."* — Anuncia desde
el inicio que se trata de un resultado con contraste interno, no de una simple lista de números.

*"PGExplainer es claramente el que mejor recupera el patron real: su plausibilidad de aristas es de
cero coma ochenta, frente a cero coma cincuenta de GNNExplainer, y la ventaja es sistematica: gana en
noventa y dos de cada cien comparaciones pareadas..."* — Establece la primera mitad del contraste con
solidez estadística: no es una diferencia casual, es una ventaja que se repite de forma consistente.

*"Y para que estas cifras signifiquen algo, las contrastamos con el azar: un explicador que eligiera
las aristas al azar, con el mismo protocolo, obtiene cero coma cuarenta."* — Ancla los números en un
punto de referencia: sin saber cuánto da el azar, "0,80" o "0,50" no se pueden interpretar. Contra el
azar de 0,40, PGExplainer lo duplica y GNNExplainer apenas lo supera.

*"...lo que ya anticipa la disociacion que viene: el fuerte de GNNExplainer no es recuperar el patron,
sino la fidelidad al modelo."* — Un adelanto deliberado dentro de la misma explicación: el hecho de
que GNNExplainer apenas supere el azar en plausibilidad ya sugiere que su fortaleza está en otra
dimensión.

*"Pero por otro lado, ese mismo PGExplainer colapsa en fidelidad: cuando medimos cuanto depende la
prediccion del modelo de las aristas que PGExplainer marca, el valor cae a cero coma once, frente a
cero coma cincuenta y seis de GNNExplainer."* — Presenta la segunda mitad del contraste, con el mismo
protagonista (PGExplainer) ahora en el lado perdedor.

*"La lectura va contra la intuicion: el explicador mas plausible no es el mas fiel."* — Formula la
tesis central de la lámina de forma directa y memorable.

*"PGExplainer recupera las aristas que definen el patron que un humano reconoce, pero GNNExplainer
recupera las aristas que el modelo realmente usa, y esos dos conjuntos no coinciden."* — Explica **por
qué** puede pasar esto: el patrón que un experto humano reconocería y el mecanismo interno real del
modelo no tienen por qué ser lo mismo, y aquí los datos muestran que efectivamente no lo son.

*"Una precision sobre la figura: compara los dos explicadores que producen mascara de aristas, que son
los unicos comparables contra esa linea base de cero coma cuarenta."* — Aclara proactivamente por qué
GNNShap no aparece: no es un descuido, es una decisión metodológica para no mezclar escalas distintas.

*"Lo que si cabe decir de GNNShap aqui es que es el mas estable internamente de los tres, el mas
consistente entre ejecuciones, aunque no lidere ni plausibilidad ni fidelidad."* — Le da a GNNShap su
propio lugar en el mapa: no gana en las dos dimensiones de esta lámina, pero tiene su propia fortaleza
(consistencia interna), que se retoma en la matriz de recomendación (lámina 32).

*"Esta disociacion solo se puede exhibir cuando tienes un patron verdadero contra el cual medir, y por
eso el eje sintetico era indispensable."* — Cierra conectando este resultado con la justificación
metodológica del eje sintético (lámina 16): sin ground-truth, esta disociación sería, literalmente,
imposible de medir.

## 🧑‍⚖️ Preguntas de jurado probables

- **"Si PGExplainer es tan plausible, ¿por qué no confiar en él para todo?"** → Porque plausibilidad y
  fidelidad son propiedades distintas: si el objetivo es auditar qué usó realmente el modelo,
  GNNExplainer es la mejor opción a pesar de ser menos plausible; la elección depende del propósito
  (ver [matriz de recomendación](lamina-32-matriz-recomendacion.md)).
- **"¿Por qué GNNShap no está en esta figura?"** → No produce máscara de aristas por diseño; su
  plausibilidad se mide sobre atributos, con otra escala y otra línea base de azar (0,075), y
  compararlo contra la línea de 0,40 de aristas lo haría parecer, injustamente, peor que el azar.
- **"El 0,80 de PGExplainer, ¿es mucho comparado con qué?"** → Comparado con el azar (0,40, calculado
  con el mismo protocolo), PGExplainer lo duplica; GNNExplainer (0,50) apenas lo supera, lo que
  refuerza la disociación.

## 🧠 En una frase

PGExplainer recupera mejor el patrón real de lavado, pero GNNExplainer refleja mejor lo que el modelo
realmente usó: el explicador más plausible no es el más fiel.
