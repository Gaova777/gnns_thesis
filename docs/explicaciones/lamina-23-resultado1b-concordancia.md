# Lámina 23 — Resultado 1b: concordancia entre regímenes

> Bloque: Resultados
> Voz: Juan Diego · Página PDF 28 (23/40) · Tiempo objetivo 70 s

## 🎯 Objetivo de la lámina

Contar uno de los resultados favoritos del propio equipo, porque nació de un error corregido: lo que
antes parecía ser una "inversión" entre los dos ejes (Elliptic disperso y el sintético denso) era, en
realidad, un artefacto de la métrica rota. Corregida, la historia real es que **ambos regímenes
concuerdan**.

## 📋 Qué dice, item por item

- **"La inversión por densidad era un artefacto de la métrica."** — Se retracta explícitamente una
  afirmación de una versión anterior de la tesis: no existe tal inversión.
- **"Corregida, Elliptic (disperso) concuerda con el sintético (denso)."** — Con la métrica ya
  arreglada, las mismas arquitecturas que son estables en el grafo denso también lo son en el
  disperso.
- **"El eje sintético separa los mismos dos grupos, sobre datos y pipeline independientes."** — No es
  el mismo código ni los mismos datos: es una confirmación completamente independiente de la misma
  estructura de dos grupos.
- **Números destacados:** la correlación de rangos entre los dos regímenes pasa de **−0,20** (con la
  métrica que tenía el bug) a **+0,80** (con la métrica corregida).
- **Nota adicional:** en el sintético, GCN (0,97) y GAT (0,96) quedan por encima de GraphSAGE (0,89) y
  TAGCN (0,89), replicando la misma partición en dos grupos observada en Elliptic.
- **Figura 6:** un gráfico que compara el orden de estabilidad entre Elliptic (disperso) y el
  sintético (denso).

## 🔑 Conceptos y técnicas que aparecen

- **"Inversión por densidad" (retractada):** una afirmación anterior de la tesis que decía que el
  orden de estabilidad entre arquitecturas se invertía al pasar de un grafo disperso a uno denso. Ya
  no es válida: era producto del bug del truncamiento, y **no debe repetirse**.
- **Correlación de rangos entre regímenes:** una forma de medir si el orden de las cuatro
  arquitecturas (de más a menos estable) es parecido entre los dos ejes (Elliptic y el sintético). Un
  valor cercano a +1 significa que el orden es casi idéntico en ambos; cercano a 0, que no hay
  relación; negativo, que el orden se invierte.
- **Régimen de densidad:** se refiere a qué tan "lleno" de conexiones está un grafo. Elliptic es un
  régimen **disperso** (vecindarios de ~2 nodos); el grafo sintético es un régimen **denso** (más
  conexiones por nodo, gracias a la simetrización explicada en la lámina 17).

## 📈 Cómo leer la figura / tabla

La Figura 6 compara, lado a lado, el orden de estabilidad de las cuatro arquitecturas en Elliptic
(disperso) y en el sintético (denso). La lectura correcta es fijarse en si el orden relativo se
mantiene: las arquitecturas que aparecen arriba en un régimen también deberían aparecer arriba en el
otro, si de verdad "concuerdan". Ese es exactamente el patrón que muestra la figura, una vez corregida
la métrica.

## 🎤 El discurso (como se dice en voz alta)

> Este resultado es uno de los que mas me gustan, porque nacio de un error corregido. En una version
> previa creiamos haber encontrado una "inversion por densidad": que el orden de estabilidad entre
> arquitecturas se daba vuelta al pasar del grafo disperso de Elliptic al grafo denso sintetico. *(pausa)*
> Cuando corregimos el bug de la metrica, esa inversion desaparecio. Lo que en realidad ocurre es lo
> contrario: los dos regimenes concuerdan. Las mismas arquitecturas que son estables en el grafo denso
> lo son en el disperso. Lo cuantificamos con la correlacion de rangos entre ambos regimenes, que pasa de
> menos cero coma veinte con la metrica defectuosa, a mas cero coma ochenta con la metrica corregida.
> *(pausa)* Y esto le da peso a la tesis, porque significa que datos reales y datos
> sinteticos cuentan la misma historia. La coherencia entre los dos ejes es lo que le da solidez a todo
> el diseno.

### Versión ampliada y explicada

*"Este resultado es uno de los que mas me gustan, porque nacio de un error corregido."* — Un tono
personal y honesto: el equipo presenta con entusiasmo un resultado que surgió, precisamente, de
corregir su propio error anterior.

*"En una version previa creiamos haber encontrado una 'inversion por densidad': que el orden de
estabilidad entre arquitecturas se daba vuelta al pasar del grafo disperso de Elliptic al grafo denso
sintetico."* — Explica con claridad qué se creía antes, para que la corrección posterior tenga
contexto.

*"Cuando corregimos el bug de la metrica, esa inversion desaparecio. Lo que en realidad ocurre es lo
contrario: los dos regimenes concuerdan."* — El giro central de la lámina: no solo desaparece el
hallazgo falso, sino que aparece uno opuesto y más sólido en su lugar.

*"Las mismas arquitecturas que son estables en el grafo denso lo son en el disperso."* — Traduce la
concordancia a lenguaje simple: no importa si el grafo es disperso o denso, GAT y GCN siguen siendo
las más estables en ambos casos.

*"Lo cuantificamos con la correlacion de rangos entre ambos regimenes, que pasa de menos cero coma
veinte con la metrica defectuosa, a mas cero coma ochenta con la metrica corregida."* — Da la
evidencia numérica exacta del cambio: de una correlación levemente negativa (que sugería inversión) a
una fuertemente positiva (que confirma concordancia).

*"Y esto le da peso a la tesis, porque significa que datos reales y datos sinteticos cuentan la misma
historia."* — Explica por qué este resultado importa más allá de sí mismo: es la evidencia de que el
grafo sintético (una construcción artificial) no está desconectado de la realidad, sino que refleja el
mismo patrón que se ve en datos reales.

*"La coherencia entre los dos ejes es lo que le da solidez a todo el diseno."* — Cierra conectando este
resultado con la decisión metodológica más importante de la tesis (el diseño de dos ejes, lámina 13):
esta concordancia es la prueba de que esa decisión fue acertada.

## 🧑‍⚖️ Preguntas de jurado probables

- **"¿Cómo puede estar tan segura la tesis de que esta concordancia no es también un artefacto?"** →
  Porque se calcula sobre datos y un pipeline de cómputo independientes entre los dos ejes, y la
  corrección de un bug conocido y documentado (el truncamiento) es justamente lo que hizo aparecer
  esta concordancia, en vez de desaparecerla; además, es coherente con la partición en dos grupos ya
  confirmada de forma robusta dentro de cada eje por separado.
- **"¿Qué tan fuerte es +0,80 como correlación de rangos?"** → Es una correlación alta, cercana al
  máximo posible (+1), lo que indica que el orden de las cuatro arquitecturas es prácticamente el
  mismo en ambos regímenes de densidad.

## 🧠 En una frase

Lo que antes parecía una inversión entre los dos regímenes de densidad era un artefacto de la métrica
rota; corregida, los dos ejes cuentan exactamente la misma historia (correlación +0,80).
