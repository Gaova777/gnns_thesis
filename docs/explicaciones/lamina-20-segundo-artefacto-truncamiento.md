# Lámina 20 — El segundo artefacto: el truncamiento de la métrica

> Bloque: Resultados
> Voz: Juan Diego · Página PDF 25 (20/40) · Tiempo objetivo 60 s

## 🎯 Objetivo de la lámina

Explicar en detalle técnico el segundo artefacto (el más instructivo de los dos, según el propio
guion): cómo un truncamiento en la métrica de Spearman mutilaba los rankings de atributos, y por qué
ese error **no afectaba a todas las arquitecturas por igual**, fabricando un liderazgo falso.

## 📋 Qué dice, item por item

- **"La métrica dimensionaba el vector de rangos por top_k en vez de por el número real de
  features."** — El código calculaba el ranking de importancia usando un parámetro fijo (`top_k=20`)
  como si ese fuera el total de atributos, en vez de usar el número real (166).
- **"Con top_k=20 sobre 166 features, toda feature con índice ≥ 20 se descartaba: sobrevivían 2 o 3 y
  el resto quedaba empatado en cero."** — El efecto práctico: de 166 atributos, prácticamente todos se
  perdían en el cálculo, y solo un puñado (2 o 3) sobrevivía con un valor real; el resto se trataba
  como si todos tuvieran exactamente la misma importancia (cero).
- **"El truncamiento no penaliza por igual: castiga más a las arquitecturas cuya importancia se
  reparte sobre muchas features."** — El punto más importante: el sesgo no era parejo. Si una
  arquitectura tiende a repartir su atención entre muchos atributos, se ve mucho más perjudicada por
  el truncamiento que una arquitectura que concentra su importancia en pocos.
- **Nota final:** por eso el truncamiento deprimía a GAT (que ganaba +0,24 al corregirlo) y a TAGCN
  (+0,28) mucho más que a GraphSAGE (+0,10), y fabricaba un liderazgo falso para GraphSAGE.
- **Recuadro "Los dos artefactos, juntos":** repite de forma resumida el fallo de memoria (favorecía a
  GAT) y el truncamiento (favorecía a GraphSAGE), remarcando que cada uno por separado ya bastaba
  para una conclusión falsa.

## 🔑 Conceptos y técnicas que aparecen

- **`top_k`:** un parámetro de configuración que le dice al código "considera solo los k elementos más
  importantes". El error fue usar este parámetro para **dimensionar** el cálculo de correlación (es
  decir, tratarlo como si fuera el tamaño total del vector), en vez de usarlo solo para mostrar un
  resumen, mientras el cálculo de fondo debía usar el número real de atributos (166).
- **Rangos empatados en cero:** cuando muchos elementos quedan fuera del cálculo por el truncamiento,
  el método los trata como si todos tuvieran el mismo puesto (un empate), lo que distorsiona
  severamente cualquier comparación basada en ese ranking.
- **Sesgo asimétrico:** un error que no afecta a todos los grupos por igual, sino que perjudica más a
  unos que a otros. Aquí, mientras más "repartida" esté la importancia de una arquitectura entre
  muchos atributos, más la perjudicaba este bug específico —y esa asimetría es justo lo que fabricó
  un ganador falso (GraphSAGE) en la versión anterior de la tesis.

## 📈 Cómo leer la figura / tabla

No aplica: esta lámina desarrolla el texto en dos columnas (el mecanismo del bug a la izquierda, el
resumen de los dos artefactos juntos a la derecha), sin figura ni tabla numérica.

## 🎤 El discurso (como se dice en voz alta)

> Me detengo en el segundo artefacto porque es el mas instructivo de los dos. *(pausa)* La metrica de
> Spearman dimensionaba el vector de rangos por el parametro de truncamiento, que estaba fijado en veinte,
> en lugar de por el numero real de atributos, que son ciento sesenta y seis. El efecto es que toda
> feature con indice mayor que veinte se descartaba en silencio. De las veinte del top sobrevivian dos o
> tres, y todo lo demas quedaba empatado en cero. Estabamos comparando rankings mutilados. *(pausa)* Y
> aqui esta lo importante, que no es el bug sino su asimetria: el truncamiento no castiga por igual a
> todas las arquitecturas. Castiga mas a aquellas cuya importancia se reparte sobre muchos atributos. Por
> eso al corregirlo GAT sube veinticuatro centesimas y TAGCN veintiocho, mientras que GraphSAGE sube solo
> diez. El liderazgo de GraphSAGE que reportaba la version anterior no era un hallazgo, era el perfil de
> sensibilidad de la metrica rota. *(pausa)* Sumado al fallo de memoria de la lamina anterior, tenemos dos
> defectos del protocolo de medida que apuntaban en direcciones opuestas y cada uno bastaba, por si solo,
> para una conclusion comparativa falsa.

### Versión ampliada y explicada

*"Me detengo en el segundo artefacto porque es el mas instructivo de los dos."* — Justifica dedicarle
una lámina completa: este bug enseña algo importante sobre cómo un error técnico sutil puede
fabricar una conclusión completamente equivocada.

*"La metrica de Spearman dimensionaba el vector de rangos por el parametro de truncamiento, que estaba
fijado en veinte, en lugar de por el numero real de atributos, que son ciento sesenta y seis."* —
Explica la causa raíz con precisión técnica: confundir un parámetro de "cuántos mostrar" con "cuántos
existen realmente" al calcular la correlación.

*"El efecto es que toda feature con indice mayor que veinte se descartaba en silencio. De las veinte
del top sobrevivian dos o tres, y todo lo demas quedaba empatado en cero. Estabamos comparando
rankings mutilados."* — Cuantifica el daño: de 166 atributos posibles, el cálculo terminaba usando
efectivamente solo 2 o 3, con el resto tratado como si no importaran en absoluto.

*"Y aqui esta lo importante, que no es el bug sino su asimetria: el truncamiento no castiga por igual
a todas las arquitecturas. Castiga mas a aquellas cuya importancia se reparte sobre muchos
atributos."* — Este es el giro conceptual clave de la lámina: un bug que afecta a todos por igual solo
añadiría ruido; un bug que afecta más a unos que a otros puede literalmente **cambiar quién parece
ganar**.

*"Por eso al corregirlo GAT sube veinticuatro centesimas y TAGCN veintiocho, mientras que GraphSAGE
sube solo diez."* — Da la evidencia numérica de esa asimetría: al arreglar el bug, GAT y TAGCN suben
mucho más que GraphSAGE, lo que confirma que el bug los perjudicaba más a ellos.

*"El liderazgo de GraphSAGE que reportaba la version anterior no era un hallazgo, era el perfil de
sensibilidad de la metrica rota."* — Es la frase que retracta explícitamente la afirmación anterior de
la tesis, con total transparencia: lo que parecía un descubrimiento científico era, en realidad, un
artefacto de un error de programación.

*"Sumado al fallo de memoria de la lamina anterior, tenemos dos defectos del protocolo de medida que
apuntaban en direcciones opuestas y cada uno bastaba, por si solo, para una conclusion comparativa
falsa."* — Cierra conectando con el primer artefacto: los dos apuntaban en direcciones **opuestas**
(uno favorecía a GAT, el otro a GraphSAGE), lo cual hace aún más contundente el mensaje de que la
medición sin corregir era, sencillamente, poco confiable.

## 🧑‍⚖️ Preguntas de jurado probables

- **"¿Por qué se usó `top_k=20` en primer lugar?"** → Era un parámetro pensado originalmente solo para
  **mostrar** un resumen visual de las top-20 features más importantes; el error fue que ese mismo
  valor terminó usándose también para dimensionar internamente el cálculo de la correlación, algo que
  no era la intención original del código.
- **"¿Por qué ahora lideran GAT y GCN y no GraphSAGE?"** → El truncamiento penalizaba más a las
  arquitecturas que reparten su importancia entre muchos atributos; al corregirlo, ese sesgo
  desaparece y emerge la partición real en dos grupos (ver lámina 21), donde GAT y GCN quedan en el
  grupo alto.

## 🧠 En una frase

El bug no solo mutilaba los rankings: los mutilaba de forma desigual entre arquitecturas, y esa
asimetría fue lo que fabricó un "liderazgo" de GraphSAGE que, al corregirse, se disuelve.
