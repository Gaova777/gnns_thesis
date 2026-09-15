# Lámina 22 — Robustez: qué sostiene esa partición

> Bloque: Resultados
> Voz: Juan Diego · Página PDF 27 (22/40) · Tiempo objetivo 55 s

## 🎯 Objetivo de la lámina

Demostrar que la partición en dos grupos (lámina 21) no es una observación frágil, sino un resultado
que resiste varias pruebas estadísticas distintas, una réplica completa del experimento, y la
extensión a otros dos explicadores. También introduce una segunda contribución: la diferencia entre
"reproducir los pesos exactos de un modelo" y "reproducir las conclusiones".

## 📋 Qué dice, item por item

- **Tabla 1 — Contrastes no paramétricos de la partición (Elliptic, 180 modelos):**
  - Global (4 arquitecturas), Kruskal-Wallis: p = 2,8×10⁻⁵.
  - Grupo alto vs. bajo, Mann-Whitney: p = 1,4×10⁻⁶.
  - Dentro del grupo alto, Kruskal-Wallis: p = 0,19.
  - Dentro del grupo bajo, Kruskal-Wallis: p = 0,08.
- **Tamaño de efecto η² = 0,13 (grande)**, el mismo estadístico que reporta la tesis en el eje
  sintético.
- **IC 95% bootstrap:** se solapan **dentro** de cada grupo, apenas se tocan **entre** grupos.
- **Sobrevive la corrección de Holm** por comparaciones múltiples.
- **Nota:** los p dentro de grupo (0,19; 0,08) son no pareados; el Wilcoxon pareado de la lámina
  anterior (0,17; 0,10) coincide en la misma conclusión: ninguna vía rechaza la igualdad dentro del
  grupo.
- **Recuadro "Reproducibilidad precisa":** al reentrenar la matriz completa, los **pesos** del modelo
  no se reproducen exactamente igual (por la forma en que la GPU suma internamente), pero las
  **conclusiones sí**: 25 de 60 configuraciones pasan el filtro de calidad (frente a 23 de 60
  originalmente), y se obtiene exactamente la misma partición.

## 🔑 Conceptos y técnicas que aparecen

- **Kruskal-Wallis y Mann-Whitney:** ver [glosario](README.md#kruskal-wallis) y
  [glosario](README.md#mann-whitney). Aquí se usan de forma complementaria: Kruskal-Wallis para
  comparar las cuatro arquitecturas a la vez (o dos subgrupos entre sí, dentro de cada bloque), y
  Mann-Whitney específicamente para comparar los dos grupos (alto vs. bajo) como bloques.
- **Corrección de Holm:** ver [glosario](README.md#corrección-de-holm). Que la separación
  "sobreviva" a esta corrección es una prueba adicional de solidez, porque descarta que el resultado
  sea un efecto de haber hecho muchas comparaciones a la vez.
- **Reproducibilidad de pesos vs. de conclusiones:** un aporte metodológico explícito de la tesis.
  Los **pesos** son los números internos exactos que aprende el modelo durante el entrenamiento; que
  no sean idénticos entre corridas es normal en GPU (las operaciones de suma no siempre se ejecutan en
  el mismo orden). Lo que sí importa, y lo que sí se comprueba, es que la **conclusión** (la partición
  en dos grupos) se mantiene sin importar esa variación en los pesos exactos.
- **Filtro de calidad:** el criterio que decide si una configuración de modelo entrenado es lo
  suficientemente buena como para incluirse en el análisis (por ejemplo, que su rendimiento supere un
  umbral mínimo).

## 📈 Cómo leer la figura / tabla

La Tabla 1 se lee de arriba hacia abajo, comparando dos bloques de filas: las dos comparaciones "entre
bloques" (global y grupo alto vs. bajo) muestran p-valores extremadamente pequeños (evidencia fuerte
de diferencia real), mientras que las dos comparaciones "dentro de bloque" (dentro del grupo alto,
dentro del grupo bajo) muestran p-valores mucho más altos (0,19 y 0,08 — compatibles con el azar). El
contraste entre ambos bloques de filas es, en sí mismo, la prueba visual de la partición.

## 🎤 El discurso (como se dice en voz alta)

> Me detengo un momento en que sostiene esa particion, porque es la diferencia entre una observacion y
> un resultado. *(pausa)* Reentrenamos la matriz completa de sesenta configuraciones tres veces, con tres
> semillas distintas, ciento ochenta modelos, y en cada una corrimos el procedimiento entero incluida su
> propia busqueda de hiperparametros. Sobre esos ciento ochenta modelos, la pregunta es simple: las diferencias que vemos, ¿son reales o
> podrian ser casualidad? Una prueba estadistica estandar responde que, si el grupo alto y el bajo fueran
> en realidad iguales, una diferencia como la que medimos apareceria menos de dos veces en un millon, asi
> que la tratamos como real. En cambio, dentro de cada grupo las diferencias son compatibles con el azar. Dicho de otro modo, toda la variacion esta entre
> grupos, ninguna dentro. Los margenes de error, calculados repitiendo el analisis muchas veces con
> remuestreo, cuentan lo mismo: se solapan dentro de cada grupo y apenas se tocan entre ellos. Y por si
> preguntan si esto sale de haber hecho muchas comparaciones a la vez, aplicamos el ajuste habitual para
> ese caso y la separacion aguanta. *(pausa)* Y esa replicacion no se limito a GNNExplainer: tambien
> corrimos GNNShap y PGExplainer en las tres semillas para GCN y GraphSAGE, y confirman su patron,
> GNNShap muy estable pero sin distinguir arquitecturas y PGExplainer degenerado. En GAT y TAGCN esa
> extension quedo en la semilla de referencia por un motivo operativo: sus modelos de las otras dos
> semillas, que si forman parte de los ciento ochenta, no estaban en el equipo donde la corrimos, y
> reentrenarlos alli agoto la memoria de la tarjeta. Aun asi, en esa semilla muestran exactamente el
> mismo patron.
> *(pausa)* Hay un segundo hallazgo aqui que tambien cuenta como contribucion. Al reentrenar
> descubrimos que el pipeline no es reproducible bit a bit: las operaciones de agregacion sobre la
> tarjeta grafica suman en un orden que no esta determinado, asi que los pesos nunca salen identicos.
> Lo que si se reproduce son las conclusiones, veinticinco configuraciones sobre el filtro de calidad
> frente a veintitres, y exactamente la misma particion. Distinguir la reproducibilidad de los pesos de
> la reproducibilidad de las conclusiones es algo que la literatura rara vez explicita, y creemos que
> deberia hacerlo.

### Versión ampliada y explicada

*"Me detengo un momento en que sostiene esa particion, porque es la diferencia entre una observacion y
un resultado."* — Marca la distinción clave de toda esta lámina: una observación es algo que se ve una
vez; un resultado es algo que resiste el escrutinio estadístico repetido.

*"Reentrenamos la matriz completa de sesenta configuraciones tres veces, con tres semillas distintas,
ciento ochenta modelos, y en cada una corrimos el procedimiento entero incluida su propia busqueda de
hiperparametros."* — Precisa el nivel de esfuerzo de la réplica: no se reutilizan los mismos
hiperparámetros, se rehace la búsqueda completa cada vez, para medir la variabilidad real de todo el
proceso.

*"Una prueba estadistica estandar responde que, si el grupo alto y el bajo fueran en realidad iguales,
una diferencia como la que medimos apareceria menos de dos veces en un millon, asi que la tratamos
como real."* — Traduce el p-valor de Mann-Whitney (1,4×10⁻⁶) a una probabilidad intuitiva.

*"En cambio, dentro de cada grupo las diferencias son compatibles con el azar."* — El contraste
directo: fuera del grupo, casi imposible que sea azar; dentro del grupo, perfectamente compatible con
azar.

*"Y por si preguntan si esto sale de haber hecho muchas comparaciones a la vez, aplicamos el ajuste
habitual para ese caso y la separacion aguanta."* — Anticipa una objeción técnica común (el problema
de comparaciones múltiples) y responde que ya se controló con la corrección de Holm.

*"Y esa replicacion no se limito a GNNExplainer: tambien corrimos GNNShap y PGExplainer en las tres
semillas para GCN y GraphSAGE, y confirman su patron..."* — Extiende la robustez más allá de un solo
explicador: los otros dos explicadores, replicados donde fue posible, muestran el mismo comportamiento
que ya se había visto (GNNShap muy estable pero sin discriminar, PGExplainer degenerado).

*"En GAT y TAGCN esa extension quedo en la semilla de referencia por un motivo operativo..."* — Es
importante notar la honestidad aquí: se explica con transparencia por qué la extensión a tres semillas
no se hizo para GAT y TAGCN (limitación de memoria en el equipo disponible), sin ocultar la limitación
ni exagerar la cobertura del resultado.

*"Al reentrenar descubrimos que el pipeline no es reproducible bit a bit: las operaciones de
agregacion sobre la tarjeta grafica suman en un orden que no esta determinado, asi que los pesos
nunca salen identicos."* — Explica el fenómeno técnico: las GPU, al sumar muchos números en paralelo,
no siempre lo hacen en el mismo orden exacto, lo que introduce diferencias mínimas pero reales en los
pesos finales del modelo.

*"Lo que si se reproduce son las conclusiones, veinticinco configuraciones sobre el filtro de calidad
frente a veintitres, y exactamente la misma particion."* — El punto que salva la credibilidad del
pipeline: aunque los pesos cambien, lo que importa (la partición en dos grupos) se mantiene igual.

*"Distinguir la reproducibilidad de los pesos de la reproducibilidad de las conclusiones es algo que
la literatura rara vez explicita, y creemos que deberia hacerlo."* — Cierra presentando esta
distinción como un aporte metodológico en sí mismo, útil más allá de esta tesis en particular.

## 🧑‍⚖️ Preguntas de jurado probables

- **"¿Por qué GNNShap y PGExplainer no tienen la misma cobertura de tres semillas que GNNExplainer en
  las cuatro arquitecturas?"** → Por una razón puramente operativa: los modelos de las semillas
  adicionales para GAT y TAGCN no estaban disponibles en el equipo donde se corrió esa extensión, y
  reentrenarlos allí agotó la memoria de la GPU; no se debe a una limitación de las cabezas de atención
  de GAT ni a que "no cupieran" por diseño.
- **"¿Qué significa exactamente que el pipeline 'no sea reproducible bit a bit'?"** → Que si se
  reentrena el mismo modelo dos veces, con la misma semilla, en la misma GPU, los pesos finales no
  serán exactamente idénticos número por número, por el orden no determinista de ciertas sumas en
  paralelo; pero las conclusiones que se extraen de esos modelos sí son consistentes.

## 🧠 En una frase

La partición en dos grupos no es una observación frágil: resiste varias pruebas estadísticas, la
corrección por comparaciones múltiples, una réplica completa con nuevas semillas, y se extiende a los
otros dos explicadores donde fue posible.
