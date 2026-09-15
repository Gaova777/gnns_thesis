# Lámina 19 — Contribución: dos artefactos de evaluación

> Bloque: Resultados
> Voz: Juan Diego · Página PDF 24 (19/40) · Tiempo objetivo 85 s

## 🎯 Objetivo de la lámina

Abrir el bloque de resultados con una contribución inesperada: antes de poder confiar en cualquier
comparación entre arquitecturas, hubo que descubrir y corregir **dos defectos en el protocolo de
medición** (no en el método científico en sí), cada uno capaz por sí solo de invertir la conclusión.
Es la base sobre la que se apoyan todos los resultados que vienen después.

## 📋 Qué dice, item por item

- **"1 · Fallo de memoria silencioso"** — Al calcular las explicaciones sobre el grafo completo, 13
  configuraciones de GAT fallaban **en silencio** (sin ningún mensaje de error visible). Esto
  favorecía a GAT de forma artificial.
- **"2 · Truncamiento del Spearman"** — La métrica descartaba atributos que quedaban fuera de un
  umbral fijo, produciendo rankings mutilados. Esto favorecía a GraphSAGE.
- **Frase destacada:** "Cada uno bastaba para una conclusión comparativa falsa." La estabilidad
  depende del protocolo de medición, no solo del método.
- **Nota adicional:** además, se reportan dos *bugs* concretos en el PGExplainer de la librería PyG
  (PyTorch Geometric), la biblioteca de software que se usó para implementar los modelos.
- **Cita al pie:** en línea con la advertencia de Kosan y colaboradores (2023) sobre la sensibilidad
  al protocolo de evaluación.

## 🔑 Conceptos y técnicas que aparecen

- **Fallo silencioso (silent failure):** un error de programación que, en vez de detener el programa
  con un mensaje visible, deja pasar el problema sin avisar, produciendo resultados incompletos o
  incorrectos que parecen normales a simple vista. Aquí, el problema era **quedarse sin memoria** al
  procesar el grafo completo (out of memory, OOM).
- **Truncamiento de una métrica:** cortar o limitar artificialmente el tamaño de los datos que entran
  a un cálculo (aquí, el número de atributos que se consideran), lo que puede distorsionar el
  resultado sin que sea evidente a simple vista. Se explica en profundidad en la
  [lámina 20](lamina-20-segundo-artefacto-truncamiento.md).
- **Artefacto de evaluación:** un problema que no está en el método científico ni en la teoría, sino
  en **cómo se calculó** algo. Es distinto de un error conceptual: aquí el diseño experimental era
  correcto, pero la implementación tenía fallas que distorsionaban los números.
- **Protocolo de evaluación:** el conjunto exacto de pasos y parámetros usados para medir algo. La
  lección de Kosan y colaboradores (citada aquí) es que, en explicabilidad de GNNs, pequeños cambios
  en el protocolo pueden cambiar radicalmente las conclusiones.

## 📈 Cómo leer la figura / tabla

No aplica: son dos recuadros de alerta en paralelo (uno por artefacto), sin figura ni tabla numérica.

## 🎤 El discurso (como se dice en voz alta)

> Gracias, Alejandro. Voy a empezar los resultados por algo que no estaba en el plan original y que
> termino siendo una de nuestras contribuciones. *(pausa)* Al analizar la estabilidad encontramos que
> dos detalles de la medicion, no del metodo, estaban distorsionando las conclusiones. El primero fue un
> fallo de memoria silencioso: al calcular las explicaciones sobre el grafo completo, trece
> configuraciones de GAT fallaban sin aviso y sus filas quedaban vacias, de modo que el promedio de GAT
> se calculaba solo sobre los casos que si terminaban, y eso lo favorecia de forma artificial. El segundo
> fue un truncamiento en la metrica de Spearman: la implementacion descartaba los atributos por debajo de
> un umbral y mutilaba los rankings, lo que esta vez favorecia a GraphSAGE. *(pausa)* Lo importante es
> esto: cada uno de estos dos detalles, por si solo, bastaba para producir una conclusion comparativa
> falsa sobre que arquitectura es mas estable. La leccion, que conecta con el trabajo de Kosan sobre
> sensibilidad al protocolo, es que la estabilidad medida depende tanto del protocolo de evaluacion como
> del metodo. Y de paso reportamos dos bugs concretos del PGExplainer de la libreria PyG. Corregimos
> todo esto y volvimos a medir. Lo que sigue son los numeros corregidos.

### Versión ampliada y explicada

*"Voy a empezar los resultados por algo que no estaba en el plan original y que termino siendo una de
nuestras contribuciones."* — Es una confesión estratégica: este hallazgo no fue buscado a propósito,
surgió del trabajo mismo, y precisamente por eso tiene valor genuino.

*"Al analizar la estabilidad encontramos que dos detalles de la medicion, no del metodo, estaban
distorsionando las conclusiones."* — Distinción muy cuidadosa: el problema no era la idea de medir
estabilidad, era **cómo** se estaba calculando en la práctica.

*"El primero fue un fallo de memoria silencioso: al calcular las explicaciones sobre el grafo
completo, trece configuraciones de GAT fallaban sin aviso y sus filas quedaban vacias, de modo que el
promedio de GAT se calculaba solo sobre los casos que si terminaban, y eso lo favorecia de forma
artificial."* — Explica el mecanismo exacto: si 13 de las corridas de GAT fallaban y solo se
promedian las que sí terminaron, ese promedio queda sesgado hacia los casos "más fáciles" (menos
exigentes de memoria), que probablemente también son los más estables, inflando artificialmente el
resultado de GAT.

*"El segundo fue un truncamiento en la metrica de Spearman: la implementacion descartaba los atributos
por debajo de un umbral y mutilaba los rankings, lo que esta vez favorecia a GraphSAGE."* — El segundo
artefacto, con su propio sesgo, en dirección contraria: aquí el detalle técnico se desarrolla a fondo
en la lámina 20.

*"Lo importante es esto: cada uno de estos dos detalles, por si solo, bastaba para producir una
conclusion comparativa falsa sobre que arquitectura es mas estable."* — Es el mensaje central: no se
necesitaban los dos juntos para llegar a una conclusión errónea; **cualquiera de los dos, por
separado**, ya era suficiente para engañar.

*"La leccion, que conecta con el trabajo de Kosan sobre sensibilidad al protocolo, es que la
estabilidad medida depende tanto del protocolo de evaluacion como del metodo."* — Sitúa el hallazgo
dentro de la literatura existente: no es un caso aislado ni una anécdota, conecta con una advertencia
ya documentada por otros investigadores sobre lo sensible que es este campo a los detalles de
implementación.

*"Corregimos todo esto y volvimos a medir. Lo que sigue son los numeros corregidos."* — Cierra
anunciando que todos los resultados que vienen después ya están libres de estos dos defectos.

## 🧑‍⚖️ Preguntas de jurado probables

- **"¿Cómo descubrieron estos dos artefactos?"** → Al revisar por qué ciertas conclusiones cambiaban
  drásticamente entre corridas o al reentrenar, se auditó el pipeline de cómputo y se encontraron
  ambos problemas de forma independiente, cada uno documentado con su propio diagnóstico.
- **"¿Cómo se sabe que ya no quedan más artefactos ocultos?"** → No se puede garantizar al 100%, pero
  se auditó todo el pipeline con un smoke test de 15 verificaciones y se replicó el hallazgo con tres
  semillas de modelo distintas, obteniendo la misma partición, lo que da confianza en la robustez del
  resultado corregido (ver lámina 22).

## 🧠 En una frase

Dos defectos silenciosos en cómo se calculaba la estabilidad —cada uno suficiente por sí solo para
invertir la conclusión— fueron encontrados y corregidos antes de reportar ningún resultado.
