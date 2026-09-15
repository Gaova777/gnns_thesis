# Explicaciones lámina por lámina — guía de estudio para la defensa

> Material de estudio profundo, sin tecnicismos, de cada lámina del deck de defensa
> (`presentacion_latex/beamer_defensa_v3.tex`, 47 páginas, 33 láminas de contenido numeradas más
> portada y cierre). Cada archivo explica el objetivo de la lámina, cada uno de sus items, los
> conceptos que menciona, su figura o tabla (si tiene), el discurso hablado y una versión ampliada
> de ese discurso, y las preguntas de jurado más probables.
>
> Generado siguiendo `docs/PLAN_explicaciones_laminas.md`. Fuentes: el deck (`.tex`), el guion
> (`DISCURSO_defensa_dos_voces.md`), `CLAUDE.md` (reglas duras) y el manuscrito (`tesis_latex/`).

---

## Cómo usar esto

1. Lee primero el **glosario** (abajo) y el **banco de analogías**: son la base que se repite en
   todas las láminas.
2. Ve lámina por lámina, **en orden**, siguiendo el índice. Cada archivo es autónomo: se puede
   estudiar solo, pero se entiende mejor en orden porque las ideas se acumulan.
3. Antes de la defensa, repasa los **guardarraíles de exactitud** (abajo): son los números y
   afirmaciones que hay que decir bien, porque corrigen versiones anteriores de la tesis.

---

## ⚠️ Guardarraíles de exactitud (léelos antes de estudiar cualquier lámina)

Estos son los puntos donde la tesis **corrigió** una versión anterior. Decir la versión vieja sería
un error grave frente al jurado.

1. **Es una partición en DOS GRUPOS, no un ranking de cuatro.** Grupo alto: GAT y GCN. Grupo bajo:
   GraphSAGE y TAGCN. Diferencias significativas **entre** grupos, no **dentro**. Nunca decir "GAT
   es la más estable": GAT y GCN se permutan entre semillas.
2. **El desbalance no degrada la estabilidad de forma significativa.** El rango va de 0,696 (1:1) a
   0,765 (1:50): siete centésimas. La prueba estadística no encuentra diferencia real entre
   escenarios, y el tamaño de efecto es pequeño (mucho menor que el de la arquitectura). El valor más
   bajo está en 1:1 (el escenario más equilibrado), es decir, lo poco que varía va **en contra** de lo
   que se esperaba. No decir "perfil plano de tres centésimas": esa cifra venía de una sola semilla.
3. **Los tamaños de efecto que reporta la tesis son η² (eta al cuadrado)**, no otro símbolo parecido
   (ε²). No mezclarlos.
4. **GNNShap mide plausibilidad sobre atributos (features), no sobre aristas**, y su línea base de
   azar es 0,075 (no 0,40). No produce máscara de aristas por diseño. Nunca compararlo en la misma
   figura o columna que la plausibilidad de aristas de GNNExplainer/PGExplainer.
5. **PGExplainer degenera en Elliptic por la dispersión del grafo**, no porque sea un mal método: en
   el grafo sintético (denso) es el mejor explicador en plausibilidad de aristas. El contraste entre
   los dos regímenes es un aporte, no una falla.
6. **Nunca reintroducir:** "inversión por densidad" o "GraphSAGE lidera en estabilidad". Ambas
   afirmaciones fueron retractadas: eran artefactos de medición. Lo vigente es que los dos regímenes
   (Elliptic disperso y el grafo sintético denso) **concuerdan** (correlación de rangos +0,80).
7. **La estabilidad se mide sobre los verdaderos positivos de validación**, no de test: el modelo
   colapsa en test por el desplazamiento temporal del dataset.
8. **Las métricas primarias de rendimiento son PR-AUC y precisión@k**, no F1 ni ROC-AUC. El ROC-AUC
   se ve engañosamente alto bajo desbalance extremo.
9. **La contribución metodológica central son dos artefactos de medición corregidos**: un fallo de
   memoria silencioso (favorecía a GAT) y un truncamiento en la métrica de Spearman (favorecía a
   GraphSAGE). Cada uno, por separado, bastaba para invertir la conclusión.
10. **El pipeline no es reproducible bit a bit** (los pesos exactos cambian entre corridas por cómo
    suma la GPU), pero **sí reproduce las conclusiones**.

---

## Glosario de conceptos recurrentes

### Grafo, nodo, arista, vecindario

Un **grafo** es una red: puntos (**nodos**) conectados por líneas (**aristas**). Aquí cada nodo es
una transacción de Bitcoin, y cada arista es un flujo de dinero entre dos transacciones. El
**vecindario** de un nodo son los nodos conectados directamente a él (o, si hablamos de varias
capas, los que están a un número de "saltos" de distancia). El **campo receptivo** es "hasta dónde
alcanza a ver" un nodo cuando el modelo agrega información de sus vecinos.

### GNN y paso de mensajes

Una **GNN** (red neuronal de grafos) aprende sobre esta red actualizando la información de cada nodo
usando la de sus vecinos, capa por capa. A esto se le llama **paso de mensajes**: cada nodo "recibe
mensajes" de sus vecinos y los combina con su propia información. Con dos capas, un nodo ve hasta dos
saltos de distancia. Ver [lámina 08](lamina-08-paso-de-mensajes.md).

### Las cuatro arquitecturas GNN

Cuatro maneras distintas de combinar la información de los vecinos: **GCN** (promedia a todos por
igual), **GraphSAGE** (muestrea un subconjunto de vecinos), **GAT** (pondera a los vecinos con
atención, no todos pesan igual) y **TAGCN** (usa filtros que alcanzan varios saltos de una vez). Ver
[lámina 09](lamina-09-cuatro-arquitecturas.md).

### Explicador post-hoc: máscara y ranking

Un **explicador post-hoc** analiza un modelo **ya entrenado**, sin tocarlo, y responde: "¿por qué
predijiste esto?". Entrega dos cosas: una **máscara de aristas** (qué conexiones del vecindario
importaron) y un **ranking de features** (qué atributos del nodo pesaron más). Los tres explicadores
de esta tesis son **GNNExplainer** (optimiza una máscara para cada caso), **PGExplainer** (entrena
una red que generaliza a todo el grafo) y **GNNShap** (reparte crédito con valores de Shapley, de la
teoría de juegos). Ver [lámina 10](lamina-10-tres-explicadores.md).

### Las tres propiedades: estabilidad, plausibilidad, fidelidad

- **Estabilidad:** ¿la explicación sale igual si repito el cálculo?
- **Plausibilidad:** ¿la explicación señala el patrón real (lo que un experto reconocería)?
- **Fidelidad:** ¿la explicación refleja de verdad lo que el modelo usó para decidir?

Son **independientes entre sí**: una explicación puede ser muy estable y estar siempre equivocada
(como un reloj parado), o puede ser plausible para un humano sin reflejar el mecanismo real del
modelo. Ver [lámina 11](lamina-11-tres-propiedades.md).

### Desbalance de clases y escenarios

El **desbalance** es que los casos de fraude son rarísimos comparados con los normales. Los
**escenarios** (1:1, 1:10, 1:50, 1:100, nativo) son proporciones artificiales que se crean para
probar qué pasa cuando el fraude es más o menos raro.

### Balanceo (la técnica) frente a desbalance (el problema)

No confundir: el **desbalance** es el problema (el fraude es raro); el **balanceo** es la **técnica**
que se usa al entrenar para compensarlo (por ejemplo, darle más peso a los casos raros). La tesis
muestra que ni uno ni otro afectan mucho la calidad de las explicaciones.

### Correlación de Spearman

Mide si dos "listas ordenadas" (rankings) se parecen. Va de −1 (órdenes opuestos) a +1 (órdenes
idénticos), pasando por 0 (no hay relación). Aquí se usa para comparar si el ranking de atributos
importantes sale igual al repetir el cálculo (estabilidad) o si dos cosas distintas suben y bajan
juntas (por ejemplo, estabilidad y plausibilidad).

### Kruskal-Wallis

Una prueba estadística que compara **varios grupos a la vez** y responde: "¿es creíble que estos
grupos sean en realidad iguales, o la diferencia que veo es demasiado grande para ser azar?". No
asume que los datos siguen una campana de Gauss (por eso es "no paramétrica"). Se usa para comparar
las cuatro arquitecturas o los cinco escenarios de desbalance a la vez.

### Mann-Whitney

Como Kruskal-Wallis, pero para comparar **dos grupos** entre sí (por ejemplo, el grupo alto contra el
grupo bajo de arquitecturas).

### Wilcoxon pareado

Compara dos conjuntos de resultados que están **emparejados** (la misma configuración medida de dos
formas distintas), en vez de grupos independientes. Se usa, por ejemplo, para comparar PGExplainer
contra GNNExplainer en las mismas 225 configuraciones.

### Bootstrap e intervalo de confianza

El **bootstrap** es una forma de estimar el margen de error de un número: se vuelve a calcular el
promedio miles de veces, cada vez sacando una muestra al azar (con reemplazo) de los mismos datos, y
se mira cuánto varía el resultado. El **intervalo de confianza (IC)** es el rango donde probablemente
cae el valor real. Si ese rango **incluye el cero**, la lectura honesta es "no hay efecto".

### Corrección de Holm

Cuando se hacen **muchas comparaciones** a la vez, algunas pueden salir "significativas" por puro
azar, solo por probar muchas veces. La corrección de Holm ajusta el criterio para que eso no engañe.
Que un resultado sobreviva a Holm significa que es robusto incluso siendo estrictos.

### El p-valor

Responde: "si en realidad no hubiera ninguna diferencia (todo fuera azar), ¿qué tan raro sería ver un
resultado como el que obtuve?". Un p-valor **pequeño** (como 0,0000... algo) dice "esto casi no puede
ser casualidad". Un p-valor **grande** (como 0,18) dice "esto es perfectamente compatible con el
azar, no hay evidencia de que sea real".

### Tamaño de efecto (η²)

El p-valor dice **si** hay una diferencia real; el tamaño de efecto dice **cuánta importancia
práctica** tiene esa diferencia. η² (eta al cuadrado) responde: "de toda la variación que veo en los
resultados, ¿qué tajada explica este factor?". Un η² pequeño (como 0,05) significa que ese factor
casi no mueve la aguja, aunque sea estadísticamente detectable.

### Qué significa "no paramétrico"

Significa que la prueba **no asume** que los datos se distribuyen en forma de campana (la campana de
Gauss). Es más prudente cuando no se sabe con certeza cómo se comportan los datos, como aquí.

### PR-AUC, ROC-AUC y precisión@k

- **ROC-AUC:** una nota general de qué tan bien el modelo separa lo positivo de lo negativo. Bajo
  desbalance extremo, se ve **engañosamente alta** porque le da mucho crédito por acertar en la
  mayoría fácil (los casos normales), aunque falle en los raros (el fraude).
- **PR-AUC:** una nota que **no se deja engañar** por el desbalance, porque solo premia si de verdad
  se encuentra lo raro.
- **Precisión@k:** de las k transacciones que el modelo marca como más sospechosas, ¿cuántas eran
  fraude real? Es la métrica más parecida al trabajo real de un analista.

### Ground-truth y validez interna vs. externa

El **ground-truth** es "la respuesta correcta conocida de antemano" (qué transacciones son
realmente parte de un patrón de lavado). Elliptic (el dato real) no tiene ground-truth de patrón, solo
de si una transacción es lícita o no. Por eso la tesis construye un **grafo sintético propio**, donde
sí se conoce el patrón verdadero. La **validez externa** viene de usar datos reales (Elliptic); la
**validez interna**, de controlar el experimento con datos sintéticos donde se sabe la respuesta.

### Shift temporal y el colapso validación→test

El **shift temporal** es que los patrones de lavado cambian con el tiempo. Un modelo entrenado con
transacciones del pasado puede fallar con las del futuro, porque el "terreno de juego" cambió. Eso
produce el **colapso validación→test**: el modelo rinde bien en los datos de práctica (validación,
del pasado) y mal en el examen real (test, del futuro).

---

## Banco de analogías (para dar coherencia entre láminas)

- **Reloj parado:** siempre da la misma hora (100% estable) y siempre está mal. Estabilidad ≠ acierto.
- **Perilla de volumen desconectada:** girar el desbalance de fácil a difícil no cambia el volumen
  (la estabilidad). La perilla no está conectada a nada.
- **El guardia que dice "aquí no hay ladrones":** acierta el 99,9% de las veces en un pueblo donde
  casi nadie es ladrón, pero atrapa cero. Así de engañoso es el ROC-AUC bajo desbalance.
- **Corredor de 100 metros frente a nadador:** GNNShap "no corre la misma carrera" que los otros dos
  explicadores (mide features, no aristas); comparar sus tiempos no tiene sentido.
- **Filtro de spam entrenado en 2015, probado en 2024:** los estafadores cambiaron de trucos; el
  filtro viejo falla. Así es el shift temporal.
- **Evaluar a un médico solo en los casos que acertó:** para saber si es consistente, se mira dónde
  sí funciona, no donde está perdido. Por eso la estabilidad se mide sobre los verdaderos positivos.
- **"¿Cuánto explica el profesor?" en las notas de un examen:** el tamaño de efecto (η²) mide qué
  tajada de la variación total le corresponde a un factor concreto.
- **Detective en una escena vacía:** sin pistas (aristas) que señalar, dice que todo es igual de
  importante. Así colapsa PGExplainer en un grafo tan disperso como Elliptic.

---

## Índice de láminas

### Portada y contenido

- [Lámina 00 — Portada](lamina-00-portada.md)
- [Lámina 01 (1/40) — Contenido de la sustentación](lamina-01-contenido.md)

### Bloque 1 · Problema y motivación (Alejandro)

- [Lámina 02 (2/40) — El lavado de dinero es un problema de red](lamina-02-lavado-problema-de-red.md)
- [Lámina 03 (3/40) — El problema: detectar sin caja negra](lamina-03-detectar-sin-caja-negra.md)
- [Lámina 04 (4/40) — Por qué la caja negra es inaceptable](lamina-04-por-que-caja-negra-inaceptable.md)
- [Lámina 05 (5/40) — Pregunta de investigación y objetivos](lamina-05-pregunta-objetivos.md)
- [Lámina 06 (6/40) — Tres hipótesis falsables](lamina-06-tres-hipotesis-falsables.md)
- [Lámina 07 (7/40) — La brecha en el estado del arte](lamina-07-la-brecha.md)

### Bloque 2 · Marco conceptual (Alejandro)

- [Lámina 08 (8/40) — Cómo funciona una GNN: paso de mensajes](lamina-08-paso-de-mensajes.md)
- [Lámina 09 (9/40) — Cuatro arquitecturas GNN](lamina-09-cuatro-arquitecturas.md)
- [Lámina 10 (10/40) — Tres explicadores post-hoc](lamina-10-tres-explicadores.md)
- [Lámina 11 (11/40) — Tres propiedades que no son lo mismo](lamina-11-tres-propiedades.md)

### Bloque 3 · Metodología (Alejandro)

- [Lámina 12 (12/40) — Qué produce, en concreto, un explicador](lamina-12-que-produce-un-explicador.md)
- [Lámina 13 (13/40) — Metodología: un diseño de dos ejes](lamina-13-dos-ejes.md)
- [Lámina 14 (14/40) — Diseño factorial](lamina-14-diseno-factorial.md)
- [Lámina 15 (15/40) — Eje 1: Elliptic Bitcoin Dataset](lamina-15-eje1-elliptic.md)
- [Lámina 16 (16/40) — Eje 2: grafo sintético con ground-truth](lamina-16-eje2-sintetico.md)
- [Lámina 17 (17/40) — Cómo se construyó el grafo sintético](lamina-17-como-se-construyo-sintetico.md)
- [Lámina 18 (18/40) — Métricas y protocolo estadístico](lamina-18-metricas-protocolo.md)

### Bloque 4 · Resultados (Juan Diego)

- [Lámina 19 (19/40) — Contribución: dos artefactos de evaluación](lamina-19-dos-artefactos.md)
- [Lámina 20 (20/40) — El segundo artefacto: el truncamiento de la métrica](lamina-20-segundo-artefacto-truncamiento.md)
- [Lámina 21 (21/40) — Resultado 1: la estabilidad separa dos grupos, no cuatro puestos](lamina-21-resultado1-dos-grupos.md)
- [Lámina 22 (22/40) — Robustez: qué sostiene esa partición](lamina-22-robustez-particion.md)
- [Lámina 23 (23/40) — Resultado 1b: concordancia entre regímenes](lamina-23-resultado1b-concordancia.md)
- [Lámina 24 (24/40) — Resultado 2: disociación plausibilidad / fidelidad](lamina-24-resultado2-disociacion.md)
- [Lámina 25 (25/40) — Resultado 3: el puente que no existe](lamina-25-resultado3-puente-nulo.md)
- [Lámina 26 (26/40) — Resultado 4: el desbalance no gobierna la estabilidad](lamina-26-resultado4-desbalance.md)
- [Lámina 27 (27/40) — Rendimiento y colapso validación → test](lamina-27-colapso-validacion-test.md)
- [Lámina 28 (28/40) — Rigor métrico: por qué PR-AUC y no ROC-AUC](lamina-28-rigor-metrico-prauc-rocauc.md)
- [Lámina 29 (29/40) — Las tres hipótesis, y qué pasó con cada una](lamina-29-tres-hipotesis-veredicto.md)

### Bloque 5 · Conclusiones (Juan Diego)

- [Lámina 30 (30/40) — Conclusiones del proyecto](lamina-30-conclusiones-proyecto.md)
- [Lámina 31 (31/40) — Los cuatro objetivos, respondidos](lamina-31-cuatro-objetivos-respondidos.md)
- [Lámina 32 (32/40) — Matriz de recomendación](lamina-32-matriz-recomendacion.md)
- [Lámina 33 (33/40) — Contribuciones y limitaciones](lamina-33-contribuciones-limitaciones.md)
- [Lámina 34 — Gracias / Preguntas (cierre)](lamina-34-cierre-gracias.md)

### Láminas de respaldo (solo si el jurado pregunta)

- [Respaldo · Disociación: valores exactos](respaldo-disociacion-valores-exactos.md)
- [Respaldo · Estabilidad por semilla de modelo](respaldo-estabilidad-por-semilla.md)
- [Respaldo · Detalle estadístico de la partición](respaldo-detalle-estadistico-particion.md)
- [Respaldo · ¿Por qué GNNExplainer para el ranking?](respaldo-por-que-gnnexplainer.md)
- [Respaldo · Curvas ROC y precisión-recall](respaldo-curvas-roc-pr.md)
