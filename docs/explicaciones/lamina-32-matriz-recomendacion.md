# Lámina 32 — Matriz de recomendación

> Bloque: Conclusiones
> Voz: Juan Diego · Página PDF 38 (32/40) · Tiempo objetivo 60 s

## 🎯 Objetivo de la lámina

Condensar todos los hallazgos de la tesis en una guía práctica y accionable: qué combinación de
arquitectura y explicador usar, según cuál sea el propósito concreto de la auditoría. Es la respuesta
formal al cuarto objetivo (O4).

## 📋 Qué dice, item por item

- **Auditabilidad / estabilidad → grupo alto: GAT o GCN.** Si lo que importa es que la explicación sea
  consistente y defendible, cualquiera de las dos arquitecturas del grupo alto es una buena base.
- **Recuperar el patrón (plausibilidad) → PGExplainer.** Si el objetivo es identificar el patrón real
  de lavado, PGExplainer es la mejor opción.
- **Fidelidad al modelo → GNNExplainer.** Si el objetivo es entender qué usó realmente el modelo para
  decidir, GNNExplainer es la mejor opción.
- **Estabilidad interna del método → GNNShap.** Si lo que importa es que el propio proceso de
  explicación sea internamente consistente entre ejecuciones, GNNShap es la mejor opción.
- **Nota al pie:** la recomendación de arquitectura es **de grupo**, no de una arquitectura concreta:
  dentro del grupo alto, la evidencia no distingue entre GAT y GCN, así que la elección final puede
  basarse en costo o disponibilidad.

## 🔑 Conceptos y técnicas que aparecen

Esta lámina no introduce conceptos técnicos nuevos, pero conviene tener presente por qué cada
recomendación corresponde a cada explicador o arquitectura:

- **Grupo alto (GAT, GCN):** la estabilidad más alta y consistente demostrada en la
  [lámina 21](lamina-21-resultado1-dos-grupos.md).
- **PGExplainer para plausibilidad:** su ventaja demostrada en la
  [lámina 24](lamina-24-resultado2-disociacion.md) (0,80 vs. 0,50 de GNNExplainer).
- **GNNExplainer para fidelidad:** su ventaja en la misma lámina 24 (0,56 vs. 0,11 de PGExplainer).
- **GNNShap para estabilidad interna:** mencionado en la lámina 24 y detallado en el
  [respaldo de disociación](respaldo-disociacion-valores-exactos.md), donde alcanza 0,98 de
  consistencia interna entre ejecuciones, la más alta de los tres explicadores.

## 📈 Cómo leer la figura / tabla

La Tabla 4 tiene dos columnas: el objetivo operativo (qué es lo que le importa a quien audita) y la
configuración recomendada. La lectura correcta es de arriba hacia abajo, entendiendo que cada fila
responde a una pregunta distinta ("¿qué necesito?"), y que no hay una única fila "ganadora": la
elección depende enteramente de cuál sea la prioridad de quien va a usar la explicación.

## 🎤 El discurso (como se dice en voz alta)

> Todo lo anterior se condensa en esta matriz de recomendacion, que responde al cuarto objetivo. La idea
> es que no existe una combinacion unica que sea la mejor para todo, precisamente porque las tres
> dimensiones son independientes. *(pausa)* Entonces la recomendacion es por objetivo. Si lo que se busca
> es auditabilidad y estabilidad, GAT o GCN. Si el objetivo es recuperar el patron de lavado, es decir
> plausibilidad, PGExplainer. Si lo que importa es la fidelidad al razonamiento del modelo, GNNExplainer.
> Y si se busca estabilidad interna del propio metodo de explicacion, GNNShap. Esta tabla es mas util que
> una recomendacion cerrada, porque obliga a hacer explicito el proposito de la auditoria antes de elegir
> la herramienta.

### Versión ampliada y explicada

*"Todo lo anterior se condensa en esta matriz de recomendacion, que responde al cuarto objetivo."* —
Marca el propósito de la lámina como el cierre práctico de todo lo demostrado en resultados.

*"La idea es que no existe una combinacion unica que sea la mejor para todo, precisamente porque las
tres dimensiones son independientes."* — Conecta directamente con la tesis central (lámina 11): si
las tres propiedades fueran lo mismo (o se implicaran unas a otras), sí existiría una única mejor
combinación; como no lo son, la recomendación debe ser condicional.

*"Entonces la recomendacion es por objetivo. Si lo que se busca es auditabilidad y estabilidad, GAT o
GCN."* — Empieza con la recomendación de arquitectura: no una arquitectura específica, sino el grupo
completo, siguiendo la precisión ya establecida en la lámina 21.

*"Si el objetivo es recuperar el patron de lavado, es decir plausibilidad, PGExplainer."* — Traduce
"recuperar el patrón" a su nombre técnico (plausibilidad), para que la conexión con los resultados sea
explícita.

*"Si lo que importa es la fidelidad al razonamiento del modelo, GNNExplainer."* — La recomendación
opuesta a la anterior, reflejando directamente la disociación demostrada en la lámina 24.

*"Y si se busca estabilidad interna del propio metodo de explicacion, GNNShap."* — Le da a GNNShap su
recomendación específica, en el único aspecto donde realmente destaca frente a los otros dos.

*"Esta tabla es mas util que una recomendacion cerrada, porque obliga a hacer explicito el proposito
de la auditoria antes de elegir la herramienta."* — Cierra defendiendo el formato de matriz (varias
opciones según propósito) frente a la alternativa de dar un único "ganador": obligar a preguntarse
"¿para qué necesito esta explicación?" antes de elegir la herramienta es, en sí mismo, un aporte
práctico valioso.

## 🧑‍⚖️ Preguntas de jurado probables

- **"¿Por qué no recomendar directamente GAT en vez del 'grupo alto'?"** → Porque la diferencia entre
  GAT y GCN dentro del grupo alto no es estadísticamente significativa (ver lámina 21); recomendar una
  sola arquitectura concreta sería sobre-interpretar la evidencia disponible.
- **"¿Qué pasa si un equipo necesita, a la vez, plausibilidad y fidelidad altas?"** → La tesis muestra
  que esa combinación no existe en los tres explicadores estudiados (ver la disociación en la lámina
  24); un equipo con esa necesidad tendría que ponderar cuál de las dos propiedades es más crítica
  para su caso de uso concreto, o considerar aplicar ambos explicadores en paralelo para tener las dos
  perspectivas.

## 🧠 En una frase

No hay una combinación única óptima: la elección de arquitectura y explicador depende de si lo que se
necesita es auditabilidad, recuperar el patrón, fidelidad al modelo, o consistencia interna del
método.
