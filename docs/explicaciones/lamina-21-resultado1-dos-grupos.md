# Lámina 21 — Resultado 1: la estabilidad separa dos grupos, no cuatro puestos

> Bloque: Resultados
> Voz: Juan Diego · Página PDF 26 (21/40) · Tiempo objetivo 75 s

## 🎯 Objetivo de la lámina

Presentar el hallazgo central sobre las cuatro arquitecturas: con la métrica ya corregida, no existe
un ranking limpio de cuatro posiciones distintas. Lo que existe es una **partición en dos grupos**, y
esa distinción precisa es la que hay que decir siempre, sin simplificarla a un "ranking".

## 📋 Qué dice, item por item

- **Grupo alto:** GAT (0,782 ± 0,013) y GCN (0,758 ± 0,025).
- **"Brecha significativa entre grupos"** (destacado en el centro).
- **Grupo bajo:** GraphSAGE (0,735 ± 0,022) y TAGCN (0,672 ± 0,077).
- **Nota:** media de 3 semillas (180 modelos). El "±" es la **desviación típica entre semillas**, no
  el intervalo de confianza que se ve en la figura (son dos cosas distintas, aunque se parezcan).
  Significativo **entre** grupos, **no dentro**: Wilcoxon pareado GAT vs. GCN da p=0,17 y GraphSAGE
  vs. TAGCN da p=0,10 (ambos altos, es decir, no hay evidencia de diferencia real dentro de cada
  grupo).
- **Figura 5:** un gráfico de barras con intervalos de confianza al 95%, mostrando la estabilidad
  (Spearman de GNNExplainer) por arquitectura, sobre 3 semillas y 180 modelos.

## 🔑 Conceptos y técnicas que aparecen

- **Partición en dos grupos (no ranking de cuatro):** la diferencia es sutil pero crucial. Un
  "ranking de cuatro" implicaría que se puede ordenar con confianza las cuatro arquitecturas, una por
  una, de mejor a peor. Lo que los datos realmente muestran es que hay **dos bloques**: dentro de cada
  bloque, las arquitecturas son estadísticamente indistinguibles entre sí; solo entre bloques hay una
  diferencia real.
- **Desviación típica entre semillas:** cuánto varía el resultado de una arquitectura cuando se
  reentrena con distintas semillas de azar. Es distinto del intervalo de confianza de la figura, que
  se calcula de otra forma (con la técnica de bootstrap sobre los 180 modelos).
- **Wilcoxon pareado, p=0,17 y p=0,10:** valores de p **altos**, que indican que no hay evidencia
  suficiente para afirmar que hay una diferencia real dentro de cada grupo (GAT vs. GCN, o GraphSAGE
  vs. TAGCN). Ver [glosario](README.md#wilcoxon-pareado) y [el p-valor](README.md#el-p-valor).

## 📈 Cómo leer la figura / tabla

La Figura 5 muestra cuatro barras (una por arquitectura), con su valor de estabilidad y un intervalo
de confianza al 95% dibujado encima de cada barra. La lectura correcta es fijarse en si esos
intervalos **se solapan** o no: entre GAT/GCN (grupo alto) y GraphSAGE/TAGCN (grupo bajo), los
intervalos están claramente separados; dentro de cada grupo, los intervalos se superponen bastante,
lo que visualmente confirma que no hay una diferencia clara entre los dos miembros de cada grupo.

## 🎤 El discurso (como se dice en voz alta)

> Con la metrica corregida, y replicando el entrenamiento completo con tres semillas de modelo, lo que
> encontramos no es un ranking de cuatro puestos sino una particion en dos grupos. Un grupo alto, con GAT
> en cero coma setenta y ocho y GCN en cero coma setenta y seis, y un grupo bajo, con GraphSAGE en cero
> coma setenta y cuatro y TAGCN en cero coma sesenta y siete. *(pausa)* Y lo importante es donde estan las
> diferencias: entre los dos grupos son estadisticamente significativas, y dentro de cada grupo no lo son.
> La prueba que compara las cuatro arquitecturas nos dice que, si en realidad no hubiera ninguna
> diferencia entre ellas, una separacion tan marcada como la que vemos apareceria menos de tres veces
> en cien mil. Pero cuando hacemos la misma pregunta dentro del grupo alto, o dentro del grupo bajo, las
> diferencias son del tamano que el azar produce con frecuencia, asi que no podemos afirmar que existan. *(pausa)* Aqui cierro nuestra
> segunda hipotesis: esperabamos que TAGCN, por su alcance multi-hop, fuera la mas estable, y aparece de
> forma consistente en el grupo bajo en las tres semillas. La hipotesis se cae.
> *(pausa)* Hay dos cosas mas que tenemos que decir. La primera, que el liderazgo de GraphSAGE que reportaba
> una version anterior de la tesis era un artefacto del truncamiento de la metrica, y corregido se
> disuelve. La segunda, que GAT y GCN se permutan entre semillas, asi que no afirmamos que ninguna de las
> dos sea la mejor: afirmamos que las dos forman el grupo alto. Decir menos seria impreciso, y decir mas
> seria sobre-interpretar.

### Versión ampliada y explicada

*"Con la metrica corregida, y replicando el entrenamiento completo con tres semillas de modelo, lo que
encontramos no es un ranking de cuatro puestos sino una particion en dos grupos."* — Abre con la
afirmación más importante de toda la lámina, formulada con cuidado: no es un ranking, es una
partición.

*"Un grupo alto, con GAT en cero coma setenta y ocho y GCN en cero coma setenta y seis, y un grupo
bajo, con GraphSAGE en cero coma setenta y cuatro y TAGCN en cero coma sesenta y siete."* — Da los
números concretos de cada grupo, en el mismo aliento en que los agrupa, para que quede clara la
estructura de dos bloques desde el primer momento.

*"Y lo importante es donde estan las diferencias: entre los dos grupos son estadisticamente
significativas, y dentro de cada grupo no lo son."* — Esta es la frase que resume por qué se habla de
"grupos" y no de "ranking": la evidencia estadística solo respalda la separación entre bloques, no un
orden fino dentro de cada uno.

*"La prueba que compara las cuatro arquitecturas nos dice que, si en realidad no hubiera ninguna
diferencia entre ellas, una separacion tan marcada como la que vemos apareceria menos de tres veces en
cien mil."* — Traduce el p-valor (2,8×10⁻⁵) a lenguaje llano: es una forma de decir "esto casi con
seguridad no es casualidad".

*"Pero cuando hacemos la misma pregunta dentro del grupo alto, o dentro del grupo bajo, las
diferencias son del tamano que el azar produce con frecuencia, asi que no podemos afirmar que
existan."* — Aplica exactamente la misma lógica, pero esta vez dentro de cada grupo, y llega a la
conclusión contraria: aquí sí es plausible que la diferencia observada sea puro azar.

*"Aqui cierro nuestra segunda hipotesis: esperabamos que TAGCN, por su alcance multi-hop, fuera la mas
estable, y aparece de forma consistente en el grupo bajo en las tres semillas. La hipotesis se cae."*
— Cierra formalmente H2 (planteada en la lámina 06): la intuición de que el alcance multi-salto
haría a TAGCN la más estable no se confirmó; de hecho, quedó en el grupo bajo de forma consistente.

*"La primera, que el liderazgo de GraphSAGE que reportaba una version anterior de la tesis era un
artefacto del truncamiento de la metrica, y corregido se disuelve."* — Reafirma, en el contexto de
este resultado concreto, la retractación ya explicada en la lámina 20.

*"La segunda, que GAT y GCN se permutan entre semillas, asi que no afirmamos que ninguna de las dos
sea la mejor: afirmamos que las dos forman el grupo alto. Decir menos seria impreciso, y decir mas
seria sobre-interpretar."* — Un ejemplo perfecto de precisión científica: se elige deliberadamente la
afirmación que la evidencia sí respalda ("las dos forman el grupo alto"), evitando tanto quedarse
corto como exagerar.

## 🧑‍⚖️ Preguntas de jurado probables

- **"¿Por qué GAT y GCN y no una sola arquitectura como la mejor?"** → Porque la diferencia entre
  ellas dentro del grupo alto no es estadísticamente significativa (Wilcoxon p=0,17); afirmar un único
  ganador sería sobre-interpretar los datos.
- **"En una tabla filtrada de una versión anterior, GCN aparecía con una sola configuración. ¿Cómo se
  sostiene que encabece?"** → El ranking que se afirma es el de la corrida completa de sesenta
  configuraciones con tres semillas de modelo, donde las cuatro arquitecturas tienen soporte
  comparable; la columna filtrada de una sola semilla se reportaba como control de robustez, no como
  estimación con validez inferencial. Además, el eje sintético (con su propia replicación de tres
  grafos por tres semillas) confirma de forma independiente que GAT y GCN encabezan.
- **"¿Esta partición sobrevive a una corrección por comparaciones múltiples?"** → Sí, la separación
  entre grupos se mantiene tras aplicar la corrección de Holm (ver [lámina 22](lamina-22-robustez-particion.md)).

## 🧠 En una frase

La estabilidad no ordena a las cuatro arquitecturas de una en una: las separa en dos grupos, un grupo
alto (GAT y GCN) y uno bajo (GraphSAGE y TAGCN), con diferencia real solo entre grupos.
