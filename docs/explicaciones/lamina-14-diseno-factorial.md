# Lámina 14 — Diseño factorial

> Bloque: Metodología
> Voz: Alejandro · Página PDF 18 (14/40) · Tiempo objetivo 55 s

## 🎯 Objetivo de la lámina

Mostrar la escala y el rigor del experimento: no es una prueba suelta, es una matriz completa que
cruza todas las combinaciones posibles de las variables de interés, repetida varias veces y validada
con pruebas estadísticas serias.

## 📋 Qué dice, item por item

- **"4 arquitecturas × 3 explicadores × 3 balanceos × 5 escenarios"** — Estas son las cuatro
  dimensiones que se combinan entre sí: arquitectura (GCN, GraphSAGE, GAT, TAGCN), explicador
  (GNNExplainer, PGExplainer, GNNShap), estrategia de balanceo (tres opciones distintas) y escenario
  de desbalance (cinco niveles, de 1:1 a 1:100 más el nativo).
- **"= 60 configuraciones por eje × 3 explicadores para el estudio de estabilidad"** — Multiplicando
  arquitecturas (4) por balanceos (3) por escenarios (5) se obtienen 60 configuraciones distintas de
  modelo, y sobre cada una se corren los tres explicadores.
- **Recuadro "Protocolo de robustez":** para la estabilidad, cada explicación se repite 5 veces con
  distintas semillas de azar (réplicas estocásticas). Para la inferencia sobre el eje sintético, se
  usan además 3 semillas de modelo por 3 grafos independientes, con pruebas estadísticas serias
  (Kruskal-Wallis, Wilcoxon e intervalos de confianza por bootstrap).

## 🔑 Conceptos y técnicas que aparecen

- **Diseño factorial:** un tipo de experimento donde se prueban **todas las combinaciones posibles**
  de varios factores a la vez (en vez de probar un factor mientras se mantienen los demás fijos), lo
  que permite ver si hay efectos cruzados entre ellos.
- **Réplica estocástica:** repetir el mismo cálculo varias veces, cambiando solo la fuente de azar
  interna (la semilla), para poder medir qué tan estable es el resultado.
- **Kruskal-Wallis, Wilcoxon, bootstrap:** ver el
  [glosario](README.md#kruskal-wallis) para las definiciones detalladas de cada una. En resumen:
  Kruskal-Wallis compara varios grupos, Wilcoxon compara pares emparejados, y el bootstrap calcula el
  margen de error de un número repitiendo el cálculo con remuestreo.

## 📈 Cómo leer la figura / tabla

No aplica: la lámina muestra la multiplicación de factores (4 × 3 × 3 × 5) de forma visual, con
números grandes y colores, seguida de un bloque de texto sobre el protocolo de robustez, sin tabla ni
gráfico numérico.

## 🎤 El discurso (como se dice en voz alta)

> El experimento es una matriz factorial completa: cuatro arquitecturas, por tres explicadores, por tres
> estrategias de balanceo, por cinco escenarios de desbalance. Eso da sesenta configuraciones por eje,
> cada una con los tres explicadores para el estudio de estabilidad. *(pausa)* Y no nos quedamos en una
> sola corrida. Cada explicacion se repite cinco veces con semillas distintas para medir su estabilidad.
> Y en el eje sintetico anadimos una capa de robustez: tres semillas de modelo por tres grafos
> independientes, con pruebas estadisticas serias, Kruskal-Wallis, Wilcoxon e intervalos de confianza
> por bootstrap. Esto es lo que convierte observaciones sueltas en evidencia con respaldo.

### Versión ampliada y explicada

*"El experimento es una matriz factorial completa: cuatro arquitecturas, por tres explicadores, por
tres estrategias de balanceo, por cinco escenarios de desbalance."* — Enumera los cuatro factores que
se van a combinar entre sí, uno por uno, para que quede claro que ninguno se deja fijo mientras se
varían los demás.

*"Eso da sesenta configuraciones por eje, cada una con los tres explicadores para el estudio de
estabilidad."* — Hace la cuenta explícita: 4 arquitecturas × 3 balanceos × 5 escenarios = 60
configuraciones de modelo distintas, y sobre cada una se corren los tres explicadores.

*"Y no nos quedamos en una sola corrida. Cada explicacion se repite cinco veces con semillas
distintas para medir su estabilidad."* — Aquí se explica de dónde sale, en la práctica, la medición de
estabilidad: no es una intuición, es literalmente repetir el cálculo cinco veces y comparar.

*"Y en el eje sintetico anadimos una capa de robustez: tres semillas de modelo por tres grafos
independientes, con pruebas estadisticas serias, Kruskal-Wallis, Wilcoxon e intervalos de confianza
por bootstrap."* — Añade una capa extra de rigor específicamente en el eje sintético: no solo se
repite la explicación (estabilidad), también se reentrena el modelo entero varias veces, sobre grafos
generados de forma independiente, para que las conclusiones no dependan de una casualidad de una sola
corrida ni de un solo grafo.

*"Esto es lo que convierte observaciones sueltas en evidencia con respaldo."* — Cierra explicando el
propósito de todo este aparato estadístico: sin él, cualquier número aislado podría ser ruido; con él,
se puede afirmar con confianza que un patrón es real.

## 🧑‍⚖️ Preguntas de jurado probables

- **"¿Por qué 5 réplicas y no más o menos?"** → Es un número que equilibra la necesidad estadística de
  tener suficientes puntos para estimar la variabilidad, con el costo computacional de repetir el
  proceso de explicación (que puede ser lento, especialmente para GNNExplainer, que optimiza una
  máscara por cada instancia).
- **"¿Por qué solo 3 semillas de modelo y no más, si eso da más robustez?"** → Es un balance práctico
  entre robustez estadística y tiempo de cómputo (reentrenar 60 configuraciones completas, con su
  propia búsqueda de hiperparámetros, varias veces, es costoso); con 3 semillas ya se logra distinguir
  con claridad la señal (diferencias entre grupos de arquitecturas) del ruido (ver láminas 21 y 22).

## 🧠 En una frase

El experimento cruza todas las combinaciones posibles de cuatro variables (60 configuraciones por
eje), y cada resultado se repite varias veces con pruebas estadísticas serias antes de llamarse
"hallazgo".
