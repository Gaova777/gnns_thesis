# Narrativa v4: una sola hipótesis, pregunta y objetivo (propuesta)

> Estado: **propuesta** del Claude de Juan Diego para THE-16 y THE-18, a partir de los
> resultados v4 (`results_v4/analysis_v4_summary.txt`). No es una decisión cerrada:
> se lleva a la revisión con Cristian (THE-19). Fecha: 30-sep-2026.

## Por qué hay que reencuadrar

La v3 se articulaba en tres hipótesis (H1 el desbalance degrada la estabilidad, H2 TAGCN
es la más estable, H3 un explicador más estable señala mejor el patrón) más un control.
Las tres se refutaron ya en la v3. Con la v4, y las etiquetas corregidas, el cuadro se
simplifica y a la vez cambia en un punto central:

- El desbalance no gobierna la estabilidad (H1, equivalencia por TOST; GNNExplainer, Friedman p=0,45).
- La arquitectura tampoco la gobierna (H2, equivalencia por TOST; GNNExplainer, Friedman
  p=0,47, Kruskal p=0,85). Las tres arquitecturas con soporte salen equivalentes (GCN queda
  fuera porque ninguno de sus modelos pasa la compuerta). La partición en dos grupos de la v3
  no sobrevive a las etiquetas corregidas.
- El balanceo sí tiene un efecto (H3; GNNExplainer, Friedman p=0,0498, al borde del umbral).
- Sensibilidad sin compuerta (incluye los 36 modelos de GCN, ninguno pasa la compuerta): las
  explicaciones de GCN son casi tan estables como las del resto (GNNExplainer 0,916 frente a
  0,927 a 0,946) aunque GCN rinde al nivel del azar, así que una estabilidad alta no certifica
  que el modelo sirva. Con más datos las diferencias entre arquitecturas se vuelven detectables,
  pero quedan dentro del margen de equivalencia de ±0,05.
- El explicador es la palanca dominante ya vista en la v3: GNNExplainer discrimina,
  PGExplainer degenera sobre variables, GNNShap (ShapleyFeatures) satura cerca del techo.
- Las tres propiedades (estabilidad, plausibilidad, fidelidad) son independientes en el
  eje sintético. La v4 lo confirma (`results_phase1_v4/`): ShapleyFeatures es el más estable
  (0,95 a 0,97) y su plausibilidad de variables queda en el azar; GNNExplainer es menos
  estable pero el más plausible y el más fiel. Matiz: en el sintético la estabilidad de
  GNNExplainer baja con el desbalance (0,92 a 0,86), pero esos escenarios quitan ilícitas y
  degradan el modelo, así que el efecto no se aísla del desbalance.

Con eso, mantener tres hipótesis sueltas confunde. Conviene una sola, que sea falsable y
que ordene toda la evidencia.

## Pregunta de investigación (propuesta)

¿Qué factores del proceso de explicación (arquitectura de la GNN, método de explicación y
estrategia de balanceo) gobiernan la **estabilidad** de las explicaciones XAI en detección
de transacciones ilícitas bajo desbalance extremo, y esa estabilidad predice la **calidad**
de la explicación medida como plausibilidad y fidelidad?

## Hipótesis única (propuesta)

**La estabilidad de las explicaciones en GNNs está gobernada por el método de explicación,
no por la arquitectura ni por el nivel de desbalance; y una explicación estable no es, por
ello, ni más plausible ni más fiel: las tres propiedades son empíricamente independientes.**

Es una sola afirmación con dos mitades falsables:
1. Qué mueve la estabilidad: el explicador, y no la arquitectura ni el desbalance (contrasta
   con H1, H2 y H3 de la v3, y el sesgo va a favor del explicador).
2. Qué no implica la estabilidad: ser estable no compra plausibilidad ni fidelidad.

La ventaja frente a las tres hipótesis viejas: cada resultado (equivalencia por arquitectura,
equivalencia por desbalance, efecto del balanceo, dominancia del explicador, disociación de
las tres propiedades) confirma o acota una parte de la misma frase, en lugar de vivir en
hipótesis separadas.

## Objetivo general (propuesta)

Caracterizar la estabilidad de los métodos XAI sobre GNNs para detección de lavado bajo
desbalance, aislar el factor que la gobierna y medir su (in)dependencia respecto de la
plausibilidad y la fidelidad, entregando una matriz de recomendación por objetivo de
auditoría.

### Objetivos específicos

1. Medir la estabilidad de tres explicadores sobre cuatro arquitecturas y varios niveles de
   desbalance, con réplicas por semilla, y decidir con pruebas de equivalencia (TOST) y de
   efecto qué factor la mueve.
2. Medir plausibilidad y fidelidad en un grafo sintético con verdad por arista, donde esas
   dos propiedades sí son observables, y contrastar su relación con la estabilidad.
3. Traducir todo en una recomendación operativa: qué combinación conviene según el objetivo
   (auditabilidad, recuperar el patrón, fidelidad al modelo).

## Esqueleto por capítulo (THE-18)

Mapa de ideas, no redacción. Cada capítulo se ancla al contexto financiero (THE-24).

1. **Introducción.** El lavado como patrón de red, el costo de los falsos positivos de las
   reglas fijas, por qué GNN y por qué explicabilidad auditable. Pregunta, hipótesis única,
   objetivos. Aporte: la estabilidad como propiedad crítica y su disociación de plausibilidad
   y fidelidad.
2. **Marco teórico.** GNN y paso de mensajes, las cuatro arquitecturas por su forma de
   agregar, los tres explicadores post-hoc, las tres propiedades y por qué no son lo mismo.
3. **Metodología.** Los dos ejes (Elliptic real y sintético con verdad por arista). Etiquetas
   correctas de Elliptic y los tres modos A/B/C (evidencia de por qué los sin etiqueta se
   tratan como lícitos, THE-20). Diseño factorial, métricas primarias (PR-AUC, no F1 ni
   ROC-AUC), protocolo estadístico (TOST, Friedman, Holm), muestreo (THE-21), qué dimensiones
   de estabilidad se miden y por qué (THE-22).
4. **Resultados Elliptic.** Rendimiento y colapso val a test. Estabilidad: equivalencia por
   desbalance y por arquitectura, efecto del balanceo. El explicador como palanca dominante.
   Límite declarado: GAT capado por VRAM (ver `runs_v4/DECISIONES.md`).
5. **Resultados sintéticos.** Plausibilidad y fidelidad con verdad por arista, disociación de
   las tres propiedades, una página en el texto y el detalle al anexo (THE-23).
6. **Discusión.** La hipótesis única contrastada pieza a pieza. Qué cambió respecto de la v3 y
   por qué (etiquetas). Implicaciones para auditoría.
7. **Conclusiones y trabajo futuro.** Matriz de recomendación. Elliptic2, AMLSim, GNN
   temporales, GraphSMOTE.

## Qué falta decidir con Cristian (THE-19)

- Aceptar la hipótesis única y su redacción exacta.
- Confirmar el modo C como principal con B de sensibilidad (ya medido en el paso 0).
- Aceptar el límite de GAT por VRAM como está declarado.
- Cómo presentar el cambio de conclusión respecto de la v3 (la partición en dos grupos que
  ya no se sostiene).
