# Validación de los resultados del 7 al 10 de octubre

Fecha: 10-oct-2026. Alcance: 3 semillas del modelo (42, 43, 44) en todo, salvo donde se dice.
Scripts: `scripts/v4/reunion_0710/` (`analisis.py`, `metrica_robusta.py`, `control_aleatorio.py`).

## Veredicto por afirmación

| Afirmación | Veredicto | Cómo se comprobó |
|---|---|---|
| GAT con pesos por clase no pasa la compuerta | Correcta | 0 de 12 modelos pasan con el umbral 0,5 y 12 de 12 con el umbral calibrado (0,82 a 0,89). |
| Integrated Gradients y Expected Gradients están bien implementados | Correcta | Exactos en un modelo lineal (error 3·10⁻⁸); gradientes iguales a diferencias finitas (≤ 2·10⁻¹⁰); error relativo de completitud de 0 a 0,014 con 100 pasos. |
| Con pesos por clase la pérdida es ½ · media(ilícitas) + ½ · media(negativos) | Correcta | Coincide al sexto decimal con la entropía cruzada ponderada de PyTorch. |
| Las cifras de estabilidad del pipeline | Correctas como cálculo | Recalculadas con otro código desde los puntajes guardados: mismo resultado. |
| «Entre escenarios el acuerdo es 0,99, más que entre semillas (0,94)» | Engañosa, corregida | Compara modelos de la misma semilla (coseno de pesos 0,90 a 1,00) contra una referencia de otra semilla. Cruzando la semilla: 0,941 frente a 0,940. |
| «El submuestreo solo desplaza el logit en log(1/β)» | No se sostiene como mecanismo observado | En los modelos sin ajuste el desplazamiento mediano es de −0,4 a 0,2 (teoría: +1,34) y la correlación de puntajes de 0,40 a 0,80. |
| «Las tres arquitecturas se apoyan en las mismas variables» | No se sostiene | Jaccard de las 10 más importantes entre arquitecturas: 0,20 a 0,50 (comparten de 3 a 7 de las 10). |
| «La estabilidad de 0,93 a 0,99 refleja lo que el modelo aprendió» | No se sostiene | Una red sin entrenar da casi la misma cifra (tabla siguiente). |

Corrección de una cifra dada en el chat el 10-oct por la mañana: se dijo «coinciden de 2 a 5
de las 10». El Jaccard de 0,20 a 0,50 equivale a compartir de 3 a 7 de las 10 (J = k / (20 − k)).

## 1. Las variables en cero inflan el Spearman

- El 46,2 % de las variables de los 30 nodos explicados vale exactamente 0 después del
  escalado robusto (39 % en todo el grafo). Mediana: 87 de 165 variables distintas de 0.
- Los cuatro explicadores de variables le dan importancia 0 a una variable en 0, por su
  ecuación (detalle en `docs/borradores/THE-41_porque_hipotesis.md`, sección 1.2).
- Solo ese bloque común da un Spearman de 0,74 (empates rotos al azar) a 0,82 (empates con
  rango medio) entre dos órdenes independientes.

## 2. Control de pesos al azar (Adebayo et al. 2018)

Escenario nativo, pesos por clase, 4 arquitecturas por 3 semillas por 4 explicadores (48
combinaciones, 30 nodos). `control_aleatorio.csv`.

| Explicador | Estabilidad entrenado | Estabilidad sin entrenar | Acuerdo entrenado y sin entrenar | Acuerdo entre dos redes sin entrenar |
|---|---|---|---|---|
| GNNExplainer | 0,949 | 0,929 | 0,896 | 0,909 |
| Shapley | 0,970 | 0,959 | 0,911 | 0,913 |
| Integrated Gradients | 1 | 1 | 0,956 | 0,958 |
| Expected Gradients | 0,929 | 0,926 | 0,688 | 0,736 |

Con el Jaccard de las 10 más importantes:

| Explicador | Estabilidad entrenado | Estabilidad sin entrenar | Acuerdo entrenado y sin entrenar |
|---|---|---|---|
| GNNExplainer | 0,657 | 0,631 | 0,206 |
| Shapley | 0,824 | 0,898 | 0,365 |
| Integrated Gradients | 1 | 1 | 0,401 |
| Expected Gradients | 0,774 | 0,774 | 0,200 |

Lectura: la estabilidad entre réplicas es una propiedad del explicador y de la entrada. Las 10
variables principales sí cambian con el entrenamiento (acuerdo de 0,20 a 0,40), pero la
estabilidad no lo registra.

## 3. Medidas que no se inflan

GNNExplainer (`robusta_config.csv`). sp_all: Spearman sobre las 165 variables; sp_nz: solo
variables distintas de 0; jac10: Jaccard de las 10 más importantes.

| Comparación | sp_all | sp_nz | jac10 |
|---|---|---|---|
| H1, pesos por clase, 3 arquitecturas: 1:10 · 1:20 · nativo | 0,946 · 0,946 · 0,947 | 0,749 · 0,751 · 0,753 | 0,62 · 0,63 · 0,62 |
| H1, estrés: 1:100 · 1:200 | 0,940 · 0,938 | 0,717 · 0,708 | 0,61 · 0,62 |
| H2, pesos por clase: GCN · GraphSAGE · GAT · TAGCN | 0,938 · 0,964 · 0,939 · 0,937 | 0,716 · 0,844 · 0,734 · 0,677 | 0,67 · 0,64 · 0,62 · 0,62 |
| H3, 3 arquitecturas: pesos por clase · sin ajuste · focal loss | 0,947 · 0,925 · 0,914 | 0,751 · 0,650 · 0,589 | 0,63 · 0,59 · 0,57 |

- H1: Friedman sobre los seis escenarios (9 modelos), p = 0,84 (sp_all) y p = 0,35 (sp_nz).
- H2 y H3: ver la subsección siguiente.

### Pruebas de H2 y H3 con las tres medidas (11-oct)

`pruebas_h2_h3_robustas.csv`. Escenarios principales; GraphSAGE, GAT y TAGCN.

**Advertencia de método.** Los modelos de una misma semilla en distintos escenarios son casi
idénticos (sección «Veredicto»), así que no son réplicas independientes. La unidad
independiente es el modelo promediado sobre los cuatro escenarios: 3 semillas por arquitectura
(9 modelos). Una prueba de Friedman con 12 bloques (4 escenarios por 3 semillas) da p = 0,0001
para H2, pero ese valor está inflado por la repetición y **no debe citarse**.

**H2, arquitectura** (pesos por clase, media por semilla):

| Explicador y medida | GraphSAGE | GAT | TAGCN | ¿Mismo orden en las 3 semillas? |
|---|---|---|---|---|
| GNNExplainer, sp_all | 0,964 | 0,939 | 0,937 | GraphSAGE primera en las 3 |
| GNNExplainer, sp_nz | 0,844 | 0,734 | 0,677 | GraphSAGE primera en las 3 |
| GNNExplainer, jac10 | 0,635 | 0,623 | 0,624 | No: cambia en cada semilla |
| Shapley, sp_all | 0,993 | 0,996 | 0,987 | GAT primera en las 3 |
| Shapley, jac10 | 0,802 | 0,861 | 0,772 | GAT primera en las 3 |

Con 3 semillas no hay prueba formal posible. Lo que se puede decir: con la medida de la tesis
las diferencias son de hasta 0,03, dentro del margen de equivalencia de 0,05; el orden depende
del explicador (GraphSAGE con GNNExplainer, GAT con Shapley) y desaparece en las 10 variables
principales de GNNExplainer. TAGCN semilla 44 es un caso atípico (sp_nz 0,46).

**H3, pérdida** (9 modelos: 3 arquitecturas por 3 semillas):

| Explicador y medida | Pesos por clase | Sin ajuste | Focal loss | Pesos por clase mayor que focal loss |
|---|---|---|---|---|
| GNNExplainer, sp_all | 0,947 | 0,925 | 0,914 | 9 de 9 (prueba de signos, p = 0,004) |
| GNNExplainer, sp_nz | 0,751 | 0,650 | 0,589 | 9 de 9 (p = 0,004) |
| GNNExplainer, jac10 | 0,627 | 0,588 | 0,573 | 7 de 9 (p = 0,18) |
| Shapley, sp_all | 0,992 | 0,994 | 0,995 | 2 de 9 |
| Shapley, jac10 | 0,812 | 0,876 | 0,894 | 1 de 9 (p = 0,04, en sentido contrario) |

La pérdida sí tiene efecto, y su sentido depende del explicador: con GNNExplainer, pesos por
clase da las explicaciones más estables; con Shapley, las menos estables. Con la medida de la
tesis el tamaño es de centésimas (0,03 y 0,003).

## 4. Escenarios de estrés 1:100 y 1:200

- Rendimiento (PR-AUC de validación, GraphSAGE con pesos por clase): 0,46 nativo, 0,37 en
  1:100, 0,36 en 1:200. Sin ajuste: 0,18, 0,02 y 0,03 (azar 0,024).
- Estabilidad entre réplicas: sin cambio significativo (ver arriba).
- Acuerdo con el modelo nativo de la misma semilla, GNNExplainer, pesos por clase:

| Medida | 1:10 | 1:20 | 1:100 | 1:200 |
|---|---|---|---|---|
| sp_nz | 0,956 | 0,966 | 0,825 | 0,822 |
| jac10 | 0,783 | 0,810 | 0,522 | 0,513 |

Referencia: nativo contra nativo de otra semilla, jac10 = 0,383.

Quitar ilícitas cambia qué variables señala el modelo; submuestrear negativos casi no. El
cambio es menor que el que produce reentrenar con otra semilla.

## 5. Límites de esta validación

- El control de pesos al azar se corrió solo en el escenario nativo con pesos por clase.
- La caída de sp_nz en 1:100 y 1:200 viene sobre todo de un modelo (GraphSAGE, semilla 44: de
  0,86 a 0,59); con 9 modelos la prueba tiene poca potencia.
- El piso de 0,74 a 0,82 es un cálculo sobre los 30 nodos explicados, no una cota general.

## Referencias

- Adebayo, J. et al. (2018). Sanity checks for saliency maps. NeurIPS.
- Dal Pozzolo, A. et al. (2015). Calibrating probability with undersampling for unbalanced classification. IEEE Symposium Series on Computational Intelligence.
- Elkan, C. (2001). The foundations of cost-sensitive learning. IJCAI.
