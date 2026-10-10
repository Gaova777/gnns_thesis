# THE-41: el porqué de cada hipótesis (borrador para Resultados y Discusión)

Pedido de Cristian (7-oct, min. 82): «el porqué tiene que estar muy bien argumentado y
explicado con las ecuaciones, la misma arquitectura, su funcionalidad, sus puntos débiles y
sus puntos fuertes. No puede ser solo un número.»

Este borrador da, para cada hipótesis, la ecuación que explica el resultado y la evidencia
que la respalda. Solo usa afirmaciones que pasaron la validación del 10-oct
(`docs/REUNION_2026-10-07_CRISTIAN.md`, sección G). Lo que no pasó está al final, en «Lo que
no se puede afirmar».

**Estado (10-oct, 15:00):** completo con la corrida de 1:100 y 1:200 y con el control de pesos
al azar terminado (48 de 48). Las diferencias de H2 y H3 con las medidas nuevas siguen sin
prueba estadística.

Alcance de las cifras: 3 semillas del modelo (42, 43, 44) en todo lo que se afirma. Donde se
usa otra cosa se dice.

---

## 1. Antes de las hipótesis: qué mide la estabilidad y por qué sale alta

### 1.1 La medida

Para un modelo y un nodo, el explicador se ejecuta R = 5 veces. Cada ejecución r da un vector
de importancia por variable, s⁽ʳ⁾ ∈ ℝ¹⁶⁵. La estabilidad del nodo es la correlación de Spearman
promedio entre los pares de ejecuciones, y la del modelo es el promedio sobre los 30 nodos
explicados:

    E = (1/30) Σ_nodos (1/10) Σ_{r<r'} ρ( s⁽ʳ⁾, s⁽ʳ'⁾ )

### 1.2 Por qué las variables en cero inflan la medida

Las variables se escalan con la mediana y el rango intercuartílico del periodo de
entrenamiento: x' = (x − mediana) / IQR. Toda variable que vale exactamente su mediana queda
en 0. En los 30 nodos explicados eso ocurre en el **46 % de las variables** (mediana: 87 de
165 distintas de 0 por nodo); en el grafo completo, en el 39 %.

Los cuatro explicadores de variables le asignan importancia 0, o casi 0, a una variable que
vale 0. No es una coincidencia: sale de la ecuación de cada uno.

| Explicador | Importancia de la variable j del nodo | Si x_j = 0 |
|---|---|---|
| GNNExplainer | Aprende una máscara m por nodo y variable, y evalúa el modelo en x ⊙ σ(m). La importancia de j es el promedio de σ(m_ij) sobre los nodos del subgrafo. | Donde x_ij = 0, ∂(x_ij σ(m_ij))/∂m_ij = 0: esa entrada no recibe gradiente y la librería la deja en 0. Como el 83 % de los nodos tiene a lo sumo 2 aristas, casi todo el promedio es el propio nodo. |
| Integrated Gradients | φ_j = (x_j − b_j) · ∫₀¹ ∂f/∂x_j (b + α(x − b)) dα, con b = 0 | φ_j = 0 exacto. |
| Expected Gradients | φ_j = E_{b, α} [ (x_j − b_j) · ∂f/∂x_j (b + α(x − b)) ], con b tomado de negativos de entrenamiento | φ_j = 0 cada vez que la referencia también vale 0 en j, lo que pasa a menudo: el 39 % de las celdas del grafo vale 0. |
| Shapley (muestreo por permutaciones) | Contribución marginal de pasar de b_j a x_j, con b = media del subgrafo | Nula exacta si todo el subgrafo vale 0 en j, porque entonces b_j = x_j. |

Consecuencia: en cualquier ejecución, y con cualquier modelo, casi la mitad de las variables
cae al fondo del orden. Dos órdenes que solo comparten ese bloque, y que por lo demás son
independientes, ya tienen un Spearman de 0,74 a 0,82. Ese es el piso de la medida.

Qué tan al fondo quedan las variables en 0 (AUC de separar variables en 0 de las demás usando
la importancia; 1 = todas al fondo): Integrated Gradients 1,00 · GNNExplainer 0,98 · Expected
Gradients 0,76 · Shapley 0,73.

### 1.3 La prueba: una red sin entrenar

Es la prueba de aleatorización de parámetros de Adebayo et al. (2018): se explica la misma
arquitectura con los pesos entrenados y con pesos al azar, sobre los mismos 30 nodos. Si la
medida dependiera de lo que el modelo aprendió, debería caer con pesos al azar.

Escenario nativo, pesos por clase, 4 arquitecturas, 3 semillas (48 combinaciones).

| Explicador | Estabilidad, modelo entrenado | Estabilidad, red sin entrenar | Acuerdo entre ambas |
|---|---|---|---|
| GNNExplainer | 0,95 | 0,93 | 0,90 |
| Shapley | 0,97 | 0,96 | 0,91 |
| Expected Gradients | 0,93 | 0,93 | 0,69 |
| Integrated Gradients | 1 (determinista) | 1 | 0,96 |

Con la coincidencia de las 10 más importantes pasa lo mismo: GNNExplainer 0,66 entrenado y
0,63 sin entrenar; Expected Gradients 0,77 y 0,77; Shapley 0,82 y 0,90. En cambio, las 10 más
importantes del modelo entrenado y de la red al azar coinciden poco (0,20 a 0,40): la red
entrenada sí señala otras variables, pero la estabilidad entre réplicas no lo registra.

La estabilidad entre réplicas es casi la misma con y sin entrenamiento. **La medida describe
al explicador y a los datos de entrada, no al modelo.**

### 1.4 Dos medidas que no se inflan

- **Spearman sobre las variables distintas de 0** del nodo (ρ_nz): la misma correlación,
  calculada solo sobre las variables que el explicador sí puede ordenar.
- **Coincidencia de las 10 más importantes** (J₁₀): Jaccard entre los conjuntos de las 10
  variables con mayor importancia, |A ∩ B| / |A ∪ B|. Si comparten k variables, J₁₀ = k / (20 − k):
  compartir 5 da 0,33, compartir 7 da 0,54 y compartir 9 da 0,82.

GNNExplainer, pesos por clase, GraphSAGE, GAT y TAGCN, escenarios principales:

| Medida | Valor |
|---|---|
| Spearman sobre las 165 variables (la de la tesis) | 0,95 |
| Spearman sobre las variables distintas de 0 | 0,75 |
| Coincidencia de las 10 más importantes | 0,62 a 0,64 |

Todo lo que sigue se reporta con las tres.

---

## 2. H1: el nivel de desbalance no cambia la estabilidad

### 2.1 Qué se observó

Con pesos por clase, la estabilidad de GNNExplainer es la misma en los cuatro escenarios
principales con las tres medidas:

| Medida | 1:10 | 1:10 con SMOTE | 1:20 | Nativo (1:38,4) |
|---|---|---|---|---|
| Spearman, 165 variables | 0,946 a 0,947 en los cuatro | | | |
| Spearman, variables distintas de 0 | 0,749 a 0,753 en los cuatro | | | |
| Coincidencia de las 10 más importantes | 0,62 a 0,64 en los cuatro | | | |

### 2.2 Por qué: la pérdida no cambia cuando se submuestrean negativos

La entropía cruzada con pesos por clase, con w_c = N / (2 N_c), es

    L = (1/Σ_i w_{y_i}) Σ_i w_{y_i} ℓ_i  =  ½ · (1/N₁) Σ_{i: y_i=1} ℓ_i  +  ½ · (1/N₀) Σ_{i: y_i=0} ℓ_i

donde ℓ_i = −log p_i(y_i), N₁ es el número de ilícitas y N₀ el de negativos. Se comprobó que es
exactamente lo que calcula PyTorch (coincide al sexto decimal).

La razón entre clases, N₁ : N₀, **no aparece** en el lado derecho. Cada clase pesa la mitad
sin importar cuántos ejemplos tenga. Al pasar del nativo (132.803 negativos) a 1:10 (34.620
negativos) el primer término no cambia, porque no se quita ninguna ilícita, y el segundo
sigue siendo el promedio de la pérdida sobre los negativos, estimado con una muestra más
pequeña del mismo conjunto. El objetivo que se optimiza es el mismo; solo cambia el ruido con
que se estima la mitad de él.

Por eso los modelos de una misma semilla, que además parten de los mismos pesos iniciales y
usan los mismos hiperparámetros, terminan casi idénticos entre escenarios: el coseno entre
sus pesos es de 0,90 a 1,00 (entre dos semillas distintas es de 0,01 a 0,06) y sus puntajes
sobre validación tienen correlación de 0,96 a 0,99.

### 2.3 La comparación que lo demuestra

La pregunta correcta no es si la estabilidad promedio se mueve, sino si cambiar el escenario
cambia **la explicación de un mismo nodo** más de lo que ya la cambia volver a entrenar con
otra semilla. GNNExplainer; GraphSAGE, GAT y TAGCN; Spearman sobre las 165 variables:

| Comparación entre dos modelos | Pesos por clase | Sin ajuste | Focal loss |
|---|---|---|---|
| Mismo escenario, otra semilla (referencia) | 0,940 | 0,936 | 0,924 |
| Otro escenario, otra semilla | 0,941 | 0,918 | 0,915 |

Con pesos por clase, cambiar el escenario no agrega nada a lo que ya mueve la semilla (0,941
frente a 0,940). Con las otras dos pérdidas agrega poco: entre una y dos centésimas.

No se debe citar el acuerdo entre escenarios con la **misma** semilla (0,987): esos modelos
son casi el mismo modelo, por lo dicho en 2.2, y compararlos contra una referencia de otra
semilla es comparar cosas distintas.

### 2.4 Qué pasa cuando sí se toca el fraude: 1:100 y 1:200

En modo C el nativo ya usa todos los negativos, así que 1:100 y 1:200 se alcanzan quitando
ilícitas: quedan 1.328 y 664 de 3.462. Aquí el primer término de la ecuación de 2.2 sí
cambia: el promedio de las ilícitas se estima con el 38 % y el 19 % de los casos.

**El rendimiento cae.** PR-AUC en validación, media de 3 semillas:

| Configuración | Nativo | 1:100 | 1:200 |
|---|---|---|---|
| GraphSAGE, pesos por clase | 0,46 | 0,37 | 0,36 |
| TAGCN, pesos por clase | 0,30 | 0,28 | 0,28 |
| GAT, pesos por clase | 0,32 | 0,34 | 0,32 |
| GraphSAGE, focal loss | 0,31 | 0,11 | 0,02 |
| GraphSAGE, sin ajuste | 0,18 | 0,02 | 0,03 |

El azar es 0,024. Sin ajuste, GraphSAGE y TAGCN no aprenden nada desde 1:100; con focal loss
dejan de aprender en 1:200. Solo pesos por clase resiste.

**La estabilidad entre réplicas no cae.** GNNExplainer, pesos por clase, GraphSAGE, GAT y TAGCN:

| Medida | 1:10 | 1:20 | Nativo | 1:100 | 1:200 |
|---|---|---|---|---|---|
| Spearman, 165 variables | 0,946 | 0,946 | 0,947 | 0,940 | 0,938 |
| Spearman, variables distintas de 0 | 0,749 | 0,751 | 0,753 | 0,717 | 0,708 |
| Coincidencia de las 10 más importantes | 0,62 | 0,63 | 0,62 | 0,61 | 0,62 |

Prueba de Friedman sobre los seis escenarios (9 modelos: 3 arquitecturas por 3 semillas):
p = 0,84 con la medida de la tesis y p = 0,35 sobre las variables distintas de 0. La baja de
0,75 a 0,71 viene casi toda de un modelo (GraphSAGE, semilla 44: de 0,86 a 0,59) y no es
significativa (Wilcoxon contra el nativo: p = 0,43 en 1:100 y p = 0,07 en 1:200). Incluso los
modelos que quedaron al nivel del azar dan 0,90 a 0,93 con la medida de la tesis.

**Lo que sí cambia es qué variables señala el modelo.** Acuerdo con el modelo nativo de la
misma semilla (GNNExplainer, pesos por clase):

| Comparación con el nativo, misma semilla | 1:10 | 1:20 | 1:100 | 1:200 |
|---|---|---|---|---|
| Spearman, variables distintas de 0 | 0,96 | 0,97 | 0,83 | 0,82 |
| Coincidencia de las 10 más importantes | 0,78 | 0,81 | 0,52 | 0,51 |

Submuestrear negativos deja al modelo casi igual (un Jaccard de 0,80 equivale a compartir 9
de las 10 variables principales); quitar ilícitas lo cambia (0,52 equivale a compartir 7). El mismo patrón aparece en
Shapley (0,86 a 0,88 frente a 0,62 a 0,63), Integrated Gradients (0,84 a 0,89 frente a 0,60 a
0,66) y Expected Gradients (0,82 a 0,86 frente a 0,52 a 0,55).

Para poner ese cambio en escala: volver a entrenar el nativo con otra semilla da 0,38, que
equivale a compartir entre 5 y 6 de las 10. Quitar el 60 % o el 80 % de las ilícitas cambia la explicación, pero menos de
lo que la cambia la semilla.

### 2.5 Cómo decirlo

H1 no se rechaza. En el rango 1:10 a 1:38,4, con pesos por clase, no podía rechazarse: los
escenarios cambian una cantidad que no entra en la función de pérdida, y el resultado respalda
la decisión de diseño de no tocar las ilícitas. En 1:100 y 1:200 tampoco se rechaza con la
estabilidad entre réplicas, pero esa medida no distingue un modelo que aprendió de uno que no
(sección 1.3). Lo que sí registra el desbalance severo es el rendimiento y el acuerdo con el
modelo nativo: el modelo aprende menos y señala otras variables.

---

## 3. H2: la arquitectura

### 3.1 La ecuación de cada capa

La diferencia entre las cuatro está en cómo entra el propio nodo i en su representación.

| Arquitectura | Capa | Cómo entra el propio nodo |
|---|---|---|
| GCN (Kipf y Welling 2017) | h_i' = σ( Σ_{j ∈ N(i) ∪ {i}} W h_j / √(d_i d_j) ) | Como un vecino más: comparte W con los vecinos y se promedia con ellos. |
| GraphSAGE (Hamilton et al. 2017) | h_i' = σ( W₁ h_i + W₂ · media_{j ∈ N(i)} h_j ) | Con una matriz propia, W₁, separada de la de los vecinos. |
| GAT (Veličković et al. 2018) | h_i' = σ( Σ_{j ∈ N(i) ∪ {i}} α_ij W h_j ) | Con un peso de atención propio, α_ii, que se aprende. |
| TAGCN (Du et al. 2017) | h_i' = σ( Σ_{k=0}^{K} (Âᵏ H)_i W_k ) | El término k = 0 es el propio nodo con su matriz W₀. |

### 3.2 Por qué GCN no aprende en Elliptic

En Elliptic la señal del fraude está en las variables del nodo, no en sus vecinos:

- solo el 12 % de los vecinos de una transacción ilícita es ilícito;
- un perceptrón multicapa que ignora el grafo rinde igual que GraphSAGE (PR-AUC en validación
  0,485 frente a 0,481).

GCN no puede separar al nodo de sus vecinos: usa la misma W para todos y los promedia. Cuando
los vecinos no se parecen al nodo, el promedio diluye la señal. Las otras tres arquitecturas
tienen un camino propio para el nodo (W₁, α_ii, W₀) y pueden ignorar a los vecinos si no
ayudan. La prueba directa: al agregarle a GCN un peso propio para el nodo, su PR-AUC en
validación sube de 0,214 a 0,464, al nivel de las demás
(`results_v4/reunion_0410/pgexpl_gcn/REPORTE.md`, parte B).

### 3.3 Puntos fuertes y débiles en este problema

| Arquitectura | Punto fuerte | Punto débil | En Elliptic (pesos por clase, nativo) |
|---|---|---|---|
| GCN | La más simple y la más barata. | Sin camino propio para el nodo: depende de que los vecinos se parezcan a él. | No pasa la compuerta en ningún escenario. |
| GraphSAGE | Separa nodo y vecinos; escala bien. | La media trata igual a todos los vecinos. | La de mejor rendimiento. |
| GAT | Pondera a cada vecino. | Más memoria (se corrió con 4 cabezas y 2 capas por el límite de 8 GB) y probabilidades desplazadas: solo pasa la compuerta con el umbral calibrado. | Rendimiento intermedio. |
| TAGCN | Mira hasta K saltos en una sola capa. | Más parámetros por capa; el campo receptivo crece rápido. | Pasa la compuerta; rendimiento por debajo de GraphSAGE. |

### 3.4 Qué pasa con la estabilidad

Con el Spearman sobre las 165 variables, las arquitecturas se ven equivalentes. Con las
medidas que no se inflan (GNNExplainer, pesos por clase, escenarios principales) aparece una
separación en una de las dos:

| Arquitectura | Spearman, variables distintas de 0 | Coincidencia de las 10 más importantes |
|---|---|---|
| GraphSAGE | 0,84 | 0,64 |
| GAT | 0,73 | 0,62 |
| GCN | 0,72 | 0,67 |
| TAGCN | 0,68 | 0,62 |

Sobre las variables distintas de 0, GraphSAGE queda por encima; en las 10 más importantes las
cuatro son indistinguibles. **A estas diferencias todavía no se les ha hecho prueba
estadística**: no se puede afirmar una separación hasta rehacer el contraste de H2 con estas
medidas.

### 3.5 Cómo decirlo

La arquitectura decide **si el modelo aprende**: la que no tiene un camino propio para el nodo
no aprende en un grafo donde los vecinos no se parecen al nodo. Entre las que aprenden, la
conclusión sobre la estabilidad depende de la medida y queda abierta hasta rehacer la prueba.

---

## 4. H3: la función de pérdida

### 4.1 Las tres pérdidas

    Sin ajuste:       L = (1/N) Σ_i ℓ_i
    Pesos por clase:  L = ½ · media_{y=1} ℓ_i + ½ · media_{y=0} ℓ_i
    Focal loss:       L = (1/N) Σ_i α_{y_i} (1 − p_i)^γ ℓ_i        (Lin et al. 2017)

con p_i la probabilidad que el modelo le da a la clase correcta, α = 0,75 para las ilícitas,
0,25 para los negativos y γ = 2.

- **Sin ajuste**, las ilícitas pesan lo que su frecuencia: 2,5 % de la pérdida en el nativo.
- **Pesos por clase** fija el reparto en mitad y mitad.
- **Focal loss** reduce el peso de los ejemplos que el modelo ya clasifica bien con el factor
  (1 − p_i)^γ. Con γ = 2, un ejemplo con p = 0,9 pesa 100 veces menos que uno con p = 0. El
  gradiente se concentra en los casos difíciles y se desvanece cerca del óptimo.

### 4.2 Por qué pesos por clase rinde mejor

Por cuánto compensa cada una el desbalance.

- **Pesos por clase** usa w_c = N / (2 N_c). En el nativo, el peso de una ilícita es 19,7 y el
  de un negativo 0,51: la razón es 38,4, exactamente el desbalance.
- **Focal loss** compensa por clase solo con α: una ilícita pesa 3 veces lo que un negativo
  (0,75 / 0,25), no 38. El resto lo debe hacer el factor (1 − p_i)^γ, que baja el peso de los
  ejemplos fáciles pero no distingue clases.

En el nativo la diferencia entre compensar 38 a 1 y compensar 3 a 1 es grande, y pesos por
clase gana. En 1:10 es 10 a 1 contra 3 a 1, y las curvas se acercan. En el 1:100 se amplía:
PR-AUC en validación de 0,37 (GraphSAGE, pesos por clase) frente a 0,11 (focal loss) y 0,02
(sin ajuste, nivel de azar).

Limitación que hay que declarar: focal loss se propuso con α y γ ajustados juntos (Lin et al.
2017). Aquí quedaron fijos en 0,75 y 2 y no se buscaron.

### 4.3 Qué pasa con la estabilidad

GraphSAGE, GAT y TAGCN, escenarios principales, GNNExplainer:

| Pérdida | Spearman, 165 variables | Spearman, variables distintas de 0 | Coincidencia de las 10 más importantes |
|---|---|---|---|
| Pesos por clase | 0,95 | 0,75 | 0,63 |
| Sin ajuste | 0,93 | 0,65 | 0,59 |
| Focal loss | 0,91 | 0,59 | 0,57 |

El orden es el mismo con las tres medidas: pesos por clase, sin ajuste, focal loss. Con la
medida de la tesis la diferencia es de cuatro centésimas; sobre las variables distintas de 0,
de dieciséis. **Tampoco tiene todavía prueba estadística con las medidas nuevas.**

Una lectura coherente con 4.1, que hay que presentar como hipótesis y no como resultado: la
estabilidad de GNNExplainer depende de qué tan nítido es el gradiente del modelo respecto a
sus variables. Focal loss produce modelos con probabilidades menos extremas y gradientes más
planos; la máscara que aprende GNNExplainer tiene entonces menos señal que seguir y depende
más de su inicio al azar.

### 4.4 PGExplainer

PGExplainer ordena aristas, no variables. El 83 % de los nodos explicados tiene a lo sumo 2
aristas: su Spearman se calcula sobre dos o tres elementos y lo que registra es ruido del
instrumento. Lo que se mueve con la pérdida en PGExplainer no es evidencia a favor de H3
(`results_v4/reunion_0410/pgexpl_gcn/REPORTE.md`, parte A).

---

## 5. Los explicadores: fuertes, débiles y por qué no coinciden

| Explicador | Qué responde | Punto fuerte | Punto débil |
|---|---|---|---|
| GNNExplainer (Ying et al. 2019) | Qué máscara sobre variables y aristas conserva la predicción | Da variables y aristas a la vez | Optimización con inicio al azar: dos ejecuciones no dan lo mismo |
| PGExplainer (Luo et al. 2020) | Qué aristas importan, con una red entrenada una sola vez | Rápido una vez entrenado | Solo aristas; inservible con nodos de 2 aristas |
| Shapley por permutaciones | Cuánto aporta cada variable en promedio sobre coaliciones | Base axiomática | Estimación por muestreo; depende del valor de referencia |
| Integrated Gradients (Sundararajan et al. 2017) | Cuánto del cambio en el logit se debe a cada variable desde una referencia | Determinista; cumple completitud | Depende de la referencia; con referencia 0 sigue de cerca a \|x\| (ρ = 0,98) |
| Expected Gradients (Erion et al. 2021) | Lo mismo, promediando sobre referencias reales | No depende de una sola referencia | Estimación por muestreo |

**No coinciden entre sí.** Sobre el mismo modelo y el mismo nodo, la coincidencia de las 10
variables más importantes entre dos explicadores va de 0,13 a 0,42. Con el Spearman sobre las
165 variables el acuerdo parece mayor (0,42 a 0,84), por el bloque de ceros de la sección 1.2.

**Tampoco coinciden entre semillas del modelo.** Las 10 variables más importantes de un nodo
coinciden en 0,39 (GNNExplainer), 0,52 (Shapley), 0,53 (Integrated Gradients) y 0,39 (Expected
Gradients) entre dos modelos idénticos salvo por la semilla.

Esto refuerza la tesis de fondo: que un explicador sea estable entre réplicas no dice que la
explicación sea correcta, ni siquiera que sea la misma si se reentrena el modelo.

---

## 6. Lo que no se puede afirmar

| Afirmación | Por qué no |
|---|---|
| «El submuestreo solo desplaza el logit en log(1/β)» (Elkan 2001; Dal Pozzolo et al. 2015), como mecanismo observado | En los modelos sin ajuste el desplazamiento mediano entre nativo y 1:10 es de −0,4 a 0,2 (la teoría predice +1,34) y la correlación de puntajes es de 0,40 a 0,80. Vale como expectativa teórica para el clasificador óptimo, no como lo que pasó. |
| «Entre escenarios el acuerdo es 0,99, más que entre semillas» | Compara modelos de la misma semilla, que son casi idénticos, contra una referencia de otra semilla. Ver 2.3. |
| «Las tres arquitecturas se apoyan en las mismas variables» | Entre arquitecturas el Jaccard de las 10 variables más importantes es de 0,20 a 0,50: comparten de 3 a 7 de las 10. |
| «La estabilidad de 0,93 a 0,99 muestra que el modelo aprendió algo consistente» | Una red sin entrenar da casi la misma cifra. Ver 1.3. |

---

## Referencias

- Adebayo, J. et al. (2018). Sanity checks for saliency maps. NeurIPS.
- Dal Pozzolo, A. et al. (2015). Calibrating probability with undersampling for unbalanced classification. IEEE Symposium Series on Computational Intelligence.
- Du, J. et al. (2017). Topology adaptive graph convolutional networks. arXiv:1710.10370.
- Elkan, C. (2001). The foundations of cost-sensitive learning. IJCAI.
- Erion, G. et al. (2021). Improving performance of deep learning models with axiomatic attribution priors and expected gradients. Nature Machine Intelligence, 3, 620-631.
- Hamilton, W. et al. (2017). Inductive representation learning on large graphs. NeurIPS.
- Kipf, T. y Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR.
- Lin, T.-Y. et al. (2017). Focal loss for dense object detection. ICCV.
- Luo, D. et al. (2020). Parameterized explainer for graph neural network. NeurIPS.
- Sundararajan, M. et al. (2017). Axiomatic attribution for deep networks. ICML.
- Veličković, P. et al. (2018). Graph attention networks. ICLR.
- Ying, R. et al. (2019). GNNExplainer: generating explanations for graph neural networks. NeurIPS.
