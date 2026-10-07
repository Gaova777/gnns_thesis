# ¿Por qué los modelos B marcan al 15,6 % de los sin etiqueta? (reunión del 4-oct, THE-34)

## La pregunta

Los modelos B se entrenan con las ilícitas contra las lícitas etiquetadas, sin ver nunca un nodo sin etiqueta. Son 9 modelos: GCN, GraphSAGE y TAGCN, cada uno con 3 pérdidas, escenario nativo y semilla 42. Aplicados al grafo completo marcan como ilícitos, en validación:

| Grupo | Tasa de marcado (mediana de los 9 B) | Rango |
|---|---|---|
| Ilícitas | 71,3 % | 45,1 a 76,9 % |
| Lícitas | 1,7 % | 1,2 a 22,8 % |
| Sin etiqueta | **15,6 %** | 12,4 a 55,6 % |

GraphSAGE y TAGCN marcan entre el 12 y el 16 % de los sin etiqueta. GCN marca entre el 23 y el 56 %, y también marca muchas lícitas (hasta 22,8 %).

Cristian advirtió que decir que esos nodos «esconden fraude» no está justificado: que el modelo los marque no implica que lo sean. Pidió uno o dos argumentos contundentes sobre qué explica ese 15,6 %. Las hipótesis que propuso fueron:

1. que B aprendió algo mal porque las lícitas etiquetadas eran pocas;
2. drift en el tiempo;
3. un efecto de la estructura del grafo («más fraude en Bogotá que en Pereira porque hay más gente»).

## Resultado en una frase

El 15,6 % se explica porque **las lícitas etiquetadas no representan a la población sin etiqueta**: B nunca vio esa región del espacio de variables y, al llegar a ella, extrapola. No es fraude conocido, no es la estructura del grafo y no es drift temporal. Es la hipótesis 1 de Cristian.

## Los argumentos

### 1. Las lícitas etiquetadas no cubren a los sin etiqueta (argumento principal)

- **Un clasificador separa lícitas etiquetadas de sin etiqueta con ROC-AUC 0,984** en validación (0,990 en train). Control: dos mitades al azar de las lícitas dan 0,485.
  - Con un modelo de mezcla, si los sin etiqueta fueran lícitas iguales a las etiquetadas más una fracción π de otra cosa, todo clasificador cumpliría AUC ≤ 0,5 + π/2.
  - Con AUC 0,984, **π ≥ 0,97**: casi toda la población sin etiqueta tiene una distribución distinta de la de las lícitas etiquetadas.
- **PSI por variable** entre sin etiqueta y lícitas de validación:
  - mediana 0,139;
  - 57 de las 165 variables por encima de 0,25 (cambio fuerte, Siddiqi 2006).
  - En train el corrimiento es mayor: mediana 0,311 y 89 variables por encima de 0,25.
- **Cuanto menos se parece un nodo a las lícitas, más lo marca B.** Ordenando los sin etiqueta por el puntaje del clasificador de dominio:
  - el decil más «lícito» se marca al 2,2 %;
  - los deciles 7 y 8 se marcan al 20 %.
- **Lectura:** B aprendió la frontera «ilícita contra este tipo concreto de lícita». Los sin etiqueta caen fuera de lo que vio como lícito, y ahí cualquier frontera es una extrapolación.
  - Es lo que esperaba Cristian: «es más factible pensar que aprendió algo un poquito mal».

Figuras: `figuras/pca_tsne_grupos.png`, `figuras/psi_boxplot.png`, `figuras/marcado_por_decil_dominio.png`.

### 2. Los marcados no se parecen al fraude conocido

- **Un clasificador separa ilícitas etiquetadas de sin etiqueta marcados con ROC-AUC 0,994** (0,993 a 0,995 según el B).
  - Por el mismo argumento de mezcla, la fracción de marcados que podría distribuirse como el fraude etiquetado es **ρ ≤ 2 (1 − AUC) ≈ 1,2 %**.
  - Con el mejor B serían a lo sumo unos 43 de 3.664 nodos.
- Los marcados quedan *entre* los dos grupos, sin ser ninguno:
  - distancia de centroides a las ilícitas 9,5 y a las lícitas 16,6;
  - el 54 % tiene mayoría de ilícitas entre sus 5 vecinos más cercanos en el espacio de variables.

  Están más cerca del fraude que de las lícitas etiquetadas, pero no son fraude del tipo etiquetado.
- **Los B no se ponen de acuerdo en a quién marcar.** Si los marcados fueran fraude, los modelos coincidirían como coinciden con las ilícitas. No pasa:

  | Grupo | κ mediano (6 B de GraphSAGE y TAGCN) | Jaccard mediano (9 B) |
  |---|---|---|
  | Ilícitas | 0,72 | 0,64 |
  | Sin etiqueta | 0,67 | 0,29 |

  Solo el 13,1 % de los sin etiqueta lo marca la mayoría (5 de 9) y solo el 2,2 % lo marcan los 9. Lo que se marca depende del modelo: es extrapolación, no una señal estable.

### 3. No es la estructura del grafo

- **Los marcados tienen menos conexiones, no más:**
  - grado medio 1,73, frente a 2,08 de los no marcados;
  - vecindario a 2 saltos con mediana de 4 nodos, frente a 5.

  La idea de «más clientes, más conexiones» va en sentido contrario.
- **Tener ilícitas cerca casi no cambia la marca:**
  - 14,0 % con un vecino ilícito directo y 13,1 % sin él;
  - 15,9 % con una ilícita a 2 saltos y 12,8 % sin ella.

  Ninguna variable estructural predice la marca mejor que AUC 0,64 (la mejor es el grado de salida).
- **Un clasificador sin grafo reproduce el fenómeno.** Un HistGradientBoosting entrenado como B, sin aristas, marca el 7,3 % de los sin etiqueta. Una regresión logística entrenada igual marca el 11,2 %.
  - El efecto ya está en las variables del nodo y en la elección de negativos, no en el paso de mensajes del GNN.

Figura: `figuras/estructura_grafo.png`.

### 4. No es drift temporal

- La tasa de marcado de los sin etiqueta es **parecida en las tres particiones**: 9,2 % en train, 13,2 % en validación y 12,6 % en test.
  - En cambio, la de las ilícitas sí se desploma con el cambio de época (84 %, 75 % y 1,8 %).
  - Si el 15,6 % fuera drift, crecería con el tiempo como crece el corrimiento de las lícitas.
- **La correlación** entre la tasa de marcado por timestep y el PSI de ese timestep es −0,12.
- **La distancia es constante en el tiempo.** El PSI entre sin etiqueta y lícitas de train está entre 0,3 y 0,9 en *todos* los timesteps, también los de train: no aparece con el tiempo.

Figura: `figuras/marcado_por_timestep.png`.

### 5. Un explicador sencillo: B marca a los sin etiqueta por razones distintas de las que usa con las ilícitas

- **Árbol de decisión que imita la marca de B sobre los sin etiqueta** (profundidad 3):
  - AUC fuera de muestra 0,846 en test;
  - se apoya en dos variables agregadas: A39 (importancia 0,62) y A65 (0,17).
- **El mismo árbol ajustado a la marca de B sobre las etiquetadas** se apoya sobre todo en L53 (0,80).
- **Las regresiones logísticas L1 de los dos casos casi no comparten variables:**
  - Jaccard 0,11 entre sus 15 variables principales;
  - Spearman 0,27 entre los coeficientes.
- **Matiz:** las atribuciones gradiente por entrada *del propio GNN* sobre los marcados sí se parecen a las de las ilícitas bien detectadas (Spearman 0,89).
  - Pero se parecen igual a las de las lícitas falsamente marcadas (0,89).
  - El modelo tiene un solo «patrón de alarma»: todo lo que lo dispara lo dispara por las mismas variables.
  - Lo que separa a los marcados de los no marcados son otras variables (A39 y A65), no las del fraude.

Figura: `figuras/atribuciones_perfil.png`.

### Control: C, que sí ve los sin etiqueta, casi no los marca

- El C de la misma configuración marca al 2,9 % de los sin etiqueta en validación.
- De los que marca B, C también marca el 12,9 %, unas 24 veces más que entre los no marcados por B.
- C no sirve como prueba, porque se entrenó con ellos como negativos. Sí muestra que, cuando el modelo ve esa población, aprende a no confundirla con fraude y conserva un recall de ilícitas del 58 %.

## Conclusión para la tesis

1. El 15,6 % **no prueba que haya fraude escondido** entre los sin etiqueta: a lo sumo un 1,2 % de los marcados se parece al fraude etiquetado.
2. Se explica porque **las lícitas etiquetadas son una muestra no representativa**: B aprendió a separar el fraude de *esas* lícitas y extrapola sobre una población que nunca vio. No es la estructura del grafo ni el drift temporal.
3. Esto **respalda usar C por efectos prácticos**, como haría un banco. Los sin etiqueta son la mayor parte de los negativos reales, en su mayoría deben ser lícitos, y un modelo entrenado sin ellos marca de forma inestable y dependiente del modelo justo la población que más importa en operación.
4. **Límite que hay que declarar:** el análisis no demuestra que los sin etiqueta sean lícitos. Muestra que el modelo B no da evidencia de lo contrario, y que la única fracción compatible con el fraude conocido es pequeña.

## Archivos

| Qué | Ruta |
|---|---|
| Scripts | `scripts/v4/reunion_0410/sinetq_01_predicciones.py` a `sinetq_05_comparacion_c.py` y `sinetq_common.py` |
| Tasas por modelo, grupo y partición | `tasas_marcado.csv` |
| PSI, PCA, t-SNE y cobertura | `psi_por_feature.csv`, `psi_todos_B.csv`, `poblacion_cobertura.json`, `distancias_centroides.csv` |
| Grafo y tiempo | `grafo_por_grupo_val.csv`, `grafo_todos_B.csv`, `aristas_entre_grupos.csv`, `marcado_por_timestep.csv`, `grafo_tiempo.json` |
| Explicador | `explicador.json`, `atribuciones_grad_x_input.csv.gz` |
| Comparación con C y controles | `comparacion_B_C.csv`, `comparacion_c_controles.json`, `jaccard_marcados_entre_B.csv` |
| Figuras | `figuras/*.png` |
| No versionado (41 MB) | `probs.npz`: se regenera con `sinetq_01_predicciones.py` |

Referencia del umbral de PSI: Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. Wiley.
