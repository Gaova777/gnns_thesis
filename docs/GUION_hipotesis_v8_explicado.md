# Guion explicado del deck de hipótesis, versión 8

Acompaña el deck «Hipótesis v4», versión 8 (32 láminas), que se genera con
`scripts/v4/reunion_0710/deck_hipotesis_v8.py`. Reemplaza, para este deck, la segunda mitad de
`docs/GUION_presentacion_v4_explicado.md`, que describe la versión de 26 láminas.

Está escrito para alguien que no conoce la tesis. Cada lámina tiene tres partes:

1. **Qué decir.** Un discurso para leer en voz alta, de 30 segundos a 1 minuto.
2. **Qué hay detrás.** De dónde sale lo que se ve y cómo se hizo. No hay que decirlo todo.
3. **Si preguntan.** Preguntas probables, con una respuesta corta.

Todas las cifras son la media de 3 entrenamientos del mismo modelo (3 semillas), salvo donde
se dice otra cosa.

---

## La historia en un minuto

1. Un banco usa un modelo para marcar transacciones sospechosas de lavado. Cuando marca una,
   alguien tiene que justificar por qué. Para eso están los **explicadores**: algoritmos que
   dicen qué datos de la transacción pesaron en la decisión.
2. Los explicadores tienen una parte aleatoria. Si al repetirlos dan una razón distinta cada
   vez, esa razón no sirve ante un auditor. Que la razón se repita es la **estabilidad**.
3. En lavado de dinero el fraude es muy escaso: en nuestros datos hay 1 transacción ilícita
   por cada 40. A eso se le llama **desbalance**. La pregunta de la tesis es si el desbalance,
   el tipo de red o la forma de compensar el desbalance vuelven inestables las explicaciones.
4. Resultado: ninguno de los tres cambia la estabilidad de forma apreciable, ni siquiera con 1
   fraude por cada 200.
5. Y el hallazgo que lo explica: **la estabilidad no dice si el modelo aprendió**. Una red sin
   entrenar da casi la misma cifra. Estable no es lo mismo que correcto.

## Diccionario

| Término | En palabras simples |
|---|---|
| Transacción | Un pago de Bitcoin. Es un punto del grafo. |
| Variables | Los 165 datos que describen cada transacción. Están anonimizados. |
| Red (GNN) | El modelo. Decide sobre cada transacción mirando sus datos y los de sus vecinas. Hay cuatro tipos: GCN, GraphSAGE, GAT y TAGCN. |
| Desbalance 1:40 | Por cada transacción ilícita hay 40 que no lo son. |
| Compensar el desbalance | Cambiar la forma de entrenar para que el fraude pese más. Tres opciones: no hacer nada («sin ajuste»), **pesos por clase** y **focal loss**. |
| Validación | Periodo que el modelo no usó para entrenar, con datos parecidos a los de entrenamiento. Ahí se mide si aprendió. |
| Prueba final (test) | El último periodo. Ahí cierra un mercado ilegal y el fraude cambia de forma: los modelos dejan de acertar. |
| PR-AUC | Nota de 0 a 1 de qué tan bien detecta fraude el modelo. Adivinar al azar da 0,024 en validación. |
| Semilla | El número que fija el azar del entrenamiento. Tres semillas son tres entrenamientos del mismo modelo. |
| Filtro de calidad (compuerta) | Regla que deja pasar solo los modelos que sí aprendieron. No tiene sentido explicar un modelo que adivina. |
| Explicador | Algoritmo que dice qué variables pesaron en una decisión. Usamos GNNExplainer, Shapley, Integrated Gradients, Expected Gradients y PGExplainer. |
| Estabilidad | De 0 a 1: cuánto coinciden dos ejecuciones del explicador sobre la misma transacción. |

---

## Lámina 1. Portada

**Qué decir.** Esta es la segunda parte del avance. En la primera contamos el problema y el
método. Aquí vamos hipótesis por hipótesis: qué esperábamos, qué resultó y por qué.

---

## Lámina 2. Qué medimos: la estabilidad

**Qué decir.** Tomamos una transacción que el modelo marcó como sospechosa y le pedimos al
explicador que diga qué variables pesaron. Lo repetimos cinco veces. Si las cinco veces
ordena las variables casi igual, la explicación es estable. La cifra va de 0 a 1.

**Qué hay detrás.** La medida es la correlación de Spearman entre los órdenes de importancia
de las 165 variables, promediada sobre los pares de ejecuciones y sobre 30 transacciones. Las
30 son las mismas para todos los modelos.

**Si preguntan.**
- *¿Por qué importa?* Porque una razón que cambia cada vez que se pide no se puede presentar a
  un auditor.
- *¿Estable quiere decir correcta?* No. Es el tema de las láminas 25 a 27.

---

## Lámina 3. Cómo se evalúa cada hipótesis

**Qué decir.** Cada hipótesis responde las mismas seis preguntas: qué esperábamos, por qué,
qué resultó, si era lo esperado, por qué resultó así y en qué impacta. Y solo usamos los
modelos que sí aprendieron: de 48 combinaciones, 28.

**Qué hay detrás.** 60 combinaciones en total: 4 redes, 5 niveles de desbalance y 3 formas de
compensarlo. Se aparta el nivel 1:1 (va en anexo) y quedan 48. Pasan el filtro de calidad 28:
GraphSAGE 11 de 12, TAGCN 11 de 12, GAT 6 de 12 y GCN 0 de 12.

**Si preguntan.**
- *¿Por qué quitar los que no aprenden?* Explicar un modelo que adivina no informa nada sobre
  el fraude.

---

## Lámina 4. Cómo leer las gráficas de rendimiento

**Qué decir.** El modelo le pone a cada transacción un puntaje de sospecha. Según dónde se
ponga el umbral de alerta, detecta más fraude pero da más falsas alarmas. Cada punto de la
curva es un umbral distinto. La curva ROC dice si separa bien las dos clases; la curva PR dice
cuántas de las alertas eran fraude de verdad, y es la que más importa cuando el fraude es
escaso.

**Qué hay detrás.** En ROC la diagonal es adivinar. Una curva por debajo de la diagonal no es
peor que el azar: el modelo ordena al revés, y eso pasa cuando los datos de la prueba final ya
no se parecen a los de entrenamiento. En PR el nivel del azar es la proporción de fraude: 2,4 %
en validación y 0,6 % en la prueba final.

**Si preguntan.**
- *¿Por qué dos curvas?* Con tan poco fraude, la ROC se ve bien aunque casi todas las alertas
  sean falsas. La PR lo delata (Saito y Rehmsmeier 2015).

---

## Lámina 5. Rendimiento de las cuatro redes

**Qué decir.** Aquí están las cuatro redes con la misma forma de compensar el desbalance, para
compararlas en igualdad. En validación, tres detectan fraude y GCN se queda sola abajo. En la
prueba final las cuatro quedan pegadas al azar. Por eso la estabilidad la medimos en
validación, donde el modelo sí aprendió.

**Qué hay detrás.** PR-AUC en validación: GraphSAGE 0,46, GAT 0,32, TAGCN 0,30 y GCN 0,10. En
la prueba final, las cuatro entre 0,008 y 0,010. La línea es la media de 3 semillas, la sombra
va del mínimo al máximo y cada punto es el umbral con que se reporta.

**Si preguntan.**
- *¿Por qué caen en la prueba final?* En ese periodo cierra un mercado ilegal y el fraude que
  queda es distinto del que el modelo vio. Es un hecho conocido de este conjunto de datos
  (Weber et al. 2019).

---

## Lámina 6. Las cuatro métricas de rendimiento

**Qué decir.** La misma historia en números y con cuatro métricas distintas: tres redes
aprenden en validación, GCN no, y todas fallan en la prueba final. GCN va al final porque no
pasa el filtro de calidad.

**Qué hay detrás.** Las métricas son ROC-AUC, PR-AUC, F1 y KS. KS mide la distancia máxima
entre la distribución de puntajes del fraude y la de lo demás.

---

## Lámina 7. Por qué el filtro de calidad usa validación

**Qué decir.** Nos preguntaron: si GCN no pasa el filtro, ¿por qué en la prueba final no es la
peor? Porque en la prueba final todos están cerca del azar, y ahí el orden lo decide la suerte
del entrenamiento. Y hay una razón de principio: si escogemos modelos con la prueba final, deja
de ser una prueba honesta.

**Qué hay detrás.** Cada punto es una semilla. En validación las tres de GCN quedan lejos de
las demás. En la prueba final se mezclan con las de GAT. Escoger con los mismos datos con que
se reporta infla el resultado (Cawley y Talbot 2010).

**Si preguntan.**
- *¿Cuál es la regla del filtro?* F1 de al menos 0,30 y MCC de al menos 0,15 en validación.
- *¿Y GAT con pesos por clase?* Solo pasa si se evalúa con el umbral ajustado. Es una decisión
  pendiente.

---

## Láminas 8 a 15. Rendimiento en cada nivel de desbalance

Las ocho siguen el mismo molde: arriba ROC, abajo PR; una columna por red; un color por forma
de compensar el desbalance. Azul es «sin ajuste», la referencia.

### Lámina 8. Desbalance real, validación

**Qué decir.** Con el desbalance real, entrenar sin compensarlo (la curva azul) da el peor
resultado en las cuatro redes. GraphSAGE con pesos por clase es la mejor.

### Lámina 9. Desbalance real, prueba final

**Qué decir.** En la prueba final ninguna curva se separa del azar. Dos curvas ROC quedan por
debajo de la diagonal: el modelo ordena al revés porque el fraude cambió.

**Qué hay detrás.** Aquí la precisión va en escala logarítmica, porque todos los valores están
cerca de 0,01.

### Lámina 10. Con 1 fraude por cada 10, validación

**Qué decir.** Aquí quitamos transacciones no ilícitas del entrenamiento hasta dejar 10 por
cada fraude. Las tres curvas se acercan: como ya hay menos desbalance, queda menos que
compensar. «Sin ajuste» no quiere decir sin balancear: quitar negativos ya es una forma de
balanceo.

### Lámina 11. Con 1 fraude por cada 10, prueba final

**Qué decir.** Se repite la caída al azar. Cambiar el desbalance del entrenamiento no arregla
que el fraude de la prueba final sea distinto.

### Lámina 12. Con fraude sintético añadido (SMOTE), validación

**Qué decir.** Aquí, además, duplicamos el fraude con casos sintéticos. El rendimiento queda
casi igual que en la lámina 10.

**Qué hay detrás.** SMOTE crea cada caso nuevo mezclando los datos de un fraude real con los de
uno parecido (Chawla et al. 2002). Solo se usa en entrenamiento.

### Lámina 13. Con fraude sintético añadido (SMOTE), prueba final

**Qué decir.** El fraude sintético tampoco evita la caída.

### Lámina 14. Con 1 fraude por cada 20, validación

**Qué decir.** Un nivel intermedio. Se repite lo del desbalance real: GraphSAGE con pesos por
clase arriba.

### Lámina 15. Con 1 fraude por cada 20, prueba final

**Qué decir.** Otra vez la caída al azar. Con esto cerramos el rendimiento y pasamos a las
hipótesis.

**Si preguntan (para las ocho).**
- *¿Por qué mostrar la prueba final si es mala?* Porque es el resultado honesto. Ocultarla
  sería reportar solo lo que salió bien.
- *¿Qué es la sombra?* El mínimo y el máximo de las 3 semillas. Sombra ancha: los tres
  entrenamientos no coinciden.

---

## Lámina 16. Hipótesis 1, desbalance: qué esperábamos y qué resultó

**Qué decir.** Esperábamos que con menos fraude las explicaciones se volvieran inestables,
porque el modelo tendría menos de dónde aprender. No pasó: entre 1 por cada 10 y 1 por cada
40, la estabilidad es la misma, 0,93.

**Qué hay detrás.** Se usó una prueba de equivalencia (TOST) con margen de 0,05: no solo «no
encontramos diferencia», sino «la diferencia, si existe, es menor que 0,05» (Lakens 2017).

**Si preguntan.**
- *¿No es raro esperar una cosa y obtener otra?* Es el resultado. La lámina siguiente explica
  por qué, y la 18 lo pone a prueba con un caso extremo.

---

## Lámina 17. Hipótesis 1: por qué

**Qué decir.** Para bajar el desbalance quitamos transacciones no ilícitas, pero no tocamos el
fraude. Con pesos por clase, además, cada clase pesa la mitad sin importar cuántos casos tenga.
Entonces el modelo aprende casi lo mismo en todos los niveles. Lo comprobamos comparando la
explicación de la misma transacción entre dos modelos: cambiar de nivel de desbalance no la
mueve más de lo que ya la mueve volver a entrenar.

**Qué hay detrás.** La función que se minimiza con pesos por clase es
L = ½ · media(ilícitas) + ½ · media(negativos). La proporción entre clases no aparece.
Acuerdo entre explicaciones: mismo nivel y otra semilla 0,940; otro nivel y otra semilla 0,941.

**Si preguntan.**
- *¿Entonces la hipótesis no podía fallar?* En ese rango y con pesos por clase, no. Por eso se
  hizo el caso extremo de la lámina 18.
- *¿No dice la teoría que submuestrear solo desplaza el puntaje?* Es la expectativa teórica
  (Elkan 2001), pero en nuestros modelos sin ajuste no se observó. No lo afirmamos.

---

## Lámina 18. Hipótesis 1, caso extremo: 1 por cada 100 y 1 por cada 200

**Qué decir.** Nos pidieron forzar el desbalance. Como ya usábamos todos los negativos, la
única forma era quitar fraude: nos quedamos con 1.328 y con 664 casos de 3.462. El modelo
detecta peor y se fija en otras variables. Pero la estabilidad entre repeticiones no cambia.

**Qué hay detrás.**
- Rendimiento de GraphSAGE con pesos por clase: 0,46 con todo el fraude, 0,37 y 0,36 en los
  dos extremos. Sin ajuste, cae al azar.
- Estabilidad: 0,947, 0,940 y 0,938. La prueba de Friedman no encuentra diferencia (p = 0,84).
- Coincidencia de las 10 variables principales con el modelo original: 0,78 a 0,81 cuando solo
  se quitan negativos; 0,52 y 0,51 cuando se quita fraude.

**Si preguntan.**
- *¿Por qué no sobremuestrear negativos en vez de quitar fraude?* Triplica el tamaño del grafo
  y no cabe en la memoria de la tarjeta gráfica. Queda declarado.
- *¿Qué significa 0,52?* Es un índice de Jaccard. Equivale a compartir 7 de las 10 variables.
  Reentrenar el mismo modelo con otra semilla da 0,38: entre 5 y 6 de 10.

---

## Lámina 19. Hipótesis 2, tipo de red: qué esperábamos y qué resultó

**Qué decir.** Esperábamos que el tipo de red cambiara la estabilidad, porque cada una combina
de forma distinta la transacción con sus vecinas. Entre las tres que aprenden, la estabilidad
es casi la misma. GCN no aprende, así que no se explica.

**Qué hay detrás.** GraphSAGE 0,94, GAT 0,93 y TAGCN 0,92, equivalentes con margen de 0,05.

**Si preguntan.**
- *¿Es definitivo?* No del todo. Con una medida más exigente (lámina 25) GraphSAGE queda por
  encima: 0,84 frente a 0,73 y 0,68. Falta la prueba estadística.

---

## Lámina 20. Hipótesis 2: por qué GCN no aprende

**Qué decir.** En estos datos la señal del fraude está en la propia transacción, no en sus
vecinas: solo el 12 % de las vecinas de un fraude es fraude. GCN mezcla la transacción con sus
vecinas y diluye la señal. La prueba: un modelo que ignora el grafo rinde igual que GraphSAGE,
y si a GCN le damos un peso propio para la transacción, sube de 0,21 a 0,46.

**Si preguntan.**
- *¿Entonces el grafo no sirve?* En este conjunto de datos aporta poco al rendimiento. Es una
  propiedad de Elliptic, no de las redes de grafos.

---

## Lámina 21. Hipótesis 2: la ecuación de cada red

**Qué decir.** La diferencia está en una sola pieza: si la transacción tiene un peso propio o
se promedia con sus vecinas. GCN no lo tiene. Las otras tres sí, cada una a su manera.

**Qué hay detrás.**

| Red | Cómo entra la propia transacción |
|---|---|
| GCN | Como una vecina más: mismo peso, se promedia. |
| GraphSAGE | Con una matriz propia, separada de la de las vecinas. |
| GAT | Con un peso de atención propio, que se aprende. |
| TAGCN | Con un término propio (el de cero saltos). |

---

## Lámina 22. Hipótesis 3, forma de compensar el desbalance

**Qué decir.** Esperábamos que no cambiara la estabilidad, y así fue en los explicadores que
ordenan variables. El único que cambia es PGExplainer, que ordena conexiones y no variables.

**Qué hay detrás.** GNNExplainer: pesos por clase 0,95, sin ajuste 0,93, focal loss 0,91.
Shapley: 0,99 en las tres.

**Si preguntan.**
- *¿Por qué pesos por clase rinde mejor que focal loss?* Pesos por clase compensa 38 a 1, el
  desbalance completo. Focal loss, como se configuró, solo 3 a 1.
- *¿Es definitivo?* Con la medida más exigente la diferencia crece: 0,75, 0,65 y 0,59. Falta
  la prueba estadística.

---

## Lámina 23. Hipótesis 3: la excepción de PGExplainer

**Qué decir.** PGExplainer da cerca de cero, pero no por el modelo sino por cómo mide. Ordena
conexiones, y el 83 % de nuestras transacciones tiene una o dos. Con dos elementos, el orden
solo puede coincidir del todo o invertirse. El mismo modelo, explicado dos veces, dio 0,04 y
0,42. Es ruido del instrumento, y lo declaramos como limitación.

---

## Lámina 24. Comparación de métodos de explicación

**Qué decir.** Como PGExplainer no sirve aquí, agregamos otro explicador, de una familia
distinta: gradientes. Los cuatro son estables consigo mismos. Pero no señalan las mismas
variables: cada uno responde una pregunta distinta. Por eso hay que decir con cuál se explicó.

**Qué hay detrás.** Integrated Gradients es determinista: repetirlo da siempre lo mismo, y su
estabilidad vale 1 por construcción. Expected Gradients es su versión con azar. El acuerdo
entre explicadores va de 0,42 a 0,84.

**Si preguntan.**
- *¿Cuál es el correcto?* En datos reales no se puede saber, porque nadie conoce la respuesta
  verdadera. Es un problema abierto (Krishna et al. 2022).

---

## Lámina 25. Límite de la medida: la red sin entrenar

**Qué decir.** Nos preguntaron por qué todo salía estable. Hicimos una prueba: explicar la
misma red con los pesos entrenados y con pesos al azar, sin entrenar. Si la estabilidad
dependiera de lo aprendido, debería caer. No cae. La causa está en los datos: casi la mitad de
las variables de una transacción vale cero, y los explicadores siempre las dejan al fondo.

**Qué hay detrás.**
- Estabilidad entrenada y sin entrenar: GNNExplainer 0,95 y 0,93; Shapley 0,97 y 0,96;
  Expected Gradients 0,93 y 0,93.
- El 46 % de las variables vale 0 después del escalado. Solo ese bloque ya da 0,74 a 0,82.
- Con medidas que no se inflan, GNNExplainer baja a 0,75 y a 0,63.
- Es la prueba de Adebayo et al. (2018).

**Si preguntan.**
- *¿Entonces la tesis está mal?* No. La hipótesis 1 se sostiene con las tres medidas. Lo que
  cambia es qué tanto se puede concluir de una estabilidad alta, que es justo la tesis de
  fondo.
- *¿Las variables principales sí cambian al entrenar?* Sí: coinciden solo 0,20 a 0,40 con las
  de la red sin entrenar. La estabilidad no lo registra.

---

## Lámina 26. ¿La explicación más estable es la más correcta?

**Qué decir.** Esperábamos que sí. Para comprobarlo usamos un grafo sintético donde sí
conocemos la respuesta. El explicador más estable acierta al nivel del azar, y el que más
acierta no es el más estable.

**Qué hay detrás.** En el grafo sintético se plantó un patrón de lavado conocido. La
plausibilidad es la fracción de las variables o conexiones señaladas que pertenecen al patrón.

---

## Lámina 27. Estable no significa correcto

**Qué decir.** Shapley promedia cientos de combinaciones: su ruido es bajo por construcción,
encuentre o no el patrón. Y GCN, que casi no aprende, tiene explicaciones tan estables como
las demás. La estabilidad es necesaria para auditar, pero no suficiente: debe ir con el
rendimiento, la plausibilidad y la fidelidad.

---

## Lámina 28. El mejor modelo, con búsqueda completa de parámetros

**Qué decir.** La matriz de experimentos sirvió para escoger un candidato: GraphSAGE con pesos
por clase. A ese le hicimos una búsqueda de parámetros 19 veces más larga. Sube de 0,46 a 0,50
en validación. En la prueba final sigue en el azar y sus explicaciones siguen igual de
estables.

**Si preguntan.**
- *¿Para qué sirve?* Descarta que los resultados dependieran de modelos poco afinados.

---

## Lámina 29. Resumen

**Qué decir.** Tres de las cuatro preguntas no salieron como esperábamos, y para cada una
tenemos una causa medida. El desbalance no cambia la estabilidad ni en el caso extremo. El
tipo de red decide si el modelo aprende. La forma de compensar el desbalance mueve centésimas.
Y la más importante: estable no es correcto.

---

## Lámina 30. Plan

**Qué decir.** Así quedan ordenados la presentación y el documento. Ya están hechos el
candidato afinado, el explicador de gradientes y los casos 1 por cada 100 y 1 por cada 200.
Queda pendiente repetir el estudio cortando los datos antes del cierre del mercado ilegal.

---

## Láminas 31 y 32. Anexo: 1 fraude por cada lícita

**Qué decir.** Solo si preguntan. Con igual cantidad de fraude y de no fraude, ningún modelo
pasa el filtro de calidad, y en la prueba final se repite la caída.

---

## Lo que no se debe decir

| Frase | Por qué no |
|---|---|
| «El submuestreo solo desplaza el puntaje del modelo» | Es teoría; en nuestros modelos no se observó. |
| «Entre escenarios las explicaciones coinciden en 0,99» | Compara modelos de la misma semilla, que son casi idénticos. |
| «Las redes se apoyan en las mismas variables» | Comparten de 3 a 7 de las 10 principales. |
| «La estabilidad alta muestra que el modelo aprendió» | Una red sin entrenar da casi lo mismo. |
| «GAT es la más estable» o cualquier orden entre redes | Las diferencias no tienen prueba estadística con las medidas nuevas. |

## Referencias

- Adebayo, J. et al. (2018). Sanity checks for saliency maps. NeurIPS.
- Cawley, G. y Talbot, N. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. JMLR, 11, 2079-2107.
- Chawla, N. et al. (2002). SMOTE: synthetic minority over-sampling technique. JAIR, 16, 321-357.
- Elkan, C. (2001). The foundations of cost-sensitive learning. IJCAI.
- Krishna, S. et al. (2022). The disagreement problem in explainable machine learning: a practitioner's perspective. arXiv:2202.01602.
- Lakens, D. (2017). Equivalence tests: a practical primer. Social Psychological and Personality Science, 8(4), 355-362.
- Saito, T. y Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLoS ONE.
- Weber, M. et al. (2019). Anti-money laundering in Bitcoin: experimenting with graph convolutional networks for financial forensics. arXiv:1908.02591.
