# Discurso de defensa a dos voces (v2)

> Sustentacion de la tesis "Estabilidad de Metodos de Explicabilidad (XAI) en Graph Neural
> Networks para la Deteccion de Lavado de Dinero bajo Desbalance Extremo". UTP, MISC, 2026.
> Autores: Alejandro Gomez Huertas y Juan Diego Garzon Ovalle. Director: Ph.D. Cristian Rosero Arias.
>
> **Duracion objetivo:** 35 a 39 minutos, ajustado a la franja de 40. **Reparto:** Alejandro presenta el Bloque A (contexto y
> metodologia, paginas 1 a 22). Juan Diego presenta el Bloque B (resultados y cierre, paginas 24 a 40).
> Cada slide indica quien habla, el tiempo objetivo, el guion hablado y el gesto de transicion.
>
> Escrito sin guiones largos ni medios. Los numeros son la version corregida (post fix R1).
> El discurso esta redactado para decirse en voz alta, no para leerse palabra por palabra:
> ensayenlo hasta que suene propio y ajusten el fraseo a su manera de hablar.
>
> **Cambios de esta v2 respecto de la v1:** (1) se enuncian de forma explicita las tres hipotesis
> falsables en el Bloque A y se cierra su refutacion en el Bloque B, (2) se responde de forma directa
> el Objetivo 1 (el nivel de desbalance no degrada la estabilidad), (3) GNNShap gana un rol claro en la
> narrativa de resultados, (4) el cierre responde de frente la pregunta de investigacion, (5) se anade
> una respuesta ensayada a la pregunta mas filosa del jurado.

---

## Convenciones

- **[A]** habla Alejandro. **[JD]** habla Juan Diego. **[RELEVO]** marca el cambio de voz.
- Los tiempos suman unos 38 minutos, ajustados a la franja de 40. Si van cortos de tiempo, las laminas mas comprimibles son las introductorias nuevas (paso de mensajes, arquitecturas).
- Donde dice *(pausa)* conviene un silencio corto de uno a dos segundos: da peso a la idea.

### Mapa del guion a la pagina del PDF

> **El deck vigente es `presentacion_latex/beamer_defensa_v3.pdf`** (tema Metropolis oscuro), de **47
> paginas**: 33 laminas de contenido numeradas (la fraccion "N/40" de Metropolis cuenta tambien las de
> respaldo), 5 separadores de seccion (no se hablan, se pasan mientras se dice la frase de transicion),
> 1 lamina de cierre, 2 de referencias y 5 de respaldo que solo se abren si el jurado pregunta. Los
> encabezados de este guion usan la numeracion de pagina de ese PDF. Duracion estimada: **35 a 39
> minutos**.

| Pag. PDF | Fracc. | Lamina | Voz |
|---|---|---|---|
| **1** | | Portada | A |
| **2** | 1/40 | Contenido de la sustentacion | A |
| *3* | | *separador: Problema y motivacion* | |
| **4** | 2/40 | El lavado de dinero es un problema de red | A |
| **5** | 3/40 | El problema: detectar sin caja negra | A |
| **6** | 4/40 | Por que la caja negra es inaceptable | A |
| **7** | 5/40 | Pregunta de investigacion y objetivos | A |
| **8** | 6/40 | Tres hipotesis falsables | A |
| **9** | 7/40 | La brecha en el estado del arte | A |
| *10* | | *separador: Marco conceptual* | |
| **11** | 8/40 | Como funciona una GNN: paso de mensajes | A |
| **12** | 9/40 | Cuatro arquitecturas GNN | A |
| **13** | 10/40 | Tres explicadores post-hoc | A |
| **14** | 11/40 | Tres propiedades que no son lo mismo | A |
| *15* | | *separador: Metodologia* | |
| **16** | 12/40 | Que produce, en concreto, un explicador | A |
| **17** | 13/40 | Metodologia: un diseno de dos ejes | A |
| **18** | 14/40 | Diseno factorial | A |
| **19** | 15/40 | Eje 1: Elliptic Bitcoin Dataset | A |
| **20** | 16/40 | Eje 2: grafo sintetico con ground-truth | A |
| **21** | 17/40 | Como se construyo el grafo sintetico | A |
| **22** | 18/40 | Metricas y protocolo estadistico | A |
| *23* | | *separador: Resultados* | **[RELEVO]** |
| **24** | 19/40 | Contribucion: dos artefactos de evaluacion | JD |
| **25** | 20/40 | El segundo artefacto: el truncamiento de la metrica | JD |
| **26** | 21/40 | Resultado 1: dos grupos, no cuatro puestos | JD |
| **27** | 22/40 | Robustez: que sostiene esa particion | JD |
| **28** | 23/40 | Resultado 1b: concordancia entre regimenes | JD |
| **29** | 24/40 | Resultado 2: disociacion plausibilidad / fidelidad | JD |
| **30** | 25/40 | Resultado 3: el puente que no existe | JD |
| **31** | 26/40 | Resultado 4: el desbalance no gobierna la estabilidad | JD |
| **32** | 27/40 | Rendimiento y colapso validacion a test | JD |
| **33** | 28/40 | Rigor metrico: por que PR-AUC y no ROC-AUC | JD |
| **34** | 29/40 | Las tres hipotesis, y que paso con cada una | JD |
| *35* | | *separador: Conclusiones* | |
| **36** | 30/40 | Conclusiones del proyecto | JD |
| **37** | 31/40 | Los cuatro objetivos, respondidos | JD |
| **38** | 32/40 | Matriz de recomendacion | JD |
| **39** | 33/40 | Contribuciones y limitaciones | JD |
| **40** | | Gracias / Preguntas (cierre) | JD |
| *41 a 42* | | *Referencias seleccionadas* | |
| *43 a 47* | | *respaldo: solo si preguntan* | |

**El [RELEVO] cae en la pagina 23**, el separador "Resultados": Alejandro cierra en la 22 (Metricas),
pasa a la 23 mientras dice la frase de entrega, y Juan Diego arranca con la 24 en pantalla. Es el punto
de cambio mas limpio posible.

**Respaldo (paginas 43 a 47):** disociacion con valores exactos y las dos lineas base de azar,
estabilidad por semilla de modelo, detalle estadistico de la particion en los dos ejes, por que el
ranking por arquitectura usa solo GNNExplainer, y las curvas ROC y de precision y exhaustividad. Sepan de memoria en que pagina esta cada una para
llegar sin buscar.

---

## BLOQUE A: Alejandro Gomez (paginas 1 a 22, aprox. 18 min)


### Pagina 1: Portada  ·  [A]  ·  40 s
Buenas tardes. Agradecemos al jurado y al director su tiempo. Somos Alejandro Gomez y Juan Diego
Garzon, y hoy defendemos nuestra tesis de maestria sobre la estabilidad de los metodos de
explicabilidad en redes neuronales de grafos, aplicados a la deteccion de lavado de dinero bajo
condiciones de desbalance extremo. La presentacion va a dos voces: yo abro con el problema y la
metodologia, y mi companero continua con los resultados y las conclusiones. *(pausa)* Empecemos por
que este problema importa.


> **En la tesis:** Portada. Cap. 1 (Introduccion).

### Pagina 2: Contenido  ·  [A]  ·  25 s
El recorrido tiene cinco partes. Primero el problema y por que la explicabilidad es critica en este
dominio. Segundo, la pregunta de investigacion y los objetivos. Tercero, la metodologia, que se apoya
en un diseno de dos ejes que es el corazon del aporte. Cuarto, los resultados. Y quinto, las
conclusiones, los limites y el trabajo futuro. *(pausa breve)* Vamos al problema.


> **En la tesis:** Cap. 1, seccion 1.8 (Estructura de la Tesis).

### Pagina 4: El lavado de dinero es un problema de red  ·  [A]  ·  55 s
Antes de nada, que es el lavado de dinero y por que lo tratamos como una red. *(pausa)* El lavado tiene
tres fases clasicas: la colocacion, cuando el dinero ilicito entra al sistema; la estratificacion,
cuando se fragmenta en muchas transacciones para borrar el rastro; y la integracion, cuando vuelve a la
economia con apariencia legal. *(pausa)* Y las tecnicas concretas que usan los criminales, lo que
llamamos tipologias, son en el fondo patrones de conexiones entre cuentas: el structuring o pitufeo, el
layering, y las formas de fan-in y fan-out. La idea que quiero dejar fijada es esta: una transaccion
mirada de forma aislada parece completamente normal; lo que delata el lavado es el patron que forman
varias cuentas juntas. *(pausa)* Por eso el problema se modela como un grafo, una red de flujos, y no
como una tabla de transacciones independientes. Esa decision es la que abre la puerta a las redes
neuronales de grafos.


> **En la tesis:** Cap. 1, seccion 1.1 (Planteamiento y Contexto del Problema). Cap. 2 (Fases y tipologias del lavado).

### Pagina 5: El problema, detectar sin caja negra  ·  [A]  ·  65 s
Con eso claro, dimensionemos el problema. El lavado mueve entre el dos y el cinco por ciento del
producto interno bruto mundial. *(pausa)* Los sistemas tradicionales funcionan por reglas fijas, y ese
enfoque genera entre el noventa y cinco y el noventa y ocho por ciento de falsos positivos: de cada
cien alertas, casi todas son ruido que un analista revisa a mano. Y son reglas rigidas, que los
criminales aprenden a esquivar. *(pausa)* Las redes de grafos ofrecen una alternativa poderosa, porque
modelan las transacciones como lo que son, una red de flujos, y superan a los metodos tabulares que
miran cada transaccion aislada. El problema es que estas redes son cajas negras: aciertan, pero no
dicen por que marcaron una transaccion. Y ahi es donde entra la explicabilidad.


> **En la tesis:** Cap. 1, seccion 1.1. Cap. 2, seccion 2.3 (Sistemas Tradicionales de Monitoreo) y seccion 2.7 (Del Monitoreo Basado en Reglas hacia Enfoques Estructurales).

### Pagina 6: Por que la caja negra es inaceptable  ·  [A]  ·  55 s
Quiero detenerme en por que, en este dominio, una caja negra no es aceptable. *(pausa)* Primero, porque
el costo de un error es alto: una alerta puede congelar cuentas o iniciar una investigacion sobre una
persona. Segundo, porque es un entorno regulado: el analista debe poder justificar cada decision ante un
auditor, no basta con decir que el modelo lo dijo. Y tercero, porque debe ser reproducible: la misma
transaccion tiene que producir la misma explicacion, hoy y dentro de un mes. *(pausa)* La consecuencia
es la que ordena toda la tesis: la explicabilidad no es un lujo, es un requisito. Y si la explicacion
cambia cada vez que se recalcula, no sirve para auditar. Por eso la propiedad critica, la que
estudiamos, es la estabilidad.


> **En la tesis:** Cap. 1, seccion 1.1. Cap. 3, seccion 3.6.4 (El Problema de la Estabilidad Explicativa).

### Pagina 7: Pregunta de investigacion y objetivos  ·  [A]  ·  60 s
Con ese marco, esta es nuestra pregunta: como se comporta la estabilidad de los metodos de
explicabilidad sobre redes de grafos para deteccion de lavado, cuando el dato esta fuertemente
desbalanceado, y que combinacion de arquitectura, explicador y estrategia de balanceo produce la
interpretacion mas robusta y auditable. *(pausa)* De ahi se desprenden cuatro objetivos. El primero,
medir si la estabilidad se degrada a medida que el desbalance se agrava. El segundo, comparar la
resiliencia de cuatro arquitecturas: GCN, GraphSAGE, GAT y TAGCN. El tercero, evaluar si las
estrategias de balanceo afectan la calidad de las explicaciones. Y el cuarto, condensar todo en una
matriz de recomendacion que diga que usar segun el objetivo de la auditoria.


> **En la tesis:** Cap. 1, seccion 1.4 (Formulacion del Problema) y seccion 1.5 (Objetivos: 1.5.1 General, 1.5.2 Especificos).

### Pagina 8: Tres hipotesis falsables  ·  [A]  ·  55 s
Y para no medir a ciegas, nos comprometimos con tres hipotesis falsables antes de ver los datos.
*(pausa)* La primera, que la estabilidad se degradaria al agravarse el desbalance: cuanto mas raro el
fraude, mas inestables las explicaciones. La segunda, que TAGCN seria la arquitectura mas estable, por
su alcance multi-salto. Y la tercera, que un explicador mas estable senalaria tambien mejor el patron
real de lavado, es decir, que la consistencia implicaria acierto. *(pausa)* Comprometerse con
predicciones concretas antes de mirar los datos es lo que hace que refutarlas signifique algo. Adelanto
que las tres se matizaron o se cayeron, y que contarlo con honestidad es parte del aporte. El desenlace
de cada una lo vera el jurado en la seccion de resultados.


> **En la tesis:** Cap. 1, seccion 1.5.3 (Hipotesis de Trabajo).

### Pagina 9: La brecha  ·  [A]  ·  60 s
Que se sabia ya. Se sabia que las redes de grafos superan a los metodos tabulares en este dominio, y
se habian estudiado la prediccion y la explicabilidad, pero por separado. *(pausa)* Que faltaba. Nadie
habia evaluado de forma sistematica la estabilidad de las explicaciones sobre grafos financieros, es
decir, si una explicacion se sostiene o cambia cuando se vuelve a calcular. Y habia un obstaculo de
fondo: el dataset de referencia, Elliptic, no trae un patron verdadero de tipologia de lavado, de modo
que la plausibilidad de una explicacion, si senala o no el patron correcto, simplemente no era medible.
Esa doble brecha, la estabilidad no evaluada y la plausibilidad no medible, es la que esta tesis viene
a cerrar.


> **En la tesis:** Cap. 1, seccion 1.2 (Revision de Literatura y Estado del Arte) y seccion 1.3 (La Arista de Investigacion). Cap. 3, seccion 3.6.4 (El Problema de la Estabilidad Explicativa).

### Pagina 11: Como funciona una GNN, paso de mensajes  ·  [A]  ·  60 s
Un marco minimo antes de los metodos, empezando por como funciona una red neuronal de grafos. *(pausa)*
La idea central es el paso de mensajes: cada nodo actualiza su representacion agregando la informacion
de sus vecinos. Y eso se repite capa por capa: con dos capas, un nodo ve a sus vecinos y a los vecinos
de sus vecinos, es decir, hasta dos saltos de distancia; a eso lo llamamos su campo receptivo. *(pausa)*
La consecuencia importante para nosotros es esta: la prediccion sobre una transaccion no depende solo de
ella, sino de todo su vecindario. Y por eso explicar, en una red de grafos, significa senalar que parte
de ese vecindario, que aristas y que atributos, sostuvo la prediccion. Ese es el objeto que toda la
tesis va a medir.


> **En la tesis:** Cap. 3, seccion 3.2 (GNN: Fundamentos Conceptuales) y seccion 3.3.

### Pagina 12: Cuatro arquitecturas GNN  ·  [A]  ·  50 s
Sobre esa base trabajamos con cuatro arquitecturas, que se diferencian justamente en como cada nodo
agrega a sus vecinos. *(pausa)* GCN simplemente los promedia, con una convolucion espectral de un salto;
es la mas simple y solida. GraphSAGE muestrea un subconjunto de vecinos y los agrega; es inductiva y
escala a grafos grandes. GAT no trata a todos los vecinos por igual: les asigna pesos con un mecanismo
de atencion. Y TAGCN usa filtros polinomicos de orden K, que alcanzan varios saltos de una sola vez.
*(pausa)* Las elegimos porque cubren las familias dominantes de agregacion, y compararlas es
precisamente nuestro segundo objetivo.


> **En la tesis:** Cap. 3, secciones 3.3 (Arquitecturas GNN Fundamentales) y 3.4 (TAGCN).

### Pagina 13: Tres explicadores post-hoc  ·  [A]  ·  50 s
Y sobre esas redes aplicamos tres explicadores post-hoc. Post-hoc significa que explican un modelo ya
entrenado, sin modificarlo. Cada uno entrega dos cosas: una mascara de aristas y un ranking de
atributos. *(pausa)* GNNExplainer optimiza una mascara por cada instancia, es decir, aprende que importa
para ese nodo en particular. PGExplainer, en cambio, entrena una red amortizada que aprende a explicar
en todo el grafo, y por eso generaliza entre nodos. Y GNNShap reparte el credito entre los elementos con
valores de Shapley, que vienen de la teoria de juegos, lo que lo hace muy consistente. *(pausa)* Cuatro
arquitecturas por tres explicadores es el nucleo de la comparacion.


> **En la tesis:** Cap. 3, seccion 3.6.2 (Metodos XAI para GNNs).

### Pagina 14: Tres propiedades  ·  [A]  ·  65 s
Esta lamina contiene la tesis central, asi que me detengo. Cuando decimos que una explicacion es
"buena", en realidad mezclamos tres preguntas distintas. La estabilidad pregunta si la explicacion se
reproduce cuando cambio la semilla o perturbo un poco la entrada. La plausibilidad pregunta si senala el
patron real de lavado, lo que un experto reconoceria. Y la fidelidad pregunta si refleja de verdad lo
que el modelo uso para decidir. *(pausa)* La tesis central es que estas tres son dimensiones distintas
que no se implican entre si. Una explicacion puede ser muy estable y aun asi apuntar al patron
equivocado, como un reloj parado, que siempre marca la misma hora y siempre esta mal. Puede ser
plausible para un humano y no reflejar el mecanismo del modelo. Gran parte de la literatura reporta una
sola de estas y la llama calidad. Nosotros vamos a mostrar, con datos, que hay que medir las tres por
separado.


> **En la tesis:** Cap. 3, seccion 3.8.4 (Metricas de Estabilidad y Fidelidad Explicativa). Cap. 6, seccion 6.2 (Estabilidad, Plausibilidad y Fidelidad como Tres Dimensiones Independientes).

### Pagina 16: Que produce un explicador  ·  [A]  ·  60 s
Antes de entrar en la metodologia quiero aterrizar que es, en concreto, una explicacion, porque toda la
tesis mide propiedades de este objeto. *(pausa)* Cuando el modelo marca una transaccion como ilicita y
le pedimos al explicador que justifique esa decision, lo que devuelve son dos cosas. Primero, una
mascara de aristas: de todas las conexiones del vecindario de esa transaccion, cuales sostienen la
prediccion, que son las que ven resaltadas en el diagrama. Y segundo, un ranking de atributos: de las
ciento sesenta y seis features que tiene cada nodo, cuales pesaron mas. *(pausa)* Sobre ese objeto se
definen las tres preguntas de la lamina anterior. La estabilidad pregunta si al repetir el calculo sale
la misma mascara y el mismo ranking. La plausibilidad, si eso coincide con el patron real de lavado. Y
la fidelidad, si es de verdad lo que el modelo uso. *(pausa)* Un detalle que va a explicar varias cosas
mas adelante: en Elliptic el vecindario tipico tiene una mediana de unos dos nodos, asi que la mascara
de aristas es casi trivial y solo el ranking de atributos discrimina. Por eso ahi medimos estabilidad
con Spearman sobre features.


> **En la tesis:** Cap. 3, seccion 3.6.2 (Metodos XAI para GNNs) y seccion 3.8.4. Cap. 4, seccion 4.1.2 (Dispersion de la Topologia).

### Pagina 17: Dos ejes  ·  [A]  ·  70 s
Como se prueba algo asi. Con un diseno de dos ejes, que es la decision metodologica mas importante de
la tesis. *(pausa)* El primer eje es Elliptic, el dataset real de transacciones de Bitcoin. Nos da
validez externa, porque son datos reales con todo su ruido y su desbalance, pero tiene dos limites:
no trae patron verdadero, asi que solo permite medir estabilidad, y sus vecindarios son minusculos,
de unos dos nodos. El segundo eje es un grafo sintetico que construimos nosotros, con patron verdadero
por cada nodo y por cada arista. Nos da validez interna: como sabemos cual es el patron correcto,
podemos medir plausibilidad y fidelidad, cosa imposible en Elliptic. *(pausa)* La clave es que los dos
ejes se complementan. Ninguno solo alcanza. Juntos permiten afirmar cosas que ninguno probaria por su
cuenta. Y aqui adelanto un punto que mi companero va a demostrar: los dos ejes, bien medidos, cuentan
la misma historia.


> **En la tesis:** Cap. 5, seccion 5.1 (Por que se Construye un Grafo Sintetico). Cap. 6, seccion 6.1 (Lectura Conjunta de los Dos Ejes).

### Pagina 18: Diseno factorial  ·  [A]  ·  55 s
El experimento es una matriz factorial completa: cuatro arquitecturas, por tres explicadores, por tres
estrategias de balanceo, por cinco escenarios de desbalance. Eso da sesenta configuraciones por eje,
cada una con los tres explicadores para el estudio de estabilidad. *(pausa)* Y no nos quedamos en una
sola corrida. Cada explicacion se repite cinco veces con semillas distintas para medir su estabilidad.
Y en el eje sintetico anadimos una capa de robustez: tres semillas de modelo por tres grafos
independientes, con pruebas estadisticas serias, Kruskal-Wallis, Wilcoxon e intervalos de confianza
por bootstrap. Esto es lo que convierte observaciones sueltas en evidencia con respaldo.


> **En la tesis:** Cap. 4, seccion 4.2 (Pipeline Experimental y Espacio Factorial). Cap. 5, seccion 5.4 (Analisis Estadistico de Robustez). Cap. 8, seccion 8.1 (Espacio de Busqueda de Hiperparametros).

### Pagina 19: Elliptic  ·  [A]  ·  60 s
El primer eje en detalle. Elliptic tiene doscientos tres mil setecientos sesenta y nueve nodos,
doscientas treinta y cuatro mil aristas, ciento sesenta y seis atributos por nodo y cuarenta y nueve
pasos temporales. Las transacciones ilicitas son apenas el dos coma dos por ciento, una razon cercana a
uno a nueve en la parte etiquetada. *(pausa)* Hicimos una particion temporal causal: entrenamos con el
pasado y evaluamos con el futuro, que es como opera un sistema real. Y hay un dato que gobierna todo lo
demas: el vecindario tipico de un nodo tiene una mediana de unos dos nodos. Es un grafo extremadamente
disperso. Por eso, como veran, la unica metrica de estabilidad que discrimina bien aqui es la
correlacion de Spearman entre rankings de atributos. Las metricas de aristas se saturan.


> **En la tesis:** Cap. 4, seccion 4.1 (Preprocesamiento y Analisis Exploratorio: 4.1.1 Composicion, 4.1.2 Dispersion de la Topologia).

### Pagina 20: Sintetico  ·  [A]  ·  65 s
El segundo eje, nuestro grafo sintetico, responde a una necesidad concreta: para medir si una
explicacion es plausible, hay que saber de antemano cual es el subgrafo correcto, y Elliptic no lo da.
Asi que lo construimos. *(pausa)* Inyectamos cuatro tipologias de lavado reconocidas: structuring,
layering, fan-in y fan-out. Y tomamos tres decisiones para que la medicion no quede servida: simetrizamos
las aristas para que el vecindario tenga estructura suficiente, anadimos aristas distractoras para que
acertar no sea trivial, y atenuamos la firma de los atributos para que la tarea no se resuelva sola.
Un detalle importante que nos van a preguntar: el patron verdadero es ciego al explicador, se define
en la construccion del grafo, antes de correr ningun metodo. No lo ajustamos para favorecer a nadie.


> **En la tesis:** Cap. 5, seccion 5.1 (Por que se Construye un Grafo Sintetico) y seccion 5.2 (Construccion del Grafo Sintetico y sus Tipologias).

### Pagina 21: Como se construyo el grafo sintetico  ·  [A]  ·  70 s
Me detengo en la construccion, porque es donde se juega la credibilidad de todo el eje sintetico y
prefiero exponerla nosotros antes de que se pregunte. *(pausa)* El grafo tiene unos nueve mil quinientos
nodos, de los cuales unos mil quinientos son ilicitos, y unas treinta y un mil aristas, con las cuatro
tipologias canonicas plantadas dentro. Pero lo importante no es el tamano, son tres decisiones que
tomamos deliberadamente para que la prueba fuera dificil. *(pausa)* La primera, simetrizar las aristas.
Con el grafo dirigido el campo receptivo cae a unos dos nodos y los patrones de estrella y de cadena
quedan invisibles al paso de mensajes, con lo que la plausibilidad de subgrafo no seria medible. La
segunda, anadir aristas distractoras desde cada nodo de patron hacia el fondo licito. Sin ellas el
subgrafo seria cien por cien patron y cualquier seleccion acertaria, con lo que la metrica no
discriminaria nada. Y la tercera, atenuar la firma de los atributos de mas cuatro a mas uno coma cinco,
para que el problema no se resolviera solo. *(pausa)* Las tres decisiones endurecen la prueba, no la
inflan. Y el punto que quiero dejar fijado: el patron verdadero lo fija el generador, que es
completamente ciego a que explicador se va a evaluar despues. Que PGExplainer gane en plausibilidad no
esta cableado en ninguna parte.


> **En la tesis:** Cap. 5, seccion 5.2 (Construccion del Grafo Sintetico y sus Tipologias). Cap. 8, seccion 8.5. Material de apoyo: `docs/DEFENSA_R2_evidencia_sintetica.md`, pilares 4 y 5.

### Pagina 22: Metricas  ·  [A]  ·  55 s
Cierro mi bloque con las metricas. La estabilidad la medimos con la correlacion de Spearman entre los
rankings de atributos, que es nuestra metrica primaria. La plausibilidad, como coincidencia con el
patron verdadero de la tipologia. La fidelidad, como cuanto cae la prediccion cuando quitamos lo que el
explicador marco como importante. Y para el rendimiento del clasificador usamos PR-AUC como metrica
primaria, porque el F1 con umbral fijo se degrada bajo desbalance extremo. *(pausa)* Todo esto se
acompana de estadistica: Kruskal-Wallis para comparar factores, Wilcoxon para comparaciones pareadas, e
intervalos de confianza por bootstrap. *(pausa, gira hacia Juan Diego)* Con la metodologia sobre la
mesa, le paso la palabra a Juan Diego para los resultados.


> **En la tesis:** Cap. 3, seccion 3.8 (Formalizacion de Metricas de Evaluacion: 3.8.2 en Escenarios de Desbalance, 3.8.4 Estabilidad y Fidelidad, 3.8.5 Nociones de Inferencia Estadistica).
**[RELEVO: Alejandro cede a Juan Diego]**


---

## BLOQUE B: Juan Diego Garzon (paginas 24 a 40, aprox. 19 min)


### Pagina 24: Dos artefactos de evaluacion  ·  [JD]  ·  85 s
Gracias, Alejandro. Voy a empezar los resultados por algo que no estaba en el plan original y que
termino siendo una de nuestras contribuciones. *(pausa)* Al analizar la estabilidad encontramos que
dos detalles de la medicion, no del metodo, estaban distorsionando las conclusiones. El primero fue un
fallo de memoria silencioso: al calcular las explicaciones sobre el grafo completo, trece
configuraciones de GAT fallaban sin aviso y sus filas quedaban vacias, de modo que el promedio de GAT
se calculaba solo sobre los casos que si terminaban, y eso lo favorecia de forma artificial. El segundo
fue un truncamiento en la metrica de Spearman: la implementacion descartaba los atributos por debajo de
un umbral y mutilaba los rankings, lo que esta vez favorecia a GraphSAGE. *(pausa)* Lo importante es
esto: cada uno de estos dos detalles, por si solo, bastaba para producir una conclusion comparativa
falsa sobre que arquitectura es mas estable. La leccion, que conecta con el trabajo de Kosan sobre
sensibilidad al protocolo, es que la estabilidad medida depende tanto del protocolo de evaluacion como
del metodo. Y de paso reportamos dos bugs concretos del PGExplainer de la libreria PyG. Corregimos
todo esto y volvimos a medir. Lo que sigue son los numeros corregidos.


> **En la tesis:** Cap. 4, seccion 4.5 (De un Artefacto de Computo a un Artefacto de Medida: dos Correcciones Metodologicas). Cap. 6, seccion 6.4 (Contribuciones Metodologicas).

### Pagina 25: El segundo artefacto en detalle  ·  [JD]  ·  60 s
Me detengo en el segundo artefacto porque es el mas instructivo de los dos. *(pausa)* La metrica de
Spearman dimensionaba el vector de rangos por el parametro de truncamiento, que estaba fijado en veinte,
en lugar de por el numero real de atributos, que son ciento sesenta y seis. El efecto es que toda
feature con indice mayor que veinte se descartaba en silencio. De las veinte del top sobrevivian dos o
tres, y todo lo demas quedaba empatado en cero. Estabamos comparando rankings mutilados. *(pausa)* Y
aqui esta lo importante, que no es el bug sino su asimetria: el truncamiento no castiga por igual a
todas las arquitecturas. Castiga mas a aquellas cuya importancia se reparte sobre muchos atributos. Por
eso al corregirlo GAT sube veinticuatro centesimas y TAGCN veintiocho, mientras que GraphSAGE sube solo
diez. El liderazgo de GraphSAGE que reportaba la version anterior no era un hallazgo, era el perfil de
sensibilidad de la metrica rota. *(pausa)* Sumado al fallo de memoria de la lamina anterior, tenemos dos
defectos del protocolo de medida que apuntaban en direcciones opuestas y cada uno bastaba, por si solo,
para una conclusion comparativa falsa.


> **En la tesis:** Cap. 4, seccion 4.5. Cap. 6, seccion 6.5 (Relacion con el Estado del Arte, la leccion de Kosan).

### Pagina 26: Ranking por arquitectura  ·  [JD]  ·  75 s
Con la metrica corregida, y replicando el entrenamiento completo con tres semillas de modelo, lo que
encontramos no es un ranking de cuatro puestos sino una particion en dos grupos. Un grupo alto, con GAT
en cero coma setenta y ocho y GCN en cero coma setenta y seis, y un grupo bajo, con GraphSAGE en cero
coma setenta y cuatro y TAGCN en cero coma sesenta y siete. *(pausa)* Y lo importante es donde estan las
diferencias: entre los dos grupos son estadisticamente significativas, y dentro de cada grupo no lo son.
La prueba que compara las cuatro arquitecturas nos dice que, si en realidad no hubiera ninguna
diferencia entre ellas, una separacion tan marcada como la que vemos apareceria menos de tres veces
en cien mil. Pero cuando hacemos la misma pregunta dentro del grupo alto, o dentro del grupo bajo, las
diferencias son del tamano que el azar produce con frecuencia, asi que no podemos afirmar que existan. *(pausa)* Aqui cierro nuestra
segunda hipotesis: esperabamos que TAGCN, por su alcance multi-hop, fuera la mas estable, y aparece de
forma consistente en el grupo bajo en las tres semillas. La hipotesis se cae.
*(pausa)* Hay dos cosas mas que tenemos que decir. La primera, que el liderazgo de GraphSAGE que reportaba
una version anterior de la tesis era un artefacto del truncamiento de la metrica, y corregido se
disuelve. La segunda, que GAT y GCN se permutan entre semillas, asi que no afirmamos que ninguna de las
dos sea la mejor: afirmamos que las dos forman el grupo alto. Decir menos seria impreciso, y decir mas
seria sobre-interpretar.


> **En la tesis:** Cap. 4, seccion 4.5 (tabla tab:ranking). Cap. 4, seccion 4.6 (Replicacion con Multiples Semillas y Estructura en Dos Grupos). Cap. 8, seccion 8.2.

### Pagina 27: Robustez de la particion  ·  [JD]  ·  55 s
Me detengo un momento en que sostiene esa particion, porque es la diferencia entre una observacion y
un resultado. *(pausa)* Reentrenamos la matriz completa de sesenta configuraciones tres veces, con tres
semillas distintas, ciento ochenta modelos, y en cada una corrimos el procedimiento entero incluida su
propia busqueda de hiperparametros. Sobre esos ciento ochenta modelos, la pregunta es simple: las diferencias que vemos, ¿son reales o
podrian ser casualidad? Una prueba estadistica estandar responde que, si el grupo alto y el bajo fueran
en realidad iguales, una diferencia como la que medimos apareceria menos de dos veces en un millon, asi
que la tratamos como real. En cambio, dentro de cada grupo las diferencias son compatibles con el azar. Dicho de otro modo, toda la variacion esta entre
grupos, ninguna dentro. Los margenes de error, calculados repitiendo el analisis muchas veces con
remuestreo, cuentan lo mismo: se solapan dentro de cada grupo y apenas se tocan entre ellos. Y por si
preguntan si esto sale de haber hecho muchas comparaciones a la vez, aplicamos el ajuste habitual para
ese caso y la separacion aguanta. *(pausa)* Y esa replicacion no se limito a GNNExplainer: tambien
corrimos GNNShap y PGExplainer en las tres semillas para GCN y GraphSAGE, y confirman su patron,
GNNShap muy estable pero sin distinguir arquitecturas y PGExplainer degenerado. En GAT y TAGCN esa
extension quedo en la semilla de referencia por un motivo operativo: sus modelos de las otras dos
semillas, que si forman parte de los ciento ochenta, no estaban en el equipo donde la corrimos, y
reentrenarlos alli agoto la memoria de la tarjeta. Aun asi, en esa semilla muestran exactamente el
mismo patron.
*(pausa)* Hay un segundo hallazgo aqui que tambien cuenta como contribucion. Al reentrenar
descubrimos que el pipeline no es reproducible bit a bit: las operaciones de agregacion sobre la
tarjeta grafica suman en un orden que no esta determinado, asi que los pesos nunca salen identicos.
Lo que si se reproduce son las conclusiones, veinticinco configuraciones sobre el filtro de calidad
frente a veintitres, y exactamente la misma particion. Distinguir la reproducibilidad de los pesos de
la reproducibilidad de las conclusiones es algo que la literatura rara vez explicita, y creemos que
deberia hacerlo.


> **En la tesis:** Cap. 4, seccion 4.6 (tablas tab:seeds y tab:ic). Cap. 6, seccion 6.4
> (Contribuciones Metodologicas). Cap. 8, seccion 8.5 (Reproducibilidad y Entorno de Computo).

### Pagina 28: Concordancia entre regimenes  ·  [JD]  ·  70 s
Este resultado es uno de los que mas me gustan, porque nacio de un error corregido. En una version
previa creiamos haber encontrado una "inversion por densidad": que el orden de estabilidad entre
arquitecturas se daba vuelta al pasar del grafo disperso de Elliptic al grafo denso sintetico. *(pausa)*
Cuando corregimos el bug de la metrica, esa inversion desaparecio. Lo que en realidad ocurre es lo
contrario: los dos regimenes concuerdan. Las mismas arquitecturas que son estables en el grafo denso
lo son en el disperso. Lo cuantificamos con la correlacion de rangos entre ambos regimenes, que pasa de
menos cero coma veinte con la metrica defectuosa, a mas cero coma ochenta con la metrica corregida.
*(pausa)* Y esto le da peso a la tesis, porque significa que datos reales y datos
sinteticos cuentan la misma historia. La coherencia entre los dos ejes es lo que le da solidez a todo
el diseno.


> **En la tesis:** Cap. 5, seccion 5.3 (Resultados de la Matriz Factorial). Cap. 6, secciones 6.1 y 6.2.

### Pagina 29: Disociacion plausibilidad y fidelidad  ·  [JD]  ·  85 s
Ahora el hallazgo central sobre los explicadores, y es un resultado con dos caras. *(pausa)* Por un
lado, PGExplainer es claramente el que mejor recupera el patron real: su plausibilidad de aristas es de
cero coma ochenta, frente a cero coma cincuenta de GNNExplainer, y la ventaja es sistematica: gana en
noventa y dos de cada cien comparaciones pareadas, algo que no ocurriria en la practica si los dos
explicadores fueran igual de buenos. Y
para que estas cifras signifiquen algo, las contrastamos con el azar: un explicador que eligiera las
aristas al azar, con el mismo protocolo, obtiene cero coma cuarenta. PGExplainer duplica ese nivel,
mientras que GNNExplainer apenas lo supera, lo que ya anticipa la disociacion que viene: el fuerte de
GNNExplainer no es recuperar el patron, sino la fidelidad al modelo. Es
decir, si el objetivo es senalar el patron de lavado, PGExplainer gana sin discusion. *(pausa)* Pero
por otro lado, ese mismo PGExplainer colapsa en fidelidad: cuando medimos cuanto depende la prediccion
del modelo de las aristas que PGExplainer marca, el valor cae a cero coma once, frente a cero coma
cincuenta y seis de GNNExplainer. La lectura va contra la intuicion: el explicador mas plausible
no es el mas fiel. PGExplainer recupera las aristas que definen el patron que un humano reconoce, pero
GNNExplainer recupera las aristas que el modelo realmente usa, y esos dos conjuntos no coinciden.
*(pausa)* Una precision sobre la figura: compara los dos explicadores que producen mascara de aristas,
que son los unicos comparables contra esa linea base de cero coma cuarenta. GNNShap no produce mascara
de aristas por diseno, su plausibilidad es de features y vive en otra escala, con su propia linea base
de cero coma cero siete cinco. Sus valores estan en la lamina de respaldo con la metrica etiquetada. Lo
que si cabe decir de GNNShap aqui es que es el mas estable internamente de los tres, el mas consistente
entre ejecuciones, aunque no lidere ni plausibilidad ni fidelidad. Cada explicador, entonces, tiene su
fortaleza en una dimension distinta. Esta disociacion solo se puede exhibir cuando tienes un patron
verdadero contra el cual medir, y por eso el eje sintetico era indispensable.


> **En la tesis:** Cap. 5, seccion 5.6 (La Disociacion entre Plausibilidad y Fidelidad). La plausibilidad de aristas en seccion 5.3.

### Pagina 30: Resultado 3, el puente que no existe  ·  [JD]  ·  60 s
Esta lamina cierra nuestra hipotesis central, y tambien se cae. *(pausa)* Esperabamos que una explicacion
mas estable fuera tambien mas plausible, es decir que la consistencia entre ejecuciones implicara acierto
sobre el patron real. Es una intuicion que esta implicita en buena parte de la literatura y nunca se
habia contrastado de frente, porque para contrastarla hace falta medir las dos cosas a la vez sobre las
mismas explicaciones, y eso exige conocer de antemano cual es el patron verdadero, algo que los datos reales no dan y el grafo sintetico si. *(pausa)* Los datos dicen que no. La correlacion entre
estabilidad y plausibilidad es de menos cero coma cero uno, con un intervalo de confianza que va de menos
cero coma cero treinta y ocho a mas cero coma cero once, o sea que incluye el cero. Es un puente nulo. Un
explicador estable no es por ello mas acertado sobre el patron real, y ambas propiedades hay que medirlas
por separado. *(pausa)* Y hay un detalle que refuerza la conclusion: cuando desagregamos por tipologia, el
signo de la relacion cambia segun cual mires, positivo en structuring y en fan-out, negativo en layering.
No hay una ley que ligue las dos dimensiones, ni siquiera dentro del mismo grafo. Es un resultado nulo, y lo
reportamos porque contradice lo que esperabamos. Si hubieramos disenado el
experimento para lucirnos, habriamos forzado una correlacion bonita, y no lo hicimos.


> **En la tesis:** Cap. 5, seccion 5.5 (La Ausencia de un Puente entre Estabilidad y Plausibilidad). Tabla `tab:synth-bridge` para el desglose por tipologia.

### Pagina 31: El desbalance no gobierna la estabilidad  ·  [JD]  ·  55 s
Cierro la primera hipotesis, y tambien se cae. *(pausa)* Esperabamos que la estabilidad se degradara a
medida que el desbalance se agravara, y que hubiera algun punto de quiebre. No ocurre ninguna de las dos
cosas. Sobre las tres semillas, la estabilidad media recorre un rango estrecho, de cero coma sesenta y
nueve seis en el escenario uno a uno a cero coma setenta y seis cinco en el uno a cincuenta: siete
centesimas en total, menos que las once centesimas que separan a las arquitecturas entre si. Y lo
decisivo es que esa variacion no pasa la prueba estadistica: la diferencia entre los cinco escenarios es
perfectamente compatible con el azar. Ademas, el escenario explica apenas un cinco por ciento de la
variacion, mientras que la arquitectura explica el trece por ciento sobre la misma medida. *(pausa)* Y conviene mirar hacia donde apunta lo poco que se mueve, porque
apunta al reves de lo que esperabamos: el valor mas bajo esta en el escenario uno a uno, que es el mas
equilibrado de todos, y los escenarios de desbalance acentuado quedan por encima. No hay deterioro
monotono, no hay pico en el escenario uno a cincuenta, y el escenario nativo no se comporta de forma
anomala. *(pausa)* Y en la misma linea, la estrategia de balanceo, que suele recibir mucha atencion en
la literatura, resulta practicamente irrelevante: su tamano de efecto es de cero coma cero uno en las
tres dimensiones. *(pausa)* Esto tiene una implicacion practica que me parece la mas util de toda la
tesis para un equipo de cumplimiento: el balanceo pueden elegirlo por rendimiento predictivo, sin temer
que al hacerlo esten degradando la interpretabilidad. Son decisiones que se pueden tomar por separado.


> **En la tesis:** Cap. 4, seccion 4.5 (perfil por escenario, tabla tab:elliptic-stab-scen). Cap. 6, seccion 6.3 (El Papel Secundario del Balanceo y de la Arquitectura). Cap. 7, seccion 7.1 (O1 y O3).

### Pagina 32: Colapso validacion a test  ·  [JD]  ·  70 s
Un resultado de rendimiento que debemos declarar con transparencia, porque enmarca todo lo anterior.
Los modelos aprenden en validacion, con un PR-AUC medio de cero coma treinta y siete, pero colapsan en
test, donde cae a cero coma cero dos. *(pausa)* La causa es el desplazamiento temporal del dataset: los
patrones de lavado cambian entre los primeros y los ultimos pasos, y un modelo entrenado con el pasado
encuentra en el futuro una distribucion distinta. Es una propiedad del dato, no un defecto de nuestro
metodo. Aqui hay un punto metodologico que quisimos remarcar: el ROC-AUC se ve enganosamente alto bajo
desbalance extremo, cero coma ochenta y ocho en validacion, y por eso no lo usamos como metrica
principal. Usamos el area de precision y exhaustividad y la precision en los primeros de la lista, que no se dejan enganar. *(pausa)* Como consecuencia, la
estabilidad la estudiamos sobre los verdaderos positivos de validacion, donde el modelo si discrimina,
y lo declaramos de forma abierta. No es esconder el colapso, es medir donde la pregunta tiene sentido.


> **En la tesis:** Cap. 4, seccion 4.4 (Rendimiento Predictivo y el Colapso de Validacion a Test). Cap. 3, seccion 3.8.2.

### Pagina 33: Rigor metrico, PR-AUC y no ROC-AUC  ·  [JD]  ·  55 s
Esta lamina desarrolla el punto metodologico que acabo de mencionar, porque creo que merece detenerse.
*(pausa)* Miren la tabla. Sobre validacion, el ROC-AUC da cero coma ochenta y ocho. Si reportaramos solo
esa cifra, cualquiera concluiria que tenemos un clasificador casi excelente. Pero sobre exactamente los
mismos modelos, el area de precision y exhaustividad da cero coma treinta y siete, y la precision en los
primeros cincuenta nodos cero coma sesenta y seis. Es una tarea mucho mas dificil de lo que el ROC-AUC
insinua. *(pausa)* Y sobre test la disociacion se vuelve extrema: el ROC-AUC se mantiene en cero coma
sesenta y cinco, que parecerian un modelo mediocre pero funcional, mientras que el area de precision y
exhaustividad se desploma a cero coma cero dos. *(pausa)* La razon es estructural: bajo desbalance
extremo, el eje de tasa de falsos positivos del ROC queda dominado por la enorme clase mayoritaria y
permanece bajo aunque el modelo no distinga la clase rara. Por eso nuestras metricas primarias son el
area de precision y exhaustividad y la precision en los primeros nodos, que ademas son las que gobiernan
el trabajo real de un analista, que revisa una lista acotada de alertas y no todo el universo de
transacciones.


> **En la tesis:** Cap. 4, seccion 4.4 (tabla tab:elliptic-rocauc). Cap. 3, seccion 3.8.2 (Metricas en Escenarios de Desbalance).

### Pagina 34: Las tres hipotesis y su veredicto  ·  [JD]  ·  60 s
Antes de pasar a las conclusiones quiero cerrar el circulo que abrio mi companero al principio. *(pausa)*
Nos comprometimos con tres predicciones falsables antes de ver los datos. La primera, que la estabilidad
se degradaria al agravarse el desbalance: refutada, el perfil es plano. La segunda, que TAGCN seria la
mas estable por su alcance multi-hop: refutada, aparece en el grupo bajo en las tres semillas. Y la
tercera, que un explicador mas estable senalaria mejor el patron: refutada, el puente es nulo. *(pausa)*
Las tres se cayeron, y las tres estan en la tesis. *(pausa)* Y quiero explicar por que presentamos esto como un
resultado y no como un fracaso. Comprometerse con predicciones concretas antes de mirar los datos es lo
que hace que refutarlas signifique algo. Si hubieramos formulado hipotesis vagas, o las hubieramos
ajustado despues de ver los numeros, no habriamos aprendido nada. Lo que estas tres refutaciones nos
dicen es que la intuicion dominante en el campo, la de que existe una nocion unica de buena explicacion
que todas las metricas capturan a la vez, es falsa. Y ese es el hallazgo central de la tesis.


> **En la tesis:** Cap. 1, seccion 1.5.3 (Hipotesis de Trabajo). Cap. 7, seccion 7.1 (Respuestas a los Objetivos de Investigacion).

### Pagina 36: Conclusiones del proyecto  ·  [JD]  ·  75 s
Antes de responder objetivo por objetivo, permitanme dejar las seis conclusiones que sostienen todo el
trabajo. *(pausa)* Primera: estabilidad, plausibilidad y fidelidad son tres dimensiones independientes.
No existe una unica buena explicacion. El mejor explicador depende del objetivo de la auditoria.
Segunda: la estabilidad por arquitectura no es un ranking de cuatro puestos sino una particion en dos
grupos, y esa particion concuerda entre los datos reales y los sinteticos, que es la evidencia mas
fuerte que tenemos de que no es un artefacto de un dataset. *(pausa)* Tercera: el explicador es la
palanca dominante, con tamanos de efecto de cero coma treinta y seis a cero coma sesenta y cuatro,
mientras que el balanceo es despreciable y el escenario de desbalance es pequeno y no significativo.
Cuarta: el mas plausible no es el mas fiel, y por eso la eleccion se hace segun se quiera reconocer el
patron o auditar el modelo. *(pausa)* Quinta, y es la que mas nos importa como aporte metodologico:
corregimos dos artefactos de medicion, y aprendimos a distinguir la reproducibilidad de los pesos de la
reproducibilidad de las conclusiones. Y sexta: las tres hipotesis que planteamos se refutaron, y lo
reportamos tal cual, sin ajustar el analisis para salvarlas.


> **En la tesis:** Cap. 7 (Conclusiones). Cap. 6 (Discusion).

### Pagina 37: Los cuatro objetivos, respondidos  ·  [JD]  ·  70 s
Esta lamina responde uno a uno los objetivos que planteamos, para que quede explicito en que medida cada
uno quedo atendido. *(pausa)* El primero, el impacto del desbalance sobre la robustez explicativa. La
respuesta es que no es el factor dominante que suponiamos: el perfil es plano y el balanceo tiene efecto
despreciable. La palanca real resulto ser el explicador, y eso reorienta la pregunta original. *(pausa)*
El segundo, la resiliencia comparada de las cuatro arquitecturas. La respuesta es que no hay cuatro
posiciones, hay dos grupos, y esa particion se replica en los dos regimenes de densidad. *(pausa)* El
tercero, el impacto de las estrategias de balanceo. Despreciable sobre las tres dimensiones, lo que
permite elegirlas por rendimiento. *(pausa)* Y el cuarto, la matriz de recomendacion. Aqui la respuesta
es que no existe una triada optima unica, precisamente porque las tres dimensiones son independientes, y
por eso la recomendacion es condicional al proposito de la auditoria, que es lo que veran en la
siguiente lamina. *(pausa)* Los cuatro objetivos quedaron respondidos, aunque dos de ellos con una
respuesta distinta de la que esperabamos al formularlos.


> **En la tesis:** Cap. 7, seccion 7.1 completa (Respuestas a los Objetivos de Investigacion).

### Pagina 38: Matriz de recomendacion  ·  [JD]  ·  60 s
Todo lo anterior se condensa en esta matriz de recomendacion, que responde al cuarto objetivo. La idea
es que no existe una combinacion unica que sea la mejor para todo, precisamente porque las tres
dimensiones son independientes. *(pausa)* Entonces la recomendacion es por objetivo. Si lo que se busca
es auditabilidad y estabilidad, GAT o GCN. Si el objetivo es recuperar el patron de lavado, es decir
plausibilidad, PGExplainer. Si lo que importa es la fidelidad al razonamiento del modelo, GNNExplainer.
Y si se busca estabilidad interna del propio metodo de explicacion, GNNShap. Esta tabla es mas util que
una recomendacion cerrada, porque obliga a hacer explicito el proposito de la auditoria antes de elegir
la herramienta.


> **En la tesis:** Cap. 7, seccion 7.1 (Respuestas a los Objetivos de Investigacion, O4). Cap. 6, seccion 6.6 (Implicaciones para la Practica de Auditoria en Entornos Regulados).

### Pagina 39: Contribuciones y limitaciones  ·  [JD]  ·  70 s
Recapitulo aportes y limites. *(pausa)* Contribuciones: mostramos que
estabilidad, plausibilidad y fidelidad son tres dimensiones independientes en este dominio. Corregimos
dos artefactos de evaluacion y reportamos dos bugs de PGExplainer. Construimos un generador sintetico
con patron verdadero por arista, y entregamos la matriz de recomendacion. *(pausa)* Limitaciones, y las
decimos nosotros antes de que el jurado las encuentre: la evidencia inferencial mas fuerte proviene del eje
sintetico, que es el unico donde plausibilidad y fidelidad son medibles. El clasificador colapsa en test
por el desplazamiento temporal, y aunque replicamos el entrenamiento con tres semillas de modelo en
ambos ejes, el numero de entrenamientos por celda sigue siendo modesto, suficiente para separar los dos
grupos de arquitecturas pero no para ordenar dentro de cada grupo, algo especialmente cierto en TAGCN
sobre Elliptic, cuya dispersion entre semillas es la mayor de las cuatro. Ninguna de estas invalida los
hallazgos, pero marcan hasta donde llegan.


> **En la tesis:** Cap. 6, secciones 6.4 (Contribuciones) y 6.8 (Limitaciones). Cap. 7, secciones 7.2 (Aportes Principales) y 7.3 (Limitaciones).

### Pagina 40: Conclusiones y cierre  ·  [JD]  ·  80 s
Para cerrar, respondo de frente nuestra pregunta de investigacion y dejo tres mensajes. *(pausa)* La
respuesta directa a la pregunta es que no existe una combinacion unica optima de arquitectura,
explicador y balanceo, y que la eleccion depende del proposito de la auditoria. Para estabilidad y
auditabilidad, GAT o GCN con un explicador consistente son el mejor punto de partida. *(pausa)* De ahi,
tres mensajes. Primero: estabilidad, plausibilidad y fidelidad no son lo mismo, y por eso el mejor
explicador depende del objetivo de quien audita. Segundo: con la medicion corregida, los datos reales y
los sinteticos cuentan una historia coherente, con GAT y GCN como las arquitecturas mas estables.
Y tercero, quiza el mas transversal: la estabilidad de una explicacion depende del protocolo de
evaluacion, no solo del metodo, y por eso corregir artefactos y retractar conclusiones apoyadas en ellos
no debilito la tesis, la hizo mas solida. *(pausa)* Como trabajo futuro, extender el eje real a datasets
con atributos no anonimizados, incorporar desplazamiento temporal al grafo sintetico, y explorar
arquitecturas temporales. *(pausa)* Con esto cerramos. Agradecemos al director y al jurado, y quedamos
atentos a sus preguntas.


> **En la tesis:** Cap. 7, secciones 7.1, 7.4 (Perspectivas Futuras) y 7.5 (Reflexion Final).
**[RELEVO: ambos de pie para la ronda de preguntas]**

---

## Notas de puesta en escena

- **Ritmo:** el bloque de resultados (paginas 24 a 34 del PDF) es el mas cargado. No lean la slide. Miren al
  jurado y usen la slide como respaldo. El discurso ya dice lo esencial. La slide tiene el detalle.
- **Los numeros que deben salir sin dudar:** grupo alto GAT 0,78 y GCN 0,76 frente a grupo bajo GraphSAGE 0,74 y TAGCN 0,67, con diferencias significativas entre grupos y no dentro, puente r = menos
  0,01, disociacion plausibilidad 0,80 (azar 0,40) contra fidelidad 0,11 para PGExplainer, concordancia de menos 0,20
  a mas 0,80 y PR-AUC de 0,37 en validacion a 0,02 en test.
- **Si se ponen nerviosos con una cifra,** digan el orden de magnitud y la direccion ("alrededor de
  cero coma ocho, muy por encima del otro"): el jurado valora que entiendan el resultado, no que
  reciten decimales.
- **Reconocer los errores juega a favor.** Cada vez que dicen "esto lo corregimos" o "esto refuto nuestra
  hipotesis", suman credibilidad. No lo escondan. Subrayenlo.

### Preguntas probables y respuestas ensayadas

- **"Por que GNNShap no esta en la lamina de disociacion?"** Respuesta: "Porque no produce mascara de
  aristas por diseno. Su plausibilidad se mide sobre features, que es otra metrica, con otra escala y
  otra linea base de azar: cero coma cero siete cinco, frente a cero coma cuarenta de aristas. Si
  pusieramos su cero coma quince junto a las barras de aristas, la figura sugeriria que esta por
  debajo del azar, cuando en realidad lo duplica. Sus tres valores estan en la lamina de respaldo,
  cada uno contra la linea base que le corresponde."
- **"El perfil por escenario, es plano de verdad o lo dibujaron en un eje de cero a uno?"** Respuesta:
  "No lo dejamos en la impresion visual, lo contrastamos. Kruskal-Wallis sobre el factor escenario da
  un valor p de cero coma dieciocho, asi que no rechazamos la igualdad entre los cinco niveles, y el
  tamano de efecto es de cero coma cero cinco, pequeno, frente al cero coma trece de la arquitectura
  sobre la misma metrica. Ademas la variacion no tiene la direccion que predecia la hipotesis: el
  minimo esta en el escenario uno a uno, que es el mas equilibrado."
- **"Por que unas tablas del Capitulo 4 usan una semilla y otras tres?"** Respuesta: "Es deliberado y
  el capitulo lo dice. La primera parte presenta la corrida original de una semilla, con la que
  detectamos los dos artefactos de medicion. Despues explicamos por que esa evidencia era insuficiente
  y replicamos con tres semillas. Todo lo que afirmamos como conclusion, la particion por arquitectura
  y el analisis por escenario, viene de las tres semillas. La tabla de ranking pone las tres columnas
  lado a lado para que la progresion se vea."
- **"Si el modelo colapsa en test, que sentido tiene medir la estabilidad de sus explicaciones?"**
  (la mas filosa, ensayenla palabra por palabra). Respuesta: "La estabilidad y el rendimiento son
  preguntas distintas. El colapso en test es un desplazamiento temporal del dato, documentado en la
  literatura de Elliptic, no un defecto del metodo. Nosotros medimos la estabilidad donde la pregunta
  tiene sentido, sobre los verdaderos positivos de validacion, donde el modelo si discrimina. Explicar
  una prediccion equivocada no aporta informacion. Y eso esta declarado en la tesis desde el principio.
  Ademas, la coherencia con el eje sintetico, donde no hay colapso, respalda que lo que medimos sobre
  validacion no es un artefacto del colapso."
- **"La evidencia fuerte viene del dataset que ustedes construyeron. No es circular?"** Respuesta: el
  patron verdadero es ciego al explicador, se fija en la construccion del grafo antes de correr ningun
  metodo, y se anadieron distractores y atenuacion de atributos para que acertar no fuera trivial. El
  eje real aporta la validez externa que el sintetico no puede dar. Apoyarse en `DEFENSA_R2_evidencia_sintetica.md`.
- **"Por que GAT y GCN y no una sola?"** Respuesta: porque la diferencia entre las de la parte alta no
  es estadisticamente significativa (Wilcoxon 0,37). Afirmar un unico ganador seria sobre-interpretar.
- **"En la tabla filtrada, GCN tiene una sola configuracion. Como sostienen que GCN encabeza?"**
  (la segunda mas filosa, y la unica que ataca directo la pagina 26, Resultado 1, ensayenla). Respuesta: "Tiene
  razon en que el n es desigual, y por eso el ranking que afirmamos es el de la corrida completa de
  sesenta configuraciones, donde las cuatro arquitecturas tienen soporte comparable: ahi GAT queda en
  cero coma setenta y ocho, GCN en cero coma setenta y seis, GraphSAGE en cero coma setenta y cuatro y
  TAGCN en cero coma sesenta y siete, sobre tres semillas de modelo. La columna filtrada de la version
  de una sola semilla la reportabamos como control de robustez, no
  como estimacion: su valor es que no invierte el orden. Con una sola configuracion, el cero coma
  ochenta y tres de GCN no admite lectura inferencial, y no lo presentamos como tal. Por eso la
  conclusion se enuncia en grueso, GAT y GCN encabezan y TAGCN queda atras, y nunca como un
  ordenamiento fino entre los primeros. Es la misma prudencia con la que declaramos que GAT y
  GraphSAGE son indistinguibles."
  *(Si insisten, la carta fuerte: el eje sintetico, que si tiene replicacion con tres grafos por tres
  semillas, pone a GCN y GAT arriba con cero coma noventa y seis. Son dos regimenes independientes
  apuntando al mismo sitio, y esa concordancia es lo que sostiene la conclusion, no una celda.)*
- **"Por que el ranking por arquitectura solo usa GNNExplainer?"** Respuesta: porque en Elliptic es el
  unico de los tres explicadores que discrimina entre arquitecturas. PGExplainer degenera y da una
  correlacion de Spearman nula para las cuatro, y GNNShap se satura cerca de cero coma noventa y cinco,
  indistinguible entre arquitecturas, igual que el indice de Jaccard. GNNExplainer es el unico con senal
  medible en este eje, y ademas es el explicador comun a los dos ejes, lo que hace directamente
  comparable la particion entre Elliptic y el sintetico. La eleccion no es arbitraria: es la unica que
  permite el contraste.
- **"Sobre la bibliografia, la referencia de 2026 existe?"** Respuesta: si, esta verificada en Crossref
  (DOI resuelve, revista Springer indexada). Tengan el DOI a la mano.

### Reparto de preguntas
- Cada quien responde con solvencia el tema del bloque que presento. El otro complementa. Tengan a mano
  el mapa de `GUION_defensa_por_capitulo.md` (parte B).
- **Slides de respaldo (ya creadas, paginas 29 a 33 del PDF):** R1 el bug de Spearman en detalle,
  R2 la estabilidad por semilla, R3 el detalle estadistico de la particion, R4 por que PR-AUC y no
  ROC-AUC, R5 la construccion del grafo sintetico. Solo se muestran si el jurado las pide. Sepan de
  memoria en que pagina esta cada una.
