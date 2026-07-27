# Discurso de defensa a dos voces (v2)

> Sustentacion de la tesis "Estabilidad de Metodos de Explicabilidad (XAI) en Graph Neural
> Networks para la Deteccion de Lavado de Dinero bajo Desbalance Extremo". UTP, MISC, 2026.
> Autores: Alejandro Gomez Huertas y Juan Diego Garzon Ovalle. Director: Ph.D. Cristian Rosero Arias.
>
> **Duracion objetivo:** 22 a 25 minutos. **Reparto:** Alejandro presenta el Bloque A (contexto y
> metodologia, slides 1 a 12); Juan Diego presenta el Bloque B (resultados y cierre, slides 13 a 21).
> Cada slide indica quien habla, el tiempo objetivo, el guion hablado y el gesto de transicion.
>
> Escrito sin guiones largos ni medios. Los numeros son la version corregida (post fix R1).
> El discurso esta redactado para decirse en voz alta, no para leerse palabra por palabra:
> ensayenlo hasta que suene propio y ajusten el fraseo a su manera de hablar.
>
> **Cambios de esta v2 respecto de la v1:** (1) se enuncian de forma explicita las tres hipotesis
> falsables en el Bloque A y se cierra su refutacion en el Bloque B; (2) se responde de forma directa
> el Objetivo 1 (el nivel de desbalance no degrada la estabilidad); (3) GNNShap gana un rol claro en la
> narrativa de resultados; (4) el cierre responde de frente la pregunta de investigacion; (5) se anade
> una respuesta ensayada a la pregunta mas filosa del jurado.

---

## Convenciones

- **[A]** habla Alejandro. **[JD]** habla Juan Diego. **[RELEVO]** marca el cambio de voz.
- Los tiempos suman unos 23 minutos, dejando margen para respirar y para el jurado.
- Donde dice *(pausa)* conviene un silencio corto de uno a dos segundos: da peso a la idea.

### Mapa del slide del guion a la pagina del PDF

> **Atencion: la numeracion del guion no coincide con el numero de pagina del PDF.** El guion tiene
> 21 slides de contenido; `presentacion_latex/beamer_defensa.pdf` tiene **26 paginas**, porque
> intercala **5 separadores de seccion** (laminas oscuras con el numero y el nombre del bloque). Los
> separadores no se hablan: se pasan mientras se dice la frase de transicion. Esta tabla dice en que
> pagina del PDF esta cada slide del guion.

| Guion | Pagina PDF | Guion | Pagina PDF |
|---|---|---|---|
| 1. Portada | **1** | 12. Metricas | **15** |
| 2. Contenido | **2** | *(separador "4. Resultados")* | *16* |
| *(separador "1. Problema y motivacion")* | *3* | 13. Dos artefactos | **17** |
| 3. El problema | **4** | 14. Ranking por arquitectura | **18** |
| 4. Pregunta, objetivos e hipotesis | **5** | 15. Concordancia entre regimenes | **19** |
| 5. La brecha | **6** | 16. Disociacion plausibilidad y fidelidad | **20** |
| *(separador "2. Marco conceptual")* | *7* | 17. Puente nulo, desbalance y balanceo | **21** |
| 6. Marco conceptual | **8** | 18. Colapso validacion a test | **22** |
| 7. Tres propiedades | **9** | *(separador "5. Conclusiones")* | *23* |
| *(separador "3. Metodologia")* | *10* | 19. Matriz de recomendacion | **24** |
| 8. Dos ejes | **11** | 20. Contribuciones y limitaciones | **25** |
| 9. Diseno factorial | **12** | 21. Conclusiones y cierre | **26** |
| 10. Elliptic | **13** | | |
| 11. Sintetico | **14** | | |

**El [RELEVO] cae en la pagina 16**, que es justamente el separador "4. Resultados": Alejandro cierra
en la 15 (Metricas), pasa a la 16 mientras dice la frase de entrega, y Juan Diego arranca ya con la
17 en pantalla. Es el punto de cambio mas limpio posible; aprovechenlo.

---

## BLOQUE A: Alejandro Gomez (slides 1 a 12, aprox. 11 min)


### Slide 1: Portada  ·  [A]  ·  40 s
Buenas tardes. Agradecemos al jurado y al director su tiempo. Somos Alejandro Gomez y Juan Diego
Garzon, y hoy defendemos nuestra tesis de maestria sobre la estabilidad de los metodos de
explicabilidad en redes neuronales de grafos, aplicados a la deteccion de lavado de dinero bajo
condiciones de desbalance extremo. La presentacion va a dos voces: yo abro con el problema y la
metodologia, y mi companero continua con los resultados y las conclusiones. *(pausa)* Empecemos por
que este problema importa.


> **En la tesis:** Portada; Cap. 1 (Introduccion).

### Slide 2: Contenido  ·  [A]  ·  25 s
El recorrido tiene cinco partes. Primero el problema y por que la explicabilidad es critica en este
dominio. Segundo, la pregunta de investigacion y los objetivos. Tercero, la metodologia, que se apoya
en un diseno de dos ejes que es el corazon del aporte. Cuarto, los resultados. Y quinto, las
conclusiones, los limites y el trabajo futuro. *(pausa breve)* Vamos al problema.


> **En la tesis:** Cap. 1, seccion 1.8 (Estructura de la Tesis).

### Slide 3: El problema  ·  [A]  ·  70 s
El lavado de dinero mueve entre el dos y el cinco por ciento del producto interno bruto mundial. Los
sistemas tradicionales de deteccion funcionan por reglas fijas, y ese enfoque genera entre el noventa
y cinco y el noventa y ocho por ciento de falsos positivos. Es decir, de cada cien alertas, casi todas
son ruido que un analista humano debe revisar a mano. *(pausa)* Las redes neuronales de grafos ofrecen
una alternativa poderosa, porque modelan las transacciones como lo que realmente son, una red de flujos
entre cuentas, y superan a los metodos tabulares que miran cada transaccion de forma aislada. El
problema es que estas redes son cajas negras. Y en un entorno regulado, donde una alerta puede
congelar cuentas o iniciar una investigacion, no basta con acertar: cada decision debe poder
justificarse ante un auditor y debe ser reproducible. *(pausa)* Ahi es donde entra la explicabilidad,
y ahi es donde encontramos que faltaba algo.


> **En la tesis:** Cap. 1, seccion 1.1 (Planteamiento y Contexto del Problema); Cap. 2, seccion 2.3 (Sistemas Tradicionales de Monitoreo) y seccion 2.7 (Del Monitoreo Basado en Reglas hacia Enfoques Estructurales).

### Slide 4: Pregunta, objetivos e hipotesis  ·  [A]  ·  80 s
Nuestra pregunta es la siguiente: como se comporta la estabilidad de los metodos de explicabilidad
sobre redes de grafos para deteccion de lavado, cuando el dato esta fuertemente desbalanceado, y que
combinacion de arquitectura, explicador y estrategia de balanceo produce la interpretacion mas robusta
y auditable. *(pausa)* De ahi se desprenden cuatro objetivos. El primero, medir como se degrada la
estabilidad de las explicaciones a medida que el desbalance se agrava. El segundo, comparar la
resiliencia de cuatro arquitecturas: GCN, GraphSAGE, GAT y TAGCN. El tercero, evaluar si las
estrategias de balanceo afectan la calidad de las explicaciones. Y el cuarto, condensar todo en una
matriz de recomendacion que diga que usar segun el objetivo. *(pausa)* Y para comprometernos con
predicciones falsables antes de ver los datos, planteamos tres hipotesis: que la estabilidad se
degradaria al agravarse el desbalance; que TAGCN, por su alcance multi-hop, seria la arquitectura mas
estable; y que un explicador mas estable senalaria tambien mejor el patron de lavado. Adelanto algo que
mi companero va a mostrar con datos: las tres se matizaron o se cayeron, y esa honestidad es parte del
aporte.


> **En la tesis:** Cap. 1, seccion 1.4 (Formulacion del Problema) y seccion 1.5 (Objetivos: 1.5.1 General, 1.5.2 Especificos, 1.5.3 Hipotesis de Trabajo).

### Slide 5: La brecha  ·  [A]  ·  60 s
Que se sabia ya. Se sabia que las redes de grafos superan a los metodos tabulares en este dominio, y
se habian estudiado la prediccion y la explicabilidad, pero por separado. *(pausa)* Que faltaba. Nadie
habia evaluado de forma sistematica la estabilidad de las explicaciones sobre grafos financieros, es
decir, si una explicacion se sostiene o cambia cuando se vuelve a calcular. Y habia un obstaculo de
fondo: el dataset de referencia, Elliptic, no trae un patron verdadero de tipologia de lavado, de modo
que la plausibilidad de una explicacion, si senala o no el patron correcto, simplemente no era medible.
Esa doble brecha, la estabilidad no evaluada y la plausibilidad no medible, es la que esta tesis viene
a cerrar.


> **En la tesis:** Cap. 1, seccion 1.2 (Revision de Literatura y Estado del Arte) y seccion 1.3 (La Arista de Investigacion); Cap. 3, seccion 3.6.4 (El Problema de la Estabilidad Explicativa).

### Slide 6: Marco conceptual  ·  [A]  ·  55 s
Un marco minimo para lo que sigue. Trabajamos con cuatro arquitecturas de red de grafos, que se
diferencian en como cada nodo agrega informacion de sus vecinos: GCN usa una convolucion espectral de
un salto, GraphSAGE muestrea y agrega de forma inductiva, GAT pondera a los vecinos con atencion, y
TAGCN usa filtros polinomicos que alcanzan varios saltos. *(pausa)* Y sobre esas redes aplicamos tres
explicadores post-hoc, que operan despues de entrenar: GNNExplainer, que optimiza una mascara por cada
instancia; PGExplainer, que entrena una red generativa para explicar; y GNNShap, que usa valores de
Shapley por muestreo. Cuatro arquitecturas por tres explicadores es el nucleo de la comparacion.


> **En la tesis:** Cap. 3, secciones 3.2 (GNN: Fundamentos Conceptuales), 3.3 (Arquitecturas GNN Fundamentales), 3.4 (TAGCN) y 3.6.2 (Metodos XAI para GNNs).

### Slide 7: Tres propiedades  ·  [A]  ·  65 s
Esta slide contiene la tesis central, asi que me detengo. Cuando decimos que una explicacion es
"buena", en realidad mezclamos tres preguntas distintas. La estabilidad pregunta si la explicacion se
reproduce cuando cambio la semilla o perturbo un poco la entrada. La plausibilidad pregunta si la
explicacion senala el patron real de lavado, lo que un experto reconoceria. Y la fidelidad pregunta si
la explicacion refleja de verdad lo que el modelo uso para decidir. *(pausa)* La tesis central es que
estas tres son dimensiones distintas que no se implican entre si. Una explicacion puede ser muy estable
y aun asi apuntar al patron equivocado. Puede ser plausible para un humano y no reflejar el mecanismo
del modelo. Gran parte de la literatura reporta una sola de estas y la llama calidad; nosotros vamos a
mostrar, con datos, que hay que medir las tres por separado.


> **En la tesis:** Cap. 3, seccion 3.8.4 (Metricas de Estabilidad y Fidelidad Explicativa); Cap. 6, seccion 6.2 (Estabilidad, Plausibilidad y Fidelidad como Tres Dimensiones Independientes).

### Slide 8: Dos ejes  ·  [A]  ·  70 s
Como se prueba algo asi. Con un diseno de dos ejes, que es la decision metodologica mas importante de
la tesis. *(pausa)* El primer eje es Elliptic, el dataset real de transacciones de Bitcoin. Nos da
validez externa, porque son datos reales con todo su ruido y su desbalance, pero tiene dos limites:
no trae patron verdadero, asi que solo permite medir estabilidad, y sus vecindarios son minusculos,
de unos dos nodos. El segundo eje es un grafo sintetico que construimos nosotros, con patron verdadero
por cada nodo y por cada arista. Nos da validez interna: como sabemos cual es el patron correcto,
podemos medir plausibilidad y fidelidad, cosa imposible en Elliptic. *(pausa)* La clave es que los dos
ejes se complementan. Ninguno solo alcanza; juntos permiten afirmar cosas que ninguno probaria por su
cuenta. Y aqui adelanto un punto que mi companero va a demostrar: los dos ejes, bien medidos, cuentan
la misma historia.


> **En la tesis:** Cap. 5, seccion 5.1 (Por que se Construye un Grafo Sintetico); Cap. 6, seccion 6.1 (Lectura Conjunta de los Dos Ejes).

### Slide 9: Diseno factorial  ·  [A]  ·  55 s
El experimento es una matriz factorial completa: cuatro arquitecturas, por tres explicadores, por tres
estrategias de balanceo, por cinco escenarios de desbalance. Eso da sesenta configuraciones por eje,
cada una con los tres explicadores para el estudio de estabilidad. *(pausa)* Y no nos quedamos en una
sola corrida. Cada explicacion se repite cinco veces con semillas distintas para medir su estabilidad.
Y en el eje sintetico anadimos una capa de robustez: tres semillas de modelo por tres grafos
independientes, con pruebas estadisticas serias, Kruskal-Wallis, Wilcoxon e intervalos de confianza
por bootstrap. Esto es lo que convierte observaciones sueltas en evidencia con respaldo.


> **En la tesis:** Cap. 4, seccion 4.2 (Pipeline Experimental y Espacio Factorial); Cap. 5, seccion 5.4 (Analisis Estadistico de Robustez); Cap. 8, seccion 8.1 (Espacio de Busqueda de Hiperparametros).

### Slide 10: Elliptic  ·  [A]  ·  60 s
El primer eje en detalle. Elliptic tiene doscientos tres mil setecientos sesenta y nueve nodos,
doscientas treinta y cuatro mil aristas, ciento sesenta y seis atributos por nodo y cuarenta y nueve
pasos temporales. Las transacciones ilicitas son apenas el dos coma dos por ciento, una razon cercana a
uno a nueve en la parte etiquetada. *(pausa)* Hicimos una particion temporal causal: entrenamos con el
pasado y evaluamos con el futuro, que es como opera un sistema real. Y hay un dato que gobierna todo lo
demas: el vecindario tipico de un nodo tiene una mediana de unos dos nodos. Es un grafo extremadamente
disperso. Por eso, como veran, la unica metrica de estabilidad que discrimina bien aqui es la
correlacion de Spearman entre rankings de atributos; las metricas de aristas se saturan.


> **En la tesis:** Cap. 4, seccion 4.1 (Preprocesamiento y Analisis Exploratorio: 4.1.1 Composicion, 4.1.2 Dispersion de la Topologia).

### Slide 11: Sintetico  ·  [A]  ·  65 s
El segundo eje, nuestro grafo sintetico, responde a una necesidad concreta: para medir si una
explicacion es plausible, hay que saber de antemano cual es el subgrafo correcto, y Elliptic no lo da.
Asi que lo construimos. *(pausa)* Inyectamos cuatro tipologias de lavado reconocidas: structuring,
layering, fan-in y fan-out. Y tomamos tres decisiones para que la medicion sea honesta: simetrizamos
las aristas para que el vecindario tenga estructura suficiente, anadimos aristas distractoras para que
acertar no sea trivial, y atenuamos la firma de los atributos para que la tarea no se resuelva sola.
Un detalle importante que nos van a preguntar: el patron verdadero es ciego al explicador, se define
en la construccion del grafo, antes de correr ningun metodo. No lo ajustamos para favorecer a nadie.


> **En la tesis:** Cap. 5, seccion 5.1 (Por que se Construye un Grafo Sintetico) y seccion 5.2 (Construccion del Grafo Sintetico y sus Tipologias).

### Slide 12: Metricas  ·  [A]  ·  55 s
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

## BLOQUE B: Juan Diego Garzon (slides 13 a 21, aprox. 12 min)


### Slide 13: Dos artefactos de evaluacion  ·  [JD]  ·  85 s
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


> **En la tesis:** Cap. 4, seccion 4.5 (De un Artefacto de Computo a un Artefacto de Medida: dos Correcciones Metodologicas); Cap. 6, seccion 6.4 (Contribuciones Metodologicas).

### Slide 14: Ranking por arquitectura  ·  [JD]  ·  75 s
Con la metrica corregida, y replicando el entrenamiento completo con tres semillas de modelo, lo que
encontramos no es un ranking de cuatro puestos sino una particion en dos grupos. Un grupo alto, con GAT
en cero coma setenta y ocho y GCN en cero coma setenta y seis, y un grupo bajo, con GraphSAGE en cero
coma setenta y tres y TAGCN en cero coma sesenta y ocho. *(pausa)* Y lo importante es donde estan las
diferencias: entre los dos grupos son estadisticamente significativas, y dentro de cada grupo no lo son.
El Kruskal-Wallis global da un valor p del orden de diez elevado a menos cinco, pero al preguntarlo
dentro del grupo alto, o dentro del grupo bajo, la igualdad no se rechaza. *(pausa)* Aqui cierro nuestra
segunda hipotesis: esperabamos que TAGCN, por su alcance multi-hop, fuera la mas estable, y aparece de
forma consistente en el grupo bajo en las tres semillas. La hipotesis se cae, y lo decimos sin rodeos.
*(pausa)* Quiero ser honesto con dos cosas mas. La primera, que el liderazgo de GraphSAGE que reportaba
una version anterior de la tesis era un artefacto del truncamiento de la metrica, y corregido se
disuelve. La segunda, que GAT y GCN se permutan entre semillas, asi que no afirmamos que ninguna de las
dos sea la mejor: afirmamos que las dos forman el grupo alto. Decir menos seria impreciso, y decir mas
seria sobre-interpretar.


> **En la tesis:** Cap. 4, seccion 4.5 (tabla tab:ranking); Cap. 8, seccion 8.2 (Resultados Completos por Configuracion sobre el Elliptic Dataset).

### Slide 15: Concordancia entre regimenes  ·  [JD]  ·  70 s
Este resultado es uno de los que mas me gustan, porque nacio de un error corregido. En una version
previa creiamos haber encontrado una "inversion por densidad": que el orden de estabilidad entre
arquitecturas se daba vuelta al pasar del grafo disperso de Elliptic al grafo denso sintetico. *(pausa)*
Cuando corregimos el bug de la metrica, esa inversion desaparecio. Lo que en realidad ocurre es lo
contrario: los dos regimenes concuerdan. Las mismas arquitecturas que son estables en el grafo denso
lo son en el disperso. Lo cuantificamos con la correlacion de rangos entre ambos regimenes, que pasa de
menos cero coma veinte con la metrica defectuosa, a mas cero coma ochenta con la metrica corregida.
*(pausa)* Y esto, lejos de debilitar la tesis, la refuerza, porque significa que datos reales y datos
sinteticos cuentan la misma historia. La coherencia entre los dos ejes es lo que le da solidez a todo
el diseno.


> **En la tesis:** Cap. 5, seccion 5.3 (Resultados de la Matriz Factorial); Cap. 6, secciones 6.1 y 6.2.

### Slide 16: Disociacion plausibilidad y fidelidad  ·  [JD]  ·  85 s
Ahora el hallazgo central sobre los explicadores, y es un resultado con dos caras. *(pausa)* Por un
lado, PGExplainer es claramente el que mejor recupera el patron real: su plausibilidad de aristas es de
cero coma ochenta, frente a cero coma cincuenta de GNNExplainer, y la diferencia es enorme
estadisticamente, con un valor de significancia del orden de diez elevado a menos treinta y cinco. Es
decir, si el objetivo es senalar el patron de lavado, PGExplainer gana sin discusion. *(pausa)* Pero
por otro lado, ese mismo PGExplainer colapsa en fidelidad: cuando medimos cuanto depende la prediccion
del modelo de las aristas que PGExplainer marca, el valor cae a cero coma once, frente a cero coma
cincuenta y seis de GNNExplainer. La lectura es contraintuitiva y potente: el explicador mas plausible
no es el mas fiel. PGExplainer recupera las aristas que definen el patron que un humano reconoce, pero
GNNExplainer recupera las aristas que el modelo realmente usa, y esos dos conjuntos no coinciden.
*(pausa)* Y para no dejar fuera al tercer explicador: GNNShap es el mas estable internamente de los
tres, el mas consistente entre ejecuciones, aunque no lidere ni plausibilidad ni fidelidad. Cada
explicador, entonces, tiene su fortaleza en una dimension distinta. Esta disociacion solo se puede
exhibir cuando tienes un patron verdadero contra el cual medir, y por eso el eje sintetico era
indispensable.


> **En la tesis:** Cap. 5, seccion 5.6 (La Disociacion entre Plausibilidad y Fidelidad); la plausibilidad de aristas en seccion 5.3.

### Slide 17: Puente nulo, desbalance y balanceo  ·  [JD]  ·  80 s
Aqui cierro las otras dos hipotesis, y las dos se caen. *(pausa)* La tercera hipotesis esperaba que una
explicacion mas estable fuera tambien mas plausible, que la consistencia implicara acierto. Los datos
dicen que no. La correlacion entre estabilidad y plausibilidad es de menos cero coma cero uno, con un
intervalo de confianza que incluye el cero. Es un puente nulo. Un explicador estable no es por ello mas
acertado sobre el patron real. Es un no-resultado, y lo reportamos con honestidad precisamente porque
contradice lo que esperabamos; si hubieramos disenado el experimento para lucirnos, habriamos forzado
una correlacion bonita, y no lo hicimos. *(pausa)* La primera hipotesis esperaba que la estabilidad se
degradara al agravarse el desbalance. Tampoco. El perfil de estabilidad de uno a uno hasta uno a cien es
esencialmente plano, sin un punto de quiebre. El nivel de desbalance, por si mismo, no gobierna la
reproducibilidad de las explicaciones. *(pausa)* Y en la misma linea, la estrategia de balanceo, que
suele recibir mucha atencion, resulta practicamente irrelevante: su tamano de efecto es menor a cero
coma cero dos en las tres dimensiones. En la practica, esto significa que el balanceo puede elegirse por
rendimiento, sin temor a degradar la interpretabilidad.


> **En la tesis:** Cap. 5, seccion 5.5 (La Ausencia de un Puente entre Estabilidad y Plausibilidad); Cap. 4, seccion 4.5 (perfil por escenario); Cap. 6, seccion 6.3 (El Papel Secundario del Balanceo y de la Arquitectura).

### Slide 18: Colapso validacion a test  ·  [JD]  ·  70 s
Un resultado de rendimiento que debemos declarar con transparencia, porque enmarca todo lo anterior.
Los modelos aprenden en validacion, con un PR-AUC medio de cero coma treinta y siete, pero colapsan en
test, donde cae a cero coma cero dos. *(pausa)* La causa es el desplazamiento temporal del dataset: los
patrones de lavado cambian entre los primeros y los ultimos pasos, y un modelo entrenado con el pasado
encuentra en el futuro una distribucion distinta. Es una propiedad del dato, no un defecto de nuestro
metodo. Aqui hay un punto metodologico que quisimos remarcar: el ROC-AUC se ve enganosamente alto bajo
desbalance extremo, cero coma ochenta y ocho en validacion, y por eso no lo usamos como metrica
principal; usamos PR-AUC y precision at k, que no se dejan enganar. *(pausa)* Como consecuencia, la
estabilidad la estudiamos sobre los verdaderos positivos de validacion, donde el modelo si discrimina,
y lo declaramos de forma abierta. No es esconder el colapso, es medir donde la pregunta tiene sentido.


> **En la tesis:** Cap. 4, seccion 4.4 (Rendimiento Predictivo y el Colapso de Validacion a Test); Cap. 3, seccion 3.8.2.

### Slide 19: Matriz de recomendacion  ·  [JD]  ·  60 s
Todo lo anterior se condensa en esta matriz de recomendacion, que responde al cuarto objetivo. La idea
es que no existe una combinacion unica que sea la mejor para todo, precisamente porque las tres
dimensiones son independientes. *(pausa)* Entonces la recomendacion es por objetivo. Si lo que se busca
es auditabilidad y estabilidad, GAT o GCN. Si el objetivo es recuperar el patron de lavado, es decir
plausibilidad, PGExplainer. Si lo que importa es la fidelidad al razonamiento del modelo, GNNExplainer.
Y si se busca estabilidad interna del propio metodo de explicacion, GNNShap. Esta tabla es mas util que
una recomendacion cerrada, porque obliga a hacer explicito el proposito de la auditoria antes de elegir
la herramienta.


> **En la tesis:** Cap. 7, seccion 7.1 (Respuestas a los Objetivos de Investigacion, O4); Cap. 6, seccion 6.6 (Implicaciones para la Practica de Auditoria en Entornos Regulados).

### Slide 20: Contribuciones y limitaciones  ·  [JD]  ·  70 s
Recapitulo aportes y limites, con la misma honestidad. *(pausa)* Contribuciones: mostramos que
estabilidad, plausibilidad y fidelidad son tres dimensiones independientes en este dominio; corregimos
dos artefactos de evaluacion y reportamos dos bugs de PGExplainer; construimos un generador sintetico
con patron verdadero por arista; y entregamos la matriz de recomendacion. *(pausa)* Limitaciones, y las
decimos sin rodeos porque el jurado las va a ver: la evidencia inferencial mas fuerte proviene del eje
sintetico, que es el unico donde plausibilidad y fidelidad son medibles; el clasificador colapsa en test
por el desplazamiento temporal; y en Elliptic trabajamos con una sola semilla de modelo. Ninguna de
estas invalida los hallazgos, pero delimitan con honestidad hasta donde llegan.


> **En la tesis:** Cap. 6, secciones 6.4 (Contribuciones) y 6.8 (Limitaciones); Cap. 7, secciones 7.2 (Aportes Principales) y 7.3 (Limitaciones).

### Slide 21: Conclusiones y cierre  ·  [JD]  ·  80 s
Para cerrar, respondo de frente nuestra pregunta de investigacion y dejo tres mensajes. *(pausa)* La
respuesta directa a la pregunta es que no existe una combinacion unica optima de arquitectura,
explicador y balanceo, y que la eleccion depende del proposito de la auditoria; para estabilidad y
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

- **Ritmo:** el bloque de resultados (slides 13 a 18) es el mas cargado. No lean la slide; miren al
  jurado y usen la slide como respaldo. El discurso ya dice lo esencial; la slide tiene el detalle.
- **Los numeros que deben salir sin dudar:** grupo alto GAT 0,78 y GCN 0,76 frente a grupo bajo GraphSAGE 0,73 y TAGCN 0,68, con diferencias significativas entre grupos y no dentro; puente r = menos
  0,01; disociacion plausibilidad 0,80 contra fidelidad 0,11 para PGExplainer; concordancia menos 0,20
  a mas 0,80; PR-AUC 0,37 en validacion a 0,02 en test.
- **Si se ponen nerviosos con una cifra,** digan el orden de magnitud y la direccion ("alrededor de
  cero coma ocho, muy por encima del otro"): el jurado valora que entiendan el resultado, no que
  reciten decimales.
- **La honestidad es su mejor carta.** Cada vez que dicen "esto lo corregimos" o "esto refuto nuestra
  hipotesis", suman credibilidad. No lo escondan; subrayenlo.

### Preguntas probables y respuestas ensayadas

- **"Si el modelo colapsa en test, que sentido tiene medir la estabilidad de sus explicaciones?"**
  (la mas filosa; ensayenla palabra por palabra). Respuesta: "La estabilidad y el rendimiento son
  preguntas distintas. El colapso en test es un desplazamiento temporal del dato, documentado en la
  literatura de Elliptic, no un defecto del metodo. Nosotros medimos la estabilidad donde la pregunta
  tiene sentido, sobre los verdaderos positivos de validacion, donde el modelo si discrimina; explicar
  una prediccion equivocada no aporta informacion. Y lo declaramos de forma abierta, no lo escondemos.
  Ademas, la coherencia con el eje sintetico, donde no hay colapso, respalda que lo que medimos sobre
  validacion no es un artefacto del colapso."
- **"La evidencia fuerte viene del dataset que ustedes construyeron. No es circular?"** Respuesta: el
  patron verdadero es ciego al explicador, se fija en la construccion del grafo antes de correr ningun
  metodo, y se anadieron distractores y atenuacion de atributos para que acertar no fuera trivial. El
  eje real aporta la validez externa que el sintetico no puede dar. Apoyarse en `DEFENSA_R2_evidencia_sintetica.md`.
- **"Por que GAT y GCN y no una sola?"** Respuesta: porque la diferencia entre las de la parte alta no
  es estadisticamente significativa (Wilcoxon 0,37); afirmar un unico ganador seria sobre-interpretar.
- **"En la tabla filtrada, GCN tiene una sola configuracion. Como sostienen que GCN encabeza?"**
  (la segunda mas filosa, y la unica que ataca directo el Slide 14; ensayenla). Respuesta: "Tiene
  razon en que el n es desigual, y por eso el ranking que afirmamos es el de la corrida completa de
  sesenta configuraciones, donde las cuatro arquitecturas tienen soporte comparable: ahi GAT queda en
  cero coma setenta y ocho, GCN en cero coma setenta y seis, GraphSAGE en cero coma setenta y tres y
  TAGCN en cero coma sesenta y ocho, sobre tres semillas de modelo. La columna filtrada de la version
  de una sola semilla la reportabamos como control de robustez, no
  como estimacion: su valor es que no invierte el orden. Con una sola configuracion, el cero coma
  ochenta y tres de GCN no admite lectura inferencial, y no lo presentamos como tal. Por eso la
  conclusion se enuncia en grueso, GAT y GCN encabezan y TAGCN queda atras, y nunca como un
  ordenamiento fino entre los primeros. Es la misma prudencia con la que declaramos que GAT y
  GraphSAGE son indistinguibles."
  *(Si insisten, la carta fuerte: el eje sintetico, que si tiene replicacion con tres grafos por tres
  semillas, pone a GCN y GAT arriba con cero coma noventa y seis. Son dos regimenes independientes
  apuntando al mismo sitio, y esa concordancia es lo que sostiene la conclusion, no una celda.)*
- **"Sobre la bibliografia, la referencia de 2026 existe?"** Respuesta: si, esta verificada en Crossref
  (DOI resuelve, revista Springer indexada); tengan el DOI a la mano.

### Reparto de preguntas
- Cada quien responde con solvencia el tema del bloque que presento; el otro complementa. Tengan a mano
  el mapa de `GUION_defensa_por_capitulo.md` (parte B).
- **Slides de respaldo sugeridas (backup, despues de la 21):** tabla factorial completa, detalle
  estadistico con intervalos, y el bug de Spearman ilustrado. Solo se muestran si el jurado los pide.
