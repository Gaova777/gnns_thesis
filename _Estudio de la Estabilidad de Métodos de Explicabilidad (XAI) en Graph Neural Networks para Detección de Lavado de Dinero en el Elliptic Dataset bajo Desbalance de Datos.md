

|  | FACULTAD DE INGENIERÍAS MAESTRIA EN  INGENIERIA EN SISTEMAS Y COMPUTACIÓN  | Version 1.0 Fecha: 24/02/2026 |
| :---: | :---: | :---: |

**1\.  TÍTULO**:  Estudio de la Estabilidad de Métodos de Explicabilidad (XAI) en Graph Neural Networks para Detección de Lavado de Dinero en el Elliptic Dataset bajo Desbalance de Datos

**ÁREA**: Ingeniería de Sistemas y Computación

**LÍNEA DE INVESTIGACIÓN**: Inteligencia Artificial y Ciencia de Datos

**2\.  PROPONENTE**: Alejandro Gomez Huertas  **CÉDULA**: 1088347904   
     **PROPONENTE**: Juan Diego Garzón Ovalle  **CÉDULA**: 1116445008  

**3\.  DIRECTOR PROPUESTO:** Ph.D. Cristian Rosero Arias

**E-MAIL:** cristianroseroa@utp.edu.co

**4\.  DEFINICIÓN DEL PROBLEMA**

**4.1. Descripción del Problema**

El lavado de dinero, conocido en la literatura internacional como Anti-Money Laundering (AML), constituye un mecanismo fundamental de financiamiento para la delincuencia organizada transnacional, permitiendo la integración de fondos ilícitos provenientes del narcotráfico, la trata de personas, el tráfico de armas y la corrupción en la economía legítima \[1\]. Este fenómeno no solo representa un problema de orden público, sino que constituye una amenaza sistémica para la estabilidad de los mercados financieros globales y el estado de derecho. El impacto económico del lavado de dinero es una amenaza de magnitud global. La Oficina de las Naciones Unidas contra la Droga y el Delito (UNODC) estima que los flujos financieros ilícitos representan anualmente entre el 2% y el 5% del Producto Interno Bruto (PIB) mundial, traduciéndose en una circulación de USD $800 mil millones a USD $2 billones cada año a través de los sistemas financieros internacionales \[2\]. En el ecosistema de criptomonedas, el volumen de lavado de dinero ha crecido exponencialmente: mientras que en 2020 se estimaron $1.3 mil millones, para 2022 la cifra alcanzó los $23.8 mil millones. Este último valor representó un incremento del 68% respecto a los niveles de 2021, consolidando una tendencia al alza en la sofisticación de estos delitos \[2\]. En 2023, el Departamento de Justicia de Estados Unidos presentó cargos contra Binance y su CEO por violar leyes AML involucrando $4.3 mil millones \[3\], evidenciando la urgencia de mecanismos de detección más efectivos.

A pesar de la magnitud billonaria de estos flujos ilícitos, la eficacia de los mecanismos de contención actuales es alarmantemente baja. Las autoridades financieras globales logran interceptar menos del 1% de los flujos financieros ilícitos mundiales \[2\]. Las instituciones financieras dependen mayoritariamente de Sistemas de Monitoreo de Transacciones (TMS) basados en reglas deterministas y umbrales estáticos heredados de décadas anteriores. Este enfoque basado en reglas genera un ruido operativo masivo: entre el 95% y el 98% de las alertas generadas resultan ser falsos positivos \[1\]. Los sistemas basados en reglas aplican criterios rígidos tipo "si-entonces" que tienen una baja tasa de detección y generan alta tasa de falsos positivos, además de ser incapaces de detectar comportamientos desconocidos \[4\].

En el contexto específico de criptomonedas, la detección de lavado de dinero se complejiza por la naturaleza seudónima de Bitcoin, donde las direcciones no están directamente vinculadas a la identidad del usuario, y las transacciones operan independientemente de bancos o sistemas financieros gubernamentales \[4\]. Los criminales explotan esta estructura empleando técnicas como mixing o tumbling (que mezclan tokens de diferentes blockchains para romper la trazabilidad), transferencias peer-to-peer y flujos cross-border para ocultar fondos ilícitos a través de las etapas de placement, layering e integration \[3\], \[5\].

Ante la insuficiencia de los métodos lineales tradicionales y la suposición errónea de independencia entre instancias de transacciones, las Graph Neural Networks (GNNs) han emergido como un nuevo paradigma tecnológico. Las GNNs permiten modelar el sistema financiero como un grafo donde las transacciones son nodos y los flujos monetarios son aristas dirigidas, capturando la naturaleza inherentemente relacional del lavado de dinero \[6\]. El Elliptic Dataset, publicado por Elliptic en colaboración con el MIT-IBM Watson AI Lab en 2019, se ha convertido en el referente estándar (benchmark) para la investigación en detección AML en criptomonedas \[7\], \[8\]. Contiene 203,769 nodos (transacciones de Bitcoin) y 234,355 aristas dirigidas distribuidos en 49 pasos temporales, donde cada transacción se describe mediante 166 características divididas en features locales (intrínsecas de la transacción) y features agregadas (comportamiento estadístico de nodos vecinos a 1–2 hops) \[7\], \[8\].

Estudios recientes han demostrado la superioridad de diversas arquitecturas GNN sobre este dataset. Weber et al. (2019), quienes contribuyeron el Elliptic Dataset, demostraron que Graph Convolutional Networks (GCN) superaron a modelos basados en gradient boosting y regresión logística en la identificación de transacciones ilícitas \[7\]. He et al. (2026) propusieron un framework basado en TAGCN (Topology Adaptive Graph Convolutional Network) que alcanzó accuracy de 98.14%, recall de 86.22%, precision de 94.23%, F1-score de 90.05% y MCC de 0.8913 sobre el Elliptic Dataset, superando baselines como GCN, GraphSAGE, GAT, LR, SVM y KNN \[8\]. Otros trabajos sobre Elliptic incluyen Inspection-L (GNN auto-supervisada con GIN \+ Random Forest, precision 0.972, recall 0.721, F1 0.828) \[9\], el modelo LSTM+GCN de Alarab et al. con accuracy de 97.77% \[10\], MDGC-LSTM de Wan y Li (macro-precision 0.8380, macro-recall 0.8161, macro-F1 0.8266) \[11\], y EG-SAN de Cui y Zhang (precision 0.964, recall 0.827, F1 0.882) \[12\].

Sin embargo, la adopción de GNNs en entornos regulados de alto riesgo entra en conflicto directo con los requisitos de transparencia, auditabilidad y supervisión humana. Las arquitecturas de redes neuronales profundas sobre grafos son inherentemente modelos de "caja negra" cuya lógica interna resulta opaca incluso para expertos técnicos \[6\], \[13\]. Esta opacidad es inaceptable en el contexto AML, donde cada alerta debe ser justificable ante reguladores, auditores y potencialmente ante cortes judiciales. El campo de Explainable Artificial Intelligence (XAI) ha emergido como un área de investigación crítica orientada a hacer los modelos más transparentes y comprensibles para stakeholders humanos \[13\], \[14\].

En este contexto, diversos métodos XAI han sido aplicados a GNNs sobre el Elliptic Dataset con resultados divergentes. He et al. (2026) integraron SHAP con su modelo TAGCN, reportando que las atribuciones identificaron variables específicas (como las Features 147, 19 y 148\) como los predictores más influyentes de ilicitud. Asimismo, introdujeron la métrica SHAP Concentration para medir la interpretabilidad global: una concentración elevada indica parsimonia en el modelo, facilitando la auditoría forense al centrar la atención en pocos factores críticos; por el contrario, una concentración baja dificulta la validación humana al dispersar la importancia entre múltiples variables \[8\]. Por otro lado, Lawal et al. (2025) exploraron el uso de GNNExplainer sobre un modelo GCN, intentando alinear subgrafos explicativos con patrones de lavado como *fan-out* y *mixing* \[6\]. No obstante, sus resultados evidenciaron que, ante el desbalance extremo del dataset (clase ilícita \< 2%), el modelo GCN colapsó hacia la clase mayoritaria (AUC ≈ 0.5), invalidando la fiabilidad práctica de cualquier explicación generada bajo esas condiciones \[6\].

El problema central de esta investigación radica en la estabilidad de la explicabilidad de métodos XAI aplicados a GNNs en el contexto del Elliptic Dataset bajo condiciones de desbalance de datos. La estabilidad de la explicabilidad se define como la consistencia con la que un método XAI identifica las mismas características o subgrafos como relevantes para una predicción dada cuando se realizan perturbaciones mínimas en las entradas o se ejecuta el explicador múltiples veces con diferentes semillas aleatorias \[15\]. En un entorno de alto riesgo regulatorio, no basta con que un modelo GNN produzca una predicción precisa; el sistema debe proporcionar una explicación comprensible, consistente y verificable del porqué de esa predicción. Si un modelo ofrece una explicación hoy para una alerta específica y, ante una perturbación imperceptible o al re-ejecutar el explicador, produce una explicación completamente distinta, entonces la confianza del analista humano se ve comprometida. y la validación se torna imposible \[15\].

La fiabilidad de la interpretabilidad en Redes Neuronales de Grafos (GNNs) ha pasado de ser una evaluación subjetiva a un campo de análisis teórico riguroso. Agarwal et al. \[17\] establecieron un precedente crítico al cuantificar, mediante cotas superiores (upper bounds), las violaciones estructurales en tres propiedades fundamentales de los explicadores: la fidelidad (faithfulness), entendida como la capacidad del método para reflejar con exactitud el mecanismo de decisión interno del modelo; la estabilidad (stability), que mide la invariancia de la explicación ante pequeñas perturbaciones o ruido en el grafo de entrada; y la preservación de la equidad (fairness preservation), que garantiza que la explicación no omita ni distorsione sesgos algorítmicos presentes en el modelo original. Su estudio demostró que algoritmos ampliamente utilizados, como GNNExplainer, presentan dificultades para recuperar el ground truth (motivos estructurales conocidos que causan la predicción), lo que pone en duda su capacidad para identificar patrones causales reales en entornos complejos.

Esta falta de robustez se ve agravada por factores intrínsecos al entrenamiento de las GNNs. Investigaciones como GNNX-BENCH \[19\] señalan que la estocasticidad (aleatoriedad en la inicialización) y la naturaleza no convexa de las superficies de pérdida donde el optimizador puede quedar atrapado en múltiples soluciones locales provocan que las explicaciones varíen significativamente incluso ante cambios mínimos en la arquitectura. Para mitigar esta ambigüedad, el benchmark GraphFramEx \[18\] introduce protocolos de evaluación sistemática basados en métricas de Fidelity+ y Fidelity-, las cuales miden objetivamente cuánto se degrada la confianza del modelo al eliminar o aislar el subgrafo identificado como importante, proporcionando así un estándar para validar la veracidad de cualquier método de explicabilidad en grafos.

Este desafío de inestabilidad explicativa se exacerba dramáticamente por el desbalance de datos inherente al Elliptic Dataset. En el subconjunto etiquetado, las transacciones ilícitas representan aproximadamente el 9.8% del total (4,545 ilícitas vs. 42,019 lícitas) \[8\], y del total del dataset incluyendo nodos no etiquetados, las transacciones ilícitas representan menos del 2% \[6\]. Investigaciones recientes en datos tabulares demuestran que el desbalance de clases degrada severamente la estabilidad de métodos XAI post-hoc como LIME y SHAP, especialmente cuando la prevalencia de la clase minoritaria cae por debajo del 5%, con el índice de Jaccard y el acuerdo de ranking cayendo a menos del 10% de consistencia en escenarios extremos \[19\]. Sin embargo, este fenómeno no ha sido evaluado sistemáticamente en el dominio de grafos aplicado al Elliptic Dataset, donde la complejidad se amplifica exponencialmente debido a las dependencias estructurales entre nodos.

Además, las técnicas de balanceo destinadas a mitigar el desbalance, como GraphSMOTE (que genera nodos sintéticos de la clase minoritaria mediante interpolación en el espacio de embeddings) \[19\], class weighting y focal loss \[6\], introducen nuevas fuentes de incertidumbre. Si bien estos enfoques han demostrado mejorar métricas de clasificación, se desconoce cómo estas modificaciones impactan la estabilidad de las explicaciones subsecuentes. Existe el riesgo de que los explicadores identifiquen como relevantes conexiones artificiales que no reflejan patrones genuinos de lavado de dinero.

**4.2. Formulación del Problema**

**Problema Principal:**

¿Cómo evaluar y cuantificar la degradación de la estabilidad de los métodos de explicabilidad (XAI) en arquitecturas de Graph Neural Networks (GNNs) aplicadas a la detección de lavado de dinero sobre el Elliptic Dataset, bajo condiciones de desbalance de datos, y qué configuraciones de balanceo y selección de modelos optimizan la estabilidad explicativa dentro de este ecosistema transaccional?

**Problemas Específicos:**

* ¿Cómo cuantificar la degradación de la estabilidad de los métodos de explicabilidad (GNNExplainer, PGExplainer y SHAP) aplicados a GNNs a medida que aumenta la razón de desbalance de clases, utilizando métricas de consistencia como el índice de Jaccard, acuerdo de ranking de características y SHAP Concentration?

* ¿Qué arquitectura GNN (GCN, GraphSAGE, GAT o TAGCN) ofrece mayor estabilidad de sus explicaciones resultantes bajo condiciones de desbalance en el Elliptic Dataset, y cómo se manifiestan las diferencias estructurales de cada arquitectura en la consistencia de los subgrafos explicativos generados?

* ¿Cuál es el impacto de técnicas de balanceo de grafos (GraphSMOTE, funciones de pérdida ponderada, focal loss) sobre la estabilidad de las explicaciones: estabilizan las explicaciones al corregir el sesgo distributivo, o introducen distorsiones en la interpretación de conexiones sospechosas al incorporar nodos y aristas sintéticos?

* ¿Qué combinación de arquitectura GNN, método de explicabilidad y técnica de balanceo presenta el mejor desempeño simultáneo en estabilidad explicativa y fidelidad predictiva para el Elliptic Dataset, permitiendo la construcción de una matriz de recomendación basada en este benchmark que oriente el despliegue de sistemas GNN auditables?

**5\.  OBJETIVO GENERAL Y OBJETIVOS ESPECÍFICOS:**   
**Objetivo general**  
Evaluar la estabilidad de los métodos de explicabilidad en Redes Neuronales de Grafos bajo las condiciones de desbalance de datos inherentes a la detección de lavado de dinero en el Elliptic Dataset, proporcionando evidencia empírica y recomendaciones técnicas sobre las combinaciones óptimas de arquitectura, explicador y técnica de balanceo que garanticen explicaciones robustas y auditables.

**Objetivos específicos**

* Cuantificar la degradación de la estabilidad mediante métricas como el índice de Jaccard, acuerdo de ranking de Spearman y SHAP Concentration en métodos de explicabilidad agnósticos (GNNExplainer), generativos (PGExplainer) y basados en teoría de juegos cooperativa (SHAP) a medida que aumenta la razón de desbalance de clases (desde 1:1 hasta 1:100 utilizando submuestreo del Elliptic Dataset).

* Comparar la robustez de arquitecturas GNN (GCN, GraphSAGE, GAT y TAGCN) en términos de la estabilidad de sus explicaciones resultantes para determinar qué topología de red ofrece mejores garantías de interpretabilidad, extendiendo los hallazgos comparativos de He et al. (2026) que reportaron diferencias significativas en SHAP Concentration entre estas arquitecturas \[8\].

* Evaluar el impacto de técnicas de balanceo, específicamente analizando si el uso de estrategias como GraphSMOTE, funciones de pérdida ponderada y focal loss estabilizan las explicaciones o introducen distorsiones ("alucinaciones") en la interpretación de conexiones sospechosas, considerando los hallazgos de Lawal et al. (2025) sobre la inestabilidad del rendimiento bajo class weighting y focal loss \[6\].

* Construir una matriz de recomendación que mapee contextos operativos a configuraciones óptimas de la tríada Arquitectura, Explicador, Balanceo incluyendo intervalos de confianza sobre métricas de estabilidad y fidelidad esperadas.

  **6\.  ANTECEDENTES Y JUSTIFICACIÓN**


  


  **6.1 ANTECEDENTES**

El dataset Elliptic, con su estructura de grafo dirigido con 203,769 nodos y 234,355 aristas, 166 features por nodo, y 49 time steps, lo convierten en un benchmark realista para evaluar modelos de clasificación de nodos en grafos \[7\], \[8\]. El dataset exhibe desbalance inherente con ratio aproximado de 1:10 entre clase ilícita y lícita en el subconjunto etiquetado (4,545 ilícitas vs. 42,019 lícitas), y menos del 2% de transacciones ilícitas sobre el total incluyendo las 157,205 transacciones no etiquetadas \[6\], \[8\].

La aplicación de GNNs al Elliptic Dataset ha sido extensivamente documentada con diversas arquitecturas. Weber et al. aplicaron GCN como enfoque semi-supervisado, demostrando que captura información estructural del grafo que los métodos tabulares ignoran \[7\], \[13\]. Lin et al. realizaron un benchmark de múltiples métodos ML y deep learning sobre Elliptic, encontrando que Random Forest alcanzó accuracy del 95% como mejor baseline tabular \[21\]. Investigaciones recientes publicadas desde 2022, múltiples estudios han propuesto arquitecturas GNN más sofisticadas, como se muestra en la Tabla 1\.

| Autor (Año) | Método | Dataset | Métricas Principales |
| :---- | :---- | :---- | :---- |
| Weber et al. (2019) \[7\] | GCN | Elliptic | P 0.812 R 0.623 F1 0.705 |
| Lo et al. (2022) \[9\] | Self-supervised GNN | Elliptic | P 0.972, R 0.721, F1 0.828, AUC 0.916 |
| Alarab et al. (2023) \[10\] | LSTM+GCN | Elliptic | Accuracy 97.77% |
| Wan y Li (2024) \[11\] | MDGC-LSTM | Elliptic | P 0.8380,R 0.8161, F1 0.8266 |
| Cui y Zhang (2024) \[12\] | EG-SAN | Elliptic | P 0.964, R 0.827, F1 0.882 |
| Adloori et al. (2024) \[22\] | GAT-ResNet | Elliptic | P 0.956, MCC 0.7334 |
| He et al. (2026) \[8\] | TAGCN | Elliptic | Acc 98.14%, R 86.22%, F1 90.05%, MCC 0.8913 |

Tabla 1: Trabajos representativos sobre Elliptic con GNNs y modelos híbridos (resumen)

**6.1.1 Métricas de Desbalance en el Elliptic Dataset**

Estos hallazgos subrayan que métricas como PR-AUC, F1-score de clase minoritaria y MCC son indispensables para evaluar modelos bajo desbalance, y no la accuracy global. El desbalance extremo del Elliptic Dataset exige un cambio de paradigma en la evaluación de modelos, desplazando a la exactitud (accuracy) como métrica de referencia debido a su carácter engañoso en este contexto. La literatura reciente, encabezada por autores como He et al. \[8\], propone el uso de indicadores robustos como el F1-score de la clase ilícita, el Coeficiente de Correlación de Matthews (MCC) y la sensibilidad bajo umbrales estrictos de precisión. Bajo este marco, se ha observado que arquitecturas como TAGCN demuestran una capacidad superior de recuperación de nodos ilícitos y una menor tasa de falsos negativos en comparación con los GCN convencionales. No obstante, la estabilidad sigue siendo un reto; Lawal et al. \[6\] evidencian que incluso con técnicas de mitigación como class weighting o focal loss, los modelos GCN tienden al colapso, convergiendo hacia predicciones de la clase mayoritaria. Este fenómeno se complementa con el compromiso crítico (trade-off) identificado por Menezes (2025) \[23\], donde una alta exhaustividad en la detección de lavado de dinero suele derivar en una precisión marginal, generando un volumen masivo de falsos positivos.

**6.1.2 Explicabilidad (XAI) Aplicada al Elliptic Dataset**

La integración de métodos de Inteligencia Artificial Explicable (XAI) en arquitecturas GNN aplicadas al Elliptic Dataset es una frontera de investigación incipiente. He et al. \[8\] exploraron la interpretabilidad post-hoc mediante la integración de SHAP con la arquitectura TAGCN, introduciendo la métrica de "concentración de atribuciones". Sus hallazgos sugieren que TAGCN logra una focalización de características significativamente más precisa que modelos como GraphSAGE, GAT o GCN, lo que facilita la identificación de variables críticas en transacciones anómalas. No obstante, la utilidad de estas explicaciones depende de la salud del modelo; como señalan Lawal et al. \[6\], el uso de GNNExplainer para visualizar subgrafos ilícitos se ve limitado cuando el modelo base sufre de colapso predictivo, lo que pone en duda la fidelidad de la explicación resultante.

**6.1.3 Estabilidad de la Explicabilidad: Brecha en el Estado del Arte**

La estabilidad de explicaciones XAI es una propiedad formalmente estudiada en el contexto general de GNNs. Agarwal et al. proporcionan análisis teórico y empírico, mostrando que explicadores como GNNExplainer pueden violar stability y faithfulness \[16\]. GNNX-BENCH propone un framework de benchmarking que incluye estabilidad frente a factores variacionales como arquitectura GNN, ruido y estocasticidad \[19\]. GraphFramEx introduce protocolos de evaluación sistemática con métricas de fidelidad y estabilidad \[18\].

En datos tabulares, estudios utilizando datos clínicos han demostrado que la consistencia de LIME y SHAP (medida por índice de Jaccard y acuerdo de ranking) disminuye significativamente con el desbalance, alcanzando niveles inferiores al 10% con prevalencia de clase minoritaria \< 1% \[19\]. Fujiwara y Kano mostraron que la destilación de conocimiento combinada con remuestreo mejora la estabilidad explicativa medida por consistencia de valores SHAP \[24\].

En el contexto de estabilidad y robustez de GNNs en aplicaciones financieras, investigaciones sobre drift semántico y estructural en knowledge graphs financieros han cuantificado degradación significativa del rendimiento de GNNs a través del tiempo, con el F1-score cayendo de 0.747 a 0.455, mientras que arquitecturas más simples demostraron mayor resistencia \[23\]. Esto sugiere que las propiedades de estabilidad dependen críticamente de la arquitectura elegida.

La brecha central que esta investigación aborda es que ningún estudio existente ha evaluado sistemáticamente cómo el desbalance de clases inherente al Elliptic Dataset degrada la estabilidad de las explicaciones generadas por métodos XAI (GNNExplainer, PGExplainer, SHAP) aplicados a múltiples arquitecturas GNN (GCN, GraphSAGE, GAT, TAGCN), ni cómo las técnicas de mitigación de desbalance interactúan con dicha estabilidad.

**6.2 Marco Conceptual** 

**6.2.1 Grafos Dirigidos y Representación de Redes Transaccionales**

En el contexto de esta investigación, un grafo es una estructura compuesta por nodos (también llamados vértices) y aristas que conectan pares de nodos entre sí. Un grafo se convierte en dirigido cuando las aristas tienen una dirección definida: van desde un nodo de origen hacia un nodo de destino. Esta propiedad es esencial para representar redes transaccionales financieras, ya que un flujo de dinero parte de una dirección de envío y llega a una dirección de recepción, no al revés.\[1\], \[2\]

En el caso del Elliptic Dataset, cada transacción de Bitcoin se representa como un nodo del grafo, y cada arista dirigida indica que los fondos de una transacción alimentaron a otra. Además, cada nodo posee un conjunto de atributos numéricos (166 características) que describen las propiedades de esa transacción: montos, comisiones, marcas de tiempo, entre otras. Este tipo de grafo se denomina atribuido, porque los nodos no solo están conectados, sino que llevan información asociada que permite distinguirlos entre sí.\[7\]

La ventaja fundamental de esta representación es que captura la naturaleza relacional del lavado de dinero. A diferencia de los enfoques tradicionales que analizan cada transacción de forma aislada, el grafo permite ver cómo las transacciones se encadenan unas con otras, revelando patrones de flujo que serían invisibles al examinar una sola transacción.\[3\]

**6.2.2 Paradigma de Paso de Mensajes en Graph Neural Networks**

Las Graph Neural Networks (GNNs) son modelos de aprendizaje profundo diseñados específicamente para procesar datos estructurados en forma de grafos. Su principio de funcionamiento se conoce como paso de mensajes (message passing), un mecanismo mediante el cual cada nodo del grafo recoge, combina y utiliza información proveniente de sus nodos vecinos para construir una representación enriquecida de sí mismo.\[4\]

El proceso de paso de mensajes se ejecuta en tres pasos que se repiten capa tras capa dentro de la red neuronal:

* Generación de mensajes: Cada nodo vecino prepara un "mensaje" que contiene información relevante sobre sí mismo (su representación actual y sus atributos). Este mensaje se envía al nodo central que está siendo actualizado.

* Agregación: El nodo central recibe todos los mensajes de sus vecinos y los combina en una única representación resumida. Esta combinación se realiza mediante funciones que no dependen del orden en que se procesan los vecinos (por ejemplo, una suma o un promedio), garantizando que el resultado sea el mismo independientemente de cómo se enumeren los nodos vecinos. \[2\]

* Actualización: El nodo central combina la información agregada de sus vecinos con su propia representación previa, produciendo una nueva representación que ahora incorpora contexto de su entorno.

Cuando este proceso se repite a lo largo de varias capas, cada nodo termina incorporando información no solo de sus vecinos directos, sino también de vecinos de vecinos, y así sucesivamente. Por ejemplo, si la red tiene tres capas, cada nodo habrá incorporado información de todos los nodos que estén a tres "saltos" de distancia en el grafo. Esto resulta clave para la detección de lavado de dinero, porque permite identificar patrones complejos que involucran cadenas de múltiples transacciones conectadas.\[3\]

**6.2.3 Arquitecturas GNN Empleadas en esta Investigación**

No todas las GNNs implementan el paso de mensajes de la misma manera. Existen diferentes arquitecturas que varían en cómo generan, agregan y transforman la información del vecindario. Esta investigación evalúa cuatro familias representativas, cada una con un mecanismo de agregación distinto:

Graph Convolutional Networks (GCN). Es la arquitectura pionera y más ampliamente utilizada. GCN aplica un enfoque análogo a una convolución (como las usadas en procesamiento de imágenes), pero adaptada al dominio de grafos. En cada capa, cada nodo promedia las representaciones de sus vecinos directos, ponderadas por la inversa de sus grados de conexión, para producir su nueva representación. Esta normalización evita que los nodos con muchas conexiones dominen la señal. GCN es simple y eficiente, pero su alcance por capa se limita a vecinos inmediatos (1-hop). Weber et al. (2019) \[7\] demostraron su utilidad en el Elliptic Dataset al superar modelos tabulares como regresión logística. Sin embargo, Lawal et al. (2025) \[6\] evidenciaron que bajo desbalance extremo (clase ilícita \< 2%), GCN puede colapsar hacia la clase mayoritaria, generando predicciones casi aleatorias.\[7\]

GraphSAGE (Sample and Aggregate). Introduce dos innovaciones respecto a GCN. Primero, en lugar de procesar todos los vecinos de un nodo, GraphSAGE muestrea aleatoriamente un subconjunto fijo de vecinos, lo que permite escalar a grafos muy grandes. Segundo, emplea funciones de agregación más flexibles (media, máximo o incluso secuencias tipo LSTM) y concatena la representación agregada del vecindario con la propia del nodo, preservando explícitamente la identidad individual. Su naturaleza inductiva lo hace relevante para escenarios donde nuevas transacciones deben clasificarse sin reentrenar el modelo completo. No obstante, el muestreo aleatorio introduce variabilidad que puede afectar la reproducibilidad de las explicaciones posteriores.

Graph Attention Networks (GAT). A diferencia de GCN, que asigna el mismo peso a todos los vecinos del mismo grado, GAT incorpora un mecanismo de atención que permite a cada nodo asignar dinámicamente diferentes niveles de importancia a cada uno de sus vecinos. En la práctica, esto significa que la red aprende a "prestar más atención" a los vecinos que son más informativos para la tarea en cuestión. Además, GAT emplea atención multi-cabeza, ejecutando varios mecanismos de atención en paralelo y combinando sus resultados, lo que estabiliza el aprendizaje y captura diferentes tipos de relaciones simultáneamente. En el contexto de detección de lavado de dinero, esta ponderación diferencial es valiosa porque no todas las transacciones conectadas a una dirección sospechosa tienen la misma relevancia para determinar su ilicitud.\[6\]

Topology Adaptive Graph Convolutional Networks (TAGCN). Extiende las capacidades de GCN mediante filtros polinomiales adaptativos de orden superior. Mientras que GCN solo considera vecinos a 1 salto por capa, TAGCN puede capturar patrones a múltiples escalas topológicas simultáneamente dentro de una sola capa, utilizando un parámetro configurable K que determina el radio de recepción del filtro \[26\]. He et al. (2026) \[8\] validaron que TAGCN con K=3 alcanza el mejor rendimiento en el Elliptic Dataset con accuracy de 98.14%, F1-score de 90.05% y MCC de 0.8913, superando a GCN, GraphSAGE y GAT. Además, TAGCN demostró una mayor concentración de atribuciones SHAP, lo que significa que su proceso de decisión se fundamenta en menos características pero más relevantes, facilitando la auditoría humana.\[8\]

| Arquitectura | Cómo procesa los vecinos | Alcance por capa | Fortaleza principal | Debilidad principal |
| ----- | ----- | ----- | ----- | ----- |
| GCN | Promedio ponderado por grado | 1 salto | Simplicidad y fundamento teórico | Colapso bajo desbalance severo \[7\] |
| GraphSAGE | Muestreo aleatorio \+ agregación flexible | Configurable | Escalabilidad e inductividad | Variabilidad por muestreo aleatorio |
| GAT | Atención dinámica multi-cabeza | 1 salto con ponderación | Ponderación inteligente de vecinos | Mayor costo computacional |
| TAGCN | Filtros polinomiales de orden K | K saltos simultáneos | Mayor rendimiento y concentración SHAP \[2\] | Mayor complejidad |

**6.2.4 Inteligencia Artificial Explicable (XAI) Aplicada a GNNs**

La Inteligencia Artificial Explicable (XAI, por Explainable Artificial Intelligence) agrupa el conjunto de técnicas y enfoques orientados a hacer que los modelos de aprendizaje profundo, inherentemente opacos, sean comprensibles para los seres humanos. En el caso de las GNNs, que funcionan como "cajas negras" cuya lógica interna es difícil de interpretar incluso para expertos técnicos, XAI busca responder una pregunta fundamental: ¿por qué el modelo clasificó esta transacción como sospechosa?\[6\]

Los métodos XAI se clasifican según el momento en que se aplican. Los métodos ante-hoc (o intrínsecos) incorporan interpretabilidad directamente en la arquitectura del modelo durante su diseño, como los pesos de atención de GAT. Los métodos post-hoc, en cambio, se aplican después del entrenamiento para analizar un modelo ya existente sin modificar su estructura. Esta investigación se centra en métodos post-hoc, ya que estos permiten evaluar la explicabilidad de múltiples arquitecturas GNN de manera comparable.\[4\]

Las explicaciones producidas por métodos XAI para GNNs adoptan generalmente dos formas: (1) subgrafos explicativos, que señalan qué nodos y aristas del vecindario fueron más determinantes para la predicción; y (2) rankings de importancia de características, que indican cuáles de los 166 atributos de cada transacción tuvieron mayor influencia en la decisión del modelo. \[3\]

**6.2.5 Métodos de Explicabilidad: GNNExplainer, PGExplainer y SHAP**

GNNExplainer fue el primer método general diseñado para explicar predicciones de cualquier GNN \[24\]. Su idea central consiste en encontrar el subgrafo más pequeño posible y el subconjunto mínimo de características que, por sí solos, sean suficientes para que el modelo mantenga su predicción original con alta confianza. En la práctica, GNNExplainer funciona creando una "máscara" sobre las aristas y características del grafo de computación de un nodo, y luego ajusta progresivamente esa máscara para identificar los elementos verdaderamente relevantes. La limitación principal de este método es que realiza una optimización individual por cada predicción que se desea explicar, lo cual es computacionalmente costoso. Además, el proceso de optimización puede converger a diferentes soluciones dependiendo de la inicialización aleatoria, lo que constituye una fuente directa de inestabilidad en las explicaciones.\[24\]

PGExplainer (Parameterized Explainer) aborda esta limitación adoptando un enfoque generativo \[25\]. En lugar de optimizar una máscara por separado para cada instancia, PGExplainer entrena una red neuronal auxiliar que aprende a generar explicaciones para múltiples instancias de manera colectiva. Dada la representación de cualquier arista del grafo, esta red auxiliar predice si dicha arista pertenece o no al subgrafo explicativo. Las ventajas de este enfoque son tres: permite generar explicaciones para nodos no vistos durante el entrenamiento, es más eficiente al no requerir optimización individual, y facilita la identificación de patrones compartidos entre transacciones de la misma clase. Sin embargo, PGExplainer también está sujeto a inestabilidad derivada de la estocasticidad propia del entrenamiento de redes neuronales y de la sensibilidad a la distribución de los datos con los que se entrena la red generadora.\[25\]

SHAP (SHapley Additive exPlanations) se fundamenta en la teoría de juegos cooperativos \[13\], \[27\] específicamente en los valores de Shapley. La intuición detrás de SHAP es tratar cada característica del modelo como un "jugador" en un juego cooperativo donde el "premio" es la predicción del modelo. El valor de Shapley de cada característica cuantifica su contribución justa promedio: se evalúa cuánto cambia la predicción del modelo al incluir o excluir esa característica en todas las posibles combinaciones con las demás características. Esto garantiza tres propiedades deseables: que las contribuciones de todas las características sumen exactamente la predicción total, que características con contribuciones idénticas reciban valores iguales, y que características que contribuyen más siempre reciban valores mayores.\[27\]

He et al. (2026) \[8\] aplicaron SHAP sobre TAGCN en el Elliptic Dataset e introdujeron la métrica de SHAP Concentration, que mide el grado de concentración de las atribuciones en pocas características. Una concentración alta indica que el modelo basa sus decisiones en un número reducido de variables críticas, facilitando la auditoría forense porque el analista puede verificar directamente esos pocos factores. Una concentración baja, en cambio, dispersa la importancia entre muchas variables, dificultando la validación humana.

| Método | Enfoque | Tipo de explicación | Ventaja | Fuente de inestabilidad |
| :---: | :---: | :---: | :---: | :---: |
| GNNExplainer | Optimización por instancia | Subgrafos \+ características | Primer método general para GNNs \[24\] | Optimización no convexa con múltiples mínimos locales  |
| PGExplainer | Red generativa entrenada colectivamente | Subgrafos | Generalización y eficiencia \[25\] | Estocasticidad del entrenamiento de la red auxiliar |
| SHAP | Teoría de juegos cooperativos | Rankings de importancia | Propiedades axiomáticas deseables \[27\] | Complejidad computacional y aproximaciones de muestreo |

**6.2.6 Estabilidad de las Explicaciones**  
La estabilidad de una explicación se refiere a la consistencia con la que un método XAI produce las mismas explicaciones cuando se le presentan entradas similares o cuando se ejecuta múltiples veces bajo condiciones ligeramente diferentes.  En términos simples: si un explicador señala hoy que las características A, B y C fueron las razones por las que una transacción fue marcada como sospechosa, y mañana, ante la misma transacción con una mínima variación o simplemente cambiando la semilla aleatoria del algoritmo, señala las características D, E y F como responsables, entonces esa explicación es inestable y, por tanto, no confiable.\[7\]

En un entorno regulado de detección de lavado de dinero, la inestabilidad explicativa tiene consecuencias directas: si un analista de cumplimiento no puede obtener explicaciones reproducibles para justificar por qué una transacción fue flaggeada, la alerta pierde validez ante auditores y reguladores. No basta con que el modelo prediga correctamente; el sistema debe poder responder consistentemente por qué hizo esa predicción.\[7\]

Es importante distinguir la estabilidad de la fidelidad (faithfulness). La fidelidad mide si la explicación refleja verdaderamente el proceso de decisión interno del modelo, mientras que la estabilidad mide si esa explicación se reproduce consistentemente. Un método puede ser estable pero infiel (siempre produce la misma explicación, pero esa explicación no corresponde a lo que el modelo realmente computa) o fiel pero inestable (la explicación captura el proceso real, pero varía entre ejecuciones). Para un despliegue confiable en AML, se requieren ambas propiedades simultáneamente.\[8\]

**6.2.7 Métricas de Estabilidad Explicativa**

Esta investigación emplea tres métricas complementarias para cuantificar la estabilidad de las explicaciones:

Índice de Jaccard. Mide la similitud entre dos conjuntos de aristas o nodos identificados como explicativos en distintas ejecuciones del explicador. Se calcula como la proporción de elementos que aparecen en ambos conjuntos respecto al total de elementos en la unión de ambos. Un valor de 1.0 indica estabilidad perfecta (los subgrafos explicativos son idénticos en ambas ejecuciones), mientras que un valor cercano a 0 indica que las explicaciones cambiaron completamente. \[14\]

Acuerdo de ranking de Spearman \[14\]. Evalúa si el orden en que las características se clasifican por importancia se mantiene consistente entre ejecuciones. Si el explicador siempre identifica la Feature 147 como la más importante, la Feature 19 como segunda, y así sucesivamente, el acuerdo de ranking será alto. Esta métrica es particularmente relevante porque muchas decisiones regulatorias se basan en cuáles son las características más influyentes, no en sus valores exactos de atribución.\[14\]

SHAP Concentration propuesto por He et al. (2026) \[8\]. Métrica específica propuesta por He et al. (2026) \[8\] que cuantifica cuánto se concentran las atribuciones SHAP en pocas características versus dispersarse entre muchas. Se evalúa analizando la distribución acumulada de los valores SHAP absolutos: una distribución fuertemente sesgada hacia pocas características indica alta concentración y, por tanto, mayor facilidad de auditoría.\[14\]

**6.2.8 Fuentes de Inestabilidad en Explicaciones de GNNs**

La inestabilidad en las explicaciones generadas por métodos XAI aplicados a GNNs puede originarse en múltiples niveles del proceso analítico:

Estocasticidad algorítmica. GNNExplainer se basa en un proceso de optimización que, al ser no convexo, puede converger a diferentes soluciones dependiendo de la semilla aleatoria inicial.  Esto significa que ejecutar el mismo explicador dos veces sobre la misma predicción puede producir subgrafos explicativos distintos. PGExplainer hereda la variabilidad del entrenamiento de redes neuronales (como el orden aleatorio de los mini-batches y el uso de dropout). SHAP, en sus aproximaciones prácticas, utiliza muestreo de coaliciones que introduce varianza en las estimaciones de importancia.\[27\]

Sensibilidad a la arquitectura. La estabilidad de las explicaciones depende críticamente de la arquitectura GNN subyacente \[23\]. GNNX-BENCH \[16\] y los protocolos de evaluación de Agarwal et al. \[15\] ha mostrado que diferentes arquitecturas producen paisajes de optimización distintos para los explicadores, lo que implica que un explicador puede ser estable sobre una arquitectura y altamente inestable sobre otra. He et al. (2026) \[8\] evidenciaron diferencias significativas en SHAP Concentration entre GCN, GraphSAGE, GAT y TAGCN, sugiriendo que la elección de arquitectura no solo afecta el rendimiento predictivo, sino también la calidad y consistencia de las explicaciones resultantes. Adicionalmente, investigaciones sobre drift semántico y estructural en grafos financieros han demostrado degradación significativa del rendimiento de GNNs a través del tiempo, con el F1-score cayendo de 0.747 a 0.455 en grafos financieros dinámicos.\[23\]

Perturbaciones en los datos. Modificaciones mínimas en los atributos de los nodos o en la estructura del grafo pueden alterar significativamente las explicaciones producidas, especialmente cuando el modelo opera cerca de la frontera de decisión entre las clases lícita e ilícita. El benchmark GraphFramEx ha propuesto protocolos de evaluación que incluyen la inyección controlada de ruido para medir la robustez de los explicadores ante estas perturbaciones.\[16\] 

**6.2.9 Desbalance de Clases y su Interacción con la Explicabilidad**

El desbalance de clases es una condición en la que una de las categorías del problema de clasificación está representada por una cantidad significativamente menor de muestras que las demás. En el Elliptic Dataset, esta condición se manifiesta en dos niveles: en el subconjunto etiquetado, las transacciones ilícitas representan apenas el 9.8% (4,545 ilícitas frente a 42,019 lícitas), configurando un ratio aproximado de 1:10; y considerando el total del dataset incluyendo las 157,205 transacciones no etiquetadas, la proporción de transacciones ilícitas cae por debajo del 2%.  Este desbalance no es un defecto del dataset, sino un reflejo fiel de la realidad: las transacciones de lavado de dinero son intrínsecamente raras comparadas con la actividad financiera legítima.\[3\]\[2\]

El impacto del desbalance sobre los métodos de explicabilidad ha sido documentado en contextos no grafos. Estudios recientes utilizando datos clínicos (Rai et al., 2025\) \[18\] demostraron que la consistencia de LIME y SHAP, medida por el índice de Jaccard y el acuerdo de ranking, disminuye drásticamente cuando la prevalencia de la clase minoritaria desciende por debajo del 5%, llegando a niveles de consistencia inferiores al 10% en escenarios extremos. Fujiwara (2024) mostró que técnicas como la destilación de conocimiento combinada con remuestreo pueden mejorar la estabilidad explicativa en contextos desbalanceados. Sin embargo, ninguno de estos estudios se realizó en el dominio de grafos, donde la complejidad se amplifica exponencialmente porque los nodos no son independientes entre sí: la alteración de un nodo afecta las explicaciones de todos los nodos dentro de su radio de alcance en el grafo. \[18\]

**6.2.10 Técnicas de Balanceo de Datos en Grafos**

Para contrarrestar el desbalance de clases, se han desarrollado diversas estrategias que pueden clasificarse en dos grandes familias: las que actúan sobre los datos (remuestreo) y las que actúan sobre la función de pérdida durante el entrenamiento.

GraphSMOTE es una extensión al dominio de grafos de la técnica clásica SMOTE (Synthetic Minority Over-sampling Technique). En datos tabulares convencionales, SMOTE genera nuevas muestras sintéticas de la clase minoritaria interpolando entre muestras existentes cercanas. GraphSMOTE adapta esta idea al mundo de los grafos, generando nodos sintéticos de la clase minoritaria (transacciones ilícitas) mediante interpolación en el espacio de representaciones latentes (embeddings) de la GNN. Lo particularmente desafiante en grafos es que no basta con generar atributos para los nuevos nodos: también se deben crear conexiones (aristas) coherentes con la topología del grafo original. La preocupación fundamental para esta investigación es que los explicadores podrían señalar como relevantes conexiones artificiales creadas por GraphSMOTE que no corresponden a patrones reales de lavado de dinero, comprometiendo la validez de la explicación. \[19\]

Class Weighting (ponderación de clases) modifica la función de pérdida durante el entrenamiento para que los errores sobre la clase minoritaria sean penalizados con mayor severidad que los errores sobre la clase mayoritaria. De esta forma, el modelo recibe una señal de aprendizaje más fuerte cuando clasifica incorrectamente una transacción ilícita. No obstante, Lawal et al. (2025) encontraron que incluso con class weighting, los modelos GCN sobre Elliptic tienden al colapso predictivo hacia la clase mayoritaria, sugiriendo que la ponderación de clases por sí sola resulta insuficiente para arquitecturas espectrales simples bajo desbalance severo.\[3\]

Focal Loss adopta un enfoque más sofisticado al modular dinámicamente la contribución de cada muestra a la pérdida total. En lugar de asignar pesos fijos por clase, focal loss reduce automáticamente el peso de las muestras que el modelo ya clasifica correctamente con alta confianza (muestras "fáciles") y concentra el esfuerzo de aprendizaje en las muestras que resultan difíciles de clasificar. El parámetro gamma controla la intensidad de este enfoque: valores más altos de gamma hacen que el modelo ignore casi completamente las muestras fáciles y se concentre cada vez más en las difíciles.\[3\]

**6.2.11 Métricas de Rendimiento Predictivo bajo Desbalance**

Bajo condiciones de desbalance extremo, la métrica de accuracy global resulta engañosa. Un clasificador trivial que predijera todas las transacciones como lícitas alcanzaría más del 98% de accuracy en el Elliptic Dataset completo, a pesar de no detectar ni una sola transacción de lavado de dinero. Por esta razón, la literatura reciente ha consolidado el uso de métricas robustas al desbalance:

* F1-score de clase ilícita: Es la media armónica entre precision (proporción de transacciones marcadas como ilícitas que realmente lo son) y recall (proporción de transacciones verdaderamente ilícitas que el modelo logra detectar). Esta métrica penaliza equilibradamente tanto los falsos positivos como los falsos negativos de la clase minoritaria.\[2\]  
* Coeficiente de Correlación de Matthews (MCC): Considera simultáneamente los cuatro posibles resultados de una clasificación binaria (verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos), proporcionando una evaluación balanceada incluso bajo desbalance severo. Un MCC de \+1 indica clasificación perfecta, 0 indica rendimiento equivalente al azar, y \-1 indica predicción completamente invertida.\[2\]  
* PR-AUC (área bajo la curva Precision-Recall): Resulta más informativa que la curva ROC tradicional cuando la clase positiva es rara, ya que la curva Precision-Recall refleja directamente el rendimiento sobre la clase minoritaria sin verse inflada por la abundancia de verdaderos negativos.\[2\]

El criterio de éxito de esta investigación requiere mantener simultáneamente un F1 de clase ilícita ≥ 0.80 y un MCC ≥ 0.70, junto con un índice de Jaccard promedio \> 0.7 entre subgrafos explicativos bajo réplicas y perturbaciones. Este criterio dual refleja la premisa de que un sistema AML confiable debe ser tanto preciso en sus predicciones como consistente en sus explicaciones. \[7\]\[2\]

**6.2.12 Arquitectura, Explicador y Balanceo**

El concepto central que articula esta investigación es que la estabilidad de las explicaciones no depende de un solo factor, sino de la interacción conjunta entre tres dimensiones: la arquitectura GNN utilizada, el método XAI aplicado y la técnica de balanceo empleada.\[7\]

Estas tres dimensiones no son independientes entre sí. La elección de arquitectura determina la topología del espacio de representaciones que el explicador debe navegar: TAGCN produce representaciones más concentradas que GCN, lo que facilita la labor del explicador. La técnica de balanceo altera la distribución de los datos sobre la cual tanto el modelo como el explicador operan: GraphSMOTE introduce nodos y conexiones sintéticas que pueden ser señalados como relevantes por los explicadores, mientras que focal loss modifica la frontera de decisión al cambiar la forma en que el modelo aprende de las diferentes clases. El método XAI, a su vez, define las propiedades formales que la explicación puede satisfacer y las fuentes de variabilidad a las que está sujeto. \[19\]

Frameworks de benchmarking como GNNX-BENCH y GraphFramEx han señalado la necesidad de evaluar estas interacciones de manera integrada, pero ningún estudio previo ha realizado esta evaluación en el dominio específico del Elliptic Dataset. La brecha central que esta investigación aborda es precisamente la ausencia de evidencia empírica sobre cómo la tríada Arquitectura–Explicador–Balanceo interactúa para determinar la estabilidad de las explicaciones en la detección de lavado de dinero sobre Bitcoin, y qué configuraciones específicas optimizan simultáneamente la fidelidad predictiva y la consistencia explicativa. \[15\]

**6.3 Justificación Económica y Operativa**  
   
El costo de la ineficiencia actual en los sistemas AML es masivo. Las estimaciones indican que el cumplimiento de normativas AML cuesta a los bancos más de USD $60 mil millones anuales, con estimaciones totales superando USD $180 mil millones globalmente \[1\]. Las multas por incumplimiento han alcanzado niveles sin precedentes (por ejemplo, sanciones de alrededor de USD $3.1 mil millones a TD Bank por deficiencias sistémicas en AML) \[1\].

La implementación de GNNs robustas y explicables sobre el ecosistema Bitcoin promete reducir costos mediante mejores tasas de detección y explicaciones estables que reduzcan el tiempo de investigación por alerta.

**6.3.1 Justificación Tecnológica**  
   
 Las arquitecturas GNN de múltiples capas (como GAT con atención multi-head o TAGCN con filtros polinomiales) disminuyen la interpretabilidad intrínseca al agregar información propagada recursivamente desde vecindarios extensos \[8\]. TAGCN emplea filtros polinomiales aprendibles hasta K-hop, donde K=3 ha sido validado como óptimo en Elliptic \[8\]. La tecnología XAI actual para GNNs enfrenta un problema de estabilidad estocástica: GNNExplainer está basado en optimización no convexa \[26\], PGExplainer adopta un enfoque generativo más eficiente \[27\], y SHAP, basado en valores de Shapley, ha mostrado distribuciones de atribución diferenciadas según arquitectura \[20\].

Esta investigación validará qué combinaciones de arquitecturas GNN, métodos XAI y estrategias de balanceo ofrecen la estabilidad explicativa necesaria para despliegues confiables, utilizando Elliptic como benchmark y capitalizando los hallazgos de He et al. \[8\] y Lawal et al. \[6\].

**6.3.2 Justificación Legal y Social**  
   
Mejorar la efectividad de la detección de lavado de dinero mediante GNNs robustas y explicables impacta directamente en la capacidad de las sociedades para desarticular redes de crimen organizado, narcotráfico, trata de personas y financiamiento del terrorismo \[1\], \[3\], \[4\]. Esta investigación, al establecer fundamentos técnicos para GNNs explicables, estables y auditables en AML sobre Bitcoin, contribuye tanto a la efectividad operativa como a la transparencia necesaria para la confianza social en sistemas de IA financieros.

**6.4 Viabilidad y Alcance**

**Disponibilidad de Datos:**  
El estudio se centrará exclusivamente en el Elliptic Dataset como benchmark principal. Se construirán escenarios de desbalance 1:1, 1:10, 1:50 y 1:100 mediante submuestreo estratificado de la clase mayoritaria, manteniendo fijo el número de transacciones ilícitas \[6\], \[8\].

**Infraestructura Computacional:**  
GPU NVIDIA RTX 4060, Google Colab Pro, PyTorch Geometric.

**Alcance Incluido:**  
Evaluación sistemática de estabilidad de explicaciones (GNNExplainer, PGExplainer, SHAP), cuatro arquitecturas (GCN, GraphSAGE, GAT, TAGCN), impacto de GraphSMOTE, class weighting y focal loss, y desarrollo de un repositorio reproducible.

**Alcance Excluido:**  
Despliegue productivo, análisis jurídico específico por jurisdicción, métricas de latencia y estudios de usabilidad con oficiales de cumplimiento, uso de otros datasets (Elliptic2, AMLSim).

**6.5 Consideraciones Éticas**

La presente investigación reconoce las implicaciones éticas inherentes al desarrollo de modelos de Inteligencia Artificial para el monitoreo de transacciones financieras. En primer lugar, respecto al uso de datos, este estudio emplea exclusivamente el Elliptic Dataset, un conjunto de datos de dominio público anonimizado. Aunque la naturaleza de Bitcoin es pseudónima y el dataset expone características topológicas de transacciones reales, la investigación se rige por un principio de uso responsable; no se realizarán intentos de desanonimización de los nodos ni de rastreo de identidades de usuarios individuales. El propósito de los datos se restringe estrictamente a la validación de la estabilidad de los métodos de explicabilidad (XAI) aplicados a la detección de lavado de dinero a nivel sistémico.

En segundo lugar, el despliegue analítico de modelos de grafos (GNNs) en entornos Anti-Money Laundering (AML) conlleva impactos directos sobre los usuarios finales, por lo que las métricas de rendimiento deben interpretarse bajo un prisma ético. Un falso positivo en un sistema AML real implica el congelamiento injustificado de los fondos de un usuario legítimo, vulnerando su presunción de inocencia y generando exclusión financiera. Por el contrario, un falso negativo permite la integración exitosa de capitales ilícitos, perpetuando el daño social derivado de los delitos subyacentes. Precisamente por este delicado equilibrio ético, esta tesis justifica la necesidad imperativa de métodos de explicabilidad (XAI) robustos y estables; un analista humano debe poder auditar de manera confiable por qué una GNN marcó una transacción como anómala antes de tomar acciones punitivas.

Finalmente, los autores de este proyecto de grado declaran formalmente la ausencia de conflictos de interés, tanto de carácter financiero como institucional, que pudieran influir en el diseño, análisis, interpretación de los datos o en las conclusiones derivadas de este estudio. El marco metodológico propuesto persigue fines netamente académicos y de avance en el estado del arte científico.

**7.0 METODOLOGÍA**

**7.1 Hipótesis**

El desbalance de datos inherente al Elliptic Dataset (\~1:10 y superiores) degrada significativamente la estabilidad de los métodos XAI (GNNExplainer, PGExplainer, SHAP) aplicados a GNNs, haciendo que las explicaciones varíen sustancialmente ante perturbaciones mínimas o cambios de semilla. Se plantea que la implementación de técnicas de re-balanceo (GraphSMOTE \[19\] y focal loss \[6\],\[28\]) podría alterar la robustez explicativa de los modelos. Asimismo, se busca determinar si arquitecturas como TAGCN y GAT presentan variaciones significativas en su estabilidad en comparación con GCN, considerando las diferencias en *SHAP Concentration* reportadas por He et al. \[8\] y las evidencias de robustez en GNNs bajo *drift* \[23\]. 

Criterios de éxito: índice de Jaccard promedio \> 0.7 entre subgrafos explicativos bajo réplicas y perturbaciones, manteniendo F1 de clase ilícita ≥ 0.80 y MCC ≥ 0.70.

   
**7.2 Diseño Metodológico**

**Fase 1: Ingeniería de Datos y Configuración de Escenarios (Meses 1–3):**  
Esta fase se centra en la preparación del Elliptic Dataset para un entorno de grafos dinámicos. Incluye la normalización estadística de características y la transmutación de los datos tabulares a estructuras compatibles con PyTorch Geometric. Se implementará una estrategia de partición temporal (time-split) para preservar la causalidad de las transacciones. Para evaluar la resiliencia del modelo, se generará un gradiente de escenarios de desbalance, desde condiciones controladas hasta el desbalance extremo observado en entornos reales de lavado de dinero. 

**Fase 2: Benchmarking de Arquitecturas GNN y Mitigación de Desbalance (Meses 3–5):**  
El objetivo de esta fase es evaluar sistemáticamente diversas familias de arquitecturas de Redes Neuronales de Grafos (GNN). Se seleccionarán modelos representativos basados en distintos mecanismos de agregación:

* Convolucionales y de vecindad: (GCN, GraphSAGE).

* Atencionales: (GAT).

* De difusión y topología adaptativa: (TAGCN).

En lugar de fijar hiperparámetros estáticos, se llevará a cabo un proceso de optimización de arquitectura mediante búsqueda de hiperparámetros (capas, dimensiones ocultas y regularización). Paralelamente, se integrarán técnicas de mitigación de desbalance que actúen tanto en el espacio de características (remuestreo sintético como *GraphSMOTE*) como en la dinámica de aprendizaje (funciones de pérdida sensibles al costo y *focal loss*). La evaluación se realizará mediante métricas robustas de discriminación de clase minoritaria, priorizando el equilibrio entre precisión y sensibilidad.

**Fase 3: Evaluación de Interpretabilidad (XAI) y Pruebas de Estrés (Meses 5–8):**  
Se aplicará un protocolo de explicabilidad post-hoc para auditar las predicciones de los modelos con mejor desempeño, utilizando específicamente los métodos GNNExplainer, PGExplainer y SHAP. El proceso se enfocará en la extracción de subgrafos explicativos (vía optimización de máscaras y aprendizaje generativo) y rankings de importancia de características (vía teoría de juegos cooperativos).

La robustez de estas explicaciones se validará mediante:

* Análisis de Estabilidad: Evaluación de la consistencia de la explicación ante múltiples inicializaciones (semillas) y la introducción de ruido controlado en los atributos de los nodos, utilizando el Índice de Jaccard y el Acuerdo de Ranking de Spearman.  
*   
* Métricas de Fidelidad: Medición de la capacidad del subgrafo explicativo para mantener la predicción original, evaluando la SHAP Concentration para determinar la densidad de la información relevante.


**Fase** **4: Análisis de Correlación y Síntesis de Resultados (Meses 8–12):**

La fase final consistirá en un análisis estadístico para determinar cómo la severidad del desbalance afecta la degradación de la estabilidad de las explicaciones. Se realizará un Análisis Factorial para identificar las interacciones críticas entre la arquitectura del modelo, la técnica de balanceo y el método XAI empleado. Como producto final, se construirá una Matriz de Recomendación Técnica que oriente la selección de configuraciones (Arquitectura, Explicador, Balanceo) optimizadas para la detección de anomalías en grafos financieros.

**8\.0 SOSTENIBILIDAD DEL PROYECTO**

**8.1 Recursos**

Humanos: maestrandos y director.  
Tecnológicos: GPU propia y cloud gratuita/paga moderada, PyTorch, PyG, MLflow.  
Bibliográficos: acceso a IEEE, Springer, arXiv.

**8.2 Estimación de Costos**

El proyecto se apoya en recursos existentes, software libre y horas de dedicación académica, por lo que el presupuesto directo en USD se estima en 0 (costos en especie).

**9.0  FUENTES DE FINANCIACIÓN**

* Recursos propios (estudiantes y grupo de investigación)

* Soporte en especie de la Universidad Tecnológica de Pereira (laboratorios, biblioteca digital).

  **10.0 CRONOGRAMA**

| Mes | Actividad | Resultado Esperado |
| :---- | :---- | :---- |
| 1–3 | Preparación de datos y escenarios | Dataset preprocesado, splits y escenarios listos |
| 3–5 | Entrenamiento de GNNs y balanceo | Modelos entrenados con métricas base |
| 5–8 | Evaluación de estabilidad XAI | Métricas de estabilidad y fidelidad calculadas |
| 8–10 | Análisis estadístico e hipótesis | Modelos de relación desbalance–estabilidad validados |
| 11–12 | Redacción de tesis y artículo | Documento final y artículo sometido |

Tabla 2.0 Cronograma del Proyecto

**11.0 BIBLIOGRAFÍA**

\[1\] Haider, K., & Akhtar, N. (2024). Money Laundering and Terrorism Financing through Virtual Currencies: Critical Analysis of International and Pakistan's Response. Pakistan Journal of Criminal Justice, 4(1), 195-210. [https://doi.org/10.62585/pjcj.v4i1.93](https://doi.org/10.62585/pjcj.v4i1.93)

\[2\] Almeida, H., Pinto, P., & Fernández Vilas, A. (2025). From Placement to Integration: A Parametric Study of Cryptocurrency-Based Money Laundering Techniques. Risks, 13(12), 249\. [doi.org/10.3390/risks13120249](https://doi.org/10.3390/risks13120249)

\[3\] Unagar, E., & Borisaniya, B. (2025). Survey on Detection of Cryptocurrency Money Laundering and Its Explanation Using XAI. SN Computer Science, 6(3), 291\. doi: 10.1007/s42979-025-03828-2.

\[4\] Subashi, R. (2024). Cryptocurrencies and Money Laundering. Balkan Journal of, 10(1).  doi: 10.2478/bjir-2024-0005.

\[5\] Europol (2022). Cryptocurrencies: Tracing the Evolution of Criminal Finances. Luxembourg: Publications Office of the European Union.

\[6\] Lawal, O., Okolie, A., & Obunadike, C. (2025). An explainable graph neural network framework for anti–money laundering in cryptocurrency transactions using the Elliptic dataset. International Journal of Network Security & Its Applications, 17(6), 27-39. doi.org/10.5121/ijnsa.2025.17602

\[7\] Weber, M., Domeniconi, G., Chen, J., Weidele, D.K., Bellei, C., Robinson, T., & Leiserson, C.E. (2019). Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics. ArXiv, abs/1908.02591.  doi.org/10.48550/arXiv.1908.02591 

\[8\] He, X., Huang, J., Ma, K., He, H., & Li, M. (2026). An explainable graph neural network framework for illicit financial transaction detection: X. He et al. Applied Intelligence, 56(4), 102\. https://doi.org/10.1007/s10489-026-07138-9

\[9\] Lo, W. W., Kulatilleke, G. K., Sarhan, M., Layeghy, S., & Portmann, M. (2023). Inspection-L: self-supervised GNN node embeddings for money laundering detection in bitcoin. Applied Intelligence, 53(16), 19406-19417. https://doi.org/10.1007/s10489-023-04504-9

\[10\] Alarab, I., & Prakoonwit, S. (2023). Graph-based lstm for anti-money laundering: Experimenting temporal graph convolutional network with bitcoin data. Neural Processing Letters, 55(1), 689-707. doi.org/10.1007/s11063-022-10904-8

\[11\] Wan, F., & Li, P. (2024). A novel money laundering prediction model based on a dynamic graph convolutional neural network and long short-term memory. Symmetry, 16(3), 378\.  https://doi.org/10.3390/sym16030378

\[12\] Cui, B., & Zhang, J. (2024, October). EG-SAN: Evolving Graph Self-attention Networks for Detecting Illicit Activities in Cryptocurrency. In International Conference on Digital Forensics and Cyber Crime (pp. 24-38). Cham: Springer Nature Switzerland. doi.org/10.1007/978-3-031-89360-5\_2

\[13\] Adadi, A., & Berrada, M. (2018). Peeking inside the black-box: a survey on explainable artificial intelligence (XAI). IEEE access, 6, 52138-52160. doi.org/10.1109/ACCESS.2018.2870052

\[14\] Gawantka, F., Just, F., Savelyeva, M., Wappler, M., & Lässig, J. (2024). A novel metric for evaluating the stability of XAI explanations. Adv. Sci. Technol. Eng. Syst. J, 9, 133-142. https://dx.doi.org/10.25046/aj090113

\[15\] Agarwal, C., Zitnik, M., & Lakkaraju, H. (2022, May). Probing gnn explainers: A rigorous theoretical and empirical analysis of gnn explanation methods. In International conference on artificial intelligence and statistics (pp. 8969-8996). PMLR. https://doi.org/10.48550/arXiv.2011.04573

\[16\] Kosan, M., Verma, S.M., Armgaan, B., Pahwa, K., Singh, A.K., Medya, S., & Ranu, S. (2023). GNNX-BENCH: Unravelling the Utility of Perturbation-based GNN Explainers through In-depth Benchmarking. ArXiv, abs/2310.01794. doi.org/10.48550/arXiv.2310.01794

\[17\] Armgaan, B., Jain, E., Pandey, H., Chandran, M., & Ranu, S. (2025). GnnXemplar: Exemplars to Explanations \- Natural Language Rules for Global GNN Interpretability. ArXiv, abs/2509.18376. https://doi.org/10.48550/arXiv.2509.18376

\[18\] Rai, T., He, J., Kaur, J., Shen, Y., Mahmud, M., Brown, D. J., . . & Baldwin, D. (2025). Evaluating XAI techniques under class imbalance using CPRD data. Frontiers in Artificial Intelligence, 8, 1682919\. dx.doi.org/10.3389/frai.2025.1682919

\[19\] Zhao, T., Zhang, X., & Wang, S. (2021, March). Graphsmote: Imbalanced node classification on graphs with graph neural networks. In Proceedings of the 14th ACM international conference on web search and data mining (pp. 833-841). https://doi.org/10.1145/3437963.3441720

\[20\] Wang, Y., Zheng, Q., Li, X., Wang, L., & Lin, L. (2025). CoSemiGNN: Blockchain fraud detection with dynamic graph neural networks based on co-association of semi-supervised. Expert Systems with Applications, 129853\. https://doi.org/10.1016/j.eswa.2025.129853

\[21\] Lin, Z., Luo, Q., Wu, D., Shen, J., Li, L., Nong, X., & Qin, Z. (2026). Detecting illicit transactions in bitcoin: a wavelet-temporal graph transformer approach for anti-money laundering. Scientific Reports, 16(1), 1548\.  https://doi.org/10.1038/s41598-025-23901-3

\[22\] Adloori, H., Dasanapu, V., & Mergu, A. C. (2024). Graph network models to detect illicit transactions in block chain. arXiv preprint arXiv:2410.07150. https://doi.org/10.48550/arXiv.2410.07150

\[23\] Menezes, R. S., & Raimir Filho, H. (2025, November). Semantic and Structural Drift in Financial Knowledge Graphs: A Robustness Analysis of GNN-Based Fraud Detectors. In 2025 IEEE International Conference on Knowledge Graph (ICKG) (pp. 285-291). IEEE. https://doi.org/10.1109/ICKG66886.2025.00044

\[24\] Ying, Z., Bourgeois, D., You, J., Zitnik, M., & Leskovec, J. (2019). Gnnexplainer: Generating explanations for graph neural networks. Advances in neural information processing systems, 32\. doi.org/10.48550/arXiv.1903.03894

\[25\] Luo, D., Cheng, W., Xu, D., Yu, W., Zong, B., Chen, H., & Zhang, X. (2020). Parameterized explainer for graph neural network. Advances in neural information processing systems, 33, 19620-19631. doi.org/10.48550/arXiv.2011.04573

\[26\] Du, J., Zhang, S., Wu, G., Moura, J.M., & Kar, S. (2017). Topology adaptive graph convolutional networks. ArXiv, abs/1710.10370. https://doi.org/10.48550/arXiv.1710.10370

\[27\] Akkas, S., & Azad, A. (2024, May). Gnnshap: Scalable and accurate gnn explanation using shapley values. In Proceedings of the ACM Web Conference 2024 (pp. 827-838). https://doi.org/10.1145/3589334.3645599

\[28\] Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision (pp. 2980-2988). https://doi.org/10.48550/arXiv.1708.02002

**Firma del Proponente:**

Alejandro Gómez Huertas  
C.C. 1088347904

Juan Diego Garzón Ovalle  
C.C. 1116445008

**Firma del Director:**

Cristian Rosero Arias  
C.C. 1088343738

