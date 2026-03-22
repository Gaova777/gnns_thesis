### 📅 Gestión Administrativa y Fechas Clave

**Plazos Inmediatos y Futuros:**
- [X] **Averiguar fecha máxima de sustentación** (evitar entrar en periodo fuera de gracia). R// 10 de junio. 
- [ ] **Entregas parciales a Cristian:** No enviar todo junto, ir entregando por capítulos.

***

### 💻 Tareas Técnicas (Código y Análisis de Datos)

**Análisis Exploratorio (EDA) y Preprocesamiento:**
- [ ] **Revisión de Literatura:** Recolectar librerías y definir pasos a seguir basados en papers.
- [X] **PCA (Principal Component Analysis):** Ejecutar sobre los datasets.
- [X] **TSNE, TCA, CDF:** Aplicar estas técnicas de reducción/visualización adicionales (Sugerencia Juan Diego).
- [X] **Etiquetado:** Investigar en el dataset cómo están reportados los "no fraudes" (si son reales o simplemente no reportados). R// se etiquetan a partir de heuristicas con base en conocimiento previo que la empresa elliptic tenía de los que movían el dinero. Si era un hacker en la DB de elliptic la transacción era ilicita. Si era un Exchanger reconocido, la transacción es licit.  
- [X] **Escalabilidad:** Evaluar si el dataset de Bitcoin es escalable al bancario y si los tipos de fraude son comparables.  R//Los tipos de fraude son parcialmente comparables: hay solapamiento en técnicas de lavado como layering y smurfing, pero Elliptic cubre ilícitos cripto-específicos (ransomware, dark markets, Ponzi on-chain) sin equivalente bancario, mientras que la banca tiene fraudes de identidad (tarjetas, account takeover) que Bitcoin simplemente no registra. Más limitante aún, el dataset no distingue entre subtipos de ilícitos dentro de sus propios nodos, lo que reduce su comparabilidad directa. Lo verdaderamente transferible no es el dataset sino la metodología: modelar transacciones como grafo temporal y aplicar GCN para clasificación de nodos es un enfoque que bancos como HSBC ya exploran, aunque enfrentan un obstáculo institucional insalvable — cada banco ve solo su fragmento del grafo, mientras el lavador cruza múltiples instituciones deliberadamente; esa fragmentación, no la técnica, es la brecha real entre el ecosistema Elliptic y la banca tradicional.

**Detección de Patrones y Modelado:**
- [ ] **Análisis Visual/Manual:** Identificar si los fraudes son individuales o grupos organizados.
    - *Nota:* Resaltar qué features permitieron ver esto (o cuáles se usaron aunque no se viera nada).
- [ ] **Isolation Forest:** Probar este modelo (no supervisado) y verificar si los fraudes salen como anomalías.
- [ ] **Clustering (K-Means):** Para agrupar comportamientos.
- [ ] **Paper Baseline:** Definir si las tareas del paper base son supervisadas o no supervisadas.
- [ ] **Extrapolar Explicabilidad:** Buscar dar un aporte extra (ej: comportamiento de clusters, esquemas mixtos supervisados/no supervisados).

***

### 📝 Escritura y Documentación (Overleaf)

- [X] **Acceso Overleaf:** Confirmar que todos pueden editar en el link compartido y que funciona bien.
- [ ] **Creación de Figuras:**
    - Hacer figuras pensando en la tesis (no hacerlas dos veces).
    - Usarlas para explicar la estructura del dataset y arquitecturas.
    - Tenerlas listas *antes* de las reuniones para discutir sobre ellas.
- [ ] **Estructura del documento:** Seguir el índice de 7 capítulos definido arriba.

***

### 🔍 Preparación para la Siguiente Reunión

**Agenda propuesta:**
- [ ] Discutir objetivos claros.
- [ ] Definir tipo de dataset.
- [ ] Definir arquitectura final a usar.
- [ ] Establecer qué resultados se esperan.
- [ ] Definir la explicabilidad (esto se hace tras entender los datos).

***
