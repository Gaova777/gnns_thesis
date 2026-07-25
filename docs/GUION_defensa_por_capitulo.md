# Guion de defensa con referencia a capítulo y sección

> Para cada slide de la sustentación y para cada afirmación clave, dónde se encuentra el enunciado
> en el manuscrito (`tesis_latex/main.pdf`). Títulos de sección tomados literalmente del manuscrito.
> Úsalo para responder al jurado "¿en qué parte de la tesis está esto?" señalando el capítulo y la sección.

---

## Parte A · Mapa slide → capítulo y sección

> **Ojo con la numeración:** el número de slide de esta tabla es el del guion (21 slides de
> contenido), **no** el número de página del PDF de la presentación (26 páginas, con 5 separadores
> de sección intercalados). La equivalencia slide ↔ página está en
> [`DISCURSO_defensa_dos_voces.md`](DISCURSO_defensa_dos_voces.md), sección *Mapa del slide del guion
> a la pagina del PDF*.

| Slide | Enunciado / qué se dice | Dónde está en la tesis |
|---|---|---|
| **1** Portada | Título y autores | Portada · Cap. 1 |
| **2** Contenido | Recorrido de la tesis | Cap. 1 · §1.8 *Estructura de la Tesis* |
| **3** El problema | Lavado, falsos positivos, GNN opacas | Cap. 1 · §1.1 *Planteamiento y Contexto del Problema* · Cap. 2 · §2.3 *Sistemas Tradicionales de Monitoreo* y §2.7 *Del Monitoreo Basado en Reglas hacia Enfoques Estructurales* |
| **4** Pregunta y objetivos | Pregunta + 4 objetivos + hipótesis | Cap. 1 · §1.4 *Formulación del Problema* · §1.5 *Objetivos* (§1.5.1 General, §1.5.2 Específicos, §1.5.3 *Hipótesis de Trabajo*) |
| **5** La brecha | Vacío en el estado del arte | Cap. 1 · §1.2 *Revisión de Literatura y Estado del Arte* y §1.3 *La Arista de Investigación* · Cap. 3 · §3.6.4 *El Problema de la Estabilidad Explicativa* |
| **6** Marco GNN/XAI | 4 arquitecturas + 3 explicadores | Cap. 3 · §3.2 *GNN: Fundamentos Conceptuales*, §3.3 *Arquitecturas GNN Fundamentales*, §3.4 *TAGCN*, §3.6.2 *Métodos XAI para GNNs* |
| **7** Tres propiedades | Estabilidad ≠ plausibilidad ≠ fidelidad | Cap. 3 · §3.8.4 *Métricas de Estabilidad y Fidelidad Explicativa* · Cap. 6 · §6.2 *…como Tres Dimensiones Independientes* |
| **8** Dos ejes | Elliptic real + sintético | Cap. 5 · §5.1 *Por qué se Construye un Grafo Sintético* · Cap. 6 · §6.1 *Lectura Conjunta de los Dos Ejes* |
| **9** Diseño factorial | Matriz + protocolo estadístico | Cap. 4 · §4.2 *Pipeline Experimental y Espacio Factorial* · Cap. 5 · §5.4 *Análisis Estadístico de Robustez* · Cap. 8 · §8.1 *Espacio de Búsqueda de Hiperparámetros…* |
| **10** Elliptic | Dataset, split, dispersión | Cap. 4 · §4.1 *Preprocesamiento y Análisis Exploratorio* (§4.1.1 *Composición…*, §4.1.2 *Dispersión de la Topología*) |
| **11** Sintético | Generador con ground-truth | Cap. 5 · §5.1 *Por qué se Construye…* y §5.2 *Construcción del Grafo Sintético y sus Tipologías* |
| **12** Métricas | Spearman, plaus., fidel., PR-AUC | Cap. 3 · §3.8 *Formalización de Métricas de Evaluación* (§3.8.2 *…en Escenarios de Desbalance*, §3.8.4 *Estabilidad y Fidelidad*, §3.8.5 *Nociones de Inferencia Estadística*) |
| **13** Dos artefactos | Contribución metodológica | Cap. 4 · §4.5 *De un Artefacto de Cómputo a un Artefacto de Medida: dos Correcciones Metodológicas* · Cap. 6 · §6.4 *Contribuciones Metodológicas* |
| **14** Ranking (GAT/GCN) | Estabilidad por arquitectura | Cap. 4 · §4.5 (tabla `tab:ranking`) · Cap. 8 · §8.2 *Resultados Completos por Configuración sobre el Elliptic Dataset* |
| **15** Concordancia (−0,20→+0,80) | Elliptic concuerda con sintético | Cap. 5 · §5.3 *Resultados de la Matriz Factorial* · Cap. 6 · §6.1 y §6.2 |
| **16** Disociación | PGExplainer plausible pero no fiel | Cap. 5 · §5.6 *La Disociación entre Plausibilidad y Fidelidad* (plausibilidad en §5.3) |
| **17** Puente nulo + balanceo | Hipótesis refutada; balanceo irrelevante | Cap. 5 · §5.5 *La Ausencia de un Puente entre Estabilidad y Plausibilidad* · Cap. 6 · §6.3 *El Papel Secundario del Balanceo y de la Arquitectura* |
| **18** Colapso val→test | ROC engañoso, PR-AUC primaria | Cap. 4 · §4.4 *Rendimiento Predictivo y el Colapso de Validación a Test* · Cap. 3 · §3.8.2 |
| **19** Matriz de recomendación | Objetivo → configuración | Cap. 7 · §7.1 *Respuestas a los Objetivos de Investigación* (O4) · Cap. 6 · §6.6 *Implicaciones para la Práctica de Auditoría…* |
| **20** Contribuciones/límites | Aportes y limitaciones | Cap. 6 · §6.4 *Contribuciones* y §6.8 *Limitaciones* · Cap. 7 · §7.2 *Aportes Principales* y §7.3 *Limitaciones* |
| **21** Conclusiones | Cierre + futuro | Cap. 7 · §7.1, §7.4 *Perspectivas Futuras*, §7.5 *Reflexión Final* |

---

## Parte B · Preguntas probables del jurado → dónde responderlas

| Pregunta que puede hacer el jurado | Respuesta corta | Sección de respaldo |
|---|---|---|
| *¿Por qué el clasificador colapsa en test?* | Shift temporal del dataset; por eso la estabilidad se mide sobre validación. | Cap. 4 · **§4.4** *…Colapso de Validación a Test* |
| *La evidencia estadística fuerte viene del dataset que ustedes construyeron.* | Ese eje es el único donde plausibilidad/fidelidad son medibles (exigen ground-truth); es robusto (3 grafos × 3 semillas) y refutó nuestra propia hipótesis. | Cap. 5 · **§5.1** (por qué) y **§5.4** (robustez) · Cap. 7 · **§7.3** (límites) · guion `docs/DEFENSA_R2_evidencia_sintetica.md` |
| *¿Por qué no usaron AMLSim?* | Bloqueado (dependencia MASON v20 inexistente) y sin ground-truth por arista; queda como trabajo futuro. | Cap. 5 · **§5.1** · Cap. 7 · **§7.4** · nota `NOTA_AUDITOR_amlsim_decision.md` |
| *¿No es circular medir plausibilidad contra un ground-truth propio?* | No: el ground-truth es ciego al explicador y el diseño refutó nuestra hipótesis del puente. | Cap. 5 · **§5.5** (puente nulo) y **§5.2** (construcción) |
| *¿Cuál es el aporte metodológico?* | Dos artefactos de evaluación corregidos + dos bugs de PGExplainer + tres dimensiones independientes. | Cap. 4 · **§4.5** · Cap. 6 · **§6.4** |
| *¿Por qué ahora lideran GAT/GCN y no GraphSAGE?* | El truncamiento del Spearman favorecía a GraphSAGE; corregido, el orden se invierte. | Cap. 4 · **§4.5** (tabla `tab:ranking`) |
| *En la tabla filtrada GCN tiene n=1. ¿Cómo sostienen que encabeza?* | El ranking que se afirma es el de la corrida completa (60), con soporte comparable entre arquitecturas; la columna filtrada es control de robustez (no invierte el orden), no una estimación. El eje sintético, con replicación 3×3, concuerda. | Cap. 4 · **§4.5** (tabla `tab:ranking`) · Cap. 5 · **§5.3** · Cap. 6 · **§6.8** (límites) · respuesta ensayada en `DISCURSO_defensa_dos_voces.md` |
| *¿El balanceo realmente no importa?* | η² < 0,02 en las tres dimensiones; es un factor secundario. | Cap. 5 · **§5.4** · Cap. 6 · **§6.3** |
| *¿Por qué PGExplainer no da estabilidad en Elliptic?* | Degenera en grafos dispersos (mode collapse); se reporta como hallazgo, no como dato. | Cap. 4 · **§4.6** *Degeneración de PGExplainer en un Grafo Disperso* |
| *¿Se cumplieron los objetivos?* | Sí; se responden uno a uno, incluidas las hipótesis refutadas. | Cap. 7 · **§7.1** *Respuestas a los Objetivos de Investigación* |
| *¿Y las implicaciones éticas/regulatorias?* | Falsos positivos, debido proceso, auditabilidad. | Cap. 6 · **§6.6** y **§6.7** *Implicaciones Más Amplias del Estudio* |

---

## Parte C · Índice rápido de la tesis (referencia)

- **Cap. 1 · Introducción** — 1.1 Planteamiento · 1.2 Revisión de Literatura · 1.3 La Arista de Investigación · 1.4 Formulación · 1.5 Objetivos (+Hipótesis) · 1.6 Justificación · 1.7 Alcance · 1.8 Estructura.
- **Cap. 2 · Marco Contextual** — 2.1 Lavado global · 2.2 Marco Regulatorio · 2.3 Sistemas por reglas · 2.4 Criptomonedas · 2.5 Tipologías · 2.6 Desafíos · 2.7 Hacia enfoques estructurales.
- **Cap. 3 · Fundamentos de IA** — 3.1 ML→DL · 3.2 GNN fundamentos · 3.3 Arquitecturas · 3.4 TAGCN · 3.5 Propiedades del aprendizaje sobre grafos · 3.6 Explicabilidad en GNNs · 3.7 Mitigación del desbalance · 3.8 Formalización de métricas · 3.9 Preparación.
- **Cap. 4 · Diseño Experimental y Resultados (Elliptic)** — 4.1 Preprocesamiento y AE · 4.2 Pipeline y espacio factorial · 4.3 Escenarios de desbalance · 4.4 Colapso val→test · 4.5 Dos correcciones metodológicas · 4.6 Degeneración de PGExplainer · 4.7 Síntesis.
- **Cap. 5 · El Eje Sintético** — 5.1 Por qué el grafo sintético · 5.2 Construcción y tipologías · 5.3 Matriz factorial · 5.4 Análisis estadístico de robustez · 5.5 Ausencia de puente · 5.6 Disociación plausibilidad/fidelidad · 5.7 Síntesis.
- **Cap. 6 · Discusión** — 6.1 Lectura conjunta · 6.2 Tres dimensiones independientes · 6.3 Papel secundario del balanceo y la arquitectura · 6.4 Contribuciones metodológicas · 6.5 Relación con el estado del arte · 6.6 Implicaciones para auditoría · 6.7 Implicaciones más amplias · 6.8 Limitaciones.
- **Cap. 7 · Conclusiones y Perspectivas** — 7.1 Respuestas a los objetivos · 7.2 Aportes principales · 7.3 Limitaciones · 7.4 Perspectivas futuras · 7.5 Reflexión final.
- **Cap. 8 · Anexos** — 8.1 Hiperparámetros · 8.2 Resultados completos Elliptic · 8.3 Matriz factorial sintética · 8.4 Matriz robusta · 8.5 Reproducibilidad.

> Nota: los números de sección corresponden al orden del manuscrito; confirma el número impreso contra el índice del `main.pdf` recompilado (los títulos son exactos).
