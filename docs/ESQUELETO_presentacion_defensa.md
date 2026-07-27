# Esqueleto — Presentación de defensa (PowerPoint)

> Guion slide por slide para la sustentación. ~21 slides · ~22-25 min · defensa a dos voces
> (Alejandro Gómez · Juan Diego Garzón). Números = versión **corregida** (post fix R1). Cada slide
> indica título, puntos clave y qué figura/visual va. Pendiente de tu aprobación antes de generar el `.pptx`.
>
> **Reparto sugerido:** Bloque A (contexto + metodología, slides 1-12) una voz; Bloque B (resultados +
> cierre, slides 13-21) la otra. Ajustable.

---

### 1 · Portada
- Título de la tesis; autores; director Ph.D. Cristian Rosero; UTP — MISC; Pereira, 2026.
- Visual: logo UTP + imagen sobria de grafo/red.

### 2 · Contenido
- Problema → Pregunta y objetivos → Metodología (dos ejes) → Resultados → Conclusiones.
- Visual: agenda en 5 hitos.

### 3 · El problema: detectar lavado sin caja negra
- AML: 2-5% del PIB mundial; sistemas por reglas → 95-98% de falsos positivos.
- GNNs modelan las transacciones como grafo y superan a lo tabular, **pero son opacas**.
- En un entorno regulado, una alerta debe ser **justificable y reproducible**.
- Visual: `chapter_2/images_ch2/enfoques_moeny_laundering.png` (reglas vs grafos).

### 4 · Pregunta de investigación y objetivos
- Pregunta: ¿cómo se comporta la **estabilidad** de los métodos XAI sobre GNNs para AML bajo desbalance, y qué combinación arquitectura–explicador–balanceo la optimiza?
- 4 objetivos: (1) degradación por desbalance, (2) resiliencia por arquitectura, (3) impacto del balanceo, (4) matriz de recomendación.
- Visual: los 4 objetivos como iconos.

### 5 · La brecha en el estado del arte
- Se estudian predicción y explicabilidad por separado; la **estabilidad** de las explicaciones sobre grafos AML no se había evaluado sistemáticamente.
- Elliptic no tiene *ground-truth* de tipología → la plausibilidad no era medible en trabajos previos.
- Visual: tabla-mini del estado del arte (Weber, He 2026, Lawal, Agarwal).

### 6 · Marco conceptual en una lámina
- GNNs: paso de mensajes; 4 arquitecturas (GCN, GraphSAGE, GAT, TAGCN).
- XAI post-hoc: GNNExplainer, PGExplainer, GNNShap.
- Visual: esquema arquitecturas × explicadores.

### 7 · Tres propiedades que NO son lo mismo
- **Estabilidad** (reproducibilidad entre semillas) · **Plausibilidad** (¿apunta al patrón real?) · **Fidelidad** (¿refleja la decisión del modelo?).
- Tesis central: son **independientes** — un explicador puede tener una y no las otras.
- Visual: diagrama de 3 círculos NO solapados.

### 8 · Metodología: diseño de dos ejes
- **Eje Elliptic (real)** → validez externa; solo permite medir estabilidad.
- **Eje sintético (propio, con ground-truth)** → validez interna; permite medir plausibilidad y fidelidad.
- Visual: `chapter_4/images_ch4/metodologia.png`.

### 9 · Diseño factorial
- Arquitectura (4) × Explicador (3) × Balanceo (3) × Escenario de desbalance (5: 1:1, 1:10, nativo, 1:50, 1:100).
- Estabilidad = 5 réplicas estocásticas; robustez = 3 semillas de modelo × 3 grafos (eje sintético).
- Visual: `chapter_5/images_ch5/heatmap_factorial.png`.

### 10 · Eje 1 — Elliptic Bitcoin Dataset
- 203.769 nodos, 234.355 aristas, 166 features, 49 pasos; ilícitas 2,2% (1:9 etiquetado).
- Split temporal causal; **campo receptivo minúsculo** (mediana ~2 nodos).
- Visual: `chapter_4/images_ch4/dispersion_elliptic.png` + `eda_clases_timestep.png`.

### 11 · Eje 2 — grafo sintético con ground-truth
- Por qué: para medir plausibilidad/fidelidad se necesita saber **cuál** subgrafo es el patrón — Elliptic no lo da.
- 4 tipologías estándar (structuring, layering, fan-in, fan-out); aristas distractoras; firma atenuada; 3 realizaciones.
- Visual: `chapter_5/images_ch5/hallazgo_estrella.png`.

### 12 · Métricas y protocolo
- Estabilidad: **Spearman** de rankings de features (primaria); Jaccard (satura).
- Plausibilidad y fidelidad vs ground-truth; **PR-AUC** como métrica predictiva primaria.
- Estadística: Kruskal-Wallis, Wilcoxon, IC bootstrap.
- Visual: tabla de métricas.

### 13 · Contribución metodológica: DOS artefactos de evaluación
- (1) Fallo de memoria silencioso (cálculo sobre grafo completo) → favorecía a GAT.
- (2) Truncamiento en la métrica de Spearman → favorecía a GraphSAGE.
- Cada uno bastaba para una conclusión comparativa **falsa**. Lección: la estabilidad depende del **protocolo de medición** (Kosan).
- + Dos bugs reportables en el PGExplainer de PyG 2.7.
- Visual: antes/después del ranking.

### 14 · Resultado 1 — estabilidad por arquitectura
- Con la métrica corregida y 3 semillas de modelo: **dos grupos**, alto (GAT 0,782 · GCN 0,758) y bajo (GraphSAGE 0,735 · TAGCN 0,676). Significativo entre grupos, no dentro.
- GAT vs GraphSAGE **no significativo** (Wilcoxon p=0,375) → ninguna domina en absoluto.
- Visual: `chapter_4/images_ch4/ranking_khop.png`.

### 15 · Resultado 1b — concordancia entre regímenes de densidad
- La "inversión por densidad" era un **artefacto**: corregida, Elliptic (disperso) **concuerda** con el sintético (denso).
- Correlación de rangos entre regímenes: **−0,20 → +0,80**.
- Visual: `chapter_5/images_ch5/contraste_regimen.png`.

### 16 · Resultado 2 — disociación plausibilidad ↔ fidelidad
- **PGExplainer** recupera mejor el patrón (plausibilidad aristas 0,80 vs 0,50; Wilcoxon p≈2,6×10⁻³⁵)…
- …pero **colapsa en fidelidad** (0,11 vs 0,56 de GNNExplainer). El más "plausible" no es el más fiel.
- Visual: `chapter_5/images_ch5/disociacion.png` + `fidelidad_arq.png`.

### 17 · Resultado 3 — puente nulo y balanceo irrelevante
- Hipótesis central (estable ⇒ plausible): **refutada**, r = −0,01 (IC incluye 0). Un no-resultado, honesto.
- Balanceo prácticamente **irrelevante** para las tres dimensiones (η² < 0,02).
- Visual: `chapter_5/images_ch5/puente_nulo.png` + `balanceo_irrelevante.png`.

### 18 · Rendimiento predictivo y colapso validación→test
- Val PR-AUC media 0,367 (mejor 0,52) → **test 0,017**; F1 test ~0,01-0,02. Colapso por shift temporal.
- **ROC-AUC engañoso** bajo desbalance (0,88 val → 0,65 test): por eso PR-AUC + precision@k.
- La estabilidad se estudia sobre TP de validación (encuadre declarado).
- Visual: `chapter_4/images_ch4/estabilidad_escenario.png` o tabla ROC/PR.

### 19 · Matriz de recomendación
- Auditabilidad/estabilidad → **GAT/GCN**; recuperar el patrón (plausibilidad) → **PGExplainer**; fidelidad → **GNNExplainer**; estabilidad interna del método → **GNNShap**.
- Visual: matriz propósito → configuración.

### 20 · Contribuciones y limitaciones
- Contribuciones: caracterización de 3 dimensiones independientes; corrección de 2 artefactos; generador sintético con ground-truth; matriz de recomendación.
- Limitaciones (honestas): inferencia fuerte proviene del eje sintético; colapso en test; 1 semilla de modelo en Elliptic.
- Visual: dos columnas (aportes / límites).

### 21 · Conclusiones y trabajo futuro
- Estabilidad ≠ plausibilidad ≠ fidelidad; el mejor explicador depende del objetivo; la medición corregida da una historia coherente entre datos reales y sintéticos.
- Futuro: Elliptic2, AMLSim, GNNs temporales, GraphSMOTE.
- Cierre: agradecimientos + "Preguntas".

---

## Notas para preparar
- Tener a mano el **guion de defensa R2** (`docs/DEFENSA_R2_evidencia_sintetica.md`) para la pregunta "la evidencia fuerte viene del dataset que ustedes construyeron".
- Slides de respaldo (backup) sugeridas: tabla factorial completa, detalle estadístico (IC95%, Wilcoxon), el bug de Spearman explicado.
