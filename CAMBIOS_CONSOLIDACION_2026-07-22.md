# Cambios de consolidación — 2026-07-22

Registro detallado de la consolidación de la tesis realizada el 2026-07-22 en la rama
`consolidacion-auditoria`. Este documento es el **punto de entrada** para ponerse al día:
dice dónde está la tesis, qué cambió y por qué, y cómo verificarlo.

Commit de referencia: **`b8afa68`** (en `origin` = `github.com/Gaova777/gnns_thesis`).

---

## Dónde está la tesis

- Fuente LaTeX: **`tesis_latex/`** — `main.tex` compila los 8 capítulos (`chapter_1/` … `chapter_8/`),
  la bibliografía (`bibliografia.bib`), las tablas (`tesis_latex/tables/`) y las figuras
  (`tesis_latex/chapter_*/images_ch*/`).
- PDF generado y commiteado: **`tesis_latex/main.pdf`** — **102 páginas**.
- Compilar desde cero (TinyTeX, biber):
  ```bash
  cd tesis_latex
  pdflatex -interaction=nonstopmode main.tex
  biber main
  pdflatex -interaction=nonstopmode main.tex
  pdflatex -interaction=nonstopmode main.tex
  ```

---

## Resumen ejecutivo del cambio central

Se corrigió un bug en la métrica de estabilidad (**fix R1** en
`src/stability/metrics.py::spearman_rank_agreement`, ya commiteado antes) que, con
`top_k_features=20`, descartaba las features con índice mayor al umbral y **mutilaba los
rankings de Spearman**. Al recomputar la estabilidad de Elliptic con la métrica corregida:

1. **Se invierte el ranking de arquitecturas.** La versión anterior decía "GraphSAGE es la
   más estable"; con la métrica corregida **GAT y GCN lideran** y GraphSAGE cae al medio.
2. **Se disuelve la "inversión por densidad".** Antes se afirmaba que el orden de estabilidad
   se invertía entre Elliptic (disperso) y el grafo sintético (denso). Corregida la métrica,
   Elliptic **concuerda** con el sintético (GAT/GCN arriba en ambos). No hay inversión.
3. **Hay dos artefactos de evaluación, no uno.** Primero uno de memoria (cálculo sobre el
   grafo completo con OOM silencioso, favorecía a GAT), y ahora el truncamiento de Spearman
   (favorecía a GraphSAGE). Cada uno bastaba para una conclusión comparativa falsa.

Este hallazgo **refuerza** la tesis: sustituye conclusiones apoyadas en artefactos por una
medición homogénea y potencia el argumento de dependencia del protocolo de evaluación (Kosan).

---

## Detalle de los cambios

### 1. Recompute de estabilidad Elliptic (código, GPU)
- Reejecutado `scripts/explain_matrix.py` sobre los 60 checkpoints con la métrica R1 corregida.
- **Cero OOM**: se usó un driver que lanza **un proceso fresco por configuración** (arch × escenario × balanceo),
  liberando toda la memoria entre configs. El OOM previo venía de correr los 60 en un solo proceso.
- Cobertura: 23 configs `quality_passed=True` con los 3 explicadores + 37 no-passed con GNNExplainer.
- Salida: `results_v3/xai-gnn-stability-B-v3.csv` (commiteado como provenance).

### 2. Ranking corregido (Spearman de GNNExplainer)

| Arquitectura | Corrida completa (60) | Corrida con filtro (23) | Sintético (denso) |
|---|---|---|---|
| GAT | **0,780** | 0,779 (n=7) | 0,964 |
| GCN | **0,778** | 0,833 (n=1) | 0,966 |
| GraphSAGE | 0,735 | 0,743 (n=11) | 0,884 |
| TAGCN | 0,580 | 0,641 (n=4) | 0,888 |

El fix no sube a todas las arquitecturas por igual (GraphSAGE +0,10, GCN +0,19, GAT +0,24,
TAGCN +0,28): el truncamiento castigaba más a las de importancia repartida. Sobre las 7 configs
donde GAT y GraphSAGE ambos pasan el filtro, el orden pasa de GraphSAGE 0,627 > GAT 0,535 (buggy)
a GAT 0,779 > GraphSAGE 0,739 (fixed).

### 3. Reescritura de prosa (narrativa de concordancia)
- **Cap 4** — sección retitulada a "dos correcciones"; se añade el segundo artefacto; números
  corregidos; `tab:ranking` (completa + filtro); claim robusto, compromiso exactitud-estabilidad
  y síntesis reescritos.
- **Cap 5** — la "inversión" pasa a "concordancia" (texto, figura, párrafo conceptual).
- **Cap 6** — discusión, implicación teórica, lección de Kosan ahora con **dos** artefactos, recomendación.
- **Cap 7** — conclusión del 2.º objetivo y matriz de recomendación (estabilidad → GAT/GCN, ya no GraphSAGE).

### 4. Tablas y figuras regeneradas
- Tablas: `elliptic_full.tex`, `elliptic_jaccard.tex`, `elliptic_stab_scenario.tex`,
  `elliptic_perf.tex`, y la tabla `tab:ranking` embebida en Cap 4.
- Figuras: `ranking_khop.png`, `estabilidad_escenario.png` (Cap 4), `contraste_regimen.png` (Cap 5).
- Regeneradas de forma determinista con `scripts/consolidacion/finalize_elliptic.py` desde el CSV.
- Nota: el Jaccard quedó igual que antes porque el bug R1 solo afectaba a Spearman, no a Jaccard.

### 5. Mejora de métricas (PR-AUC primaria + ROC-AUC / precision@k)
- Se elevó **PR-AUC** a métrica primaria (F1 en umbral es degenerado bajo desbalance extremo).
- Se añadió un bloque de contraste con **ROC-AUC** y **precision@k**, recomputados por inferencia
  sobre los 60 checkpoints (`scripts/consolidacion/reeval_rocauc.py` → `results_v3/reeval_metrics.csv`).
- Hallazgo (23 modelos que aprenden): val ROC-AUC 0,884 / PR-AUC 0,367 / p@50 0,657, colapsando a
  test ROC-AUC 0,653 / PR-AUC 0,017 / p@50 0,020. El ROC-AUC alto es **engañoso** bajo imbalance;
  se presenta como argumento de rigor métrico, no como "resultado mejor".

### 6. Estadística añadida
- **Cap 4** — GAT vs GraphSAGE **no significativo**: Wilcoxon p=0,375, t pareado p=0,380; IC95%
  solapados (GAT [0,705; 0,853], GraphSAGE [0,720; 0,766]) → indistinguibles.
- **Cap 5** — correlación de rangos entre regímenes Elliptic↔sintético: **−0,20 (buggy) → +0,80 (fixed)**,
  que cuantifica el paso de inversión a concordancia.

### 7. Auditoría de consistencia
- Cap 1/2/8 revisados: ya eran consistentes (las hipótesis refutadas —TAGCN no es la más estable,
  desbalance secundario, puente estabilidad-plausibilidad nulo— siguen válidas; el anexo remite a
  las tablas regeneradas). La narrativa corregida está contenida en Cap 4/5/6/7.

---

## Artefactos nuevos en el repo

- `tesis_latex/main.pdf` (102 páginas) y los `.tex`/tablas/figuras actualizados.
- `results_v3/xai-gnn-stability-B-v3.csv` — estabilidad recomputada (provenance).
- `results_v3/reeval_metrics.csv` — ROC-AUC / PR-AUC / precision@k por checkpoint y partición.
- `scripts/consolidacion/finalize_elliptic.py` — regenera tablas y figuras desde el CSV (sin GPU).
- `scripts/consolidacion/reeval_rocauc.py` — recomputa ROC-AUC/precision@k por inferencia (usa checkpoints).
- `RUNBOOK_CONSOLIDACION.md` — checklist de consolidación actualizado.

## Lo que NO se tocó

- **`phase1/` (eje sintético) intacto.** Sus números usan `top_k=None`, no los afecta el bug R1.
- El código del pipeline (`src/`) salvo el fix R1 ya commiteado previamente.

## Estado del repo

- Rama `consolidacion-auditoria`, commit `b8afa68`, pusheado a `origin` (Gaova777).
- Working tree limpio; PDF y fuentes al día; el manuscrito completo (41 archivos) está trackeado.
