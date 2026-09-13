# Plan de cierre: integración de la robustez a 3 semillas (GNNShap + PGExplainer + reeval)

Este documento es la guía EXACTA para el paso de cierre ("al final concluye y cambia la
tesis, la presentación y el discurso, y haz push") de la cadena de ejecuciones
`B -> C -> D`. Está pensado para ejecutarse de forma autónoma (tarea programada
`finalize-3seed-sweep`), posiblemente desde una sesión sin memoria de la conversación
original, así que es autocontenido.

- **Repo:** `/home/juan/Escritorio/gnn_thesis/gnns_thesis` (rama `main`, remoto `origin` = `Gaova777/gnns_thesis`).
- **Identidad de commit:** `gaova777 <gaova777@utp.edu.co>`.
- **Estilo de prosa:** español formal, términos técnicos en inglés, SIN rayas ni guiones largos (— –).
- **Pie de mensaje de commit:** terminar con `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## 0. Precondiciones (abortar si alguna falla)

1. Si existe `FINALIZE_DONE` en la raíz del repo: ya se cerró. No hacer nada (y deshabilitar la tarea programada).
2. Si NO existe `CHAIN_DONE` en la raíz: la cadena `B->C->D` sigue corriendo. No hacer nada; se reintentará luego.
3. `ps aux | grep -E 'run_gnnshap_seedsweep|run_phase_D' | grep -v grep` debe estar vacío. Si hay un driver vivo, abortar.

Principio rector: **todos los resultados nuevos son confirmatorios, no correctivos.** El
manuscrito ya afirma (Cap. 4, §"Replicación con tres semillas") que GNNShap se satura cerca
de la unidad sin discriminar arquitecturas y que PGExplainer degenera en Elliptic. La
replicación a 3 semillas CONFIRMA eso; no invierte nada. Si algún número contradijera la
narrativa vigente (no debería), **detenerse y dejar una nota en vez de forzar un push**.

**ASIMETRÍA DE SEMILLAS (crítico, no sobre-afirmar).** GCN y GraphSAGE tienen las tres semillas
(42/43/44). GAT y TAGCN **solo tienen la semilla 42**: sus pesos no cabían para reentrenar en la GPU
de 8 GB disponible (GAT usa ocho cabezas de atención sobre el grafo completo; el entrenamiento
reventó por OOM en el backward). Por tanto: (a) el IC a 3 semillas y la frase "tres semillas" valen
para GCN/GraphSAGE, NO para GAT/TAGCN; (b) `robustez_3semillas.py` ya marca las celdas de GAT/TAGCN
como `(s42)` y solo pone IC donde `n_seeds>=2`, así que respetar esa tabla tal cual; (c) el bloque
ROC-AUC del manuscrito se mantiene en la semilla 42 (ver 2a); (d) las curvas se generan con
`--seed 42` (por defecto) para que sus etiquetas cuadren con esas cifras. El resultado de arquitectura
(GNNExplainer, las 4 arqs a 3 semillas) NO se ve afectado.

## 1. Regenerar artefactos (determinista, sin GPU)

```bash
cd /home/juan/Escritorio/gnn_thesis/gnns_thesis
~/.local/bin/uv run python scripts/consolidacion/escenario_3semillas.py     # GNNExplainer por escenario (ya en tesis; refresca)
~/.local/bin/uv run python scripts/consolidacion/robustez_3semillas.py      # NUEVO: GNNShap+PGExplainer 3 semillas + reeval
~/.local/bin/uv run python scripts/make_fig_curves.py                       # NUEVO: curvas PR/ROC (semilla 42 por defecto: cuadra con las cifras del manuscrito)
cp presentacion_latex/fig/curvas_pr_roc.png tesis_latex/chapter_4/images_ch4/curvas_pr_roc.png
```

Driver D ya dejó `results_v3/reeval_curves.csv` (los puntos de curva); `make_fig_curves.py` genera
`presentacion_latex/fig/curvas_pr_roc.png`. El pipeline de curvas está validado de punta a punta.

`robustez_3semillas.py` imprime por stdout y escribe:
- `tesis_latex/tables/elliptic_robustez_3semillas.tex` (Spearman features: GNNExplainer vs GNNShap, con IC95).
- `tesis_latex/chapter_4/images_ch4/robustez_explicadores.png`.
- `results_seedsweep/robustez_3semillas_summary.csv` (todas las familias + reeval; **esta es la fuente de las cifras** de abajo).

Leer `results_seedsweep/robustez_3semillas_summary.csv` (la columna `n_seeds` dice cuántas semillas respaldan cada celda):
- Filas `explainer=GNNShap` -> media de Spearman por arquitectura (~0,9x, plana). GCN/GraphSAGE con `n_seeds=3`; GAT/TAGCN con `n_seeds=1` (solo seed 42).
- Filas `explainer=PGExplainer` -> media de Jaccard por arquitectura (confirma degeneración). Misma asimetría de semillas.
- Filas `explainer=reeval42/val` y `reeval42/test` -> `roc_auc`, `pr_auc_trap`, `prec_at_50` sobre la SEMILLA 42; deben reproducir las cifras ya publicadas (val ROC 0,884 / PR 0,367 / p@50 0,657; test 0,653 / 0,017 / 0,020).

## 2. Editar prosa (mínima y aditiva)

### 2a. `tesis_latex/chapter_4/Chapter_4.tex`

- **Bloque ROC-AUC** (buscar "ROC-AUC medio de 0,884" y la tabla `tab:elliptic-rocauc`, ~líneas 97-112):
  **NO cambiar las cifras.** Son de la semilla 42 y el reeval las reproduce (filas `reeval42/*` del
  summary). No relabelar como "tres semillas": GAT y TAGCN solo tienen seed 42, así que un promedio a
  tres semillas sería mixto y engañoso. Este bloque queda igual; su refuerzo visual es la figura de
  curvas (2d).
- **Replicación de GNNShap y PGExplainer** (buscar "se recalculó la estabilidad de GNNExplainer",
  ~línea 145): inmediatamente DESPUÉS de la sección por escenario (tras
  `\input{tables/elliptic_stab_scenario.tex}`, ~línea 223) añadir un párrafo corto + la tabla nueva:
  ```latex
  La replicación con tres semillas se extendió a GNNShap y PGExplainer para las dos arquitecturas
  entrenables en la GPU de 8 GB disponible, GCN y GraphSAGE. GNNShap reproduce su saturación en las
  tres semillas, con una correlación de Spearman de features próxima a la unidad
  (Tabla \ref{tab:elliptic-robustez-3s}), lo que ratifica que su altísima estabilidad convive con una
  incapacidad para ordenar arquitecturas. GAT y TAGCN, cuyos pesos con ocho cabezas de atención sobre
  el grafo completo no cabían para reentrenar en esa tarjeta, se reportan sobre la semilla original,
  en la que exhiben la misma saturación, de modo que la observación es transversal a las cuatro
  arquitecturas aunque la replicación formal cubra dos. PGExplainer mantiene su degeneración.
  [Rellenar las cifras concretas desde el summary.csv; respetar las marcas (s42).]
  \input{tables/elliptic_robustez_3semillas.tex}
  ```
  Ajustar las cifras entre corchetes con los valores reales del `summary.csv`, sin inventar IC para
  las celdas `(s42)`.

### 2b. `presentacion_latex/beamer_defensa_v3.tex`

Añadir, en la lámina de respaldo sobre estabilidad/replicación (o como nota al pie de la lámina de
ranking a 3 semillas), una frase: que la replicación a tres semillas cubre GNNShap y PGExplainer para
GCN y GraphSAGE (saturación y degeneración confirmadas), y que GAT/TAGCN se reportan en la semilla 42
por límite de memoria, con el mismo patrón. No alterar la narrativa ni las cifras de GNNExplainer.

### 2c. `docs/DISCURSO_defensa_dos_voces.md`

En la respuesta ensayada sobre replicación (buscar "semillas"), añadir: "La replicación a tres
semillas no se limitó a GNNExplainer; GNNShap y PGExplainer se replicaron en GCN y GraphSAGE (las
arquitecturas que caben en la GPU de 8 GB), confirmando saturación y degeneración; GAT y TAGCN
muestran el mismo patrón ya en la semilla original."

### 2d. Curvas PR/ROC (tesis, deck y discurso)

La figura `presentacion_latex/fig/curvas_pr_roc.png` ya existe (paso 1) y se copió a
`tesis_latex/chapter_4/images_ch4/`. Se genera con `--seed 42` (por defecto), así que sus etiquetas
de AUC cuadran exactamente con las cifras seed-42 del bloque ROC-AUC; el caption del doc de pendiente
sirve tal cual, sin cambios. Insertarla siguiendo los TRES huecos de
`docs/PENDIENTE_curvas_pr_roc.md` (sección "Qué queda por insertar cuando exista la figura"):

- **Tesis** (`tesis_latex/chapter_4/Chapter_4.tex`): insertar el bloque `\begin{figure}...
  \includegraphics{chapter_4/images_ch4/curvas_pr_roc.png}...\label{fig:curvas}\end{figure}` del doc
  de pendiente justo después de la tabla `tab:elliptic-rocauc` (~línea 114), con el caption del doc de
  pendiente sin cambios (es semilla 42), y una frase de enlace en el párrafo de la línea 97 que remita
  a la Figura \ref{fig:curvas}.
- **Deck** (`presentacion_latex/beamer_defensa_v3.tex`): lámina 28 ("Rigor métrico: PR-AUC y no
  ROC-AUC"), hoy solo tabla. Pasarla a dos columnas, tabla a la izquierda y `\figcard{curvas_pr_roc.png}`
  a la derecha con su `\figcap` (la figura se resuelve por `\graphicspath`, que incluye `fig/`).
- **Discurso** (`docs/DISCURSO_defensa_dos_voces.md`): en la sección "Pagina 28", añadir una o dos
  frases que apunten a los dos paneles de la figura. Los números no cambian.

## 3. Recompilar y verificar

```bash
cd tesis_latex && pdflatex -interaction=nonstopmode main.tex && biber main \
  && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
cd ../presentacion_latex && lualatex -interaction=nonstopmode beamer_defensa_v3.tex \
  && lualatex -interaction=nonstopmode beamer_defensa_v3.tex
```

Verificar: ambos PDF se generan, sin `! LaTeX Error`/`! Undefined control sequence` fatales en el
`.log`, la referencia `tab:elliptic-robustez-3s` resuelve (no `??`), y el número de páginas de
`main.pdf` es razonable (~114 ± unas pocas). Si la compilación falla, **no hacer push**: dejar una
nota en `FINALIZE_BLOCKED.md` con el error y abortar.

## 4. Commit y push

```bash
cd /home/juan/Escritorio/gnn_thesis/gnns_thesis
git add tesis_latex/chapter_4/Chapter_4.tex tesis_latex/tables/elliptic_robustez_3semillas.tex \
        tesis_latex/tables/elliptic_stab_scenario.tex tesis_latex/chapter_4/images_ch4/ \
        tesis_latex/main.pdf presentacion_latex/beamer_defensa_v3.tex presentacion_latex/beamer_defensa_v3.pdf \
        docs/DISCURSO_defensa_dos_voces.md presentacion_latex/fig/curvas_pr_roc.png \
        scripts/consolidacion/run_phase_D_pg_reeval.py scripts/consolidacion/robustez_3semillas.py \
        scripts/consolidacion/reeval_rocauc.py scripts/make_fig_curves.py
git add -f results_seedsweep/xai-gnn-stability-seedsweep.csv results_seedsweep/robustez_3semillas_summary.csv results_v3/reeval_metrics.csv results_v3/reeval_curves.csv
git -c user.name=gaova777 -c user.email=gaova777@utp.edu.co commit -m "$(cat <<'EOF'
resultados: robustez a 3 semillas de GNNShap y PGExplainer + reeval ROC-AUC/PR-AUC

Extiende la replicación de tres semillas a los tres explicadores (no solo
GNNExplainer) y recalcula ROC-AUC/PR-AUC/precision@k sobre los 180 modelos.
GNNShap confirma su saturación cercana a 1 sin discriminar arquitecturas y
PGExplainer su degeneración en Elliptic; la narrativa del manuscrito no cambia,
se refuerza con replicación. Actualiza tesis, presentación y discurso.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
git push origin main
```

## 5. Cierre

```bash
printf 'finalize completo\n' > /home/juan/Escritorio/gnn_thesis/gnns_thesis/FINALIZE_DONE
```

Luego deshabilitar la tarea programada: `list_scheduled_tasks` -> `update_scheduled_task`
con `taskId='finalize-3seed-sweep'` y `enabled=false`. Informar al usuario del push (hash y
resumen de cifras).
