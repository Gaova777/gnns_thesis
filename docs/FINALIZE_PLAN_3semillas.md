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

## 1. Regenerar artefactos (determinista, sin GPU)

```bash
cd /home/juan/Escritorio/gnn_thesis/gnns_thesis
~/.local/bin/uv run python scripts/consolidacion/escenario_3semillas.py     # GNNExplainer por escenario (ya en tesis; refresca)
~/.local/bin/uv run python scripts/consolidacion/robustez_3semillas.py      # NUEVO: GNNShap+PGExplainer 3 semillas + reeval
~/.local/bin/uv run python scripts/make_fig_curves.py                       # NUEVO: curvas PR/ROC (promedia los quality_passed de las 3 semillas)
cp presentacion_latex/fig/curvas_pr_roc.png tesis_latex/chapter_4/images_ch4/curvas_pr_roc.png
```

Driver D ya dejó `results_v3/reeval_curves.csv` (los puntos de curva); `make_fig_curves.py` genera
`presentacion_latex/fig/curvas_pr_roc.png`. El pipeline de curvas está validado de punta a punta.

`robustez_3semillas.py` imprime por stdout y escribe:
- `tesis_latex/tables/elliptic_robustez_3semillas.tex` (Spearman features: GNNExplainer vs GNNShap, con IC95).
- `tesis_latex/chapter_4/images_ch4/robustez_explicadores.png`.
- `results_seedsweep/robustez_3semillas_summary.csv` (todas las familias + reeval; **esta es la fuente de las cifras** de abajo).

Leer `results_seedsweep/robustez_3semillas_summary.csv` y quedarse con:
- Filas `explainer=GNNShap` -> media de Spearman por arquitectura (se espera ~0,9x, plana).
- Filas `explainer=PGExplainer` -> media de Jaccard por arquitectura (confirma degeneración).
- Filas `explainer=reeval/val` y `reeval/test` -> `roc_auc`, `pr_auc_trap`, `prec_at_50` a 3 semillas (modelos que pasan el filtro).

## 2. Editar prosa (mínima y aditiva)

### 2a. `tesis_latex/chapter_4/Chapter_4.tex`

- **Bloque ROC-AUC** (buscar "ROC-AUC medio de 0,884" y la tabla `tab:elliptic-rocauc`, ~líneas 97-112):
  sustituir las seis cifras (val/test × ROC-AUC, PR-AUC, precisión@50) por las medias a 3 semillas
  del `summary.csv` (reeval/val, reeval/test), y añadir en la prosa y el caption que ahora están
  "promediadas sobre las tres semillas de modelo". Si una cifra redondea igual que la actual,
  mantenerla y solo añadir la cláusula de tres semillas. La narrativa (ROC alto y engañoso, PR-AUC
  bajo, colapso en test) NO cambia.
- **Replicación a 3 semillas** (buscar "se recalculó la estabilidad de GNNExplainer", ~línea 145):
  inmediatamente DESPUÉS de la sección por escenario (tras `\input{tables/elliptic_stab_scenario.tex}`,
  ~línea 223) añadir un párrafo corto + la tabla nueva:
  ```latex
  La replicación con tres semillas se extendió también a los otros dos explicadores, de modo que la
  caracterización de una sola semilla queda ahora confirmada por ciento ochenta modelos. GNNShap
  reproduce su saturación, con una correlación de Spearman de features próxima a la unidad y
  estadísticamente indistinguible entre las cuatro arquitecturas (Tabla \ref{tab:elliptic-robustez-3s}),
  lo que ratifica que su altísima estabilidad convive con una incapacidad para ordenar arquitecturas.
  PGExplainer mantiene su degeneración en las tres semillas, con una proporción de valores vacíos que
  no cede ante el cambio de inicialización, en coherencia con que la limitación es del grafo disperso
  y no del ajuste. [Rellenar con las cifras concretas del summary.csv.]
  \input{tables/elliptic_robustez_3semillas.tex}
  ```
  Ajustar las cifras entre corchetes con los valores reales del `summary.csv`.

### 2b. `presentacion_latex/beamer_defensa_v3.tex`

Añadir, en la lámina de respaldo sobre estabilidad/replicación (o como nota al pie de la lámina de
ranking a 3 semillas), una frase: que la replicación a tres semillas cubre **las tres familias** de
explicador, no solo GNNExplainer, y que GNNShap (saturado) y PGExplainer (degenerado) confirman su
comportamiento de una semilla. No alterar la narrativa ni las cifras de GNNExplainer.

### 2c. `docs/DISCURSO_defensa_dos_voces.md`

En la respuesta ensayada sobre replicación (buscar "semillas"), añadir: "La replicación a tres
semillas no se limitó a GNNExplainer; también se corrieron GNNShap y PGExplainer en las tres
semillas, y ambos confirman su patrón (saturación y degeneración respectivamente)."

### 2d. Curvas PR/ROC (tesis, deck y discurso)

La figura `presentacion_latex/fig/curvas_pr_roc.png` ya existe (paso 1) y se copió a
`tesis_latex/chapter_4/images_ch4/`. Insertarla siguiendo los TRES huecos documentados en
`docs/PENDIENTE_curvas_pr_roc.md` (sección "Qué queda por insertar cuando exista la figura"), con UN
ajuste: como ahora se promedia sobre las TRES semillas, el caption NO debe decir "veintitrés
configuraciones" sino "las configuraciones que superan el filtro de calidad, promediadas sobre las
tres semillas de modelo".

- **Tesis** (`tesis_latex/chapter_4/Chapter_4.tex`): insertar el bloque `\begin{figure}...
  \includegraphics{chapter_4/images_ch4/curvas_pr_roc.png}...\label{fig:curvas}\end{figure}` del doc
  de pendiente justo después de la tabla `tab:elliptic-rocauc` (~línea 114), con el caption ajustado a
  3 semillas, y una frase de enlace en el párrafo de la línea 97 que remita a la Figura \ref{fig:curvas}.
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
