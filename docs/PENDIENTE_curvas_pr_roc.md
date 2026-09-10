# Pendiente: curvas PR / ROC para la lámina de rendimiento

> **Estado (2026-09-10):** el código está listo y verificado, pero **falta correrlo en la máquina que
> tiene los checkpoints**. Este documento es el traspaso: qué falta, quién puede hacerlo, y qué queda
> por insertar cuando la figura exista.

## Por qué no se pudo correr en la máquina de Alejandro

`scripts/consolidacion/reeval_rocauc.py` hace inferencia sobre los pesos entrenados. Necesita dos
cosas que **no están en git por diseño** y que no existen en la máquina de Windows:

| Necesita | Dónde está | En git |
|---|---|---|
| `results_models_v3/*_best.pt` + `*_meta.json` (60 checkpoints) | máquina con GPU | no (`.gitignore`) |
| `data/` (Elliptic, ~300 MB) | se autodescarga con PyG | no |

Los agregados de esos checkpoints **sí** están versionados en `results_v3/reeval_metrics.csv`, y de
ahí salen los ROC-AUC y PR-AUC que ya reporta la Tabla 4.4 del manuscrito. Lo que falta son los
**puntos de curva**, que no se pueden reconstruir a partir de los agregados.

## Qué hay que correr (máquina con GPU, dos comandos)

```bash
# 1. Vuelca los puntos de curva de los 60 checkpoints (inferencia, sin reentrenar)
uv run python scripts/consolidacion/reeval_rocauc.py --dump-curves

# 2. Grafica (promedia sobre los 23 que superan el filtro de calidad)
uv run python scripts/make_fig_curves.py
```

Salidas:

- `results_v3/reeval_curves.csv` (unos 12.000 puntos: 60 checkpoints x 2 splits x 101 puntos)
- `presentacion_latex/fig/curvas_pr_roc.png`

Antes de lanzarlo conviene correr la comprobación, que no necesita GPU ni checkpoints:

```bash
uv run python scripts/consolidacion/check_curvas.py
```

## Bug corregido antes de la corrida

`curve_points_on_grid` remuestreaba las curvas con `np.interp`, que exige un eje x estrictamente
creciente. `precision_recall_curve` de sklearn emite **dos puntos en recall 0**, el centinela
`(0, precisión=1)` y un degenerado `(0, precisión=0)`, y con empates `np.interp` se queda con el
último. Resultado: la curva PR arrancaba en precisión **0** en lugar de 1, justo en el borde
izquierdo, que es donde mira el ojo. El área apenas cambiaba, pero la figura quedaba al revés en la
zona que importa.

Corregido con `_dedupe_x`, que colapsa los empates quedándose con la y máxima, en las dos curvas.
`check_curvas.py` lo cubre: verifica que la curva PR arranque en 1 y termine en la prevalencia, que
el área remuestreada coincida con el AP y el ROC-AUC reales, que la ROC sea monótona y que un split
sin positivos devuelva NaN en vez de reventar.

## Qué queda por insertar cuando exista la figura

Nada de esto se escribió todavía en el manuscrito ni en el guion, **a propósito**: describir una
figura que nadie ha visto es exactamente el tipo de afirmación sin respaldo que esta tesis se dedicó
a eliminar. Los tres huecos son estos.

### 1. Manuscrito, `tesis_latex/chapter_4/Chapter_4.tex`

Insertar después de la Tabla \ref{tab:elliptic-rocauc} (hoy en la línea 114), copiando la figura a
`tesis_latex/chapter_4/images_ch4/curvas_pr_roc.png`:

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.95\textwidth]{chapter_4/images_ch4/curvas_pr_roc.png}
    \caption{Curvas ROC y de precisión y exhaustividad sobre validación y test, promediadas
    punto a punto sobre las veintitrés configuraciones que superan el filtro de calidad. El
    contraste entre ambos paneles hace visible lo que las medias de la Tabla
    \ref{tab:elliptic-rocauc} resumen, a saber que el ROC-AUC se sostiene mientras la precisión
    se desploma para cualquier nivel de cobertura.}
    \label{fig:curvas}
\end{figure}
```

Y una frase de enlace en el párrafo de la línea 97, después de "que son ademas las que gobiernan el
trabajo real de triaje en una auditoría antilavado": remitir a la Figura \ref{fig:curvas} y describir
lo que **efectivamente se vea**, sin adelantarlo aquí.

### 2. Deck, `presentacion_latex/beamer_defensa_v3.tex`

La lámina 28 ("Rigor métrico: por qué PR-AUC y no ROC-AUC") es hoy solo tabla. Pasarla a dos
columnas, tabla a la izquierda y `\figcard{curvas_pr_roc.png}` a la derecha con su `\figcap`. La
figura ya se resuelve por `\graphicspath`, que incluye `fig/`.

### 3. Guion, `docs/DISCURSO_defensa_dos_voces.md`

La sección `### Pagina 28: Rigor metrico, PR-AUC y no ROC-AUC` sigue siendo válida tal como está: los
números que dice son los de la tabla y no cambian. Solo hay que añadir una o dos frases que apunten a
la figura, del tipo "miren los dos paneles", una vez se sepa qué se ve en ellos.

## Qué NO cambia

Los valores ya publicados salen de `results_v3/reeval_metrics.csv` y están verificados contra ese
CSV: ROC-AUC 0,884 y 0,653; PR-AUC 0,367 y 0,017; precisión en los primeros 50, 0,657 y 0,020; en los
primeros 100, 0,640 y 0,022. La figura ilustra esos números, no los sustituye, y `make_fig_curves.py`
toma las etiquetas de AUC de ese mismo CSV en lugar de integrarlas de la curva, precisamente para que
figura y texto no puedan desincronizarse.
