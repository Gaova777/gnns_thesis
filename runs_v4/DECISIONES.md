# Log de decisiones: corrida v4 (Elliptic)

Cualquier desviación del `RUNBOOK_V4.md` se anota aquí con fecha, qué se cambió y por qué.
El manuscrito (THE-25) y la validación con Cristian (THE-19) se escriben a partir de este archivo.

---

## 2026-09-29 · GAT: heads 8 → 4 y 2 capas para caber en 8 GB (OOM)

**Qué se cambió.** Solo para GAT, en `src/training/hyperopt.py`:
- prior `heads`: 8 → 4 (el valor por defecto de GAT en `src/models/gat.py`).
- espacio de búsqueda `heads`: {4, 8} → {2, 4}.
- espacio de búsqueda `num_layers`: tope en 2 (las otras tres siguen con {2, 3}).
- `hidden_dim` NO se toca: sigue en {64, 128, 148}, igual que las otras tres arquitecturas.

Y para todas las arquitecturas, sin cambiar el cómputo: se libera la VRAM al terminar cada trial de
Optuna (`hyperopt.py`) y antes del reentrenamiento final (`scripts/train_matrix.py`).

**Por qué.** GAT con `hidden=148` y `heads=8` sobre el grafo completo de Elliptic (203.769 nodos,
234.355 aristas) hace OOM en la RTX 4060 (7,62 GiB usables). El cuello es el tensor de mensajes por
arista (234k × ancho efectivo 148×8 = 1184), que pide ~9,5 GiB. No es contención: OOM también con la
GPU vacía (sonda 29-sep). fp16/AMP no ayuda porque `autocast` no reduce ese tensor de *message
passing*. Barrido empírico de memoria (pico de forward+backward, grafo completo):

| hidden | heads | ancho efectivo | pico |
|---|---|---|---|
| 148 | 8 | 1184 | **OOM** |
| 148 | 4 | 592 | 4,50 GiB |
| 128 | 4 | 512 | 3,92 GiB |
| 64 | 8 | 512 | 3,96 GiB |
| 128 | 2 | 256 | 2,04 GiB |
| 64 | 4 | 256 | 2,05 GiB |

El barrido se hizo con 2 capas. Al relanzar quedó a la vista que con **3 capas**, heads=4 y
hidden=148 el pico pasa de ~7 GiB y también hace OOM, y que ese trial tumbaba la configuración
entera. Por eso el tope de 2 capas: con `heads` ∈ {2,4}, `hidden` ∈ {64,128,148} y 2 capas ninguna
combinación hace OOM (peor caso 4,5 GiB). Dos capas equivalen a un campo receptivo de 2 saltos, el
estándar en Elliptic. Verificado: la corrida final de GAT (semillas 42, 43 y 44, 36 modelos) no tuvo
ningún OOM.

**Encaje con el runbook.** La tabla de alertas (≥2 OOM) autoriza «restringir solo para esa
arquitectura, declararlo aquí y avisar a Alejandro». Se restringe `heads` en vez de `hidden` porque
es más suave (conserva `hidden=148`, el mismo tope que las otras tres) y porque `heads=4` es el
default estándar de GAT, no un recorte arbitrario. No se viola C2: C2 prohíbe *subir* presupuesto
para GAT; esto lo *baja*.

**Efecto esperado en los resultados.** El corrimiento en la estabilidad de GAT se estima en
entre ±0,02 y ±0,03 (del orden del ruido entre semillas de la v3: 0,766/0,789/0,790). La conclusión de fondo
(partición en dos grupos) solo cambiaría si GAT cruzara de grupo (caída > 0,05), lo que la v3, con un
GAT aún más pequeño que quedó en el grupo alto, hace poco probable. Las otras tres arquitecturas y
el eje sintético no se ven afectados. **Se declara como límite en el manuscrito y se valida con
Cristian (THE-19).**

**Alternativas descartadas.** GAT completo (8 heads/148) en GPU de 16 GB: no hay acceso a esa
máquina. GAT en CPU a tamaño completo: ~1 día por semilla, inviable para el 15-oct. Muestreo de
vecinos: cambiaría el régimen de entrenamiento respecto a las otras tres (confound peor).

**Nota (02-oct) · consistencia con el eje sintético.** El eje sintético (`phase1/`) no pasa por
`hyperopt.py` y ya construía GAT con `heads=4` (`phase1/v4_common.py` y los scripts de la v3). El
recorte de Elliptic no afecta a esa corrida y deja los dos ejes con el mismo número de cabezas.
La sonda del barrido de memoria quedó en `scripts/v4/probe_gat_memoria.py`.

---

## 2026-10-02 · Sensibilidad sin compuerta y estabilidad entre semillas principal

**Qué se hizo.** Se explicaron los 87 modelos que no pasan la compuerta (`explain_matrix
--include-gated --resume`, un proceso por arquitectura, escenario y semilla, igual que
`run_all_v4.sh`), para que la sección SENSIBILIDAD de `analysis_v4_summary.txt` deje de coincidir
con la principal, como pide el runbook. 48/48 procesos sin fallos, en 3 h 46 min. Logs en
`runs_v4/SENS_*.log`.

**Cambio de procedimiento.** `cross_seed_stability.py` no filtra por compuerta por defecto. Por eso,
después de explicar los modelos sin compuerta, la versión principal se regenera con `--gate-only`
(`results_v4/elliptic_v4_cross_seed.csv`, idéntica a la del commit `decaa23`: 153 pares, diferencia
0,0) y la de todos los modelos se guarda aparte en `results_v4/elliptic_v4_cross_seed_sinfiltro.csv`.
La sección PRINCIPAL del análisis no cambia (verificado contra el commit `decaa23`).
