# Eje sintético v4 — resultados

La corrida completa son 144 configuraciones: 4 arquitecturas × 4 escenarios × 3 balanceos × 3 semillas de modelo, con 3 explicadores cada una. Terminó el 1-oct-2026 en la máquina de Alejandro, solo en CPU, con `bash scripts/v4/run_phase1_v4.sh --resume`.

Según la reunión con Cristian del 28-sep, el sintético va a **anexo**: sirve como «sensor lateral» para la discusión de plausibilidad y no es la contribución principal.

| Archivo | Qué es |
|---|---|
| `summary_v4.csv` | Resumen por arquitectura × explicador: medias reales frente al azar, ganancia y p-valor |
| `results_v4.csv` | Una fila por configuración × explicador (432 filas) |
| `results_v4_pernode.csv` | Lo mismo, nodo a nodo |
| `alignment.json` | Prueba de alineamiento modelo↔tipología (margen ≥ 0,10). Las 4 la pasan; GCN, al borde con 0,138 |
| `shortcuts_v4.txt` | AUC de una sola variable en el grafo v4: entre 0,50 y 0,62, sin atajos |
| `*_legacy.*` | Las mismas pruebas sobre el grafo legado, para comparar |

No se suben los grafos (`synthetic_*_g42.pt`): `phase1/synthetic_aml_generator.py --v4` los reconstruye con la semilla 42.

## Antes de usar estas cifras

1. **La plausibilidad de aristas no es comparable entre arquitecturas.**
   - TAGCN ve 6 saltos: en la mediana, su subgrafo tiene 9.311 nodos de 11.115 y 47.311 aristas.
   - GCN, GAT y GraphSAGE ven 2 saltos: unas 36 nodos y 78 aristas.
   - Por eso el azar vale ≈ 0,001 en TAGCN y ≈ 0,47 en las demás. La «ganancia» grande de TAGCN (+0,35 / +0,43) sale de esa línea base, **no** de que explique mejor.
2. **Los escenarios son los del diseño del 23-sep.** 1:50 y 1:100 se obtienen quitando ilícitas, al contrario que en Elliptic v4, donde ningún escenario quita ilícitas. Hay que declararlo en el anexo.
3. **PGExplainer falló en 24 de 432 filas** (`status = pg_failed`), todas en 1:50 y 1:100.

## Lo que se ve

- **ShapleyFeatures** es el más estable: Spearman de 0,95 a 0,97, sin efecto del desbalance. Pero su plausibilidad de variables está en el azar (0,11 a 0,16, con el azar en 0,15). Estable no es lo mismo que correcto.
- **GNNExplainer** pierde estabilidad con el desbalance: 0,92 en natural y 1:1, 0,86 en 1:100. Es el más plausible en variables (0,24 a 0,45) y el más fiel: su fid+ supera al azar por 0,12 a 0,32.
- **Plausibilidad de aristas:** en las arquitecturas de 2 saltos, los explicadores apenas superan al azar (+0,01 a +0,03). PGExplainer queda por debajo.
