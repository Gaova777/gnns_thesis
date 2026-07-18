# Verificación de citas (búsqueda web + lectura de PDFs) — 2026-07-16

Se buscaron y **se leyeron los PDFs reales** (no snippets) para verificar cada cita. Regla del estudiante:
deben existir, con DOI real (no inventado) y de tipo article. Resultado abajo.

---

## 1. "He et al. (2026)" con accuracy 98.14% / F1 90.05% / MCC 0.8913 (TAGCN + SHAP) → **NO EXISTE**

- **No hay ningún paper real** que reporte esos números con TAGCN+SHAP sobre Elliptic.
- El "overview" automático del buscador SÍ los mostraba, pero **al abrir el PDF del paper que citaba, el
  paper dice lo CONTRARIO** (ver §2). Los números trazan al propio anteproyecto del estudiante, no a una
  fuente. **Cautela demostrada:** nunca citar por el snippet del buscador; hay que leer el PDF.
- **ACCIÓN:** retirar del anteproyecto los números 98.14/90.05/0.8913, la atribución a "He et al. 2026", la
  "SHAP Concentration metric (He et al.)" y "TAGCN K=3 óptimo para Elliptic (He et al.)". Ninguno tiene
  fuente verificable.

## 2. Cita REAL que sí existe y que APOYA el hallazgo verdadero de la tesis

**Lawal, O., Okolie, A. & Obunadike, C. (2025).** *An Explainable Graph Neural Network Framework for
Anti-Money Laundering in Cryptocurrency Transactions Using the Elliptic Dataset.* **International Journal of
Network Security & Its Applications (IJNSA), 17(5/6), noviembre 2025.** **DOI: 10.5121/ijnsa.2025.17602.**
- Tipo: **journal article** (revista, con DOI real verificado en el PDF, pág. 27).
- Verificado leyendo el PDF: usa **GCN + GNNExplainer** (no TAGCN+SHAP); **enfatiza las LIMITACIONES**:
  *"emphasizes the limitations rather than the performance... instead of treating the model's high training
  accuracy as a success, we demonstrate how imbalance, structural sparsity, and label noise impede reliable
  learning... performance remains unstable"*; discute alineación con fan-out/mixing.
- **USO CORRECTO:** apoya el hallazgo real de esta tesis (accuracy engañosa + colapso VAL→TEST +
  inestabilidad bajo imbalance). Es una cita **más fuerte y honesta** que el "TAGCN 98.14%" inventado.
  (Nota: aparece también listada en IJCNC/aircconline; la versión con DOI verificado es la de IJNSA.)

```bibtex
@article{lawal2025explainable,
  author={Lawal, Oluwatosin and Okolie, Awele and Obunadike, Callistus},
  title={An Explainable Graph Neural Network Framework for Anti-Money Laundering in Cryptocurrency Transactions Using the Elliptic Dataset},
  journal={International Journal of Network Security \& Its Applications (IJNSA)},
  volume={17}, number={5/6}, year={2025}, doi={10.5121/ijnsa.2025.17602}}
```

## 3. Citas fundacionales reales (para el dataset y para TAGCN)

**Weber, M., Domeniconi, G., Chen, J., Weidele, D.K.I., Bellei, C., Robinson, T. & Leiserson, C.E. (2019).**
*Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics.*
**KDD '19 Workshop on Anomaly Detection in Finance. arXiv:1908.02591.**
- El paper **fundacional del Elliptic Dataset** (203.769 nodos, 234.355 edges, 166 features). Verificado.
- Tipo: **workshop paper** (revisado por pares, ampliamente citado). No es "journal article" estricto, pero
  es LA referencia canónica del dataset — estándar citarla.

**Du, J., Zhang, S., Wu, G., Moura, J.M.F. & Kar, S. (2017).** *Topology Adaptive Graph Convolutional
Networks.* **arXiv:1710.10370** (Carnegie Mellon).
- La fuente **real de TAGCN** (implementado como `TAGConv` en PyG). Verificado.
- Tipo: **arXiv preprint** (no journal article). Es la referencia canónica de TAGCN; sirve para justificar el
  uso/parámetro K **sin** afirmar "óptimo para Elliptic" (eso no lo dice).

```bibtex
@inproceedings{weber2019aml,
  author={Weber, Mark and Domeniconi, Giacomo and Chen, Jie and Weidele, Daniel Karl I. and Bellei, Claudio and Robinson, Tom and Leiserson, Charles E.},
  title={Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics},
  booktitle={KDD Workshop on Anomaly Detection in Finance}, year={2019}, note={arXiv:1908.02591}}

@article{du2017tagcn,
  author={Du, Jian and Zhang, Shanghang and Wu, Guanhang and Moura, Jos\'e M. F. and Kar, Soummya},
  title={Topology Adaptive Graph Convolutional Networks}, journal={arXiv preprint arXiv:1710.10370}, year={2017}}
```

## 4. arXiv:2602.23599 (warm-start priors) — PENDIENTE / no recomendada como "article"

Existe el ID y el título (*Normalisation and Initialisation Strategies for GNNs in Blockchain Anomaly
Detection*, feb 2026), pero: (a) es **preprint**, no journal article; (b) autores/números **no verificados
leyendo el PDF**; (c) NO es fuente de "warm-start priors" (trata init/normalización). **No apoyarse en ella**
para números; si se usa, leer el PDF y citarla solo por lo que dice (init/norm), no por warm-start.

---

## Resumen de acciones (integridad)
| Afirmación en el anteproyecto | Estado | Acción |
|---|---|---|
| "He et al. 2026", TAGCN 98.14%/90.05%/0.8913 | **fabricada / sin fuente** | RETIRAR |
| "SHAP Concentration (He et al.)" | sin fuente | RETIRAR atribución (definirla como métrica propia si se usa) |
| "TAGCN K=3 óptimo para Elliptic (He et al.)" | sin fuente | citar Du et al. 2017 por el default, sin "óptimo" |
| Elliptic dataset | real | **Weber et al. 2019** (arXiv:1908.02591) |
| TAGCN | real | **Du et al. 2017** (arXiv:1710.10370) |
| GNN+XAI+Elliptic con limitaciones/inestabilidad | real | **Lawal et al. 2025** (DOI 10.5121/ijnsa.2025.17602) — apoya el hallazgo de la tesis |

**Verificación final la hace el estudiante** abriendo cada DOI/arXiv, pero estas tres SÍ fueron leídas y
confirmadas aquí. El planteamiento no queda sin respaldo: se reconstruye sobre Weber (dataset), Du (TAGCN) y
Lawal (limitaciones/XAI), que dicen lo que la tesis realmente encontró.
