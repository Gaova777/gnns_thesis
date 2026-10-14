# Auditoría de la bibliografía del manuscrito v4

Fecha: 14 de octubre de 2026.

## 1. Existencia

`scripts/verificar_referencias.py` genera `bibliografia.bib` consultando Crossref por DOI. Una
referencia se acepta solo si:

- el DOI existe;
- es un artículo de revista (`journal-article`, que incluye las revisiones);
- no tiene retractación registrada.

Los datos de autores, título, revista, año, volumen, número y páginas se copian de Crossref; no
se transcribe nada a mano.

Las únicas excepciones están en `excepciones.bib`. Son los artículos originales de los métodos y
del conjunto de datos usados (GCN, GraphSAGE, GAT, TAGCN, GNNExplainer, PGExplainer, SHAP,
Integrated Gradients, la prueba de Adebayo y Elliptic). Su título se verificó en arXiv o en las
actas de la conferencia. Resultado en `verificacion.csv`.

## 2. Coherencia de cada cita

1. Se extrajo cada uso de `\cite` con la oración en que aparece (`auditoria/citas.json`).
2. Se obtuvo el resumen de cada artículo desde OpenAlex o Semantic Scholar
   (`auditoria/resumenes.json`).
3. Se contrastó cada oración con el resumen (`auditoria/revision.txt`). Criterio: la oración no
   puede atribuir al trabajo nada que no esté en su resumen o, si no hay resumen, en su título.

### Sin resumen disponible

- **Spearman, Friedman, Wilcoxon, Jaccard y LeCun.** Son trabajos fundacionales, citados solo
  para la definición del método; el título lo confirma.
- **Motie et al. y Wang et al.** Se citan solo por lo que dice su título.

### Corregido en esta auditoría

Se ajustaron unas 20 oraciones que decían más de lo que respalda el resumen o citaban el trabajo
equivocado:

- **Chen et al. (2018).** Se usaba para las tres etapas del lavado y para la reducción de falsas
  alarmas; ahora solo se cita por las técnicas que revisa.
- **Hilal et al. y Labanca et al.** La distinción entre modelos supervisados y no supervisados es
  de Labanca et al.
- **Almeida et al. y Subashi.** Se les atribuían técnicas que no mencionan; ahora se describe lo
  que dicen.
- **Kute et al.** Se usaba como prueba de la adopción del aprendizaje profundo; su resumen dice
  que se ha usado poco por falta de interpretabilidad.
- **Agarwal et al., Longa et al., Nauta et al. y Bodria et al.** Se reformularon según lo que
  declaran.
- **Rai et al.** Comparan datos desbalanceados con balanceados, no «distintos grados».
- **Vassallo et al.** Su resumen no menciona Elliptic; se cita por el cambio de concepto.
- **Rudin.** Se usaba para una afirmación propia; ahora se discute su objeción de fondo.
- **Motie et al.** Se le atribuían dominios de aplicación que no se pueden verificar.

## 3. Relación de los resultados con la literatura

La Sección «Relación con los trabajos previos» del Capítulo 7 (Tabla de resultados frente a la
literatura) confronta cada resultado con los trabajos citados e indica si coincide, difiere o no
es comparable.
