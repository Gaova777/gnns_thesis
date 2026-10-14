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
- **Rai et al.** Con el resumen se redujo a «balanceados frente a desbalanceados»; el texto
  completo muestra que sí variaron el grado (ver la Sección 4).
- **Vassallo et al.** Su resumen no menciona Elliptic; se cita por el cambio de concepto.
- **Rudin.** Se usaba para una afirmación propia; ahora se discute su objeción de fondo.
- **Motie et al.** Se le atribuían dominios de aplicación que no se pueden verificar.

## 3. Relación de los resultados con la literatura

La Sección «Relación con los trabajos previos» del Capítulo 7 (Tabla de resultados frente a la
literatura) confronta cada resultado con los trabajos citados e indica si coincide, difiere o no
es comparable.

## 4. Revisión del texto completo de las referencias clave (14-oct-2026)

Se revisó el texto completo de las referencias que sostienen las comparaciones de la discusión.
Los textos se obtuvieron solo de copias de acceso abierto (arXiv, repositorios institucionales y
editoriales abiertas) y se leyeron como texto plano; no se guardaron los PDF ni se saltó ningún
control de acceso.

| Referencia | Fuente del texto completo | Resultado |
|---|---|---|
| Weber et al. 2019 | arXiv 1908.02591 | Confirmado: LR, RF, MLP y GCN; partición 34/15; fallo de todos los métodos tras el cierre del paso 43; 4.545 ilícitas y 42.019 lícitas |
| Alarab y Prakoonwit 2023 (recurrente) | repositorio de Bournemouth | Confirmado, con dos precisiones añadidas: sus variables ANF son el número de vecinas antecedentes lícitas e ilícitas (usan etiquetas de las vecinas) y con ellas reportan F1 > 0,80 en el paso 43 |
| Lin et al. 2026 | Scientific Reports (abierto) | Precisado: excluyen las transacciones sin etiqueta y usan una partición aleatoria estratificada 80/10/10, no temporal |
| Lawal et al. 2025 | IJNSA (abierto) | **Corregido.** El resumen no coincide con el texto: tratan «sin etiqueta» como tercera clase, la prueba tiene 8 nodos y el texto no presenta resultados con pesos por clase, focal loss ni GNNExplainer. Se retiró de la tabla de comparación y de la Tabla 4.1 como trabajo con explicaciones |
| Rai et al. 2025 | Frontiers (abierto) | **Corregido.** Sí varían el grado de desbalance (40 % a 1 % y original). Su «consistencia» es el acuerdo con un modelo balanceado de referencia, no la estabilidad entre réplicas; lo distinguen ellos mismos de Chen et al. Usan Jaccard top-10, como esta tesis |
| Agarwal et al. 2023 | arXiv 2208.09339 | **Corregido.** Sí miden estabilidad, frente a perturbaciones mínimas de la entrada. Hallazgo nuevo para la discusión: la estabilidad de las explicaciones sobre variables se parece más a la de explicaciones aleatorias que la de las explicaciones sobre nodos |
| Adebayo et al. 2018 | arXiv 1810.03292 | Confirmado, con un hallazgo nuevo: con Integrated Gradients, el Spearman sobre valores absolutos sigue alto tras aleatorizar y sin valor absoluto cae a cero. Esta tesis usa valores absolutos (ahora declarado en la Metodología) |
| Longa et al. 2024 | arXiv 2210.15304 | Precisado: miden plausibilidad y fidelidad; la estabilidad solo de forma cualitativa |
| Zheng et al. 2026 | arXiv 2202.07082 | Confirmado; se añadió el ejemplo de fraude (defraudadores conectados con clientes) y la separación nodo/vecinas |
| Gawantka et al. 2024 | ASTESJ (abierto) | Confirmado |
| Kute et al. 2021 | IEEE Access (abierto) | Confirmado |
| Alarab y Prakoonwit 2022 (remuestreo) | repositorio de Bournemouth | **Añadida.** Referencia real (Crossref, journal-article) que estaba en el guion pero no en el manuscrito. Elliptic etiquetado, bosque aleatorio, importancia por permutación; cambios significativos solo con algunas técnicas (SMOTE-SF en prueba, p = 0,001; ninguna en entrenamiento) |

**No revisadas en texto completo:**

- Chen et al. 2023 (EJOR) y Fujiwara et al. 2024 (Results in Engineering) son de acceso abierto,
  pero los servidores (ScienceDirect y el repositorio de Edimburgo) responden con un desafío
  anti-bots, que no se intentó saltar. Para Chen et al., Rai et al. confirman de forma
  independiente que estudian la variabilidad entre entrenamientos repetidos bajo desbalance.
- Mammadov et al. 2026 (IEEE TEM), He et al. 2026 (Applied Intelligence) y Vassallo et al. 2021
  (SN Computer Science) son de pago.

Para estas cinco se mantiene lo que dice el resumen, y las tablas lo declaran.
