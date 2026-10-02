# Agenda para la reunión con Cristian (THE-19)

> Preparada por el Claude de Juan Diego a partir de la corrida v4 y sus decisiones.
> Fecha objetivo de la reunión: 03-oct-2026. Fuentes: `results_v4/analysis_v4_summary.txt`,
> `runs_v4/DECISIONES.md`, `results_v4_labels/label_comparison.md`, `docs/NARRATIVA_v4.md`.
>
> Cada punto marca si es **[Decisión de Cristian]** (necesita su visto bueno) o **[Informar]**
> (solo dejarlo claro).

## A. El giro de fondo

1. **[Informar] Las etiquetas de Elliptic estaban invertidas en la v3.** El eje se entrenó
   tratando los 157.205 nodos sin etiqueta como lícitos y dejando fuera los 42.019 lícitos
   reales. Es la razón de ser de toda la v4.

2. **[Decisión de Cristian] La conclusión central de la v3 no sobrevive.** Con las etiquetas
   corregidas: la partición en dos grupos de estabilidad por arquitectura desaparece (las
   cuatro salen equivalentes, TOST, Friedman p=0,11), y el balanceo pasa a tener efecto
   (Friedman p=0,039) cuando antes se daba por despreciable. Cómo narrar esta reversión de
   forma honesta es la conversación clave.

## B. Decisiones de diseño del PR a validar

3. **[Decisión de Cristian] Modo de etiqueta.** Se eligió C (lícitas + sin etiqueta) como
   principal y B (solo lícitas) como sensibilidad. El criterio pre-registrado E1 se cumple
   (C rinde como B, mediana de la diferencia de PR-AUC = -0,007). Dos señales piden cautela:
   los modelos B marcan como ilícita el 13,98% de los sin-etiqueta (no son limpiamente
   lícitos), y B produce 9/9 modelos que pasan la compuerta frente a 4/9 de C. Pregunta:
   ¿C como principal es defendible, o se llevan C y B en paralelo?

4. **[Decisión de Cristian] GAT capado por memoria.** GAT con 8 cabezas y hidden 148 no cabe
   en la GPU de 8 GB (pico ~9,5 GiB). Se bajó a heads=4 y 2 capas, con hidden intacto en 148,
   documentado en `runs_v4/DECISIONES.md`. El confound queda neutralizado de hecho porque la
   conclusión de arquitectura es equivalencia y GAT sale con las demás (0,928 frente a 0,945
   y 0,953). Además, el eje sintético ya construía GAT con heads=4, así que los dos ejes
   quedan consistentes. Pregunta: ¿se acepta como límite declarado?

5. **[Decisión de Cristian] GCN casi no aprende en modo C** (val PR-AUC 0,095) y queda fuera
   del soporte de H2. Qué significa que GCN, que en la v3 estaba en el grupo alto, ahora falle
   como clasificador, y cómo se reporta.

6. **[Informar] Escenarios.** La v4 corrió 4 escenarios (native, 1:10, 1:1, 1:10_os), no los
   6 del runbook original. Confirmar que basta para H1.

## C. Narrativa

7. **[Decisión de Cristian] Pasar de tres hipótesis a una sola.** Propuesta redactada en
   `docs/NARRATIVA_v4.md`: la estabilidad la gobierna el método de explicación, no la
   arquitectura ni el desbalance, y ser estable no implica ser plausible ni fiel. Cristian
   dijo que propondría su propia redacción; se contrasta con la suya.

8. **[Decisión de Cristian] Pregunta de investigación y objetivo**, redactados para el jurado.

9. **[Decisión de Cristian] Dónde van las pruebas estadísticas** (TOST, refutar la nula):
   cuerpo del documento o anexo de marco teórico. Lo pidió él explícitamente.

## D. Puntos técnicos menores

10. **[Informar] Rama "no detectado con esta potencia"** para PGExplainer en H1 y H2, frente
    a "equivalencia" de GNNExplainer y ShapleyFeatures: cómo se reporta esa diferencia entre
    explicadores.

11. **[Informar] Colapso validación a test** (shift temporal): se mantiene, y por eso la
    estabilidad se mide sobre los verdaderos positivos de validación. Confirmar el encuadre.

12. **[Decisión de Cristian] Eje sintético (THE-13)**, en la máquina de Alejandro: GCN quedó
    al borde en la prueba de alineamiento (margen +0,138, criterio ≥ 0,10). Decidir si su
    plausibilidad se interpreta como calidad del explicador o va a anexo.

13. **[Informar] Qué no se rehízo:** los CSV de la v3 y `phase1/` viejos se conservan; el
    manuscrito citará la v4 para el eje Elliptic.

## Cierre esperado de la reunión

- Redacción final de la hipótesis única, la pregunta y el objetivo (desbloquea THE-25).
- Visto bueno al modo C, al límite de GAT y al tratamiento de GCN.
- Ubicación de las pruebas estadísticas.
- Lo acordado se anota en `runs_v4/DECISIONES.md` y se refleja en Linear.
