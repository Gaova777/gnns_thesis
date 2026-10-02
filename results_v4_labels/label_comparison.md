# Paso 0 — ¿los sin etiqueta se pueden tratar como lícitos?

Escenario nativo, semilla 42. Varas comunes en **test** (ts 43-49); el umbral de cada modelo se calibró en su propia validación.

| Modo | n | PR-AUC vs lícitas | PR-AUC vs sin etiqueta | PR-AUC vs todo | recall ilícitas | marcadas: lícitas | marcadas: sin etiqueta |
|---|---|---|---|---|---|---|---|
| C (lícitas + sin etiqueta) | 9 | 0.039 | 0.011 | 0.009 | 0.012 | 1.33% | 2.66% |
| B (lícitas) | 9 | 0.049 | 0.006 | 0.005 | 0.030 | 2.58% | 13.98% |
| A (sin etiqueta, v3) | 9 | 0.041 | 0.013 | 0.009 | 0.018 | 2.55% | 1.13% |

_Medianas sobre las configuraciones (arquitectura × balanceo)._

## E1 · C rinde como B separando ilícitas de lícitas reales
- Mediana(C − B) de PR-AUC vs lícitas = -0.007 (rango -0.014 a +0.148, n = 9, Wilcoxon p = 0.570). Criterio |mediana| ≤ 0.05: **se cumple**.

## E2 · Un modelo que nunca vio los sin etiqueta los trata como lícitos
- Modelos B, fracción marcada como ilícita en test: lícitas 2.58% · sin etiqueta 13.98% · ilícitas 2.96%.
- Probabilidad media de ilícita: lícitas 0.114 · sin etiqueta 0.292 · ilícitas 0.217.
- Lectura: si «sin etiqueta» queda cerca de «lícitas» y lejos de «ilícitas», los sin etiqueta son en su mayoría lícitos; el exceso sobre las lícitas acota el fraude no detectado que esconden.

## E3 · A y C llegan a lo mismo
- Mediana(C − A) de PR-AUC vs todo = -0.001 (rango -0.019 a +0.019, n = 9).

## Compuerta de calidad (validación propia de cada modo)
- C (lícitas + sin etiqueta): 4/9 pasan.
- B (lícitas): 9/9 pasan.
- A (sin etiqueta, v3): 4/9 pasan.
