# Respaldo · Estabilidad por semilla de modelo

> Bloque: Respaldo (solo si el jurado pregunta)
> Página PDF 44 · Sin discurso propio en el guion (se abre solo si se pide más detalle)

## 🎯 Objetivo de la lámina

Mostrar el detalle fila por fila, semilla por semilla, de la estabilidad de cada arquitectura, para
poder responder con precisión si el jurado pregunta por la variabilidad exacta detrás de los promedios
mostrados en la [lámina 21](lamina-21-resultado1-dos-grupos.md).

## 📋 Qué dice, item por item

- **Tabla 7 — Estabilidad por semilla de modelo (Elliptic, GNNExplainer):**
  - **GAT:** semilla 42 = 0,766; semilla 43 = 0,789; semilla 44 = 0,790 → media 0,782, IC 95%
    de 0,758 a 0,805.
  - **GCN:** semilla 42 = 0,783; semilla 43 = 0,733; semilla 44 = 0,757 → media 0,758, IC 95%
    de 0,724 a 0,787.
  - **GraphSAGE:** semilla 42 = 0,756; semilla 43 = 0,713; semilla 44 = 0,738 → media 0,735, IC 95%
    de 0,710 a 0,756.
  - **TAGCN:** semilla 42 = 0,586; semilla 43 = 0,693; semilla 44 = 0,736 → media 0,672, IC 95%
    de 0,632 a 0,717.
- **Nota:** GAT y GCN **se permutan** entre semillas (en la semilla 42, GCN incluso supera a GAT); por
  eso no se nombra un ganador único dentro del grupo alto. TAGCN tiene la mayor dispersión entre
  semillas (0,077, el triple que las demás): su lugar en el grupo bajo es firme, pero su valor central
  admite un margen de incertidumbre mayor.

## 🔑 Conceptos y técnicas que aparecen

- **Permutación entre semillas:** que el orden relativo de dos valores cambie de una corrida a otra.
  Aquí, en la semilla 42, GCN (0,783) supera a GAT (0,766); en las semillas 43 y 44, es al revés. Esto
  es precisamente lo que justifica no afirmar que una de las dos sea "la mejor": el orden depende de la
  semilla, no es consistente.
- **Dispersión entre semillas:** cuánto varía el resultado de una misma arquitectura al cambiar
  únicamente la semilla de entrenamiento. TAGCN va de 0,586 a 0,736 (una diferencia de 0,15 entre su
  peor y mejor semilla), mucho más que cualquier otra arquitectura, lo que indica que su comportamiento
  es menos predecible de una corrida a otra.

## 📈 Cómo leer la figura / tabla

La Tabla 7 muestra, para cada arquitectura, sus tres valores individuales (uno por semilla), la media,
y el intervalo de confianza al 95%. La lectura correcta no es solo mirar la columna "Media" (que es la
que aparece resumida en la lámina 21), sino comparar las tres columnas de semillas entre sí para cada
arquitectura: eso revela si el resultado es consistente (poca diferencia entre semillas, como GAT y
GCN) o más volátil (mucha diferencia, como TAGCN).

## 🕐 Cuándo abrir esta lámina

Abrir esta lámina si el jurado pregunta: "¿cuánto varía cada arquitectura entre semillas?", "¿por qué
no dicen que GAT es la más estable si tiene el promedio más alto?", o "¿qué tan confiable es el valor
de TAGCN, dado que es el más bajo?".

## 🧑‍⚖️ Preguntas de jurado probables

- **"Si GAT tiene el promedio más alto (0,782), ¿por qué no se afirma que es la arquitectura más
  estable?"** → Porque en la semilla 42, GCN (0,783) en realidad supera a GAT (0,766); el orden se
  invierte según la semilla, lo que demuestra que la diferencia entre ambas, dentro del grupo alto, no
  es una diferencia real y consistente, sino ruido de muestreo.
- **"¿Por qué TAGCN tiene tanta más dispersión que las demás arquitecturas?"** → Es un hallazgo
  reconocido como limitación en la tesis (ver [lámina 33](lamina-33-contribuciones-limitaciones.md)):
  con solo tres semillas, se puede afirmar con confianza que TAGCN pertenece al grupo bajo, pero no se
  puede fijar con la misma precisión su valor central exacto, precisamente por esta mayor variabilidad.

## 🧠 En una frase

GAT y GCN se permutan entre semillas (ninguna es consistentemente mejor que la otra), y TAGCN tiene la
mayor dispersión de las cuatro arquitecturas, lo que respalda por qué la tesis habla de "dos grupos" y
no de un ranking fino.
