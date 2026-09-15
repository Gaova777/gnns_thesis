# Lámina 04 — Por qué la caja negra es inaceptable aquí

> Bloque: Problema y motivación
> Voz: Alejandro · Página PDF 6 (4/40) · Tiempo objetivo 55 s

## 🎯 Objetivo de la lámina

Explicar, con tres razones concretas, por qué en este dominio específico (detección de lavado, un
entorno regulado) no basta con que un modelo acierte: tiene que poder **explicarse**. Y de ahí extraer
la consecuencia central de toda la tesis: la propiedad que hay que estudiar es la **estabilidad**.

## 📋 Qué dice, item por item

- **"Una alerta puede congelar cuentas o iniciar una investigación: el costo de un error es alto."**
  — No es un entorno donde equivocarse sea barato. Una decisión automática puede afectar de forma
  directa la vida de una persona o de una empresa.
- **"En un entorno regulado, el analista debe justificar cada decisión ante un auditor."** — No basta
  con que el modelo tenga razón; alguien tiene que poder **defender** esa decisión frente a un
  regulador o un juez, y para eso necesita entender el motivo.
- **"Y debe ser reproducible: la misma entrada tiene que dar la misma explicación."** — Si se audita
  la misma transacción dos veces, la explicación no puede cambiar de un día para otro; si cambia, no
  sirve como evidencia.
- **Recuadro "Consecuencia":** la explicabilidad no es un lujo, es un **requisito**. Y si la
  explicación cambia cada vez que se recalcula, no sirve para auditar. Por eso la propiedad crítica —
  y el foco de toda la tesis — es la **estabilidad**.

## 🔑 Conceptos y técnicas que aparecen

- **Entorno regulado:** un contexto donde existen reglas legales y normativas (por ejemplo, contra el
  lavado de activos) que obligan a las instituciones a justificar sus decisiones ante autoridades.
- **Reproducibilidad de una explicación:** que, dado el mismo caso, el mismo método de explicación
  produzca el mismo resultado si se vuelve a calcular. Es distinto de que el **modelo** sea
  reproducible; aquí se habla específicamente de que la **explicación** lo sea.
- **Estabilidad (adelanto):** la propiedad que mide justamente esa reproducibilidad de la explicación.
  Se define y mide en detalle a partir de la [lámina 08](lamina-08-paso-de-mensajes.md) en adelante.
  Ver también el [glosario](README.md#las-tres-propiedades-estabilidad-plausibilidad-fidelidad).

## 📈 Cómo leer la figura / tabla

No aplica: esta lámina no tiene figura ni tabla.

## 🎤 El discurso (como se dice en voz alta)

> Quiero detenerme en por que, en este dominio, una caja negra no es aceptable. *(pausa)* Primero, porque
> el costo de un error es alto: una alerta puede congelar cuentas o iniciar una investigacion sobre una
> persona. Segundo, porque es un entorno regulado: el analista debe poder justificar cada decision ante un
> auditor, no basta con decir que el modelo lo dijo. Y tercero, porque debe ser reproducible: la misma
> transaccion tiene que producir la misma explicacion, hoy y dentro de un mes. *(pausa)* La consecuencia
> es la que ordena toda la tesis: la explicabilidad no es un lujo, es un requisito. Y si la explicacion
> cambia cada vez que se recalcula, no sirve para auditar. Por eso la propiedad critica, la que
> estudiamos, es la estabilidad.

### Versión ampliada y explicada

*"Quiero detenerme en por que, en este dominio, una caja negra no es aceptable."* — Anuncia que va a
dar razones puntuales, no solo una afirmación general.

*"Primero, porque el costo de un error es alto: una alerta puede congelar cuentas o iniciar una
investigacion sobre una persona."* — La primera razón es de consecuencias: un falso positivo aquí no
es solo un correo de spam mal filtrado, es una acción que afecta la vida real de alguien.

*"Segundo, porque es un entorno regulado: el analista debe poder justificar cada decision ante un
auditor, no basta con decir que el modelo lo dijo."* — La segunda razón es institucional: existen
leyes y normas que exigen justificación humana de las decisiones automatizadas en este sector.
Responder "porque el modelo lo dijo" no es una justificación válida ante un regulador.

*"Y tercero, porque debe ser reproducible: la misma transaccion tiene que producir la misma
explicacion, hoy y dentro de un mes."* — La tercera razón es práctica: una explicación que cambia
cada vez que se recalcula no puede usarse como evidencia sólida en una investigación.

*"La consecuencia es la que ordena toda la tesis: la explicabilidad no es un lujo, es un requisito."*
— Es la frase bisagra: convierte las tres razones anteriores en una conclusión operativa.

*"Y si la explicacion cambia cada vez que se recalcula, no sirve para auditar. Por eso la propiedad
critica, la que estudiamos, es la estabilidad."* — Cierra apuntando directamente al objeto de estudio
de toda la tesis: no basta con tener una explicación, tiene que ser **estable**.

## 🧑‍⚖️ Preguntas de jurado probables

- **"¿Por qué la estabilidad y no, por ejemplo, la precisión del modelo, es la propiedad crítica
  aquí?"** → Porque la precisión ya se mide con métricas estándar de clasificación; lo que faltaba
  evaluar, y lo que un entorno auditado exige además, es si la **explicación** de esa predicción es
  consistente en el tiempo, algo que la literatura previa no había evaluado de forma sistemática
  (ver [lámina 07](lamina-07-la-brecha.md)).
- **"¿No sería suficiente con documentar manualmente cada decisión?"** → No a escala: un sistema real
  procesa miles de transacciones, y depender de justificación manual caso por caso anula la ventaja
  de automatizar la detección.

## 🧠 En una frase

En un entorno regulado, la explicabilidad no es opcional: es un requisito, y para que sirva de
evidencia debe ser además **estable**.
