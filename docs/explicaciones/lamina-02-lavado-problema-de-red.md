# Lámina 02 — El lavado de dinero es un problema de red

> Bloque: Problema y motivación
> Voz: Alejandro · Página PDF 4 (2/40) · Tiempo objetivo 55 s

## 🎯 Objetivo de la lámina

Explicar, antes de hablar de modelos o de inteligencia artificial, **qué es** el lavado de dinero y
por qué la única forma sensata de detectarlo es mirando la **red de conexiones** entre cuentas, no
cada transacción por separado. Esta lámina es la semilla de toda la decisión de usar grafos.

## 📋 Qué dice, item por item

- **Tres fases clásicas:**
  - **Colocación:** el dinero ilícito (por ejemplo, de un delito) entra por primera vez al sistema
    financiero formal.
  - **Estratificación:** ese dinero se fragmenta en muchas transacciones pequeñas y encadenadas, para
    que sea difícil rastrear su origen.
  - **Integración:** el dinero, ya "limpio" en apariencia, vuelve a la economía normal (compras,
    inversiones) sin levantar sospechas.
- **Tipologías = patrones entre cuentas:** las técnicas concretas que usan los criminales no son
  trucos aislados, son formas de **conectar cuentas**: *structuring* (o "pitufeo", fraccionar una
  suma grande en muchas pequeñas para no llamar la atención), *layering* (crear varias capas de
  transacciones intermedias), *fan-in* (muchas cuentas que envían dinero hacia una sola) y *fan-out*
  (una cuenta que reparte dinero hacia muchas otras).
- **La idea clave (recuadro):** una transacción mirada sola, de forma aislada, parece perfectamente
  normal. Lo que realmente delata el lavado es el **patrón de conexiones** entre varias cuentas.
- **Nota final:** por esa razón el problema se modela como un **grafo** (una red), y no como una
  tabla de transacciones que se miran una por una, sin relación entre sí.

## 🔑 Conceptos y técnicas que aparecen

- **Grafo:** una red de puntos (nodos) conectados por líneas (aristas). Aquí, cada nodo es una
  transacción y cada arista es un flujo de dinero entre dos transacciones. Ver
  [glosario](README.md#grafo-nodo-arista-vecindario).
- **Tipologías de lavado:** son, en el fondo, **formas geométricas** dentro de la red: una estrella
  que converge hacia un punto (fan-in), una estrella que se abre desde un punto (fan-out), una cadena
  de pasos intermedios (layering), o muchas transacciones pequeñas repartidas (structuring). Pensarlas
  como "formas en la red" es justo lo que después permite construir un grafo sintético con estas
  tipologías "plantadas" a propósito (ver [lámina 16](lamina-16-eje2-sintetico.md)).

## 📈 Cómo leer la figura / tabla

No aplica: esta lámina no tiene figura ni tabla, solo texto organizado en dos columnas.

## 🎤 El discurso (como se dice en voz alta)

> Antes de nada, que es el lavado de dinero y por que lo tratamos como una red. *(pausa)* El lavado tiene
> tres fases clasicas: la colocacion, cuando el dinero ilicito entra al sistema; la estratificacion,
> cuando se fragmenta en muchas transacciones para borrar el rastro; y la integracion, cuando vuelve a la
> economia con apariencia legal. *(pausa)* Y las tecnicas concretas que usan los criminales, lo que
> llamamos tipologias, son en el fondo patrones de conexiones entre cuentas: el structuring o pitufeo, el
> layering, y las formas de fan-in y fan-out. La idea que quiero dejar fijada es esta: una transaccion
> mirada de forma aislada parece completamente normal; lo que delata el lavado es el patron que forman
> varias cuentas juntas. *(pausa)* Por eso el problema se modela como un grafo, una red de flujos, y no
> como una tabla de transacciones independientes. Esa decision es la que abre la puerta a las redes
> neuronales de grafos.

### Versión ampliada y explicada

*"Antes de nada, que es el lavado de dinero y por que lo tratamos como una red."* — Anuncia que esta
lámina va a sentar las bases conceptuales del problema, antes de tocar cualquier tecnicismo.

*"El lavado tiene tres fases clasicas: la colocacion... la estratificacion... y la integracion..."* —
Se recorren las tres fases en orden temporal: entra el dinero sucio, se disfraza a través de muchos
pasos, y sale limpio al otro lado. Entender estas tres fases ayuda a ver por qué el rastro del dinero
pasa necesariamente por **muchas cuentas conectadas**, no por una sola transacción sospechosa.

*"Y las tecnicas concretas que usan los criminales, lo que llamamos tipologias, son en el fondo
patrones de conexiones entre cuentas..."* — Aquí se traduce la jerga (structuring, layering, fan-in,
fan-out) a algo visual: son formas que dibuja el dinero al moverse entre cuentas. Esta traducción es
la que después permite "dibujar" esas formas a propósito en un grafo sintético.

*"La idea que quiero dejar fijada es esta: una transaccion mirada de forma aislada parece
completamente normal..."* — Es la frase más importante de la lámina: resume por qué mirar
transacción por transacción (como hacen los métodos tradicionales) no basta.

*"Por eso el problema se modela como un grafo... Esa decision es la que abre la puerta a las redes
neuronales de grafos."* — Cierra con la consecuencia práctica: si el problema es de red, la
herramienta natural para resolverlo es un modelo diseñado para redes (una GNN), tema que se
desarrolla más adelante.

## 🧑‍⚖️ Preguntas de jurado probables

- **"¿Por qué no basta con analizar cada transacción con reglas de monto o frecuencia?"** → Porque el
  lavado se define por el **patrón entre transacciones**, no por ninguna transacción aislada; una
  regla sobre montos individuales no puede capturar, por ejemplo, que diez cuentas distintas le
  envíen dinero a una sola (fan-in).
- **"¿Estas cuatro tipologías son las únicas que existen?"** → Son las tipologías canónicas y más
  reconocidas en la literatura de lavado de dinero; la tesis las usa como base para construir el
  grafo sintético con patrones conocidos (ver lámina 16 y 17).

## 🧠 En una frase

El lavado de dinero se delata en el **patrón de conexiones** entre cuentas, no en ninguna transacción
aislada, y por eso el problema se modela como una red (un grafo).
