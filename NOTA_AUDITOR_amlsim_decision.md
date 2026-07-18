# Nota para el auditor — Fase 1 (II.1): AMLSim bloqueado, opciones + recomendación

**Fecha:** 2026-07-16
**Contexto:** ejecutando el `ENCARGO_claude_code_amlsim.md` (montar AMLSim en la 4060 del estudiante).
Hice la diligencia completa de montaje y topé con un bloqueo de dependencia real. Abajo: qué funciona, el
bloqueo con evidencia, un hallazgo metodológico, las opciones y mi recomendación.

---

## 1. Lo que SÍ quedó montado
- **Java 8** (Temurin 1.8.0_492) y **Maven**, instalados en espacio de usuario vía SDKMAN (sin sudo).
- Repo **AMLSim clonado** (IBM), pom inspeccionado, build de Maven ejecutado.

## 2. El bloqueo: MASON v20 no es obtenible como jar
AMLSim depende de **MASON versión 20** (framework de simulación multi-agente de GMU). El build falla en:
```
Could not find artifact mason:mason:jar:20 in central
```
Evidencia de que no hay jar distribuible de la v20:
- **Maven Central**: solo existen versiones viejas bajo otros groupId (`edu.gmu.cs:mason:14`,
  `net.sf.sociaal:mason:16`, `...:mason:18`). **No hay v20.**
- **GitHub `eclab/mason` release `v20`**: existe el tag pero tiene **0 assets** (solo código fuente).
- **Sitio de GMU**: la URL histórica `mason20.zip` devuelve HTML (el sitio se rediseñó); ya no sirve el zip.
- → Para tener MASON v20 habría que **compilarlo desde fuente** (otro build Java, con sus propias deps).

## 3. Dos problemas más, aguas abajo (aunque se resolviera MASON)
- **Las muestras pre-generadas de AMLSim** (`sample/20K_fanin200.tgz`, etc., 20K nodos) **solo traen
  `isFraud` a nivel de NODO** — `nodes.csv = nodeid,isFraud,init_balance,fraudStep` y
  `transactions.csv = source,target,value,time`. **No hay ground-truth por-edge ni por-tipología**, que es
  justo lo que la métrica de plausibilidad (item 7) necesita.
- **Generar datos nuevos** con AMLSim requiere su entorno Python: `networkx==1.11` (muy viejo) +
  **`pygraphviz`**, que normalmente necesita la librería `graphviz` del sistema (paquete `apt` → **tu
  sudo**, que el asistente no tiene).

## 4. Hallazgo metodológico (aplica a CUALQUIER generador, también AMLSim)
Al validar el prototipo Python medí su receptive field con el mismo `k_hop_subgraph` de Elliptic:
- **Aristas dirigidas (as-is): mediana 2 nodos** — FALLA el criterio de densidad (igual que Elliptic).
- **Aristas no dirigidas (simetrizadas): mediana 17 nodos** — PASA (objetivo ≥ ~10).

Es decir: **el problema de dispersión es sobre todo la DIRECCIÓN de las aristas**, no el número de cuentas
por patrón. Con message passing dirigido (source→target) un nodo solo "ve" a sus predecesores, y los
patrones tipo estrella/cadena quedan invisibles. **Decisión de diseño para el sintético (sea prototipo o
AMLSim): usar aristas no dirigidas / simétricas** para que la tipología esté en el receptive field y la
plausibilidad sea medible. (Elliptic tampoco simetriza, lo que explica parte de su dispersión.)

## 5. Lo que YA funciona: el prototipo Python
El generador `phase1/synthetic_aml_generator.py` (dejado por la sesión de análisis), **con simetrización**:
- Densidad: mediana 17 nodos de receptive field ✓ (vs 2-3 de Elliptic).
- **Ground-truth exacto por nodo Y por edge** (`typology_node`, `typology_edge`) — 4 tipologías
  (STRUCTURING, LAYERING, FAN_IN, FAN_OUT).
- Salida PyG `Data` directa → corre con el pipeline corregido (k-hop, GNNExplainer, estabilidad) sin build
  de Java.
- ~9.3k nodos, 14% ilícito, control total de densidad por construcción.

---

## 6. Opciones (con esfuerzo y trade-offs)

| # | Opción | Esfuerzo / riesgo | Pro | Contra |
|---|--------|-------------------|-----|--------|
| **A** | **Prototipo Python ahora** | Bajo, inmediato | Ground-truth exacto por nodo/edge; densidad ✓; corre ya con el pipeline; desbloquea el item 7 | No lleva la "marca" AMLSim (menos apelación de herramienta reconocida) |
| **B** | **AMLSim completo** | Alto, incierto | Dataset "de marca" AMLSim para el capítulo de datos | Compilar MASON v20 desde fuente + `pygraphviz` probablemente necesita **tu sudo** (graphviz); multi-hora sin garantía |
| **C** | **Muestras pre-generadas AMLSim** | Bajo | Sin build; provenance AMLSim | Ground-truth **solo por nodo** (isFraud); habría que **reconstruir** los edges de patrón (aproximado); una tipología por muestra |
| **D** | **Híbrido** | Bajo ahora | Prototipo desbloquea item 7 YA; AMLSim después si instalas graphviz (sudo) con el método ya validado | Dos datasets que mantener |

## 7. Recomendación

**Opción D (híbrido), empezando por A.** Razones:
1. **El aporte científico del item 7 no depende de la marca del generador**: es "¿la estabilidad predice la
   plausibilidad?" sobre tipologías AML estándar (structuring/layering/fan-in-out), que el prototipo ya
   inyecta con ground-truth exacto y densidad verificada.
2. **AMLSim está genuinamente bloqueado** por MASON v20 (solo-fuente) y probablemente por `graphviz` (tu
   sudo). No debe bloquear la investigación.
3. El prototipo permite **construir y validar YA** la métrica de plausibilidad (II.3) y el análisis puente
   (II.4) sobre un caso con ground-truth conocido — que es exactamente el uso que el propio encargo le dio
   ("banco de validación").
4. **AMLSim queda como robustez/credibilidad**: si decides instalar `graphviz` con sudo, intento el build
   completo (incl. compilar MASON desde fuente) con la métrica y el pipeline ya validados, y se reporta como
   confirmación sobre un dataset de marca. Si no, el capítulo de datos se defiende con el generador
   controlado + la limitación declarada (misma honestidad que en Elliptic).

**Si prefieres AMLSim sí o sí (Opción B):** dime y (a) intento compilar MASON v20 desde fuente, y (b) te
paso el comando exacto de `sudo apt install` para graphviz/graphviz-dev que necesitarías correr tú.

---

## 8. Estado actual (para que no haya sorpresas)
- Instalado (usuario): SDKMAN + Java 8 + Maven en `~/.sdkman`. AMLSim clonado en `~/AMLSim`.
- Prototipo en `phase1/synthetic_aml_generator.py` (validado; falla densidad si se usa dirigido, pasa si se
  simetriza).
- Nada del pipeline corregido se tocó. No se generó dataset de tesis todavía (espera esta decisión).
