# Fase 2 — Creación de la Base de Conocimiento de Documentos

## Identificación de Documentos

Tipos de documentos relevantes para el sistema de atención al cliente de
EcoMarket:

1. **FAQ** — preguntas frecuentes (envíos, pagos, contacto, etc.).
2. **Spreadsheet de inventario** — catálogo de productos, stock, precios.
3. **PDF de política de devoluciones** — reglas por categoría de producto.
4. **Historial de casos complejos** — registro de casos pasados que
   requirieron intervención humana, útil como referencia de cómo se
   resolvieron situaciones similares.
5. **Documento de casos de escalamiento** — criterios explícitos de
   cuándo el sistema debe dejar de responder automáticamente e invocar a
   un agente humano.
6. **Documento de preguntas sin respuesta en el corpus** — registro de
   preguntas que el chatbot no pudo responder por no estar en la base de
   conocimiento, útil para identificar huecos de contenido y priorizar
   qué documentar a futuro.

## Segmentación (Chunking)

Estrategia en dos pasos:

1. **Chunking conceptual primero**: dividir por tipo de información
   (inventario, políticas de escalamiento, devoluciones, FAQ, etc.) antes
   de cualquier otra segmentación. Cada tipo de documento tiene una
   estructura y un propósito distinto, así que mezclarlos en el mismo
   proceso de chunking no tiene sentido.

2. **Dentro de cada tipo de documento, chunking por tamaño fijo con
   overlapping**: una vez separados por categoría, se prueban distintos
   tamaños de chunk contra un **golden set de preguntas base**, para
   encontrar el tamaño que mejor balancea precisión (recall de la
   información correcta) y latencia (chunks más pequeños y numerosos
   cuestan más en tiempo de búsqueda y re-ranking). El overlapping entre
   chunks consecutivos evita cortar una idea a la mitad en el límite
   entre dos fragmentos.

**Relación con el chunking recursivo**: esta estrategia es, en esencia,
similar al chunking recursivo, pero con el primer nivel de corte definido
**a alto nivel por humanos** (la categorización por tipo de documento) en
vez de por separadores sintácticos automáticos, y el segundo nivel fijo
(tamaño fijo con overlap) en vez de seguir bajando recursivamente por
párrafos y oraciones. Bajar un nivel más y dividir por párrafos sería
demasiado granular para este caso, fragmentando información que debería
mantenerse junta (por ejemplo, una política completa de devolución para
una categoría de producto).

**Impacto de un chunking deficiente en la calidad de las respuestas**:
- **Sin overlapping**: el LLM puede perder el contexto de la sección que
  está buscando, recibiendo datos incompletos justo en el límite entre
  dos chunks.
- **Chunks muy grandes**: consumen más tokens por consulta, lo que
  aumenta tanto el costo como la latencia de cada respuesta.
- **Chunks muy pequeños**: el modelo puede no tener el contexto completo
  necesario, incluso si el chunk correcto está dentro del top-k
  recuperado, porque la información relevante quedó repartida en
  fragmentos separados.

## Indexación

Proceso de principio a fin:

1. **Extracción de texto puro** de cada documento (PDF, spreadsheet, etc.).
2. **Chunking** según la estrategia descrita arriba (por tipo de
   documento, luego por tamaño fijo con overlap).
3. **Generación de embeddings**: cada chunk se pasa por el modelo de
   embeddings elegido (ver Fase 1), que genera el vector numérico
   correspondiente al contenido del fragmento.
4. **Almacenamiento**: se guarda, por cada chunk, el vector generado, el
   **texto original** del chunk, y **metadatos** (título del documento de
   origen, número de página, tipo de documento, etc.).
5. **Indexación en la base de datos vectorial**: el indexador construye
   un mapa de coordenadas de los vectores usando algoritmos de búsqueda
   aproximada de vecinos cercanos (ANN) como **HNSW**, **IVF** o **PQ**,
   para que la búsqueda por similitud sea eficiente incluso con muchos
   vectores.
6. **Texto original disponible para búsqueda híbrida**: conservar el
   texto original (no solo el vector) permite combinar la búsqueda
   semántica por embeddings con búsqueda por palabras clave (ej. BM25),
   si en el futuro se quisiera implementar retrieval híbrido.
