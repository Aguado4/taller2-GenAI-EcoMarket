# Fase 1 — Selección de Componentes Clave del Sistema RAG

## Modelo de Embeddings

La elección depende de dos cosas que hay que verificar primero: el idioma
real de los documentos y el idioma de las consultas de los clientes. Si
ambos son en español, la decisión se reduce a comparar un modelo
**multilingüe** contra uno **específico de español**.

En cualquier caso, la forma correcta de decidir no es elegir a priori,
sino correr una evaluación de desempeño (ej. recall@k sobre un set de
preguntas de prueba) comparando ambas opciones, incluyendo también
modelos propietarios de pago (OpenAI, Cohere, etc.) como punto de
referencia de calidad máxima.

Para el caso específico de EcoMarket, un modelo **open-source es más que
suficiente**:
- Las consultas son mayormente preguntas frecuentes (FAQ) de dominio
  acotado (pedidos, devoluciones, catálogo), no preguntas abiertas de
  conocimiento general que requieran máxima precisión semántica.
- El volumen de documentos de la base de conocimiento es relativamente
  pequeño, por lo que no se necesita el embedding "más potente del
  mercado" para obtener buen recall.
- Evita el costo recurrente por token de un modelo de pago, que no se
  justifica para este volumen y este caso de uso.

En el código de este repositorio se usó
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` como punto
de partida (open-source, corre en CPU, multilingüe), pero la
recomendación formal es: antes de fijarlo en producción, correr la
comparación de desempeño descrita arriba contra un modelo específico de
español y contra un modelo de pago, para confirmar que la pérdida de
calidad frente a las opciones de pago es aceptable para este volumen de
preguntas.

## Base de Datos Vectorial

Comparación de las opciones evaluadas, incluyendo pgvector:

| Opción | Escalabilidad | Costo | Facilidad de uso | Notas para EcoMarket |
|---|---|---|---|---|
| **Pinecone** | Alta (manejado, escala automáticamente) | De pago (desde plan gratuito limitado, luego por uso) | Muy alta (SaaS, sin infraestructura que mantener) | Buena opción si el catálogo de documentos crece mucho y no se quiere operar infraestructura propia; el costo recurrente es el principal contra para un caso de uso con pocos documentos |
| **Weaviate** | Alta (se puede auto-hospedar o usar su cloud managed) | Gratis si se auto-hospeda; de pago en su versión cloud | Media (más configuración que Pinecone, pero más flexible: búsqueda híbrida nativa, filtros por metadatos avanzados) | Buen punto medio si se quiere control propio de la infraestructura y búsqueda híbrida (vector + keyword) desde el día uno |
| **ChromaDB** | Baja-media (pensado para un solo nodo; no es la opción para escalar a millones de vectores distribuidos) | Gratis, corre local o embebido | Muy alta (se integra en pocas líneas con LangChain, ideal para prototipar) | La elegida en este repo: para el volumen inicial de EcoMarket (unos pocos documentos, cientos de chunks) es más que suficiente, y el costo cero permite iterar rápido. Si el catálogo de EcoMarket creciera órdenes de magnitud, migrar a otra opción |
| **pgvector** | Media-alta (heredada de Postgres: escala tan bien como la instancia de Postgres que la hospede) | Gratis (extensión de Postgres); el costo es el de operar la base de datos relacional en sí | Media (requiere ya tener o montar una instancia de Postgres) | Ventaja clave para EcoMarket: si el inventario, pedidos y clientes ya viven en una base de datos relacional, pgvector permite guardar los embeddings **en la misma base de datos transaccional**, con joins directos entre datos vectoriales y datos estructurados (ej. filtrar por categoría de producto o estado de pedido en la misma consulta), sin sincronizar dos sistemas distintos |

**Elección para este taller:** ChromaDB, por ser gratuita, local y de
integración inmediata con LangChain — adecuada para el volumen actual de
documentos de EcoMarket y para iterar rápido durante el desarrollo.

**Elección general / recomendación de producción: Weaviate.** En el caso
general (no solo para prototipar este taller), **Weaviate** es el mejor
punto medio entre las opciones evaluadas:

- Permite **búsqueda híbrida** (vectorial + por palabras clave) nativa,
  lo cual puede mejorar los resultados frente a una búsqueda puramente
  semántica, especialmente para consultas con términos exactos (nombres
  de producto, números de pedido).
- Es **escalable**.
- Da la **flexibilidad de elegir** entre operarlo en la nube (managed) o
  auto-hospedarlo, sin estar atado a un solo modelo de despliegue como sí
  ocurre con Pinecone (siempre managed) o ChromaDB (pensado para un solo
  nodo).

El único trade-off real frente a las otras opciones es la **complejidad**:
requiere más configuración que ChromaDB o Pinecone para aprovechar todas
sus capacidades. Para EcoMarket, esa complejidad adicional se justifica
en cuanto el sistema pase de prototipo a producción real.

Si en cambio EcoMarket ya tuviera su inventario y pedidos en una base de
datos relacional, **pgvector** seguiría siendo la opción más simple para
evitar mantener dos sistemas de datos separados, a costa de no tener
búsqueda híbrida nativa ni la escalabilidad horizontal de Weaviate.
