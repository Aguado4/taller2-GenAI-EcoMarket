# Casos de Prueba — Evaluación del Pipeline RAG

Casos reales corridos con `python -m rag.evaluate` (salida completa en
[`outputs/evidencia_rag.txt`](../outputs/evidencia_rag.txt)). Sirven para
que el evaluador reproduzca y entienda por qué el sistema se comporta así.

## Por qué estos casos

- **Dentro de dominio**: confirman que el retrieval + re-ranking trae el
  chunk correcto y que el LLM responde anclado a ese texto (no inventa).
- **Fuera de dominio**: confirman el mecanismo de defensa en dos capas:
  1. Umbral numérico sobre el score de re-ranking (`RERANK_SCORE_THRESHOLD
     = -3.0` en `rag/pipeline.py`): si el mejor chunk recuperado está muy
     lejos semánticamente, se activa el fallback sin llamar al LLM.
  2. Instrucción del prompt ("si no sabes, dilo"): cubre los casos
     borderline que no activan el umbral pero tampoco tienen respuesta
     real en la base de conocimiento.
- **Con RAG vs. sin RAG**: evidencia concreta de que, sin retrieval, el
  modelo alucina (inventa correos, teléfonos, precios en otra moneda).

## Resultados

| # | Pregunta | Tipo | Score re-rank (mejor chunk) | Fallback | Resultado esperado | Resultado obtenido |
|---|---|---|---|---|---|---|
| 1 | ¿Puedo devolver un champú sólido si ya abrí el empaque? | Dentro de dominio | 4.709 | No | Explica que no aplica salvo defecto de fábrica | ✅ Correcto, anclado a `politica_devoluciones.md` |
| 2 | ¿Cuánto cuesta la mochila de fibras recicladas y se puede devolver? | Dentro de dominio | 4.853 | No | Precio exacto + regla de devolución de hogar (30 días) | ✅ Correcto: "$120.000 COP", devolución a 30 días |
| 3 | ¿Qué métodos de pago acepta EcoMarket? | Dentro de dominio | 8.666 | No | Lista de métodos del FAQ | ✅ Correcto: tarjetas, PSE, contraentrega |
| 4 | Compré un snack de frutas deshidratadas y llegó vencido, ¿qué hago? | Dentro de dominio | 6.056 | No | Excepción a la regla de "no devolvible" por producto vencido | ✅ Correcto: ofrece reemplazo/reembolso |
| 5 | ¿Cuál es la capital de Australia? | Fuera de dominio (claro) | -9.083 | **Sí** | Fallback sin invocar al LLM | ✅ Correcto, umbral activado |
| 6 | ¿EcoMarket vende laptops o celulares? | Fuera de dominio (borderline) | -1.263 | No (umbral no se activa) | El LLM debe decir que no sabe, no inventar | ✅ Correcto: refusal vía prompt, no vía umbral |
| 7 | ¿Cuál es el CEO de EcoMarket y cuánto gana? | Fuera de dominio (borderline) | -1.139 | No (umbral no se activa) | El LLM debe decir que no sabe, no inventar | ✅ Correcto: refusal vía prompt |
| 8 | (1) y (2) repetidas, con RAG **desactivado** | Baseline sin contexto | — | — | El modelo debería alucinar datos | ✅ Alucina: correo `devoluciones@ecomarket.com` y teléfono `+34 900 123 456` (inventados), precio en euros (`49,99 €`) en vez de `$120.000 COP` |

## Cómo reproducirlo

```bash
python -m rag.ingest      # construye el índice (una vez, o si cambia la knowledge_base)
python -m rag.evaluate    # corre esta batería y la agrega a outputs/evidencia_rag.txt
```

También se puede probar interactivamente en `streamlit run app_streamlit.py`,
usando el toggle de RAG y el panel de transparencia para ver los chunks
recuperados y sus scores en cada pregunta.
