"""UI de Streamlit para el asistente de atención al cliente de EcoMarket
con RAG (retrieval + re-ranking).

Uso:
    streamlit run app_streamlit.py

Requiere haber corrido antes `python -m rag.ingest` para construir el
índice de Chroma con la knowledge base.
"""

from __future__ import annotations

import streamlit as st

from rag.pipeline import (
    RERANK_SCORE_THRESHOLD,
    RagResult,
    answer_question,
    generate_answer_without_rag,
)
from rag.prompt_loader import latest_version, list_versions

st.set_page_config(page_title="EcoMarket - Asistente RAG", page_icon="🛒", layout="wide")

st.title("🛒 Asistente de atención al cliente de EcoMarket")
st.caption(
    "Taller 2 — RAG con retrieval (ChromaDB) + re-ranking (cross-encoder) + generación (Groq). "
    "Continuación del Taller 1."
)

with st.sidebar:
    st.header("Configuración")

    use_rag = st.toggle("Usar RAG (retrieval + re-ranking)", value=True)

    st.divider()

    top_k = st.slider("Chunks a recuperar (etapa 1, Chroma)", min_value=3, max_value=20, value=10)
    top_n = st.slider("Chunks tras re-ranking (etapa 2, cross-encoder)", min_value=1, max_value=10, value=3)

    st.divider()

    try:
        versions = list_versions()
        prompt_version = st.selectbox(
            "Versión del prompt", options=versions, index=versions.index(latest_version())
        )
    except FileNotFoundError:
        st.error("No hay prompts en rag/prompts/. Crea al menos rag/prompts/v1.toml.")
        prompt_version = None

    st.divider()
    st.caption(
        f"Umbral de fallback (score del re-ranker): {RERANK_SCORE_THRESHOLD}. "
        "Si el mejor chunk queda por debajo, el asistente admite que no sabe "
        "en vez de inventar una respuesta."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Escribe tu pregunta como cliente de EcoMarket...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            if use_rag:
                result: RagResult = answer_question(
                    question,
                    top_k_retrieval=top_k,
                    top_n_rerank=top_n,
                    prompt_version=prompt_version,
                )
                answer = result.answer

                if result.used_fallback:
                    st.warning("El asistente no encontró suficiente evidencia y activó el fallback.")

                st.markdown(answer)

                with st.expander("🔍 Ver pipeline RAG (transparencia)"):
                    st.markdown(f"**Versión de prompt usada:** v{result.prompt_version}")

                    st.markdown("**Etapa 1 — Recuperados por Chroma (similitud vectorial):**")
                    st.caption("Menor distancia = más similar.")
                    for i, chunk in enumerate(result.retrieved_chunks, start=1):
                        st.text(
                            f"{i}. [{chunk.source}] distancia={chunk.retrieval_score:.4f} — {chunk.content[:120]}..."
                        )

                    st.markdown("**Etapa 2 — Tras re-ranking (cross-encoder):**")
                    st.caption("Mayor score = más relevante para la pregunta.")
                    if result.reranked_chunks:
                        for i, chunk in enumerate(result.reranked_chunks, start=1):
                            st.text(
                                f"{i}. [{chunk.source}] score={chunk.rerank_score:.4f} — {chunk.content[:120]}..."
                            )
                    else:
                        st.text("(sin chunks: no se encontró nada en la base de conocimiento)")
            else:
                answer = generate_answer_without_rag(question)
                st.markdown(answer)
                st.caption(
                    "⚠️ RAG desactivado: esta respuesta viene solo del conocimiento general del "
                    "modelo, sin consultar la base de conocimiento de EcoMarket. Compárala con "
                    "el modo RAG activado para ver el efecto del grounding."
                )

    st.session_state.messages.append({"role": "assistant", "content": answer})
