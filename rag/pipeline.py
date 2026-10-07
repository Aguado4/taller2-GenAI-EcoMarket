"""Pipeline RAG de dos etapas: retrieval (Chroma) -> re-ranking (CrossEncoder) -> generación (Groq).

Reutiliza el patrón del Taller 1 (taller 1 genAI/app.py) para la conexión
con el LLM: mismo proveedor (Groq), misma convención de .env/.env.local,
mismo arreglo de encoding UTF-8 para Windows, y el mismo esquema de
prompts versionados (rag/prompts/vN.toml).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import CrossEncoder

from rag.ingest import load_vectorstore
from rag.prompt_loader import load_prompt

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env.local")
load_dotenv(BASE_DIR / ".env")

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Umbral del score del cross-encoder por debajo del cual se considera que
# ningún chunk recuperado es realmente relevante para la pregunta, y se
# activa el fallback de "no tengo información" en vez de alucinar.
# Calibrado empíricamente probando preguntas dentro y fuera de dominio
# (ver respuestas/fase3_ingenieria_rag.md).
RERANK_SCORE_THRESHOLD = -3.0

FALLBACK_MESSAGE = (
    "No tengo información suficiente en la base de conocimiento de EcoMarket "
    "para responder esto con seguridad. Te recomiendo que un agente humano "
    "revise tu caso directamente."
)

_cross_encoder: CrossEncoder | None = None


def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "Falta GROQ_API_KEY.\n"
            "1. Crea una key gratis en https://console.groq.com/keys\n"
            "2. Copia .env.example a .env (o .env.local) y pega tu key."
        )
    return Groq(api_key=api_key)


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    return _cross_encoder


@dataclass
class RetrievedChunk:
    content: str
    source: str
    retrieval_score: float  # distancia L2 de Chroma (menor = más similar)
    rerank_score: float | None = None  # score del cross-encoder (mayor = más relevante)


def retrieve(query: str, top_k: int = 10) -> list[RetrievedChunk]:
    """Etapa 1: recuperación rápida por similitud vectorial en Chroma."""
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    return [
        RetrievedChunk(
            content=doc.page_content,
            source=doc.metadata.get("source", "desconocido"),
            retrieval_score=float(score),
        )
        for doc, score in results
    ]


def rerank(query: str, chunks: list[RetrievedChunk], top_n: int = 3) -> list[RetrievedChunk]:
    """Etapa 2: re-ranking preciso con un cross-encoder sobre los candidatos de la etapa 1."""
    if not chunks:
        return []

    cross_encoder = get_cross_encoder()
    pairs = [[query, chunk.content] for chunk in chunks]
    scores = cross_encoder.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk.rerank_score = float(score)

    reranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)
    return reranked[:top_n]


def build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[Fragmento {i} - fuente: {chunk.source}]\n{chunk.content}")
    return "\n\n---\n\n".join(parts)


def _fill_template(template: str, **values: str) -> str:
    result = template
    for key, value in values.items():
        result = result.replace("{" + key + "}", value)
    return result


def generate_answer(question: str, context: str, prompt_version: int | None = None) -> str:
    prompt_data = load_prompt(prompt_version)
    role_prompt = prompt_data["prompts"].get("role_prompt", "").strip()
    user_message = _fill_template(
        prompt_data["prompts"]["instruction_prompt"],
        question=question,
        context=context,
    )

    messages = []
    if role_prompt:
        messages.append({"role": "system", "content": role_prompt})
    messages.append({"role": "user", "content": user_message})

    response = get_client().chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content


def generate_answer_without_rag(question: str) -> str:
    """Baseline sin RAG: el modelo responde solo con su conocimiento general,
    sin ningún fragmento de la knowledge base de EcoMarket. Sirve para
    comparar (toggle 'RAG on/off' en la UI) y evidenciar el efecto del
    grounding, igual que la comparación v1/v2 del Taller 1."""
    messages = [
        {
            "role": "system",
            "content": "Actúa como un agente de servicio al cliente de EcoMarket, una tienda de e-commerce de productos sostenibles.",
        },
        {"role": "user", "content": question},
    ]
    response = get_client().chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content


@dataclass
class RagResult:
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    reranked_chunks: list[RetrievedChunk]
    used_fallback: bool
    prompt_version: int


def answer_question(
    question: str,
    top_k_retrieval: int = 10,
    top_n_rerank: int = 3,
    prompt_version: int | None = None,
) -> RagResult:
    """Ejecuta el pipeline completo: retrieval -> re-ranking -> (fallback o generación)."""
    retrieved = retrieve(question, top_k=top_k_retrieval)
    reranked = rerank(question, retrieved, top_n=top_n_rerank)

    best_score = reranked[0].rerank_score if reranked else float("-inf")
    resolved_version = load_prompt(prompt_version)["_resolved_version"]

    if not reranked or best_score < RERANK_SCORE_THRESHOLD:
        return RagResult(
            answer=FALLBACK_MESSAGE,
            retrieved_chunks=retrieved,
            reranked_chunks=reranked,
            used_fallback=True,
            prompt_version=resolved_version,
        )

    context = build_context(reranked)
    answer = generate_answer(question, context, prompt_version=prompt_version)
    return RagResult(
        answer=answer,
        retrieved_chunks=retrieved,
        reranked_chunks=reranked,
        used_fallback=False,
        prompt_version=resolved_version,
    )
