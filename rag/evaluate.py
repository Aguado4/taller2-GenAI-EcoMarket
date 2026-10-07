"""Batería de pruebas del pipeline RAG: corre preguntas dentro y fuera de
dominio, con y sin RAG, y guarda la evidencia en outputs/evidencia_rag.txt.

Uso:
    python -m rag.evaluate

Requiere haber corrido antes `python -m rag.ingest`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from rag.pipeline import answer_question, generate_answer_without_rag

BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUT_FILE = OUTPUTS_DIR / "evidencia_rag.txt"

# Preguntas "dentro de dominio": la respuesta debería estar en la
# knowledge base (política de devoluciones, catálogo, FAQ).
IN_DOMAIN_QUESTIONS = [
    "¿Puedo devolver un champú sólido si ya abrí el empaque?",
    "¿Cuánto cuesta la mochila de fibras recicladas y se puede devolver?",
    "¿Cuánto tiempo tarda en llegar mi pedido a Cali?",
    "¿Qué métodos de pago acepta EcoMarket?",
    "Compré un snack de frutas deshidratadas y llegó vencido, ¿qué hago?",
]

# Preguntas "fuera de dominio": no hay nada en la knowledge base sobre
# esto, así que el pipeline debería activar el fallback en vez de
# inventar una respuesta.
OUT_OF_DOMAIN_QUESTIONS = [
    "¿Cuál es la capital de Australia?",
    "¿EcoMarket vende laptops o celulares?",
    "¿Cuál es el CEO de EcoMarket y cuánto gana?",
]


def run_battery() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [f"{'=' * 70}\nEvidencia de corrida: {timestamp}\n{'=' * 70}\n"]

    lines.append("\n########## PREGUNTAS DENTRO DE DOMINIO (con RAG) ##########\n")
    for question in IN_DOMAIN_QUESTIONS:
        print(f"[con RAG] {question}")
        result = answer_question(question)
        lines.append(f"\n--- Pregunta: {question} ---")
        lines.append(f"Fallback activado: {result.used_fallback}")
        if result.reranked_chunks:
            best = result.reranked_chunks[0]
            lines.append(f"Mejor chunk (fuente={best.source}, score={best.rerank_score:.3f})")
        lines.append(f"Respuesta:\n{result.answer}\n")

    lines.append("\n########## PREGUNTAS FUERA DE DOMINIO (con RAG, debe activar fallback) ##########\n")
    for question in OUT_OF_DOMAIN_QUESTIONS:
        print(f"[fuera de dominio] {question}")
        result = answer_question(question)
        lines.append(f"\n--- Pregunta: {question} ---")
        lines.append(f"Fallback activado: {result.used_fallback}")
        if result.reranked_chunks:
            best = result.reranked_chunks[0]
            lines.append(f"Mejor chunk (fuente={best.source}, score={best.rerank_score:.3f})")
        lines.append(f"Respuesta:\n{result.answer}\n")

    lines.append("\n########## COMPARACIÓN CON RAG DESACTIVADO (baseline) ##########\n")
    for question in IN_DOMAIN_QUESTIONS[:2]:
        print(f"[sin RAG] {question}")
        answer = generate_answer_without_rag(question)
        lines.append(f"\n--- Pregunta: {question} ---")
        lines.append(f"Respuesta SIN RAG (solo conocimiento general del modelo):\n{answer}\n")

    with OUTPUT_FILE.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nEvidencia agregada a {OUTPUT_FILE}")


if __name__ == "__main__":
    run_battery()
