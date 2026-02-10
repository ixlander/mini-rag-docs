from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict


SYSTEM_PROMPT = """You are a documentation QA assistant.
Rules:
- Answer ONLY using the provided CONTEXT.
- If the answer is not present in the CONTEXT, say exactly: "I couldn't find this in the documentation."
- Be concise and accurate.
- Always include citations: list of chunk_ids that support the answer.
Return JSON with keys: answer, citations, confidence.
Confidence must be one of: low, medium, high.
"""


def build_context_block(chunks: List[Dict], max_chars: int = 12000) -> str:
    """
    Build a context block from top chunks.
    chunks: list of dicts with keys: chunk_id, title, section, source_group, text
    """
    parts = []
    total = 0
    for i, c in enumerate(chunks, start=1):
        header = f"[{i}] chunk_id={c['chunk_id']} | source={c.get('source_group','')} | title={c.get('title','')} | section={c.get('section','')}"
        body = (c.get("text") or "").strip()
        block = header + "\n" + body + "\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts).strip()


def build_user_prompt(question: str, context_block: str) -> str:
    return f"""CONTEXT:
{context_block}

QUESTION:
{question}

Return JSON only.
"""
