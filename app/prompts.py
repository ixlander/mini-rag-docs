from __future__ import annotations

from typing import List, Dict


SYSTEM_PROMPT = """You are a documentation QA assistant.

LANGUAGE RULE (STRICT, NON-NEGOTIABLE):
- Detect the language of the user's QUESTION.
- Always answer in THE SAME language as the question.
- If the question is in Russian, answer in Russian.
- If the question is in English, answer in English.
- Never answer in Chinese unless the question itself is in Chinese.

GROUNDING RULES (STRICT):
- Use ONLY the information from the provided CONTEXT.
- Do NOT use any external knowledge.
- If the answer is not explicitly stated in the CONTEXT, reply exactly:
  "I couldn't find this in the documentation."
- Do NOT guess, infer, or paraphrase beyond the context.

CITATIONS RULE (STRICT):
- citations must be a JSON array of chunk_id strings.
- Each citation MUST be an exact chunk_id from the CONTEXT.
- Example of a valid citation:
  "docs/user_management.md::chunk0001"
- Do NOT include titles, section names, indices like [1], or any extra text.

OUTPUT FORMAT (STRICT):
Return a valid JSON object with EXACTLY these keys:
{
  "answer": string,
  "citations": string[],
  "confidence": "low" | "medium" | "high"
}

Do NOT return markdown.
Do NOT return explanations.
Do NOT return any text outside JSON.
"""


def build_context_block(chunks: List[Dict], max_chars: int = 12000) -> str:
    parts: List[str] = []
    total = 0

    for i, c in enumerate(chunks, start=1):
        header = f"[{i}] chunk_id={c['chunk_id']} | title={c.get('title','')}"
        body = (c.get("text") or "").strip()

        block = f"{header}\n{body}\n"
        if total + len(block) > max_chars:
            break

        parts.append(block)
        total += len(block)

    return "\n".join(parts).strip()


def build_user_prompt(question: str, context_block: str, conversation_history: str = "") -> str:
    parts: List[str] = []

    if conversation_history:
        parts.append("CONVERSATION HISTORY:")
        parts.append(conversation_history)
        parts.append("")

    parts.append("CONTEXT:")
    parts.append(context_block)
    parts.append("")
    parts.append("QUESTION:")
    parts.append(question)
    parts.append("")
    parts.append("Return JSON only.")

    return "\n".join(parts)


def build_conversation_history(messages: List[Dict], max_chars: int = 2000) -> str:
    """Format recent messages into a compact history block.

    Parameters
    ----------
    messages : list[dict]
        Each dict must have ``role`` and ``content`` keys.
    max_chars : int
        Truncate from the *oldest* side so the most recent context fits.
    """
    if not messages:
        return ""

    lines: List[str] = []
    total = 0

    # Walk newest → oldest, stop when budget is exhausted.
    for msg in reversed(messages):
        role = msg["role"].upper()
        content = (msg.get("content") or "").strip()
        line = f"{role}: {content}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)

    # Reverse back to chronological order.
    lines.reverse()
    return "\n".join(lines)
