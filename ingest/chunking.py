from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from ingest.parsers import Document, Section

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source_group: str
    title: str
    section: str
    text: str
    url: Optional[str] = None
    doc_type: str = "unknown"
    section_path: str = "Main"
    source_quality: float = 1.0
    ingested_at: str = ""

def estimate_tokens(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    return max(1, len(text) // 4)


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")


def split_into_paragraphs(text: str) -> List[str]:
    text = text.strip().replace("\r\n", "\n").replace("\r", "\n")
    raw_blocks = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    parts: List[str] = []

    for block in raw_blocks:
        lines = [ln.rstrip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue

        if "```" in block:
            parts.append(block)
            continue

        if all(_TABLE_ROW_RE.match(ln) for ln in lines):
            parts.append("\n".join(lines))
            continue

        grouped: List[str] = []
        buf: List[str] = []
        for ln in lines:
            if _HEADING_RE.match(ln):
                if buf:
                    grouped.append(" ".join(buf).strip())
                    buf = []
                grouped.append(ln.strip())
                continue
            if _LIST_ITEM_RE.match(ln):
                if buf:
                    grouped.append(" ".join(buf).strip())
                    buf = []
                grouped.append(ln.strip())
                continue
            buf.append(ln.strip())
        if buf:
            grouped.append(" ".join(buf).strip())

        parts.extend([p for p in grouped if p])

    return parts


def _estimate_source_quality(doc: Document) -> float:
    non_empty = sum(1 for s in doc.sections if (s.text or "").strip())
    if non_empty == 0:
        return 0.0
    total_chars = sum(len((s.text or "").strip()) for s in doc.sections)
    avg_len = total_chars / non_empty
    score = 0.4
    if non_empty >= 3:
        score += 0.3
    if avg_len >= 250:
        score += 0.2
    if doc.doc_type in {"md", "html", "htm"}:
        score += 0.1
    return min(1.0, max(0.0, score))


def take_overlap_tail(prev_text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0:
        return ""
    words = prev_text.split()
    if not words:
        return ""
    tail = words[-overlap_tokens:]
    return " ".join(tail)


def chunk_paragraphs(
    paragraphs: List[str],
    max_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    chunks: List[str] = []
    cur_parts: List[str] = []
    cur_tokens = 0

    def flush():
        nonlocal cur_parts, cur_tokens
        if not cur_parts:
            return
        chunk_text = "\n\n".join(cur_parts).strip()
        if chunk_text:
            chunks.append(chunk_text)
        cur_parts = []
        cur_tokens = 0

    for p in paragraphs:
        p_tokens = estimate_tokens(p)

        if p_tokens > max_tokens:
            flush()
            sents = _SENT_SPLIT_RE.split(p)
            buf: List[str] = []
            buf_tokens = 0
            for s in sents:
                s = s.strip()
                if not s:
                    continue
                st = estimate_tokens(s)
                if buf and (buf_tokens + st > max_tokens):
                    chunks.append(" ".join(buf).strip())
                    tail = take_overlap_tail(chunks[-1], overlap_tokens)
                    buf = [tail] if tail else []
                    buf_tokens = estimate_tokens(" ".join(buf)) if buf else 0
                buf.append(s)
                buf_tokens += st
            if buf:
                chunks.append(" ".join(buf).strip())
            continue

        if cur_parts and (cur_tokens + p_tokens > max_tokens):
            flush()
            if chunks:
                tail = take_overlap_tail(chunks[-1], overlap_tokens)
                if tail:
                    cur_parts = [tail]
                    cur_tokens = estimate_tokens(tail)

        cur_parts.append(p)
        cur_tokens += p_tokens

    flush()
    return chunks

def chunk_document(
    doc: Document,
    max_tokens: int = 550,
    overlap_tokens: int = 80,
) -> List[Chunk]:
    out: List[Chunk] = []
    counter = 0
    source_quality = _estimate_source_quality(doc)
    ingested_at = doc.ingested_at or datetime.now(timezone.utc).isoformat()

    for sec in doc.sections:
        section_title = sec.title.strip() or "Main"
        section_path = (getattr(sec, "path", "") or section_title).strip()
        text = (sec.text or "").strip()
        if not text:
            continue

        paragraphs = split_into_paragraphs(text)
        texts = chunk_paragraphs(paragraphs, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

        for t in texts:
            counter += 1
            chunk_id = f"{doc.doc_id}::chunk{counter:04d}"
            out.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    source_group=doc.source_group,
                    title=doc.title,
                    section=section_title,
                    text=t.strip(),
                    url=doc.url,
                    doc_type=doc.doc_type or "unknown",
                    section_path=section_path,
                    source_quality=source_quality,
                    ingested_at=ingested_at,
                )
            )

    return out


def chunk_corpus(docs: List[Document], max_tokens: int = 550, overlap_tokens: int = 80) -> List[Chunk]:
    chunks: List[Chunk] = []
    for d in docs:
        chunks.extend(chunk_document(d, max_tokens=max_tokens, overlap_tokens=overlap_tokens))
    return chunks
