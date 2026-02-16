from __future__ import annotations

import re
from dataclasses import dataclass
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

def estimate_tokens(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    return max(1, len(text) // 4)


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_into_paragraphs(text: str) -> List[str]:
    text = text.strip().replace("\r\n", "\n").replace("\r", "\n")
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts


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

    for sec in doc.sections:
        section_title = sec.title.strip() or "Main"
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
                )
            )

    return out


def chunk_corpus(docs: List[Document], max_tokens: int = 550, overlap_tokens: int = 80) -> List[Chunk]:
    chunks: List[Chunk] = []
    for d in docs:
        chunks.extend(chunk_document(d, max_tokens=max_tokens, overlap_tokens=overlap_tokens))
    return chunks