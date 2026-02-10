from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from html.parser import HTMLParser


@dataclass(frozen=True)
class Section:
    title: str
    text: str


@dataclass(frozen=True)
class Document:
    doc_id: str
    source_path: str
    source_group: str  
    title: str
    sections: List[Section]
    url: Optional[str] = None


_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _WHITESPACE_RE.sub(" ", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = _BLANK_LINES_RE.sub("\n\n", s)
    return s.strip()


def _make_doc_id(path: Path, root_dir: Path) -> str:
    rel = path.resolve().relative_to(root_dir.resolve())
    return str(rel).replace("\\", "/")


def _guess_group(path: Path, root_dir: Path) -> str:
    rel = path.resolve().relative_to(root_dir.resolve()).parts
    if len(rel) >= 3 and rel[0] == "data" and rel[1] == "raw":
        return rel[2]
    return "unknown"


def _read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")

