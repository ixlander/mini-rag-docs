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


_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def parse_markdown(md: str) -> Tuple[str, List[Section]]:
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    lines = md.split("\n")

    headings: List[Tuple[int, str, int]] = []
    for i, line in enumerate(lines):
        m = _MD_HEADING_RE.match(line.strip())
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            headings.append((level, title, i))

    if not headings:
        text = _normalize_text(md)
        title = "Untitled"
        return title, [Section(title="Main", text=text)] if text else [Section(title="Main", text="")]

    doc_title = next((t for lvl, t, _ in headings if lvl == 1), headings[0][1]) or "Untitled"

    sections: List[Section] = []
    for idx, (lvl, title, start_i) in enumerate(headings):
        end_i = headings[idx + 1][2] if idx + 1 < len(headings) else len(lines)
        block = "\n".join(lines[start_i + 1 : end_i])
        block = _normalize_text(block)
        sections.append(Section(title=title or "Untitled Section", text=block))

    return doc_title, sections


class _HTMLToSections(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_script = False
        self._in_style = False
        self._capture_heading_level: Optional[int] = None
        self._heading_buf: List[str] = []
        self._text_buf: List[str] = []

        self.sections: List[Section] = []
        self.doc_title: str = "Untitled"

        self._current_title: str = "Main"
        self._current_text_parts: List[str] = []

        self._seen_h1 = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        if t == "script":
            self._in_script = True
        elif t == "style":
            self._in_style = True
        elif t in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._capture_heading_level = int(t[1])
            self._heading_buf = []
        elif t in ("p", "br", "li", "div"):
            self._current_text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t == "script":
            self._in_script = False
        elif t == "style":
            self._in_style = False
        elif t in ("h1", "h2", "h3", "h4", "h5", "h6"):
            heading = _normalize_text(" ".join(self._heading_buf))
            if heading:
                self._finalize_section()

                if t == "h1" and not self._seen_h1:
                    self.doc_title = heading
                    self._seen_h1 = True

                self._current_title = heading
            self._capture_heading_level = None
            self._heading_buf = []

    def handle_data(self, data: str) -> None:
        if self._in_script or self._in_style:
            return
        txt = data.strip()
        if not txt:
            return

        if self._capture_heading_level is not None:
            self._heading_buf.append(txt)
        else:
            self._current_text_parts.append(txt)

    def _finalize_section(self) -> None:
        text = _normalize_text(" ".join(self._current_text_parts))
        if text:
            self.sections.append(Section(title=self._current_title, text=text))
        self._current_text_parts = []

    def close(self) -> None:
        super().close()
        self._finalize_section()
        if not self.doc_title and self.sections:
            self.doc_title = self.sections[0].title

