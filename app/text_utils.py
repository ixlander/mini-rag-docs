from __future__ import annotations

import re
from typing import List

EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\-\s().]{7,}\d)(?!\d)")
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


def tokenize_text(text: str) -> List[str]:
    return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if len(t) > 1]


def redact_pii(text: str) -> str:
    s = EMAIL_RE.sub("[REDACTED_EMAIL]", text or "")
    s = PHONE_RE.sub("[REDACTED_PHONE]", s)
    s = CARD_RE.sub("[REDACTED_CARD]", s)
    return s
