from __future__ import annotations

import re
from typing import List

EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s()\-]{8,}\d)(?!\d)")
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


def tokenize_text(text: str) -> List[str]:
    return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if len(t) > 1]


def redact_pii(text: str) -> str:
    s = EMAIL_RE.sub("[REDACTED_EMAIL]", text or "")
    s = PHONE_RE.sub("[REDACTED_PHONE]", s)
    s = CARD_RE.sub(
        lambda m: "[REDACTED_CARD]" if _looks_like_card(m.group(0)) else m.group(0),
        s,
    )
    return s


def _looks_like_card(candidate: str) -> bool:
    digits = "".join(ch for ch in candidate if ch.isdigit())
    if not (13 <= len(digits) <= 19):
        return False
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0
