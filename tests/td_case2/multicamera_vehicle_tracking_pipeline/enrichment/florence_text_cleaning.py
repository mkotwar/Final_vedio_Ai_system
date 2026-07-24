from __future__ import annotations

import re


_SPECIAL_TOKEN_PATTERN = re.compile(r"</?(?:s|pad|unk|bos|eos)>", flags=re.IGNORECASE)


def clean_florence_text(raw_output: str) -> str:
    text = str(raw_output or "")
    cleaned = _SPECIAL_TOKEN_PATTERN.sub(" ", text)
    cleaned = cleaned.replace("\r", "\n")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()
