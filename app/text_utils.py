"""Text extraction and splitting, shared by the API, the UI and the evaluation script."""
from __future__ import annotations

import re
from io import BytesIO

from pypdf import PdfReader

_LEADING_BULLETS = re.compile(r"^[\s\-•*·▪►–]+")
_SPACES = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")


def extract_pdf_text(data: bytes) -> str:
    """Return the text of every page in a PDF, one line per line of the page."""
    reader = PdfReader(BytesIO(data))
    # extract_text() can return None for pages with no text layer
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _clean(line: str) -> str:
    return _SPACES.sub(" ", _LEADING_BULLETS.sub("", line)).strip()


def split_lines(text: str, min_words: int = 3) -> list[str]:
    """Split text into lines, re-joining bullets that PDF extraction wrapped.

    A line starting with a lowercase letter is treated as the continuation
    of the previous line, which is how wrapped bullet points usually look.
    """
    merged: list[str] = []
    for raw in text.splitlines():
        line = _clean(raw)
        if not line:
            continue
        if merged and line[0].islower():
            merged[-1] = f"{merged[-1]} {line}"
        else:
            merged.append(line)
    return [line for line in merged if len(line.split()) >= min_words]


def split_requirements(job_text: str, min_words: int = 4) -> list[str]:
    """Break a job description into sentence-sized requirement candidates."""
    requirements: list[str] = []
    for line in split_lines(job_text, min_words=1):
        for sentence in _SENTENCE_END.split(line):
            if len(sentence.split()) >= min_words:
                requirements.append(sentence)
    return requirements