from __future__ import annotations

import csv
import io
import re
from typing import Iterable, List, Sequence


def extract_tagged_value(text: str, label: str) -> str | None:
    pattern = rf"{re.escape(label)}\s*[:：]\s*(.+)"
    match = re.search(pattern, text)
    if match:
        value = match.group(1).strip()
        # Stop at the first line break to avoid spilling into other fields
        return value.splitlines()[0].strip()
    return None


def extract_section(text: str, label: str) -> str:
    pattern = rf"{re.escape(label)}\s*[:：]"
    start = re.search(pattern, text)
    if not start:
        return ""
    section_start = start.end()
    remainder = text[section_start:]
    end = re.search(r"\n[\w\-<>·•가-힣 ]{1,30}\s*[:：]", remainder)
    if end:
        return remainder[: end.start()].strip()
    return remainder.strip()


def sentence_tokenize(text: str) -> List[str]:
    cleaned = text.replace("\r", " ")
    parts = re.split(r"(?<=[.!?。!?])\s+|\n", cleaned)
    sentences = [segment.strip() for segment in parts if segment and segment.strip()]
    return sentences


def clamp(value: int, lower: int = 0, upper: int = 5) -> int:
    return max(lower, min(upper, value))


def to_csv(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(headers)
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue().strip()
