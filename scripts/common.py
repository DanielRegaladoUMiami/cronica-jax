"""Shared utilities for all scrapers: HTTP politeness, schema, hashing."""
from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import requests

logger = logging.getLogger(__name__)

USER_AGENT = (
    "cronica-jax/0.0.1 (research; https://github.com/DanielRegaladoUMiami/cronica-jax; "
    "contact: dxr1491@miami.edu)"
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"


@dataclass
class Cronica:
    """Unified schema for one football match report."""

    liga: str            # e.g. "primera_nacional_arg", "liga_expansion_mx"
    temporada: str       # e.g. "2024", "2024-2025"
    fecha: str           # ISO-8601 "YYYY-MM-DD" or "" if unknown
    local: str           # home team, may be "" if not extracted
    visitante: str       # away team, may be "" if not extracted
    resultado: str       # e.g. "2-1" or "" if unknown
    fuente: str          # "wikinoticias" | "wikipedia" | "ole" | "promiedos" | "record" | ...
    url: str
    titulo: str
    cronica: str         # the narrative text itself
    licencia: str        # "CC-BY-SA-4.0" | "fair-use-research"
    n_palabras: int = 0
    content_hash: str = ""

    def __post_init__(self) -> None:
        self.cronica = clean_text(self.cronica)
        self.n_palabras = count_words(self.cronica)
        self.content_hash = hash_content(self.cronica)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- text cleaning ----------

_WS_RE = re.compile(r"\s+")
_BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*(Foto|Imagen|Video|Crédito):.*$", re.M),
    re.compile(r"^\s*Lee también:.*$", re.M | re.I),
    re.compile(r"^\s*Te puede interesar:.*$", re.M | re.I),
    re.compile(r"^\s*Suscríbete .*$", re.M | re.I),
    re.compile(r"^\s*Compartir en .*$", re.M | re.I),
    re.compile(r"^\s*Sigue a .*$", re.M | re.I),
    re.compile(r"\[\s*\d+\s*\]"),         # wiki refs [1], [12]
    re.compile(r"\(\s*editar\s*\)", re.I),
]


def clean_text(text: str) -> str:
    if not text:
        return ""
    for pat in _BOILERPLATE_PATTERNS:
        text = pat.sub("", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def count_words(text: str) -> int:
    return len([w for w in text.split() if w])


def hash_content(text: str) -> str:
    """Content hash for exact-dedup."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------- HTTP politeness ----------


class PoliteSession:
    """requests Session with rate limiting and retries.

    Use ONE PoliteSession per host to keep delays correct.
    """

    def __init__(self, delay_seconds: float = 1.5, max_retries: int = 3) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.delay = delay_seconds
        self.max_retries = max_retries
        self._last_request: float = 0.0

    def get(self, url: str, **kwargs) -> requests.Response | None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, timeout=30, **kwargs)
                self._last_request = time.monotonic()
                if resp.status_code == 200:
                    return resp
                if resp.status_code in (429, 503):
                    backoff = 2**attempt
                    logger.warning("Rate-limited at %s; sleeping %ss", url, backoff)
                    time.sleep(backoff)
                    continue
                logger.warning("HTTP %s for %s", resp.status_code, url)
                return None
            except requests.RequestException as e:
                logger.warning("Error GET %s (attempt %d): %s", url, attempt, e)
                time.sleep(2**attempt)
        return None


# ---------- IO ----------


def write_jsonl(records: Iterator[Cronica], path: Path) -> int:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    logger.info("Wrote %d records to %s", n, path)
    return n


def read_jsonl(path: Path) -> Iterator[dict]:
    import json

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
