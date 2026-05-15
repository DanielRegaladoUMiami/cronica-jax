"""Scrape football match reports from Wikinoticias ES and Wikipedia ES.

These two sources are CC-BY-SA-4.0 — redistributable on HF Hub.

- Wikinoticias: narrative articles about specific matches (best quality).
- Wikipedia: season articles with embedded narrative paragraphs per matchday.

Usage:
    python -m scripts.scrape_wikimedia --out data/raw/wikimedia.jsonl
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

from scripts.common import Cronica, PoliteSession, RAW_DIR, write_jsonl

logger = logging.getLogger(__name__)

WIKINEWS_API = "https://es.wikinews.org/w/api.php"
WIKIPEDIA_API = "https://es.wikipedia.org/w/api.php"

# Wikinoticias categories with football match coverage in Spanish.
WIKINEWS_CATEGORIES = [
    "Categoría:Fútbol",
    "Categoría:Fútbol_de_Argentina",
    "Categoría:Fútbol_de_México",
    "Categoría:Fútbol_de_Colombia",
    "Categoría:Fútbol_de_Uruguay",
    "Categoría:Fútbol_de_Chile",
    "Categoría:Fútbol_de_Perú",
]

# Wikipedia: season articles for second/lower divisions LATAM.
# These titles use Spanish naming conventions and follow predictable patterns.
WIKIPEDIA_SEED_TITLES = [
    # Primera Nacional Argentina
    *(f"Primera Nacional {y}" for y in range(2018, 2026)),
    *(f"Torneo de la Primera B Nacional {y}" for y in range(2014, 2018)),
    # Liga de Expansión MX (formerly Ascenso MX)
    *(f"Temporada {y}-{y+1} de la Liga de Expansión MX" for y in range(2020, 2026)),
    *(f"Temporada {y}-{y+1} del Ascenso MX" for y in range(2013, 2020)),
    # Primera B Colombia
    *(f"Torneo {tag} {y} (Colombia)" for y in range(2018, 2026) for tag in ("Apertura", "Finalización")),
    # Segunda División Profesional Uruguay
    *(f"Segunda División Profesional de Uruguay {y}" for y in range(2018, 2026)),
    # Primera B Chile
    *(f"Primera B de Chile {y}" for y in range(2018, 2026)),
]


def fetch_wikinews_category_members(
    sess: PoliteSession, category: str, limit: int = 500
) -> list[str]:
    """List article titles in a Wikinoticias category."""
    titles: list[str] = []
    cont: dict = {}
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": str(min(limit, 500)),
            "cmtype": "page",
            "format": "json",
            **cont,
        }
        resp = sess.get(WIKINEWS_API, params=params)
        if not resp:
            break
        data = resp.json()
        for m in data.get("query", {}).get("categorymembers", []):
            titles.append(m["title"])
        if "continue" in data:
            cont = data["continue"]
        else:
            break
        if len(titles) >= limit:
            break
    return titles[:limit]


def fetch_article_text(sess: PoliteSession, api_url: str, title: str) -> tuple[str, str] | None:
    """Fetch plain-text extract of an article. Returns (title, text) or None."""
    params = {
        "action": "query",
        "prop": "extracts|info",
        "titles": title,
        "explaintext": "1",
        "exsectionformat": "plain",
        "inprop": "url",
        "format": "json",
        "redirects": "1",
    }
    resp = sess.get(api_url, params=params)
    if not resp:
        return None
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        if "extract" in page:
            return page.get("title", title), page["extract"]
    return None


# Heuristics to detect if a Wikinoticias article is about a specific match.
_MATCH_HINT = re.compile(
    r"\b(\d+)\s*[\-–:a]\s*(\d+)\b"     # "2-1" "3 a 0" "2:1"
    r"|gan(ó|a)\b|derrot(ó|a)\b|empat(ó|a|aron)\b|venc(ió|e)\b",
    re.I,
)


def is_match_article(text: str) -> bool:
    """Cheap filter: must contain a scoreline AND a result verb."""
    if not text or len(text) < 200:
        return False
    return bool(_MATCH_HINT.search(text[:1500]))


# Parse league/season hints from titles.
_LIGA_PATTERNS = [
    (re.compile(r"Primera Nacional", re.I), "primera_nacional_arg"),
    (re.compile(r"Liga de Expansión MX|Ascenso MX", re.I), "liga_expansion_mx"),
    (re.compile(r"Primera B.*Colombia|\(Colombia\)", re.I), "primera_b_col"),
    (re.compile(r"Segunda División.*Uruguay", re.I), "segunda_uy"),
    (re.compile(r"Primera B.*Chile", re.I), "primera_b_cl"),
]
_SEASON_RE = re.compile(r"(20\d{2}(?:-20\d{2})?|20\d{2}-\d{2})")


def detect_liga_from_title(title: str) -> str:
    for pat, liga in _LIGA_PATTERNS:
        if pat.search(title):
            return liga
    return "wikinews_general"


def detect_season_from_title(title: str) -> str:
    m = _SEASON_RE.search(title)
    return m.group(1) if m else ""


def harvest_wikinews(sess: PoliteSession, limit_per_cat: int = 300):
    """Yield Cronica objects from Wikinoticias football articles."""
    seen_titles: set[str] = set()
    for cat in WIKINEWS_CATEGORIES:
        logger.info("Wikinews category: %s", cat)
        titles = fetch_wikinews_category_members(sess, cat, limit=limit_per_cat)
        for title in titles:
            if title in seen_titles:
                continue
            seen_titles.add(title)
            result = fetch_article_text(sess, WIKINEWS_API, title)
            if not result:
                continue
            real_title, text = result
            if not is_match_article(text):
                continue
            url = f"https://es.wikinews.org/wiki/{real_title.replace(' ', '_')}"
            yield Cronica(
                liga=detect_liga_from_title(real_title),
                temporada=detect_season_from_title(real_title),
                fecha="",
                local="",
                visitante="",
                resultado="",
                fuente="wikinoticias",
                url=url,
                titulo=real_title,
                cronica=text,
                licencia="CC-BY-SA-4.0",
            )


def harvest_wikipedia_seasons(sess: PoliteSession):
    """Yield Cronica objects from Wikipedia season articles (long, less narrative)."""
    for title in WIKIPEDIA_SEED_TITLES:
        result = fetch_article_text(sess, WIKIPEDIA_API, title)
        if not result:
            continue
        real_title, text = result
        # Season articles are long mixes of tables+prose. Keep only if they contain
        # at least 3 scoreline markers (proxy for actual narrative content).
        if len(re.findall(r"\b\d+[\-–]\d+\b", text)) < 3:
            continue
        url = f"https://es.wikipedia.org/wiki/{real_title.replace(' ', '_')}"
        yield Cronica(
            liga=detect_liga_from_title(real_title),
            temporada=detect_season_from_title(real_title),
            fecha="",
            local="",
            visitante="",
            resultado="",
            fuente="wikipedia",
            url=url,
            titulo=real_title,
            cronica=text,
            licencia="CC-BY-SA-4.0",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RAW_DIR / "wikimedia.jsonl")
    parser.add_argument("--limit-per-cat", type=int, default=300)
    parser.add_argument("--skip-wikipedia", action="store_true")
    parser.add_argument("--skip-wikinews", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sess = PoliteSession(delay_seconds=1.0)  # Wikimedia APIs accept ~1 req/sec polite

    def records():
        if not args.skip_wikinews:
            yield from harvest_wikinews(sess, limit_per_cat=args.limit_per_cat)
        if not args.skip_wikipedia:
            yield from harvest_wikipedia_seasons(sess)

    write_jsonl(records(), args.out)


if __name__ == "__main__":
    main()
