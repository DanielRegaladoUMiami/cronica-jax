"""Scrape football match reports from sports news sites (Olé, Promiedos, Récord).

These are NOT CC-licensed — they are scraped under fair-use for non-commercial
research. The scraper:
- respects robots.txt
- rate-limits aggressively (1 request per 2 seconds per host)
- identifies itself in User-Agent
- stores only what is needed: title, URL, body text + metadata
- redistribution on HF Hub will use the "fair-use-research" license tag
  and provide attribution + opt-out instructions in the data card.

Sources implemented:
  - ole.com.ar (Ascenso section)
  - promiedos.com.ar (Primera Nacional resumens)
  - record.com.mx (Liga de Expansión MX section)

Usage:
    python -m scripts.scrape_news --source ole --out data/raw/ole.jsonl
    python -m scripts.scrape_news --source promiedos --out data/raw/promiedos.jsonl
    python -m scripts.scrape_news --source record --out data/raw/record.jsonl
"""
from __future__ import annotations

import argparse
import logging
import re
import urllib.parse as up
import urllib.robotparser as rp
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup

from scripts.common import (
    Cronica,
    PoliteSession,
    RAW_DIR,
    USER_AGENT,
    write_jsonl,
)

logger = logging.getLogger(__name__)


def load_robots(host: str) -> rp.RobotFileParser:
    parser = rp.RobotFileParser()
    parser.set_url(f"{host}/robots.txt")
    try:
        parser.read()
    except Exception as e:
        logger.warning("Failed to read robots.txt for %s: %s", host, e)
    return parser


def can_fetch(parser: rp.RobotFileParser, url: str) -> bool:
    try:
        return parser.can_fetch(USER_AGENT, url)
    except Exception:
        return True  # permissive on parser failure; we still rate-limit


# ---------- generic article extraction ----------

_MIN_PARAGRAPH_CHARS = 40
_SCORE_RE = re.compile(r"\b(\d{1,2})\s*[-–:]\s*(\d{1,2})\b")


def extract_article_text(html: str) -> tuple[str, str]:
    """Return (title, body_text) from raw HTML using generic heuristics."""
    soup = BeautifulSoup(html, "html.parser")

    # remove obvious non-content
    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                     "form", "noscript", "iframe", "figure"]):
        tag.decompose()
    for tag in soup.find_all(attrs={"class": re.compile(
            r"(comment|share|related|recommend|ad-|advert|newsletter|tag)",
            re.I)}):
        tag.decompose()

    title = ""
    for sel in ("h1", "meta[property='og:title']", "title"):
        t = soup.select_one(sel)
        if t:
            title = (t.get("content") or t.get_text() or "").strip()
            if title:
                break

    article = soup.find("article") or soup.find("main") or soup.body
    if article is None:
        return title, ""

    paragraphs: list[str] = []
    for p in article.find_all(["p", "h2", "h3"]):
        text = p.get_text(" ", strip=True)
        if len(text) >= _MIN_PARAGRAPH_CHARS:
            paragraphs.append(text)
    body = "\n".join(paragraphs)
    return title, body


def extract_score(text: str) -> str:
    m = _SCORE_RE.search(text[:500])
    return f"{m.group(1)}-{m.group(2)}" if m else ""


# ---------- source: Olé (Ascenso) ----------


def scrape_ole(sess: PoliteSession, max_articles: int = 800) -> Iterable[Cronica]:
    """Scrape Olé Ascenso section.

    Strategy: paginated index → article URLs → fetch each.
    """
    host = "https://www.ole.com.ar"
    robots = load_robots(host)
    index_paths = [
        "/ascenso",
        "/ascenso/primera-nacional",
        "/ascenso/primera-b",
        "/ascenso/primera-c",
    ]
    seen_urls: set[str] = set()
    n_yield = 0
    for path in index_paths:
        for page in range(1, 11):
            url = f"{host}{path}?page={page}"
            if not can_fetch(robots, url):
                logger.warning("Disallowed by robots: %s", url)
                continue
            resp = sess.get(url)
            if not resp:
                break
            soup = BeautifulSoup(resp.text, "html.parser")
            links = {
                up.urljoin(host, a["href"])
                for a in soup.find_all("a", href=True)
                if "/ascenso/" in a["href"]
                and a["href"].endswith(".html")
            }
            new = links - seen_urls
            if not new:
                break
            for art_url in new:
                if n_yield >= max_articles:
                    return
                if not can_fetch(robots, art_url):
                    continue
                ar = sess.get(art_url)
                if not ar:
                    continue
                title, body = extract_article_text(ar.text)
                if not body or len(body.split()) < 120:
                    continue
                score = extract_score(body)
                liga = (
                    "primera_nacional_arg" if "primera-nacional" in art_url
                    else "primera_b_arg" if "primera-b" in art_url
                    else "primera_c_arg" if "primera-c" in art_url
                    else "ascenso_arg"
                )
                seen_urls.add(art_url)
                n_yield += 1
                yield Cronica(
                    liga=liga,
                    temporada="",
                    fecha="",
                    local="",
                    visitante="",
                    resultado=score,
                    fuente="ole",
                    url=art_url,
                    titulo=title,
                    cronica=body,
                    licencia="fair-use-research",
                )


# ---------- source: Promiedos (Primera Nacional resúmenes) ----------


def scrape_promiedos(sess: PoliteSession, max_articles: int = 600) -> Iterable[Cronica]:
    host = "https://www.promiedos.com.ar"
    robots = load_robots(host)
    # Promiedos uses tournament IDs; we crawl the news/resumen section.
    seed = f"{host}/noticias/"
    if not can_fetch(robots, seed):
        logger.warning("Promiedos disallows scraping; skipping.")
        return
    resp = sess.get(seed)
    if not resp:
        return
    soup = BeautifulSoup(resp.text, "html.parser")
    article_urls = {
        up.urljoin(host, a["href"])
        for a in soup.find_all("a", href=True)
        if "/noticia/" in a["href"]
    }
    n_yield = 0
    for art_url in article_urls:
        if n_yield >= max_articles:
            return
        if not can_fetch(robots, art_url):
            continue
        ar = sess.get(art_url)
        if not ar:
            continue
        title, body = extract_article_text(ar.text)
        if not body or len(body.split()) < 100:
            continue
        score = extract_score(body)
        n_yield += 1
        yield Cronica(
            liga="primera_nacional_arg",
            temporada="",
            fecha="",
            local="",
            visitante="",
            resultado=score,
            fuente="promiedos",
            url=art_url,
            titulo=title,
            cronica=body,
            licencia="fair-use-research",
        )


# ---------- source: Récord (Liga de Expansión MX) ----------


def scrape_record(sess: PoliteSession, max_articles: int = 600) -> Iterable[Cronica]:
    host = "https://www.record.com.mx"
    robots = load_robots(host)
    index_paths = ["/futbol-liga-de-expansion-mx", "/futbol/ascenso-mx"]
    seen_urls: set[str] = set()
    n_yield = 0
    for path in index_paths:
        for page in range(0, 10):
            url = f"{host}{path}" + (f"?page={page}" if page else "")
            if not can_fetch(robots, url):
                continue
            resp = sess.get(url)
            if not resp:
                break
            soup = BeautifulSoup(resp.text, "html.parser")
            links = {
                up.urljoin(host, a["href"])
                for a in soup.find_all("a", href=True)
                if any(k in a["href"] for k in ("liga-de-expansion", "ascenso-mx"))
                and a["href"].count("/") >= 4
            }
            new = links - seen_urls
            if not new:
                break
            for art_url in new:
                if n_yield >= max_articles:
                    return
                if not can_fetch(robots, art_url):
                    continue
                ar = sess.get(art_url)
                if not ar:
                    continue
                title, body = extract_article_text(ar.text)
                if not body or len(body.split()) < 120:
                    continue
                score = extract_score(body)
                seen_urls.add(art_url)
                n_yield += 1
                yield Cronica(
                    liga="liga_expansion_mx",
                    temporada="",
                    fecha="",
                    local="",
                    visitante="",
                    resultado=score,
                    fuente="record",
                    url=art_url,
                    titulo=title,
                    cronica=body,
                    licencia="fair-use-research",
                )


SOURCES = {
    "ole": scrape_ole,
    "promiedos": scrape_promiedos,
    "record": scrape_record,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=sorted(SOURCES))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max", type=int, default=600)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out = args.out or RAW_DIR / f"{args.source}.jsonl"
    sess = PoliteSession(delay_seconds=2.0)  # 2s between requests
    scraper = SOURCES[args.source]
    write_jsonl(scraper(sess, max_articles=args.max), out)


if __name__ == "__main__":
    main()
