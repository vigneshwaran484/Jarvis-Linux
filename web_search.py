"""
web_search.py — give search_web real answers instead of just opening a browser.

Backends, tried in order until one returns something:
  1. Tavily            — real web results; only when TAVILY_API_KEY is set.
  2. DuckDuckGo IA     — instant answers / abstracts, no key.
  3. Wikipedia         — best-matching article summary, no key.

Each backend returns a short plain-text snippet (capped so it fits a small
model's context) or "" on no result / any error. Adapted from
AIproductfactory python-backend/intelligence/web_tools.py.
"""
import requests

from config import TAVILY_API_KEY, WEB_SEARCH_TIMEOUT
from jarvis_log import log

MAX_CHARS = 500
_HEADERS = {"User-Agent": "Jarvis-Linux/1.0 (personal voice assistant)"}


def _clip(text, limit=MAX_CHARS):
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + "…"


def _tavily(query):
    if not TAVILY_API_KEY:
        return ""
    resp = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": 3,
            "include_answer": True,
            "include_raw_content": False,
        },
        timeout=WEB_SEARCH_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("answer"):
        return data["answer"]
    snippets = [
        f"{r.get('title', '')}: {r.get('content', '')}"
        for r in data.get("results", [])[:3]
        if isinstance(r, dict) and r.get("content")
    ]
    return " | ".join(snippets)


def _duckduckgo(query):
    resp = requests.get(
        "https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
        headers=_HEADERS,
        timeout=WEB_SEARCH_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    for key in ("Answer", "AbstractText", "Definition"):
        if data.get(key):
            return str(data[key])
    for topic in data.get("RelatedTopics", []):
        if isinstance(topic, dict) and topic.get("Text"):
            return topic["Text"]
    return ""


def _wikipedia(query):
    resp = requests.get(
        "https://en.wikipedia.org/w/rest.php/v1/search/page",
        params={"q": query, "limit": 1},
        headers=_HEADERS,
        timeout=WEB_SEARCH_TIMEOUT,
    )
    resp.raise_for_status()
    pages = resp.json().get("pages", [])
    if not pages:
        return ""
    key = pages[0]["key"]
    summary = requests.get(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(key, safe='')}",
        headers=_HEADERS,
        timeout=WEB_SEARCH_TIMEOUT,
    )
    summary.raise_for_status()
    extract = summary.json().get("extract", "")
    return f"{pages[0].get('title', key)}: {extract}" if extract else ""


BACKENDS = [("tavily", _tavily), ("duckduckgo", _duckduckgo), ("wikipedia", _wikipedia)]


def lookup(query):
    """Return (snippet, backend_name) for the first backend with an answer,
    or ("", None) if none found anything."""
    query = (query or "").strip()
    if not query:
        return "", None
    for name, backend in BACKENDS:
        try:
            text = backend(query)
        except Exception as e:  # network down, bad JSON, rate limit, ...
            log.warning("web search backend %s failed for %r: %s", name, query, e)
            continue
        if text and text.strip():
            log.info("web search: %s answered %r", name, query)
            return _clip(text), name
    return "", None
