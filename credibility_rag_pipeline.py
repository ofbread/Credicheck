#!/usr/bin/env python3
"""
Core credibility scoring library.

Provides CredibilityScorer (CrediGraph-based domain scoring with caching),
url_to_domain helper, and key constants used across the pipeline.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Callable
from urllib.parse import urlparse

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Constants ---
CREDIGRAPH_TOKEN = os.environ.get("CREDIGRAPH_TOKEN", "")
DEFAULT_CREDIBILITY_SCORE = 0.35

SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
SERPER_ENDPOINT = "https://google.serper.dev/search"
BLOCKED_DOMAINS = {"politifact.com", "snopes.com"}


# --- Credibility Scorer ---
class CredibilityScorer:
    """CrediGraph-based domain credibility scorer with caching.

    Uses query_internal (inference scores) for continuous scores, which has
    broader coverage than query_GT (ground truth from DQR, ~11.5K domains).
    """

    def __init__(self, token: str, default_score: float = DEFAULT_CREDIBILITY_SCORE):
        self.token = token
        self.default_score = default_score
        self.cache: dict[str, float] = {}
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from credigraph import CrediGraphClient
                self._client = CrediGraphClient(token=self.token)
            except ImportError:
                logger.warning("CrediGraph not installed, using default scores")
        return self._client

    def get_score(self, domain: str) -> float:
        if not domain:
            return self.default_score
        domain = domain.lower().strip()
        if domain.startswith("www."):
            domain = domain[4:]
        if domain in self.cache:
            return self.cache[domain]

        client = self._get_client()
        if client is None:
            self.cache[domain] = self.default_score
            return self.default_score

        try:
            results = client.query_internal_batch([domain])
            if isinstance(results, list) and results:
                result = results[0]
                if isinstance(result, dict):
                    score = result.get("credibility_level")
                    if score is not None:
                        self.cache[domain] = float(score)
                        return float(score)
        except Exception as e:
            logger.debug(f"CrediGraph error for {domain}: {e}")

        self.cache[domain] = self.default_score
        return self.default_score

    def prefetch_scores(
        self,
        domains: list[str],
        batch_size: int = 50,
        progress_callback: Optional[Callable[[int, int, list[str], bool, str], None]] = None,
    ) -> None:
        """Prefetch credibility scores for many domains in batch API calls."""
        to_fetch: list[str] = []
        seen: set[str] = set()
        for d in domains:
            if not d:
                continue
            d = d.lower().strip()
            if d.startswith("www."):
                d = d[4:]
            if d not in seen and d not in self.cache:
                seen.add(d)
                to_fetch.append(d)
        if not to_fetch:
            return
        client = self._get_client()
        if client is None:
            for d in to_fetch:
                self.cache[d] = self.default_score
            return

        batch_size = max(1, min(batch_size, 100))
        total_batches = (len(to_fetch) + batch_size - 1) // batch_size
        for start in range(0, len(to_fetch), batch_size):
            batch_num = (start // batch_size) + 1
            chunk = to_fetch[start : start + batch_size]
            results = None
            try:
                results = client.query_internal_batch(chunk)
            except Exception as e:
                logger.debug(f"CrediGraph batch chunk failed: {e}")
                results = None
            if results is None or not isinstance(results, list):
                for d in chunk:
                    _ = self.get_score(d)
                if progress_callback:
                    progress_callback(batch_num, total_batches, chunk, False, "fallback per-domain")
                continue
            for i, d in enumerate(chunk):
                if i < len(results):
                    r = results[i]
                    score = r.get("credibility_level") if isinstance(r, dict) else None
                    self.cache[d] = float(score) if score is not None else self.default_score
                else:
                    self.cache[d] = self.default_score
            if progress_callback:
                progress_callback(batch_num, total_batches, chunk, True, f"{len(results)} scores")


# --- Web Search ---
def serper_search(query: str, api_key: str = SERPER_API_KEY, max_results: int = 10) -> list[dict]:
    """Search via Serper (Google) API. Returns list of result dicts."""
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": max_results}
    try:
        resp = requests.post(SERPER_ENDPOINT, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Serper search failed: {e}")
        return []

    results = []
    for i, item in enumerate(data.get("organic", [])):
        url = item.get("link", "")
        domain = url_to_domain(url)
        if domain in BLOCKED_DOMAINS:
            continue
        results.append({
            "title": item.get("title", ""),
            "url": url,
            "snippet": item.get("snippet", ""),
            "description": item.get("snippet", ""),
            "search_rank": len(results),
        })
        if len(results) >= max_results:
            break
    return results



# --- URL helpers ---
def url_to_domain(url: str) -> str:
    """Extract domain from URL."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().strip()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""
