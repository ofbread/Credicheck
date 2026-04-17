#!/usr/bin/env python3
"""
SciFact claim verification with multi-hop web search.

Two-phase pipeline per claim:
  Phase A — Evidence gathering (runs ONCE per claim):
    Agentic loop requests web search via tool calls, collecting raw evidence.
    Agent sees evidence in neutral (rank-only) format.

  Phase B — Prediction (runs ONCE per retrieval mode):
    Pool all raw evidence from Phase A.
    Apply retrieval mode selection (no_credibility / reranked / stratified / filtered).
    Single LLM call produces final label: SUPPORTS / REFUTES / NOT_ENOUGH_INFO.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import requests

from credibility_rag_pipeline import (
    CREDIGRAPH_TOKEN, DEFAULT_CREDIBILITY_SCORE, SERPER_API_KEY,
    CredibilityScorer, serper_search,
)
from src.common import f1_stats


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_AGENT_MODEL = "google/gemini-2.0-flash-001"
VALID_LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]

MAX_HOPS = 5
MAX_RESULTS = 20
TOP_K = 10
ALPHA = 0.4
FILTER_THRESHOLD = 0.55


SYSTEM_PROMPT = """You are a scientific claim verification assistant with web search access.
Use multi-hop search to verify a biomedical/scientific claim.

Rules:
1) You may call the search tool multiple times to gather evidence.
2) Use search iteratively when evidence is incomplete or conflicting.
3) Base your final decision only on the provided search evidence.
4) If evidence is insufficient, contradictory without resolution, or tangential to the claim, choose NOT_ENOUGH_INFO rather than guessing.
5) Final output must include exactly one final label from:
   SUPPORTS, REFUTES, NOT_ENOUGH_INFO
6) End with:
   Final Answer: <LABEL>
"""


SEARCH_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for evidence about the claim.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query text"}
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
]


VERDICT_SYSTEM_PROMPT = """You are a claim verification assistant.
Given the claim and the retrieved web evidence below, determine whether the evidence supports, refutes, or provides insufficient information for the claim.

Rules:
1) Base your decision only on the provided evidence.
2) If evidence is insufficient, contradictory without resolution, or tangential to the claim, choose NOT_ENOUGH_INFO.
3) Output exactly one label from: SUPPORTS, REFUTES, NOT_ENOUGH_INFO
4) End with: Final Answer: <LABEL>
"""


@dataclass
class Claim:
    id: str
    text: str
    label: str | None
    dataset: str = "scifact"


def normalize_scifact_label(raw_label: str | None) -> str | None:
    if not raw_label:
        return None
    t = str(raw_label).strip().upper()
    if t == "SUPPORT":
        return "SUPPORTS"
    if t == "CONTRADICT":
        return "REFUTES"
    if t in {"NOT_ENOUGH_INFO", "NOT ENOUGH INFO", "NEI"}:
        return "NOT_ENOUGH_INFO"
    return None


def derive_label_from_scifact_evidence(evidence: Any) -> str:
    # SciFact encodes SUPPORT/CONTRADICT labels inside evidence groups.
    if not isinstance(evidence, dict) or not evidence:
        return "NOT_ENOUGH_INFO"
    for doc_groups in evidence.values():
        if not isinstance(doc_groups, list):
            continue
        for group in doc_groups:
            if not isinstance(group, dict):
                continue
            mapped = normalize_scifact_label(group.get("label"))
            if mapped:
                return mapped
    return "NOT_ENOUGH_INFO"


def load_claims(path: Path, offset: int, limit: int) -> list[Claim]:
    claims: list[Claim] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            claim_id = str(row.get("id", ""))
            text = str(row.get("claim", ""))
            explicit = normalize_scifact_label(row.get("label"))
            label = explicit if explicit in VALID_LABELS else None
            if label is None and "evidence" in row:
                label = derive_label_from_scifact_evidence(row.get("evidence"))
            if label not in VALID_LABELS:
                label = None
            claims.append(Claim(id=claim_id, text=text, label=label))
    if offset > 0:
        claims = claims[offset:]
    if limit > 0:
        claims = claims[:limit]
    return claims


def openrouter_chat(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.2,
    max_tokens: int = 700,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    for attempt in range(3):
        try:
            resp = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=90)
            if resp.status_code >= 500 and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            resp.raise_for_status()
            break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    return resp.json()



def url_to_domain(url: str) -> str:
    if not url:
        return ""
    try:
        d = urlparse(url).netloc.lower().strip()
        return d[4:] if d.startswith("www.") else d
    except Exception:
        return ""


def rank_score(search_rank: int, max_rank: int) -> float:
    max_rank = max(1, max_rank)
    return 1.0 - (search_rank / max_rank)


def apply_retrieval_mode(
    *,
    results: list[dict],
    scorer: CredibilityScorer,
    mode: str,
    alpha: float,
    threshold: float,
    top_k: int,
) -> list[dict]:
    enriched: list[dict] = []
    for r in results:
        domain = url_to_domain(r.get("url", ""))
        rank = int(r.get("search_rank", 0))
        rank_based = rank_score(rank, max_rank=max(len(results), 1))
        if domain:
            cred = scorer.get_score(domain)
        else:
            cred = scorer.default_score
        if mode in {"raw_search_rank", "no_credibility"}:
            combined = rank_based
        else:
            combined = (1.0 - alpha) * rank_based + alpha * cred
        item = dict(r)
        item["domain"] = domain
        item["credibility_score"] = float(cred)
        item["combined_score"] = float(combined)
        enriched.append(item)

    by_url: dict[str, dict] = {}
    for e in enriched:
        url = e.get("url", "")
        prev = by_url.get(url)
        if prev is None or e["combined_score"] > prev["combined_score"]:
            by_url[url] = e
    dedup = list(by_url.values())

    def _enforce_domain_diversity(candidates: list[dict], k: int, max_per_domain: int = 2) -> list[dict]:
        selected: list[dict] = []
        domain_counts: dict[str, int] = {}
        deferred: list[dict] = []
        for c in candidates:
            d = c.get("domain", "")
            if domain_counts.get(d, 0) < max_per_domain:
                selected.append(c)
                domain_counts[d] = domain_counts.get(d, 0) + 1
            else:
                deferred.append(c)
            if len(selected) >= k:
                break
        for c in deferred:
            if len(selected) >= k:
                break
            selected.append(c)
        return selected[:k]

    if mode in {"raw_search_rank", "no_credibility"}:
        selected = sorted(dedup, key=lambda x: x["search_rank"])
    elif mode == "reranked":
        ranked = sorted(dedup, key=lambda x: x["combined_score"], reverse=True)
        selected = _enforce_domain_diversity(ranked, top_k)
    elif mode == "filtered":
        selected = [
            x
            for x in sorted(dedup, key=lambda x: x["combined_score"], reverse=True)
            if x["credibility_score"] >= threshold
        ]
        if not selected:
            selected = sorted(dedup, key=lambda x: x["combined_score"], reverse=True)[: min(3, len(dedup))]
    elif mode == "stratified":
        high = [x for x in dedup if x["credibility_score"] >= 0.6]
        med = [x for x in dedup if 0.45 <= x["credibility_score"] < 0.6]
        low = [x for x in dedup if x["credibility_score"] < 0.45]
        for tier in (high, med, low):
            tier.sort(key=lambda x: x["search_rank"])
        selected = []
        seen_domains: set[str] = set()
        for tier in (high, med, low):
            for r in tier:
                if len(selected) >= top_k:
                    break
                d = r.get("domain", "")
                if d not in seen_domains or sum(1 for s in selected if s.get("domain") == d) < 2:
                    selected.append(r)
                    seen_domains.add(d)
        remaining = [r for r in high + med + low if r not in selected]
        for r in remaining:
            if len(selected) >= top_k:
                break
            selected.append(r)
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}")

    return selected[:top_k]


def format_raw_evidence(selected: list[dict]) -> str:
    """Format selected evidence as raw text blocks without credibility info."""
    if not selected:
        return "No search results."
    blocks = []
    for i, r in enumerate(selected, start=1):
        title = r.get("title", "")
        domain = r.get("domain", "")
        snippet = r.get("snippet", "") or r.get("description", "")
        blocks.append(
            f"[{i}] (Domain: {domain})\n"
            f"    Title: {title}\n"
            f'    "{snippet}"'
        )
    return "\n".join(blocks)



def parse_search_from_text(text: str) -> str | None:
    m = re.search(r"SEARCH:\s*(.+)", text, re.IGNORECASE)
    if not m:
        return None
    q = m.group(1).strip()
    return q if q else None


def parse_final_label(text: str) -> str | None:
    try:
        data = json.loads(text)
        candidate = str(data.get("prediction", data.get("label", ""))).strip().upper().replace(" ", "_")
        if candidate in VALID_LABELS:
            return candidate
    except (json.JSONDecodeError, AttributeError):
        pass
    m = re.search(r"Final Answer:\s*([A-Z_ ]+)", text, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip().upper().replace(" ", "_")
        if candidate in VALID_LABELS:
            return candidate
    return None


def search_evidence_multihop(
    *,
    claim: Claim,
    openrouter_key: str,
    agent_model: str,
    scorer: CredibilityScorer,
    max_hops: int,
    max_results: int,
    top_k: int,
    raw_cache: dict[str, list[dict]],
    search_fn: Any = None,
) -> dict:
    """Phase A: Agentic search loop — gathers evidence only, no prediction."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Claim: {claim.text}\nPlease verify this claim."},
    ]
    searches: list[dict] = []
    _search = search_fn or (lambda q: serper_search(q, max_results=max_results))
    queries_used: list[str] = []

    for hop in range(max_hops):
        resp = openrouter_chat(
            api_key=openrouter_key,
            model=agent_model,
            messages=messages,
            tools=SEARCH_TOOL_SCHEMA,
            temperature=0.2,
            max_tokens=550,
        )
        msg = resp["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        content = msg.get("content") or ""

        if tool_calls:
            messages.append(
                {"role": "assistant", "content": content, "tool_calls": tool_calls}
            )
            for tc in tool_calls:
                fn = tc.get("function", {}).get("name")
                if fn != "web_search":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": json.dumps({"error": f"Unknown tool: {fn}"}),
                    })
                    continue
                args_raw = tc.get("function", {}).get("arguments", "{}")
                try:
                    fn_args = json.loads(args_raw)
                except json.JSONDecodeError:
                    fn_args = {"query": claim.text}
                query = str(fn_args.get("query", claim.text))
                if query in raw_cache:
                    raw = raw_cache[query]
                else:
                    raw = _search(query)
                    raw_cache[query] = raw
                queries_used.append(query)
                neutral_selected = apply_retrieval_mode(
                    results=raw, scorer=scorer, mode="no_credibility",
                    alpha=0.0, threshold=0.0, top_k=top_k,
                )
                summary = format_raw_evidence(neutral_selected)
                searches.append({"hop": hop + 1, "query": query, "raw_results": raw})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": summary,
                })
            continue

        q = parse_search_from_text(content)
        if q:
            if q in raw_cache:
                raw = raw_cache[q]
            else:
                raw = _search(q)
                raw_cache[q] = raw
            queries_used.append(q)
            neutral_selected = apply_retrieval_mode(
                results=raw, scorer=scorer, mode="no_credibility",
                alpha=0.0, threshold=0.0, top_k=top_k,
            )
            summary = format_raw_evidence(neutral_selected)
            searches.append({"hop": hop + 1, "query": q, "raw_results": raw})
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Search result: {summary}"})
            continue

        break

    return {"queries": queries_used, "searches": searches, "num_hops": len(searches)}


def predict_with_evidence(
    *,
    claim: Claim,
    retrieval_mode: str,
    openrouter_key: str,
    agent_model: str,
    scorer: CredibilityScorer,
    top_k: int,
    alpha: float,
    threshold: float,
    all_raw_results: list[dict],
    query_plan: list[str],
) -> dict:
    """Phase B: Apply retrieval mode to pooled evidence and make one LLM prediction."""
    selected = apply_retrieval_mode(
        results=all_raw_results, scorer=scorer, mode=retrieval_mode,
        alpha=alpha, threshold=threshold, top_k=top_k,
    )
    evidence_text = format_raw_evidence(selected)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": VERDICT_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Claim: {claim.text}\n\n"
            f"Retrieved evidence:\n{evidence_text}\n\n"
            "Based on the evidence above, provide your final verdict."
        )},
    ]
    resp = openrouter_chat(
        api_key=openrouter_key, model=agent_model,
        messages=messages, tools=None, temperature=0.2, max_tokens=550,
    )
    content = resp["choices"][0]["message"].get("content") or ""
    prediction = parse_final_label(content)
    if prediction is None:
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": "Please provide final verdict in format: Final Answer: SUPPORTS/REFUTES/NOT_ENOUGH_INFO"})
        resp = openrouter_chat(
            api_key=openrouter_key, model=agent_model,
            messages=messages, tools=None, temperature=0.2, max_tokens=200,
        )
        content = resp["choices"][0]["message"].get("content") or ""
        prediction = parse_final_label(content) or "NOT_ENOUGH_INFO"
    return {
        "claim_id": claim.id, "claim": claim.text, "claim_label": claim.label,
        "retrieval_mode": retrieval_mode, "prediction": prediction, "num_hops": 0,
        "searches": [{"hop": 1, "query": "; ".join(query_plan), "requested_query": "",
                       "raw_results": all_raw_results, "selected": selected, "summary": evidence_text}],
        "final_response": content,
    }


def rows_to_csv(rows: list[dict], out_csv: Path) -> None:
    import csv

    cols = ["claim_id", "claim", "claim_label", "prediction", "retrieval_mode", "num_hops"]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


def main() -> None:
    parser = argparse.ArgumentParser(description="SciFact multi-hop verification.")
    parser.add_argument("--claims-jsonl", type=Path, default=_PROJECT_ROOT / "data" / "scifact" / "claims_dev.jsonl")
    parser.add_argument("--output-dir", type=Path, default=_PROJECT_ROOT / "outputs" / "retrieval_phase" / "scifact")
    parser.add_argument("--num-claims", type=int, default=100, help="<=0 for all.")
    parser.add_argument("--retrieval-modes", nargs="+",
                        default=["no_credibility", "reranked", "stratified"],
                        choices=["no_credibility", "raw_search_rank", "reranked", "filtered", "stratified"])
    parser.add_argument("--openrouter-api-key", type=str, default=os.environ.get("OPENROUTER_API_KEY", ""))
    parser.add_argument("--serper-api-key", type=str, default=SERPER_API_KEY)
    parser.add_argument("--credigraph-token", type=str, default=CREDIGRAPH_TOKEN)
    args = parser.parse_args()

    if not args.openrouter_api_key:
        raise SystemExit("Set OPENROUTER_API_KEY.")
    if not args.serper_api_key:
        raise SystemExit("Set SERPER_API_KEY.")
    needs_credibility = any(m in {"reranked", "filtered", "stratified"} for m in args.retrieval_modes)
    if needs_credibility and not args.credigraph_token:
        raise SystemExit("Set CREDIGRAPH_TOKEN.")

    claims = load_claims(args.claims_jsonl, 0, args.num_claims)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scorer = CredibilityScorer(token=args.credigraph_token, default_score=DEFAULT_CREDIBILITY_SCORE)
    if not args.credigraph_token:
        scorer._get_client = lambda: None  # type: ignore[method-assign]

    _search_fn = lambda q: serper_search(q, api_key=args.serper_api_key, max_results=MAX_RESULTS)
    print(f"Output: {args.output_dir}")

    print("\n=== Phase A: Evidence gathering ===")
    raw_cache_by_claim: dict[str, dict[str, list[dict]]] = {}
    query_plan_by_claim: dict[str, list[str]] = {}
    search_hops_by_claim: dict[str, int] = {}

    for i, claim in enumerate(claims, start=1):
        print(f"  [{i}/{len(claims)}] claim_id={claim.id}  searching...", end="", flush=True)
        claim_raw_cache = raw_cache_by_claim.setdefault(claim.id, {})
        search_result = search_evidence_multihop(
            claim=claim,
            openrouter_key=args.openrouter_api_key,
            agent_model=DEFAULT_AGENT_MODEL,
            scorer=scorer,
            max_hops=MAX_HOPS,
            max_results=MAX_RESULTS,
            top_k=TOP_K,
            raw_cache=claim_raw_cache,
            search_fn=_search_fn,
        )
        query_plan_by_claim[claim.id] = search_result["queries"]
        search_hops_by_claim[claim.id] = search_result["num_hops"]
        print(f"  {search_result['num_hops']} hops, {len(claim_raw_cache)} queries cached")
        time.sleep(0.3)

    all_mode_summaries: dict[str, dict] = {}
    for mode in args.retrieval_modes:
        print(f"\n=== Phase B: mode '{mode}' ===")
        results = []
        y_true: list[str] = []
        y_pred: list[str] = []

        for i, claim in enumerate(claims, start=1):
            claim_raw_cache = raw_cache_by_claim.get(claim.id, {})
            query_plan = query_plan_by_claim.get(claim.id, [])
            all_raw: list[dict] = []
            for q in query_plan:
                all_raw.extend(claim_raw_cache.get(q, []))

            rec = predict_with_evidence(
                claim=claim, retrieval_mode=mode,
                openrouter_key=args.openrouter_api_key,
                agent_model=DEFAULT_AGENT_MODEL, scorer=scorer,
                top_k=TOP_K, alpha=ALPHA,
                threshold=FILTER_THRESHOLD,
                all_raw_results=all_raw, query_plan=query_plan,
            )
            rec["num_hops"] = search_hops_by_claim.get(claim.id, 0)
            results.append(rec)
            if claim.label in VALID_LABELS:
                y_true.append(claim.label)  # type: ignore[arg-type]
                y_pred.append(rec["prediction"])
            if i % 20 == 0 or i == len(claims):
                print(f"  [{i}/{len(claims)}]")
            time.sleep(0.2)

        metrics = f1_stats(y_true, y_pred, VALID_LABELS)
        out_jsonl = args.output_dir / f"predictions_{mode}.jsonl"
        out_csv = args.output_dir / f"predictions_{mode}.csv"
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        rows_to_csv(results, out_csv)

        all_mode_summaries[mode] = {
            "metrics": metrics,
            "num_claims": len(results),
            "num_labeled_claims": len(y_true),
            "output_jsonl": str(out_jsonl),
            "output_csv": str(out_csv),
        }
        print(json.dumps({"mode": mode, "metrics": metrics, "num_labeled_claims": len(y_true)}, indent=2))

    run_summary = {
        "claims_jsonl": str(args.claims_jsonl),
        "num_claims": len(claims),
        "retrieval_modes": args.retrieval_modes,
        "alpha": ALPHA,
        "filter_threshold": FILTER_THRESHOLD,
        "max_hops": MAX_HOPS,
        "max_results": MAX_RESULTS,
        "top_k": TOP_K,
        "agent_model": DEFAULT_AGENT_MODEL,
        "freeze_retrieval_comparison": True,
        "mode_summaries": all_mode_summaries,
    }
    with open(args.output_dir / "run_summary_verification.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    print("\nSaved run summary:", args.output_dir / "run_summary_verification.json")


if __name__ == "__main__":
    main()
