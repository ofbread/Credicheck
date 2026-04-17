#!/usr/bin/env python3
"""
CONFACT claim verification with provided evidence and retrieval baselines.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import requests

from confact.confact_loader import load_confact
from credibility_rag_pipeline import CREDIGRAPH_TOKEN, DEFAULT_CREDIBILITY_SCORE, CredibilityScorer
from src.common import f1_stats

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_AGENT_MODEL = "google/gemini-2.0-flash-001"
VALID_LABELS = ["SUPPORTS", "REFUTES"]



CONFACT_LABEL_TEMPLATE = """
You are a fact-checking verifier.
Given a claim and an evidence summary, classify the claim as exactly one label:
- SUPPORTS
- REFUTES

End with:
Final Answer: <LABEL>

Claim: {claim}

Evidence Summary:
{evidence_summary}
""".strip()


@dataclass
class ConfactClaim:
    id: str
    text: str
    label: str | None
    evidence: list[dict[str, str]]


def openrouter_chat(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int = 48,
    retries: int = 2,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("choices"):
                raise RuntimeError(f"OpenRouter response missing choices: {str(data)[:400]}")
            return data
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
    raise RuntimeError(f"OpenRouter call failed after retries: {last_error}")


def resolve_archive_url(url: str) -> str:
    """Resolve web.archive.org URL to original URL when encoded in path."""
    if not url or "web.archive.org/web/" not in url:
        return url
    marker = "/web/"
    start = url.find(marker)
    if start < 0:
        return url
    suffix = url[start + len(marker):]
    # Typical format: <14-digit timestamp>/<original-url>
    if len(suffix) >= 15 and suffix[14:15] == "/":
        original = suffix[15:]
        if original.startswith(("http://", "https://")):
            return original
    return url


def url_to_domain(url: str) -> str:
    if not url:
        return ""
    try:
        resolved = resolve_archive_url(url)
        d = urlparse(resolved).netloc.lower().strip()
        return d[4:] if d.startswith("www.") else d
    except Exception:
        return ""


def apply_retrieval_mode(
    *,
    raw_results: list[dict],
    scorer: CredibilityScorer,
    mode: str,
    alpha: float,
    threshold: float,
    top_k: int,
) -> list[dict]:
    _ = alpha
    enriched: list[dict] = []
    for r in raw_results:
        domain = url_to_domain(r.get("url", ""))
        if domain:
            cred = scorer.get_score(domain)
        else:
            cred = scorer.default_score
        if mode in {"no_credibility", "raw_search_rank"}:
            combined = 0.0
        else:
            combined = cred
        x = dict(r)
        x["domain"] = domain
        x["credibility_score"] = float(cred)
        x["combined_score"] = float(combined)
        enriched.append(x)

    by_url: dict[str, dict] = {}
    for e in enriched:
        u = e.get("url", "")
        prev = by_url.get(u)
        if prev is None or e["combined_score"] > prev["combined_score"]:
            by_url[u] = e
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

    if mode in {"no_credibility", "raw_search_rank"}:
        selected = sorted(dedup, key=lambda x: x["search_rank"])
    elif mode == "reranked":
        ranked = sorted(dedup, key=lambda x: x["combined_score"], reverse=True)
        selected = _enforce_domain_diversity(ranked, top_k)
    elif mode == "filtered":
        selected = [x for x in sorted(dedup, key=lambda x: x["combined_score"], reverse=True) if x["credibility_score"] >= threshold]
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
        raise ValueError(f"Unknown mode: {mode}")
    return selected[:top_k]


def normalize_confact_label(label: str) -> str | None:
    t = (label or "").strip().lower()
    if "support" in t:
        return "SUPPORTS"
    if "refut" in t:
        return "REFUTES"
    return None


def parse_predicted_label(text: str) -> str:
    upper = (text or "").strip().upper().replace(" ", "_")
    if "SUPPORTS" in upper or "SUPPORTED" in upper:
        return "SUPPORTS"
    if "REFUTES" in upper or "REFUTED" in upper:
        return "REFUTES"
    token = upper.split()[0] if upper else ""
    if token in VALID_LABELS:
        return token
    # Binary fallback for CONFACT.
    return "REFUTES"


def load_confact_claims(confact_dir: Path, split: str, offset: int, num_claims: int) -> list[ConfactClaim]:
    out: list[ConfactClaim] = []
    instances = load_confact(split=split, data_dir=confact_dir)
    for inst in instances:
        ev = [
            {
                "evidence_id": e.evidence_id,
                "url": e.original_link,
                "content": e.content or "",
            }
            for e in inst.evidence
            if (e.original_link or "").strip()
        ]
        out.append(
            ConfactClaim(
                id=f"{split}_{inst.id}",
                text=inst.claim or inst.question,
                label=normalize_confact_label(inst.label),
                evidence=ev,
            )
        )
    out = out[offset:]
    if num_claims > 0:
        out = out[:num_claims]
    return out


def build_raw_results_from_evidence(evidence: list[dict[str, str]]) -> list[dict]:
    raw: list[dict] = []
    for i, e in enumerate(evidence):
        content = (e.get("content") or "").strip()
        raw.append(
            {
                "title": e.get("evidence_id", f"evidence_{i+1}"),
                "url": e.get("url", ""),
                "description": content[:1200],
                "extra_snippets": [content[1200:1600]] if len(content) > 1200 else [],
                "search_rank": i,
            }
        )
    return raw


def chunk_raw_results(raw_results: list[dict], chunk_size: int, max_hops: int) -> list[list[dict]]:
    if chunk_size <= 0:
        chunk_size = max(1, len(raw_results))
    chunks = [raw_results[i : i + chunk_size] for i in range(0, len(raw_results), chunk_size)]
    if max_hops > 0:
        chunks = chunks[:max_hops]
    return chunks


def dedup_selected(selected_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_url: dict[str, dict[str, Any]] = {}
    for e in selected_items:
        u = str(e.get("url", ""))
        prev = by_url.get(u)
        if prev is None or float(e.get("combined_score", 0.0)) > float(prev.get("combined_score", 0.0)):
            by_url[u] = e
    return list(by_url.values())


def format_raw_evidence(selected_evidence: list[dict[str, Any]]) -> str:
    """Format selected evidence as raw text blocks without credibility info."""
    if not selected_evidence:
        return "No evidence selected."
    blocks: list[str] = []
    for i, r in enumerate(selected_evidence, start=1):
        domain = r.get("domain", "")
        title = r.get("title", "")
        snippet = r.get("description", "") or r.get("snippet", "")
        blocks.append(
            f"[{i}] (Domain: {domain})\n"
            f"    Title: {title}\n"
            f'    "{snippet}"'
        )
    return "\n".join(blocks)



def predict_label(
    *,
    verifier_model: str,
    openrouter_api_key: str,
    claim: str,
    evidence_summary: str,
) -> tuple[str, str, str]:
    if not evidence_summary.strip():
        return "REFUTES", "", "Final Answer: REFUTES"
    prompt = CONFACT_LABEL_TEMPLATE.format(claim=claim, evidence_summary=evidence_summary)
    try:
        resp = openrouter_chat(
            api_key=openrouter_api_key,
            model=verifier_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=128,
        )
        text = str(resp["choices"][0]["message"].get("content") or "").strip()
        return parse_predicted_label(text), text, text
    except Exception:
        return "REFUTES", "", "Final Answer: REFUTES"





def rows_to_csv(rows: list[dict], out_csv: Path) -> None:
    import csv

    cols = ["claim_id", "claim", "claim_label", "prediction", "retrieval_mode", "num_hops"]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


def main() -> None:
    parser = argparse.ArgumentParser(description="CONFACT claim verification with provided evidence baselines.")
    parser.add_argument("--confact-dir", type=Path, default=_PROJECT_ROOT / "data" / "confact")
    parser.add_argument("--confact-split", choices=["HumC"], default="HumC")
    parser.add_argument("--output-dir", type=Path, default=_PROJECT_ROOT / "outputs" / "retrieval_phase" / "confact")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--num-claims", type=int, default=100, help="Use <=0 for all.")
    parser.add_argument("--num-examples", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--retrieval-modes",
        nargs="+",
        choices=["no_credibility", "raw_search_rank", "reranked", "filtered", "stratified"],
        default=["no_credibility", "reranked", "filtered"],
        help="Retrieval baselines to run in this execution.",
    )
    parser.add_argument("--max-hops", type=int, default=0, help="Maximum hops per claim (0 = auto, process all evidence chunks).")
    parser.add_argument("--max-results", type=int, default=20, help="Evidence chunk size per hop.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--filter-threshold", type=float, default=0.55)
    parser.add_argument("--agent-model", type=str, default=DEFAULT_AGENT_MODEL)
    parser.add_argument("--openrouter-api-key", type=str, default=os.environ.get("OPENROUTER_API_KEY", ""))
    parser.add_argument("--credigraph-token", type=str, default=CREDIGRAPH_TOKEN)
    parser.add_argument("--allow-default-credibility", action="store_true")
    parser.add_argument("--default-credibility", type=float, default=DEFAULT_CREDIBILITY_SCORE)
    parser.add_argument("--sleep-per-claim", type=float, default=0.1)
    args = parser.parse_args()
    if args.num_examples is not None:
        args.num_claims = args.num_examples

    if not args.openrouter_api_key:
        raise SystemExit("Missing OpenRouter key.")
    needs_credibility = any(m in {"reranked", "filtered", "stratified"} for m in args.retrieval_modes)
    if needs_credibility and not args.credigraph_token and not args.allow_default_credibility:
        raise SystemExit("Missing CrediGraph token; pass --allow-default-credibility for fallback.")

    claims = load_confact_claims(
        confact_dir=args.confact_dir,
        split=args.confact_split,
        offset=args.offset,
        num_claims=args.num_claims,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scorer = CredibilityScorer(token=args.credigraph_token, default_score=args.default_credibility)
    if not args.credigraph_token:
        scorer._get_client = lambda: None  # type: ignore[method-assign]

    run_summary: dict[str, Any] = {
        "claims_source": str(args.confact_dir / f"{args.confact_split}.pkl"),
        "confact_split": args.confact_split,
        "num_claims": len(claims),
        "retrieval_modes": args.retrieval_modes,
        "max_hops": args.max_hops,
        "max_results": args.max_results,
        "top_k": args.top_k,
        "agent_model": args.agent_model,
        "alpha": args.alpha,
        "filter_threshold": args.filter_threshold,
        "mode_summaries": {},
    }

    for retrieval_mode in args.retrieval_modes:
        print(f"\n=== Running mode: {retrieval_mode} ===")
        results: list[dict[str, Any]] = []
        y_true: list[str] = []
        y_pred: list[str] = []
        jsonl_path = args.output_dir / f"predictions_{retrieval_mode}.jsonl"

        for idx, claim in enumerate(claims, start=1):
            print(f"[{idx}/{len(claims)}] claim_id={claim.id}")
            try:
                raw_results = build_raw_results_from_evidence(claim.evidence)
                auto_hops = math.ceil(len(raw_results) / max(1, args.max_results)) if raw_results else 0
                effective_max_hops = auto_hops if args.max_hops <= 0 else args.max_hops
                hop_chunks = chunk_raw_results(raw_results, args.max_results, effective_max_hops)
                searches: list[dict[str, Any]] = []
                selected_all: list[dict[str, Any]] = []
                for hop_idx, hop_raw in enumerate(hop_chunks, start=1):
                    selected = apply_retrieval_mode(
                        raw_results=hop_raw,
                        scorer=scorer,
                        mode=retrieval_mode,
                        alpha=args.alpha,
                        threshold=args.filter_threshold,
                        top_k=args.top_k,
                    )
                    selected_all.extend(selected)
                    hop_summary = format_raw_evidence(selected)
                    searches.append(
                        {
                            "hop": hop_idx,
                            "query": claim.text,
                            "requested_query": claim.text,
                            "raw_results": hop_raw,
                            "selected": selected,
                            "summary": hop_summary,
                        }
                    )

                selected_dedup = dedup_selected(selected_all)
                evidence_summary = format_raw_evidence(selected_dedup)
                prediction, _raw_model_output, final_response = predict_label(
                    verifier_model=args.agent_model,
                    openrouter_api_key=args.openrouter_api_key,
                    claim=claim.text,
                    evidence_summary=evidence_summary,
                )
                rec = {
                    "claim_id": claim.id,
                    "claim": claim.text,
                    "claim_label": claim.label,
                    "retrieval_mode": retrieval_mode,
                    "prediction": prediction,
                    "num_hops": len(searches),
                    "searches": searches,
                    "final_response": final_response,
                }
            except Exception as exc:
                rec = {
                    "claim_id": claim.id,
                    "claim": claim.text,
                    "claim_label": claim.label,
                    "retrieval_mode": retrieval_mode,
                    "prediction": "REFUTES",
                    "num_hops": 0,
                    "searches": [],
                    "final_response": f"Error: {exc}",
                }
            results.append(rec)
            if claim.label in VALID_LABELS:
                y_true.append(claim.label)  # type: ignore[arg-type]
                y_pred.append(rec["prediction"])
            if args.sleep_per_claim > 0:
                time.sleep(args.sleep_per_claim)

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        csv_path = args.output_dir / f"predictions_{retrieval_mode}.csv"
        rows_to_csv(results, csv_path)

        metrics = f1_stats(y_true, y_pred, VALID_LABELS)
        mode_summary = {
            "metrics": metrics,
            "num_claims": len(results),
            "num_labeled_claims": len(y_true),
            "output_jsonl": str(jsonl_path),
            "output_csv": str(csv_path),
        }
        run_summary["mode_summaries"][retrieval_mode] = mode_summary
        print(json.dumps({"mode": retrieval_mode, "metrics": metrics, "num_labeled_claims": len(y_true)}, indent=2))

    with open(args.output_dir / "run_summary_verification.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    print("\nSaved run summary:", args.output_dir / "run_summary_verification.json")


if __name__ == "__main__":
    main()

