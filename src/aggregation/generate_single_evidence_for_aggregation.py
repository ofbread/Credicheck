#!/usr/bin/env python3
"""
Generate single-evidence LLM predictions for credibility-weighted aggregation.
"""
from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import csv
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

import requests

from project_paths import OUTPUTS

from credibility_rag_pipeline import DEFAULT_CREDIBILITY_SCORE

from src.common import (
    EvidenceItem,
    average,
    build_dataset_paths,
    build_single_evidence_prompt as build_prompt,
    compute_conflict_score,
    compute_spread_and_std,
    evidence_key,
    evidence_text,
    load_jsonl,
    majority_label,
    parse_single_evidence_label as parse_label,
)

DEFAULT_MODEL = "google/gemini-2.0-flash-001"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
RETRIEVAL_MODES = ("no_credibility", "reranked", "filtered", "stratified")


def _looks_like_real_key(s: str) -> bool:
    """Reject obvious placeholders like 'sk-or-v1-...' or empty strings."""
    s = s.strip()
    return bool(s) and len(s) > 20 and "..." not in s


def resolve_openrouter_api_key(cli_value: str) -> str:
    """Resolve API key from CLI argument, environment variable, or module default."""
    s = (cli_value or "").strip()
    if _looks_like_real_key(s):
        return s
    env = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if _looks_like_real_key(env):
        return env
    return DEFAULT_OPENROUTER_KEY


def openrouter_chat_user_message(
    *,
    api_key: str,
    model: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> str:
    """Send a single user-message prompt to OpenRouter and return the response text."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                OPENROUTER_BASE_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data.get("choices"):
                raise RuntimeError(f"No choices: {str(data)[:300]}")
            return str(data["choices"][0]["message"].get("content") or "").strip()
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"OpenRouter failed: {last_error}")


def evidence_progress(
    n: int,
    total: int,
    *,
    desc: str,
    dataset: str,
    claim_id: str,
    width: int = 32,
    file: Any = None,
) -> None:
    """In-place progress bar on one line (no tqdm)."""
    f = file if file is not None else sys.stderr
    if total <= 0:
        print(f"\r{desc} | {dataset} | claim_id={claim_id}", end="", flush=True, file=f)
        return
    pct = min(1.0, n / total)
    filled = min(width, int(width * pct))
    bar = "=" * filled + (">" if filled < width else "") + " " * (width - filled - 1)
    cid = claim_id[:24] + "…" if len(claim_id) > 24 else claim_id
    print(
        f"\r{desc} [{bar}] {n}/{total} ({100 * pct:.1f}%) | {dataset} | {cid}   ",
        end="",
        flush=True,
        file=f,
    )


def evidence_progress_done(file: Any = None) -> None:
    print(file=file if file is not None else sys.stderr)


def safe_float_cred(x: Any, default: float) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except (TypeError, ValueError):
        pass
    return default


def collect_selected_evidences_as_items(
    record: dict[str, Any],
    *,
    default_credibility: float,
) -> list[EvidenceItem]:
    """Dedupe by URL; build EvidenceItem list from searches[*].selected.

    Credibility scores are read directly from the retrieval phase data
    (present for all modes including no_credibility).
    """
    seen_urls: set[str] = set()
    out: list[EvidenceItem] = []
    for hop in record.get("searches", []) or []:
        for item in hop.get("selected", []) or []:
            url = str(item.get("url", "") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            text = evidence_text(item)
            key = evidence_key(item)
            title = str(item.get("title", "") or "")
            raw = item.get("credibility_score")
            score = safe_float_cred(raw, default_credibility) if raw is not None else default_credibility

            out.append(
                EvidenceItem(
                    key=key,
                    title=title,
                    url=url,
                    text=text,
                    credibility_score=score,
                )
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Single-evidence predictions over *selected* evidences for one retrieval mode; "
            "output summary for credibility_weighted_aggregation_verify.py."
        )
    )
    parser.add_argument("--root-dir", type=Path, default=_PROJECT_ROOT)
    parser.add_argument(
        "--retrieval-mode",
        type=str,
        choices=list(RETRIEVAL_MODES),
        required=True,
        help="Which predictions_*.jsonl to read (selected evidences).",
    )
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default="",
        help="OpenRouter key. Falls back to OPENROUTER_API_KEY env var.",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--timeout",
        type=float,
        default=30,
        help="Request timeout in seconds.",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--search-engine", type=str, default="serper",
                        help="Which search engine's retrieval results to read.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS / "aggregation_phase" / "single_evidence",
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        default="",
        help="Summary JSON file name inside output-dir. Default: summary_{retrieval_mode}.json",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="",
        help="Cache file name inside output-dir. Default: single_evidence_selected_{mode}_cache.jsonl",
    )
    parser.add_argument(
        "--dataset-averages-file",
        type=str,
        default="",
        help="Optional dataset averages JSON. Default: dataset_averages_selected_{mode}.json",
    )
    parser.add_argument(
        "--claim-metrics-csv",
        type=str,
        default="",
        help="Optional claim metrics CSV. Default: claim_metrics_selected_{mode}.csv",
    )
    parser.add_argument("--max-evidences", type=int, default=0, help="0 = all.")
    parser.add_argument("--max-claims-per-dataset", type=int, default=0, help="0 = all.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["climate_fever", "scifact", "confact"],
    )
    parser.add_argument("--default-credibility", type=float, default=DEFAULT_CREDIBILITY_SCORE)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable stderr progress bar for evidence processing.",
    )
    args = parser.parse_args()

    openrouter_key = resolve_openrouter_api_key(args.openrouter_api_key)

    mode = args.retrieval_mode
    summary_name = args.summary_file or f"summary_{mode}.json"
    cache_name = args.cache_file or f"single_evidence_selected_{mode}_cache.jsonl"
    averages_name = args.dataset_averages_file or f"dataset_averages_selected_{mode}.json"
    metrics_csv_name = args.claim_metrics_csv or f"claim_metrics_selected_{mode}.csv"

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / summary_name
    cache_path = output_dir / cache_name
    dataset_averages_path = output_dir / averages_name
    claim_metrics_csv_path = output_dir / metrics_csv_name

    dataset_paths = build_dataset_paths(args.root_dir, search_engine=args.search_engine)
    pred_paths: dict[str, Path] = {}
    records_by_dataset: dict[str, dict[str, dict[str, Any]]] = {}

    for dataset in args.datasets:
        folder = dataset_paths.get(dataset)
        if folder is None:
            print(f"[skip] unknown dataset: {dataset}")
            continue
        path = folder / f"predictions_{mode}.jsonl"
        if not path.exists():
            print(f"[skip] missing file: {path}")
            continue
        pred_paths[dataset] = path
        rows = load_jsonl(path)
        records_by_dataset[dataset] = {str(r.get("claim_id", "")): r for r in rows}

    if not records_by_dataset:
        raise SystemExit("No dataset records loaded; check paths and --datasets.")

    total_claims = 0
    total_evidences = 0
    for dataset, id_map in records_by_dataset.items():
        n = 0
        for claim_id, rec in id_map.items():
            if args.max_claims_per_dataset > 0 and n >= args.max_claims_per_dataset:
                break
            n += 1
            total_claims += 1
            total_evidences += len(
                collect_selected_evidences_as_items(
                    rec,
                    default_credibility=args.default_credibility,
                )
            )
    if args.max_evidences > 0:
        total_evidences = min(total_evidences, args.max_evidences)

    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cache[str(obj["cache_key"])] = obj

    overall_summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "retrieval_mode": mode,
        "evidence_source": "selected",
        "datasets": {},
    }

    total_calls = 0
    processed_evidences = 0
    cache_hits = 0
    claim_metrics_rows: list[dict[str, Any]] = []

    if not args.no_progress and total_evidences > 0:
        print(
            f"Progress: {total_evidences} evidence items across {len(records_by_dataset)} dataset(s), "
            f"mode={mode}",
            file=sys.stderr,
        )

    for dataset, id_map in records_by_dataset.items():
        labels = (
            ["SUPPORTS", "REFUTES"]
            if dataset == "confact"
            else ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
        )
        pred_path = pred_paths[dataset]
        claim_entries: list[dict[str, Any]] = []
        dataset_label_counter: Counter[str] = Counter()
        dataset_gold_counter: Counter[str] = Counter()
        pred_by_gold: dict[str, Counter[str]] = defaultdict(Counter)
        conflict_scores: list[float] = []
        label_credibility_claim_averages: dict[str, list[float]] = {lab: [] for lab in labels}
        dataset_credibility_scores_all_evidences: list[float] = []
        majority_match_flags: list[float] = []
        claims_processed = 0

        for claim_id, rec in id_map.items():
            if args.max_claims_per_dataset > 0 and claims_processed >= args.max_claims_per_dataset:
                break
            if args.max_evidences > 0 and processed_evidences >= args.max_evidences:
                break

            claim = str(rec.get("claim", ""))
            gold = str(rec.get("claim_label", "")).upper()
            dataset_gold_counter[gold] += 1
            evidences = collect_selected_evidences_as_items(
                rec,
                default_credibility=args.default_credibility,
            )

            cred_scores = [e.credibility_score for e in evidences if e.credibility_score is not None]
            spread, std = compute_spread_and_std([float(x) for x in cred_scores])
            pred_counter: Counter[str] = Counter()
            evidence_rows: list[dict[str, Any]] = []
            full_prediction = str(rec.get("prediction", "") or "").upper()

            for ev in evidences:
                if args.max_evidences > 0 and processed_evidences >= args.max_evidences:
                    break
                cache_key = f"{mode}||{dataset}||{claim_id}||{ev.key}"
                if cache_key in cache:
                    label = str(cache[cache_key]["label"])
                    cache_hits += 1
                else:
                    prompt = build_prompt(claim=claim, evidence=ev, labels=labels)
                    raw = openrouter_chat_user_message(
                        api_key=openrouter_key,
                        model=args.model,
                        user_prompt=prompt,
                        temperature=0.0,
                        max_tokens=512,
                        timeout=args.timeout,
                        retries=args.retries,
                    )
                    label = parse_label(raw, labels)
                    cache_obj = {
                        "cache_key": cache_key,
                        "retrieval_mode": mode,
                        "dataset": dataset,
                        "claim_id": claim_id,
                        "evidence_key": ev.key,
                        "label": label,
                    }
                    cache[cache_key] = cache_obj
                    with cache_path.open("a", encoding="utf-8") as cf:
                        cf.write(json.dumps(cache_obj, ensure_ascii=False) + "\n")
                    total_calls += 1
                    if args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)

                processed_evidences += 1
                if not args.no_progress:
                    evidence_progress(
                        processed_evidences,
                        total_evidences,
                        desc=f"evidences[{mode}]",
                        dataset=dataset,
                        claim_id=claim_id,
                    )
                pred_counter[label] += 1
                dataset_label_counter[label] += 1
                pred_by_gold[gold][label] += 1

                evidence_rows.append(
                    {
                        "url": ev.url,
                        "title": ev.title,
                        "credibility_score": ev.credibility_score,
                        "single_evidence_prediction": label,
                    }
                )
                if ev.credibility_score is not None:
                    dataset_credibility_scores_all_evidences.append(float(ev.credibility_score))

            n_e = len(evidence_rows)
            conflict_score = compute_conflict_score(pred_counter, labels, n_e)
            avg_credibility_by_single_label: dict[str, float | None] = {}
            for lab in labels:
                label_scores = [
                    float(row["credibility_score"])
                    for row in evidence_rows
                    if row["single_evidence_prediction"] == lab and row["credibility_score"] is not None
                ]
                avg_val = average(label_scores)
                avg_credibility_by_single_label[lab] = avg_val
                if avg_val is not None:
                    label_credibility_claim_averages[lab].append(avg_val)

            majority_single_label = majority_label(pred_counter)
            match_majority = (
                None
                if not majority_single_label or majority_single_label.startswith("TIE:")
                else (full_prediction == majority_single_label)
            )
            if match_majority is not None:
                majority_match_flags.append(1.0 if match_majority else 0.0)
            conflict_scores.append(conflict_score)

            claim_entries.append(
                {
                    "claim_id": claim_id,
                    "claim_label": gold,
                    "run_predictions_by_mode": {mode: full_prediction},
                    "full_evidence_prediction": full_prediction,
                    "single_evidence_majority_label": majority_single_label,
                    "full_prediction_matches_single_evidence_majority": match_majority,
                    "num_evidences": n_e,
                    "credibility_score_min": min(cred_scores) if cred_scores else None,
                    "credibility_score_max": max(cred_scores) if cred_scores else None,
                    "credibility_score_spread": spread,
                    "credibility_score_stddev": std,
                    "conflict_score": conflict_score,
                    "avg_credibility_score_by_single_evidence_label": avg_credibility_by_single_label,
                    "single_evidence_prediction_counts": dict(pred_counter),
                    "single_evidence_prediction_rates": {
                        k: (v / n_e if n_e else 0.0) for k, v in pred_counter.items()
                    },
                    "evidences": evidence_rows,
                }
            )
            claim_metrics_rows.append(
                {
                    "dataset": dataset,
                    "claim_id": claim_id,
                    "claim_label": gold,
                    f"full_prediction_{mode}": full_prediction,
                    "single_evidence_majority_label": majority_single_label,
                    "full_prediction_matches_single_evidence_majority": match_majority,
                    "num_evidences": n_e,
                    "credibility_score_min": min(cred_scores) if cred_scores else None,
                    "credibility_score_max": max(cred_scores) if cred_scores else None,
                    "credibility_score_spread": spread,
                    "credibility_score_stddev": std,
                    "conflict_score": conflict_score,
                    "count_supports_single_evidence": pred_counter.get("SUPPORTS", 0),
                    "count_refutes_single_evidence": pred_counter.get("REFUTES", 0),
                    "count_nei_single_evidence": pred_counter.get("NOT_ENOUGH_INFO", 0),
                }
            )
            claims_processed += 1

        if claim_entries:
            avg_spread = statistics.mean(float(c["credibility_score_spread"]) for c in claim_entries)
            avg_std = statistics.mean(float(c["credibility_score_stddev"]) for c in claim_entries)
        else:
            avg_spread = 0.0
            avg_std = 0.0

        avg_conflict = average(conflict_scores) or 0.0
        avg_match_majority = average(majority_match_flags)
        avg_credibility_by_single_label_across_claims = {
            lab: average(vals) for lab, vals in label_credibility_claim_averages.items()
        }
        avg_credibility_across_all_evidences = average(dataset_credibility_scores_all_evidences)

        overall_summary["datasets"][dataset] = {
            "paths": {"filtered": str(pred_path)},
            "label_space": labels,
            "num_claims": len(claim_entries),
            "num_evidence_items": int(sum(c["num_evidences"] for c in claim_entries)),
            "avg_credibility_score_across_all_evidences": avg_credibility_across_all_evidences,
            "retrieval_mode_used_for_selected_evidences": mode,
            "gold_label_distribution": dict(dataset_gold_counter),
            "single_evidence_prediction_distribution": dict(dataset_label_counter),
            "single_evidence_prediction_distribution_by_gold_label": {
                g: dict(cnt) for g, cnt in pred_by_gold.items()
            },
            "conflict_summary": {
                "avg_conflict_score_across_claims": avg_conflict,
                "max_conflict_score_across_claims": max(conflict_scores, default=0.0),
            },
            "credibility_spread_summary": {
                "avg_spread_across_claims": avg_spread,
                "avg_stddev_across_claims": avg_std,
                "max_spread_across_claims": max(
                    (float(c["credibility_score_spread"]) for c in claim_entries),
                    default=0.0,
                ),
                "max_stddev_across_claims": max(
                    (float(c["credibility_score_stddev"]) for c in claim_entries),
                    default=0.0,
                ),
            },
            "avg_credibility_score_by_single_evidence_label_across_claims": (
                avg_credibility_by_single_label_across_claims
            ),
            "agreement_summary": {
                "avg_full_prediction_matches_single_evidence_majority": avg_match_majority,
                "num_claims_used_for_agreement": len(majority_match_flags),
            },
            "claims": claim_entries,
        }

    if not args.no_progress and total_evidences > 0:
        evidence_progress_done()

    overall_summary["run_stats"] = {
        "retrieval_mode": mode,
        "openrouter_calls_made_this_run": total_calls,
        "cache_hits": cache_hits,
        "processed_evidence_items": processed_evidences,
        "output_dir": str(output_dir),
        "summary_file": str(summary_path),
        "predictions_source_per_dataset": {k: str(v) for k, v in pred_paths.items()},
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)

    dataset_averages: dict[str, Any] = {}
    for dataset, ds_obj in overall_summary["datasets"].items():
        dataset_averages[dataset] = {
            "num_claims": ds_obj.get("num_claims"),
            "num_evidence_items": ds_obj.get("num_evidence_items"),
            "avg_credibility_score_across_all_evidences": ds_obj.get(
                "avg_credibility_score_across_all_evidences"
            ),
            "conflict_summary": ds_obj.get("conflict_summary"),
            "credibility_spread_summary": ds_obj.get("credibility_spread_summary"),
            "avg_credibility_score_by_single_evidence_label_across_claims": ds_obj.get(
                "avg_credibility_score_by_single_evidence_label_across_claims"
            ),
            "agreement_summary": ds_obj.get("agreement_summary"),
        }

    with dataset_averages_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_averages, f, ensure_ascii=False, indent=2)

    with claim_metrics_csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "dataset",
            "claim_id",
            "claim_label",
            f"full_prediction_{mode}",
            "single_evidence_majority_label",
            "full_prediction_matches_single_evidence_majority",
            "num_evidences",
            "credibility_score_min",
            "credibility_score_max",
            "credibility_score_spread",
            "credibility_score_stddev",
            "conflict_score",
            "count_supports_single_evidence",
            "count_refutes_single_evidence",
            "count_nei_single_evidence",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in claim_metrics_rows:
            writer.writerow(row)

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote dataset averages: {dataset_averages_path}")
    print(f"Wrote claim metrics CSV: {claim_metrics_csv_path}")
    print(
        "\nNext step (example):\n"
        f"  python credibility_weighted_aggregation_verify.py --evidence-source {mode} "
        f"--summary-json {summary_path}"
    )
    print(json.dumps(overall_summary["run_stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
