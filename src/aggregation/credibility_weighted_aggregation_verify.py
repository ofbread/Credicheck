#!/usr/bin/env python3
"""
Aggregation-phase credibility integration.

Consumes per-evidence single-label predictions and aggregates them into
a final claim label using credibility-weighted voting. Supports power
weighting with adaptive power selection.
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
from typing import Any

from credibility_rag_pipeline import DEFAULT_CREDIBILITY_SCORE
from project_paths import OUTPUTS

from src.common import f1_stats, load_jsonl


def safe_float(x: Any, default: float) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    if not math.isfinite(v):
        return default
    return v


def choose_weighted_label(
    *,
    evidences: list[dict[str, Any]],
    label_space: list[str],
    default_credibility: float,
    credibility_power: float,
) -> tuple[str, dict[str, float], dict[str, int]]:
    weighted_sum = {label: 0.0 for label in label_space}
    counts = {label: 0 for label in label_space}

    for ev in evidences:
        label = str(ev.get("single_evidence_prediction", "")).upper().strip()
        if label not in weighted_sum:
            continue
        cred = safe_float(ev.get("credibility_score"), default_credibility)
        weight = max(0.0, cred) ** credibility_power
        weighted_sum[label] += weight
        counts[label] += 1

    total_count = sum(counts.values())
    if total_count == 0:
        if "NOT_ENOUGH_INFO" in label_space:
            return "NOT_ENOUGH_INFO", weighted_sum, counts
        if "REFUTES" in label_space:
            return "REFUTES", weighted_sum, counts
        return label_space[0], weighted_sum, counts

    # Tie-break order:
    # 1) larger weighted score
    # 2) larger raw count
    # 3) label_space order for deterministic output
    label_rank = {label: idx for idx, label in enumerate(label_space)}
    best = max(
        label_space,
        key=lambda label: (
            weighted_sum[label],
            counts[label],
            -label_rank[label],
        ),
    )
    return best, weighted_sum, counts



def choose_label_adaptive_power(
    *,
    evidences: list[dict[str, Any]],
    label_space: list[str],
    default_credibility: float,
) -> tuple[str, dict[str, float], dict[str, int]]:
    """Per-claim adaptive power: high power when credibility gap is informative,
    near-zero when evidence credibility is uniform."""
    import statistics as _stats
    from collections import defaultdict as _dd

    creds = [safe_float(ev.get("credibility_score"), default_credibility) for ev in evidences]
    cred_std = _stats.stdev(creds) if len(creds) >= 2 else 0.0

    label_creds: dict[str, list[float]] = _dd(list)
    for ev in evidences:
        lab = str(ev.get("single_evidence_prediction", "")).upper().strip()
        if lab in label_space:
            label_creds[lab].append(safe_float(ev.get("credibility_score"), default_credibility))

    if len(label_creds) >= 2:
        avg_creds = [_stats.mean(v) for v in label_creds.values() if v]
        cred_gap = max(avg_creds) - min(avg_creds) if len(avg_creds) >= 2 else 0.0
    else:
        cred_gap = 0.0

    if cred_gap > 0.20 and cred_std > 0.15:
        power = 2.0
    elif cred_gap < 0.08:
        power = 0.01
    else:
        power = 1.0

    return choose_weighted_label(
        evidences=evidences,
        label_space=label_space,
        default_credibility=default_credibility,
        credibility_power=power,
    )


def choose_label_margin_gated(
    *,
    evidences: list[dict[str, Any]],
    label_space: list[str],
    default_credibility: float,
    credibility_power: float,
    margin_threshold: float = 0.12,
) -> tuple[str, dict[str, float], dict[str, int]]:
    """Only apply credibility weighting when the credibility margin between the
    top two label groups exceeds margin_threshold. Otherwise use uniform weights."""
    import statistics as _stats
    from collections import Counter as _Counter, defaultdict as _dd

    label_creds: dict[str, list[float]] = _dd(list)
    label_counts: _Counter = _Counter()
    for ev in evidences:
        lab = str(ev.get("single_evidence_prediction", "")).upper().strip()
        if lab in label_space:
            label_creds[lab].append(safe_float(ev.get("credibility_score"), default_credibility))
            label_counts[lab] += 1

    top2 = label_counts.most_common(2)
    if len(top2) >= 2:
        avg1 = _stats.mean(label_creds[top2[0][0]]) if label_creds[top2[0][0]] else 0.0
        avg2 = _stats.mean(label_creds[top2[1][0]]) if label_creds[top2[1][0]] else 0.0
        margin = abs(avg1 - avg2)
    else:
        margin = 0.0

    effective_power = credibility_power if margin >= margin_threshold else 0.01
    return choose_weighted_label(
        evidences=evidences,
        label_space=label_space,
        default_credibility=default_credibility,
        credibility_power=effective_power,
    )


def choose_label_tiered_consensus(
    *,
    evidences: list[dict[str, Any]],
    label_space: list[str],
    default_credibility: float,
    high_threshold: float = 0.6,
) -> tuple[str, dict[str, float], dict[str, int]]:
    """Separate voting by credibility tier, then combine with structured logic:
    unanimous HIGH tier wins; tier disagreement → NEI; fallback → overall majority."""
    from collections import Counter as _Counter

    high_votes: _Counter = _Counter()
    low_votes: _Counter = _Counter()
    all_votes: _Counter = _Counter()
    counts = {lab: 0 for lab in label_space}
    weighted_sum = {lab: 0.0 for lab in label_space}

    for ev in evidences:
        lab = str(ev.get("single_evidence_prediction", "")).upper().strip()
        if lab not in label_space:
            continue
        cred = safe_float(ev.get("credibility_score"), default_credibility)
        all_votes[lab] += 1
        counts[lab] += 1
        weighted_sum[lab] += cred
        if cred >= high_threshold:
            high_votes[lab] += 1
        else:
            low_votes[lab] += 1

    total_all = sum(all_votes.values())
    if total_all == 0:
        fallback = "NOT_ENOUGH_INFO" if "NOT_ENOUGH_INFO" in label_space else label_space[0]
        return fallback, weighted_sum, counts

    total_high = sum(high_votes.values())
    if total_high >= 2:
        high_maj, high_maj_count = high_votes.most_common(1)[0]
        high_agreement = high_maj_count / total_high

        if high_agreement >= 0.7:
            return high_maj, weighted_sum, counts

        low_maj = low_votes.most_common(1)[0][0] if low_votes else None
        if low_maj and low_maj != high_maj and "NOT_ENOUGH_INFO" in label_space:
            return "NOT_ENOUGH_INFO", weighted_sum, counts

    overall_maj = all_votes.most_common(1)[0][0]
    return overall_maj, weighted_sum, counts


def build_claim_text_map(filtered_predictions_jsonl: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not filtered_predictions_jsonl.exists():
        return out
    for row in load_jsonl(filtered_predictions_jsonl):
        cid = str(row.get("claim_id", ""))
        if cid:
            out[cid] = str(row.get("claim", ""))
    return out


def _extract_url_scores_from_jsonl(path: Path) -> dict[str, float]:
    """Extract {url: credibility_score} from any predictions JSONL with searches[*].selected."""
    if not path.exists():
        return {}
    url_scores: dict[str, float] = {}
    for rec in load_jsonl(path):
        for hop in rec.get("searches", []) or []:
            for item in hop.get("selected", []) or []:
                url = str(item.get("url", "") or "").strip()
                raw = item.get("credibility_score")
                if url and raw is not None:
                    url_scores[url] = safe_float(raw, 0.0)
    return url_scores



def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Credibility-weighted aggregation using cached single-evidence predictions. "
            "No API calls are made."
        )
    )
    parser.add_argument(
        "--evidence-source",
        type=str,
        default="no_credibility",
        choices=["no_credibility", "reranked", "filtered", "stratified"],
        help="Which retrieval mode's single-evidence summary to use.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Path to single-evidence summary JSON. Default: summary_{source}.json under aggregation_phase/single_evidence.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUTS / "aggregation_phase" / "evaluation",
        help="Root output directory; one subdirectory per dataset.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["climate_fever", "scifact", "confact"],
        help="Subset of datasets to process.",
    )
    parser.add_argument(
        "--mode-name",
        type=str,
        default="credibility_weighted",
        help="Retrieval mode tag written into outputs.",
    )
    parser.add_argument(
        "--default-credibility",
        type=float,
        default=DEFAULT_CREDIBILITY_SCORE,
        help="Fallback credibility when cached value is missing/non-finite.",
    )
    parser.add_argument(
        "--credibility-power",
        type=float,
        default=1.0,
        help="Weight transform: weight = max(0, credibility)^power.",
    )
    parser.add_argument(
        "--adaptive-power",
        action="store_true",
        default=False,
        help="Use adaptive per-claim power selection based on credibility gap between label groups.",
    )
    parser.add_argument(
        "--margin-gated",
        action="store_true",
        default=False,
        help="Only apply credibility weighting when margin between top labels is meaningful.",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.12,
        help="Credibility margin threshold for margin-gated mode.",
    )
    parser.add_argument(
        "--tiered-consensus",
        action="store_true",
        default=False,
        help="Use tiered consensus: separate HIGH/LOW tier voting with structured combination.",
    )
    parser.add_argument(
        "--informative-only",
        action="store_true",
        default=False,
        help="Filter out NEI single-evidence predictions before aggregation (keep only informative evidence).",
    )
    parser.add_argument(
        "--conflict-only-weighting",
        action="store_true",
        default=False,
        help="Apply credibility weighting only when non-NEI evidence conflicts (both S and R present). Use uniform weights otherwise.",
    )
    args = parser.parse_args()

    if args.credibility_power <= 0:
        raise SystemExit("--credibility-power must be > 0.")


    base_dir = OUTPUTS / "aggregation_phase" / "single_evidence"
    if args.summary_json is None:
        args.summary_json = base_dir / f"summary_{args.evidence_source}.json"
    if not args.summary_json.exists():
        raise SystemExit(
            f"Summary not found: {args.summary_json}. "
            f"Run generate_single_evidence_for_aggregation.py --retrieval-mode {args.evidence_source}."
        )

    blob = json.loads(args.summary_json.read_text(encoding="utf-8"))
    datasets_obj = blob.get("datasets", {})
    if not isinstance(datasets_obj, dict):
        raise SystemExit("Invalid summary format: missing datasets object.")

    run_summary: dict[str, Any] = {
        "evidence_source": args.evidence_source,
        "summary_json": str(args.summary_json),
        "datasets": args.datasets,
        "mode_name": args.mode_name,
        "default_credibility": float(args.default_credibility),
        "credibility_power": float(args.credibility_power),
        "dataset_summaries": {},
    }

    for dataset in args.datasets:
        ds_obj = datasets_obj.get(dataset)
        if not isinstance(ds_obj, dict):
            print(f"[skip] dataset not found in summary: {dataset}")
            continue

        label_space = [str(x).upper() for x in (ds_obj.get("label_space") or [])]
        if not label_space:
            print(f"[skip] label_space missing: {dataset}")
            continue

        claims = ds_obj.get("claims") or []
        if not isinstance(claims, list):
            print(f"[skip] claims malformed: {dataset}")
            continue

        paths = ds_obj.get("paths") or {}
        predictions_jsonl = Path(str(paths.get("filtered", ""))) if isinstance(paths, dict) else None
        claim_text_by_id = build_claim_text_map(predictions_jsonl) if predictions_jsonl else {}

        out_dir = args.output_root / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        out_jsonl = out_dir / f"predictions_{args.mode_name}.jsonl"
        out_csv = out_dir / f"predictions_{args.mode_name}.csv"

        rows: list[dict[str, Any]] = []
        y_true: list[str] = []
        y_pred: list[str] = []

        for c in claims:
            claim_id = str(c.get("claim_id", ""))
            claim_label = str(c.get("claim_label", "")).upper().strip()
            evidences = c.get("evidences") or []
            if not isinstance(evidences, list):
                evidences = []

            # Filter to informative evidence (non-NEI single-evidence predictions)
            if args.informative_only:
                evidences = [
                    ev for ev in evidences
                    if str(ev.get("single_evidence_prediction", "")).upper().strip() != "NOT_ENOUGH_INFO"
                ]

            # Conflict-only: use credibility power only when S and R both present
            if args.conflict_only_weighting:
                non_nei_labels = set()
                for ev in evidences:
                    lab = str(ev.get("single_evidence_prediction", "")).upper().strip()
                    if lab in label_space and lab != "NOT_ENOUGH_INFO":
                        non_nei_labels.add(lab)
                effective_power = float(args.credibility_power) if len(non_nei_labels) >= 2 else 0.01
            else:
                effective_power = float(args.credibility_power)

            if args.adaptive_power:
                pred, weighted_sum, counts = choose_label_adaptive_power(
                    evidences=evidences,
                    label_space=label_space,
                    default_credibility=float(args.default_credibility),
                )
            elif args.margin_gated:
                pred, weighted_sum, counts = choose_label_margin_gated(
                    evidences=evidences,
                    label_space=label_space,
                    default_credibility=float(args.default_credibility),
                    credibility_power=float(args.credibility_power),
                    margin_threshold=float(args.margin_threshold),
                )
            elif args.tiered_consensus:
                pred, weighted_sum, counts = choose_label_tiered_consensus(
                    evidences=evidences,
                    label_space=label_space,
                    default_credibility=float(args.default_credibility),
                )
            else:
                pred, weighted_sum, counts = choose_weighted_label(
                    evidences=evidences,
                    label_space=label_space,
                    default_credibility=float(args.default_credibility),
                    credibility_power=effective_power,
                )

            rec = {
                "claim_id": claim_id,
                "claim": claim_text_by_id.get(claim_id, ""),
                "claim_label": claim_label,
                "retrieval_mode": args.mode_name,
                "prediction": pred,
                "num_hops": 0,
                "searches": [],
                "aggregation": {
                    "method": "credibility_weighted_vote",
                    "weighted_sum_by_label": weighted_sum,
                    "count_by_label": counts,
                    "num_evidences": len(evidences),
                },
                "final_response": f"Final Answer: {pred}",
            }
            rows.append(rec)

            if claim_label in label_space:
                y_true.append(claim_label)
                y_pred.append(pred)

        metrics = f1_stats(y_true, y_pred, label_space)
        with out_jsonl.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        with out_csv.open("w", encoding="utf-8", newline="") as f:
            cols = ["claim_id", "claim", "claim_label", "prediction", "retrieval_mode", "num_hops"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in cols})

        run_summary["dataset_summaries"][dataset] = {
            "mode": args.mode_name,
            "metrics": metrics,
            "num_claims": len(rows),
            "num_labeled_claims": len(y_true),
            "output_jsonl": str(out_jsonl),
            "output_csv": str(out_csv),
        }
        per_dataset_run_summary = {
            "dataset": dataset,
            "evidence_source": args.evidence_source,
            "summary_json": str(args.summary_json),
            "num_claims": len(rows),
            "retrieval_modes": [args.mode_name],
            "default_credibility": float(args.default_credibility),
            "credibility_power": float(args.credibility_power),
            "mode_summaries": {
                args.mode_name: {
                    "metrics": metrics,
                    "num_claims": len(rows),
                    "num_labeled_claims": len(y_true),
                    "output_jsonl": str(out_jsonl),
                    "output_csv": str(out_csv),
                }
            },
        }
        per_dataset_summary_path = out_dir / "run_summary_verification.json"
        per_dataset_summary_path.write_text(
            json.dumps(per_dataset_run_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "mode": args.mode_name,
                    "metrics": metrics,
                    "num_labeled_claims": len(y_true),
                },
                indent=2,
            )
        )

    run_summary_path = args.output_root / "run_summary_verification.json"
    run_summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved run summary: {run_summary_path}")


if __name__ == "__main__":
    main()
