"""Shared utilities for the TrustNetworkResearch pipeline."""
from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Credibility thresholds for label derivation
HIGH_THRESHOLD = 0.60
LOW_THRESHOLD = 0.45

DATASET_LABELS: dict[str, list[str]] = {
    "climate_fever": ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"],
    "scifact": ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"],
    "confact": ["SUPPORTS", "REFUTES"],
}

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------

def evidence_text(item: dict[str, Any]) -> str:
    if "snippet" in item:
        return str(item.get("snippet", "") or "").strip()
    desc = str(item.get("description", "") or "").strip()
    extras = item.get("extra_snippets", []) or []
    extra_txt = " ".join(str(x).strip() for x in extras if str(x).strip())
    return f"{desc}\n{extra_txt}".strip()


def evidence_key(item: dict[str, Any]) -> str:
    url = str(item.get("url", "") or "").strip()
    text = evidence_text(item)
    return f"{url}||{text}"


@dataclass(frozen=True)
class EvidenceItem:
    key: str
    title: str
    url: str
    text: str
    credibility_score: float | None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def f1_stats(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    per_label = {}
    f1_values = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_label[lab] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        f1_values.append(f1)
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0
    return {"accuracy": acc, "macro_f1": (sum(f1_values) / len(f1_values) if f1_values else 0.0), "per_label": per_label}


def compute_conflict_score(pred_counter: Counter, label_space: list[str], total: int) -> float:
    if total <= 0:
        return 0.0
    max_ent = math.log(len(label_space)) if len(label_space) > 1 else 1.0
    ent = 0.0
    for lab in label_space:
        p = pred_counter.get(lab, 0) / total
        if p > 0:
            ent -= p * math.log(p)
    return ent / max_ent if max_ent > 0 else 0.0


def compute_spread_and_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    spread = max(values) - min(values)
    std = statistics.stdev(values) if len(values) >= 2 else 0.0
    return spread, std


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def majority_label(pred_counter: Counter) -> str:
    if not pred_counter:
        return ""
    return pred_counter.most_common(1)[0][0]



# ---------------------------------------------------------------------------
# Credibility
# ---------------------------------------------------------------------------

def credibility_label(score: float, is_default: bool = False) -> str:
    if is_default:
        return "UNRATED"
    if score >= HIGH_THRESHOLD:
        return "HIGH"
    if score >= LOW_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def normalize_credibility_fields(
    evidences: list[dict[str, Any]],
    default_score: float = 0.35,
) -> list[dict[str, Any]]:
    """Normalize retrieval-phase credibility fields for downstream use.

    Derives credibility_label (HIGH/MEDIUM/LOW/UNRATED) from credibility_score.
    Domains where CrediGraph returned no score get UNRATED.
    """
    from urllib.parse import urlparse

    out: list[dict[str, Any]] = []
    for ev in evidences:
        item = dict(ev)
        if not item.get("domain"):
            url = str(item.get("url", ""))
            try:
                d = urlparse(url).netloc.lower().strip()
                item["domain"] = d[4:] if d.startswith("www.") else d
            except Exception:
                item["domain"] = ""
        score = float(item.get("credibility_score", default_score))
        item["credibility_score"] = score
        # Mark as UNRATED if score is exactly the pipeline default (domain not in CrediGraph)
        is_default = abs(score - default_score) < 0.01
        item["credibility_label"] = credibility_label(score, is_default=is_default)
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# Single-evidence prompts
# ---------------------------------------------------------------------------

def build_single_evidence_prompt(claim: str, evidence: EvidenceItem, labels: list[str]) -> str:
    label_text = ", ".join(labels)
    return (
        "You are a fact-checking assistant.\n"
        f"Task: Using ONLY the single evidence below, classify the claim as one of: {label_text}.\n"
        "Rules:\n"
        "- Use only the given evidence text.\n"
        "- If evidence is insufficient/unclear and NOT_ENOUGH_INFO is available in label set, choose NOT_ENOUGH_INFO.\n"
        "- Be strict and avoid guessing.\n"
        f'- Output JSON only: {{"label":"<one of {label_text}>","reason":"<one short sentence>"}}\n\n'
        f"Claim:\n{claim}\n\n"
        f"Evidence title: {evidence.title}\n"
        f"Evidence URL: {evidence.url}\n"
        f"Evidence text:\n{evidence.text}\n"
    )


def parse_single_evidence_label(raw: str, allowed_labels: list[str]) -> str:
    try:
        data = json.loads(raw)
        label = str(data.get("label", "")).strip().upper()
        if label in allowed_labels:
            return label
    except Exception:
        pass
    text = raw.upper()
    for label in allowed_labels:
        if re.search(rf"\b{re.escape(label)}\b", text):
            return label
    return allowed_labels[-1]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def build_dataset_paths(root: Path, search_engine: str = "serper") -> dict[str, Path]:
    base = root / "outputs" / "retrieval_phase"
    if search_engine != "serper":
        base = base / search_engine
    return {
        "climate_fever": base / "climate_fever",
        "scifact": base / "scifact",
        "confact": base / "confact",
    }
