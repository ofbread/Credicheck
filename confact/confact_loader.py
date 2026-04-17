"""Load the CONFACT (Conflicting Evidence for Fact-Checking) dataset.

CONFACT provides claims with conflicting evidence from different sources.
- HumC.pkl: Human-level conflict (287 instances)
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFACT_DIR = Path(__file__).resolve().parent.parent / "data" / "confact"


@dataclass
class EvidenceItem:
    evidence_id: str
    original_link: str
    content: str


@dataclass
class ConfactInstance:
    id: int | str
    claim: str
    label: str
    question: str
    evidence: list[EvidenceItem]
    claim_date: str | None
    review_date: str | None
    country: str | None
    original_claim_url: str | None
    fact_checking_article: str | None

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> ConfactInstance:
        evidence_url = raw.get("evidence_url") or []
        evidence = [
            EvidenceItem(
                evidence_id=str(e.get("evidence_id", "")),
                original_link=str(e.get("original_link", "")),
                content=str(e.get("content", "")),
            )
            for e in evidence_url
        ]
        return cls(
            id=raw.get("id", 0),
            claim=str(raw.get("claim", "")),
            label=str(raw.get("label", "")),
            question=str(raw.get("question", "")),
            evidence=evidence,
            claim_date=raw.get("claim_date"),
            review_date=raw.get("review_date"),
            country=raw.get("country"),
            original_claim_url=raw.get("original_claim_url"),
            fact_checking_article=raw.get("fact_checking_article"),
        )

def load_confact(
    split: str = "HumC",
    data_dir: Path | None = None,
) -> list[ConfactInstance]:
    """Load CONFACT dataset.

    Args:
        split: "HumC" (human-level conflict, 287) or "ModC" (model-level, 611)
        data_dir: Directory containing HumC.pkl and ModC.pkl. Default: data/confact/
    """
    data_dir = data_dir or CONFACT_DIR
    if split not in ("HumC", "ModC"):
        raise ValueError("split must be 'HumC' or 'ModC'")
    pkl_path = data_dir / f"{split}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"CONFACT file not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        raw_list = pickle.load(f)
    return [ConfactInstance.from_raw(r) for r in raw_list]
