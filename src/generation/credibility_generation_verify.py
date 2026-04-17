#!/usr/bin/env python3
"""
Generation-phase credibility integration.

Uses selected evidences from the retrieval phase, then prompts an LLM
with few-shot chain-of-thought examples that demonstrate reasoning over
credibility when making a claim verification decision.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random as _random
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import requests

from project_paths import OUTPUTS
from src.common import (
    credibility_label,
    evidence_text,
    f1_stats,
    load_jsonl,
    normalize_credibility_fields,
    HIGH_THRESHOLD,
    LOW_THRESHOLD,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.0-flash-001"

DATASETS = {
    "climate_fever": {
        "labels": ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"],
        "default_prediction": "NOT_ENOUGH_INFO",
    },
    "scifact": {
        "labels": ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"],
        "default_prediction": "NOT_ENOUGH_INFO",
    },
    "confact": {
        "labels": ["SUPPORTS", "REFUTES"],
        "default_prediction": "REFUTES",
    },
}





def resolve_archive_url(url: str) -> str:
    if not url or "web.archive.org/web/" not in url:
        return url
    marker = "/web/"
    start = url.find(marker)
    if start < 0:
        return url
    suffix = url[start + len(marker):]
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


def sanitize_model_for_path(model: str) -> str:
    """Make model name safe for use in directory names (e.g. 'google/gemini-2.0-flash-001' -> 'google_gemini-2.0-flash-001')."""
    if not model:
        return "unknown"
    return model.replace("/", "_").replace("\\", "_").strip() or "unknown"


def collect_selected_evidences(rec: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for hop in rec.get("searches", []) or []:
        for item in hop.get("selected", []) or []:
            url = str(item.get("url", "")).strip()
            if url in seen_urls:
                continue
            seen_urls.add(url)
            out.append(item)
    return out


FEW_SHOT_3LABEL = """
Example 1 (HIGH-credibility sources agree, LOW contradicts):
Claim: "Arctic ice has been increasing since 2012."
Evidence:
[1] (Domain: nasa.gov, Credibility: HIGH, Score: 0.82)
    "NASA data shows Arctic sea ice extent has continued its long-term declining trend."
[2] (Domain: climatechangefacts.blog, Credibility: LOW, Score: 0.28)
    "Arctic ice has been growing rapidly since 2012, disproving climate alarmists."
[3] (Domain: nature.com, Credibility: HIGH, Score: 0.79)
    "Multi-year ice volume remains well below the 1979-2000 average."

Reasoning:
- Evidence [1] from nasa.gov (HIGH credibility) directly contradicts the claim with authoritative satellite data.
- Evidence [2] from climatechangefacts.blog (LOW credibility) supports the claim, but this source has low credibility and its claim lacks supporting data.
- Evidence [3] from nature.com (HIGH credibility) corroborates [1] with peer-reviewed ice volume measurements.
- Two highly credible sources agree that ice is declining. The one supporting source has low credibility.
Final Answer: REFUTES

Example 2:
Claim: "Drinking green tea prevents cancer."
Evidence:
[1] (Domain: webmd.com, Credibility: MEDIUM, Score: 0.58)
    "Some studies suggest green tea polyphenols may have anti-cancer properties, but results are inconclusive."
[2] (Domain: healthline.com, Credibility: MEDIUM, Score: 0.55)
    "While lab studies show promise, clinical trials have not confirmed that green tea prevents cancer in humans."

Reasoning:
- Evidence [1] from webmd.com (MEDIUM credibility) says results are inconclusive, neither clearly supporting nor refuting.
- Evidence [2] from healthline.com (MEDIUM credibility) similarly notes lab promise but no clinical confirmation.
- Both sources have moderate credibility and both indicate the evidence is insufficient to confirm the claim.
- There is no high-credibility source providing a definitive answer either way.
Final Answer: NOT_ENOUGH_INFO

Example 3 (HIGH-credibility sources disagree):
Claim: "Electric vehicles produce more lifetime emissions than gasoline cars."
Evidence:
[1] (Domain: nature.com, Credibility: HIGH, Score: 0.79)
    "Lifecycle analysis shows that EVs produce 50-70% fewer emissions than conventional vehicles over their full lifespan, even accounting for battery manufacturing."
[2] (Domain: science.org, Credibility: HIGH, Score: 0.76)
    "In regions with coal-heavy electricity grids, EVs can produce comparable or slightly higher lifecycle emissions than efficient gasoline vehicles during the first 3-5 years of ownership."
[3] (Domain: epa.gov, Credibility: HIGH, Score: 0.88)
    "On average across the US electricity mix, EVs produce significantly fewer lifecycle greenhouse gas emissions than conventional cars."

Reasoning:
- Evidence [1] from nature.com (HIGH credibility) refutes the claim with lifecycle analysis data.
- Evidence [2] from science.org (HIGH credibility) partially supports the claim but only under specific conditions (coal-heavy grids, first 3-5 years).
- Evidence [3] from epa.gov (HIGH credibility) refutes the claim based on the US average.
- Two of three HIGH-credibility sources refute the claim. The third provides a narrow exception, not a general rule. The claim makes an absolute statement ("produce more") which is contradicted by the weight of credible evidence.
Final Answer: REFUTES

Example 4 (UNRATED sources with specific corroborating content):
Claim: "Nigeria has approximately 1 doctor per 5,000 patients."
Evidence:
[1] (Domain: who.int, Credibility: HIGH, Score: 0.90)
    "WHO reports severe physician shortages in sub-Saharan Africa, with ratios varying widely by country."
[2] (Domain: dailytrust.com, Credibility: UNRATED)
    "Nigeria's doctor-to-patient ratio stands at approximately 1:4,000 to 1:5,000 according to NMA."
[3] (Domain: premiumtimesng.com, Credibility: UNRATED)
    "The Nigerian Medical Association confirms approximately 45,000 doctors serve over 200 million people."

Reasoning:
- Evidence [1] from who.int (HIGH credibility) confirms physician shortages in the region but doesn't give Nigeria-specific ratios.
- Evidence [2] from dailytrust.com (UNRATED — no credibility data) provides the specific ratio matching the claim, citing a local authority (NMA).
- Evidence [3] from premiumtimesng.com (UNRATED) provides numbers that compute to ~1:4,444, consistent with the claim.
- Although [2] and [3] are UNRATED, they cite specific local authorities and are mutually consistent. The HIGH-credibility WHO source corroborates the broader trend.
Final Answer: SUPPORTS
""".strip()

FEW_SHOT_2LABEL = """
Example 1:
Claim: "The WHO declared a global health emergency due to Ebola in 2019."
Evidence:
[1] (Domain: who.int, Credibility: HIGH, Score: 0.90)
    "On 17 July 2019, WHO declared the Ebola outbreak in the Democratic Republic of the Congo a Public Health Emergency of International Concern."
[2] (Domain: reuters.com, Credibility: HIGH, Score: 0.78)
    "The World Health Organization declared the Ebola epidemic a global emergency on Wednesday."

Reasoning:
- Evidence [1] from who.int (HIGH credibility) is the primary source and directly confirms the claim.
- Evidence [2] from reuters.com (HIGH credibility) corroborates with independent reporting.
- Both highly credible sources agree: the WHO did declare an emergency related to Ebola in 2019.
Final Answer: SUPPORTS

Example 2:
Claim: "Brain drain has no impact on healthcare in developing countries."
Evidence:
[1] (Domain: theguardian.com, Credibility: HIGH, Score: 0.75)
    "Africa struggles to stop brain drain of doctors and nurses, leaving healthcare systems critically understaffed."
[2] (Domain: lancet.com, Credibility: HIGH, Score: 0.85)
    "Migration of health workers from low-income countries significantly reduces healthcare access for local populations."

Reasoning:
- Evidence [1] from theguardian.com (HIGH credibility) describes serious healthcare staffing problems caused by brain drain.
- Evidence [2] from lancet.com (HIGH credibility) provides peer-reviewed evidence that migration of health workers reduces healthcare access.
- Both highly credible sources directly contradict the claim that brain drain has "no impact."
Final Answer: REFUTES

Example 3 (HIGH-credibility sources disagree):
Claim: "Remote work increases employee productivity."
Evidence:
[1] (Domain: hbr.org, Credibility: HIGH, Score: 0.72)
    "A Stanford study found that remote workers were 13% more productive than their in-office counterparts."
[2] (Domain: economist.com, Credibility: HIGH, Score: 0.80)
    "Research from multiple firms shows that fully remote workers are 10-20% less productive than hybrid workers, with collaboration and innovation suffering most."
[3] (Domain: forbes.com, Credibility: MEDIUM, Score: 0.55)
    "Companies report mixed results from remote work policies, with productivity gains in individual tasks but losses in team projects."

Reasoning:
- Evidence [1] from hbr.org (HIGH credibility) supports the claim with specific study data.
- Evidence [2] from economist.com (HIGH credibility) partially refutes it, noting reduced productivity compared to hybrid workers.
- Evidence [3] from forbes.com (MEDIUM credibility) suggests mixed results depending on task type.
- Two HIGH-credibility sources disagree. However, [2] compares remote to hybrid (not to in-office), while [1] compares remote to in-office. The claim asks about productivity vs. the default, and [1] directly addresses that. On balance, the evidence leans toward supporting the claim with caveats.
Final Answer: SUPPORTS
""".strip()






# --- Additional few-shot example for relevance_gated / combined modes ---
FEW_SHOT_TANGENTIAL_3LABEL = """
Example 5 (HIGH-credibility sources are tangential — none address the specific claim):
Claim: "The Great Barrier Reef lost 50% of its coral cover between 1995 and 2017."
Evidence:
[1] (Domain: nature.com, Credibility: HIGH, Score: 0.79)
    "Climate change is the greatest threat to coral reef ecosystems worldwide, with rising sea temperatures causing mass bleaching events."
[2] (Domain: science.nasa.gov, Credibility: HIGH, Score: 0.75)
    "Satellite monitoring shows ocean temperatures in the Coral Sea have increased by 0.8°C over the past three decades."
[3] (Domain: aims.gov.au, Credibility: HIGH, Score: 0.82)
    "The Australian Institute of Marine Science conducts long-term monitoring of reef health across multiple sites."
[4] (Domain: theguardian.com, Credibility: HIGH, Score: 0.68)
    "Scientists warn that coral reefs face an uncertain future as ocean acidification accelerates."

Reasoning:
- Evidence [1] from nature.com (HIGH credibility) discusses threats to coral reefs globally but does not mention the Great Barrier Reef specifically or any percentage of coral cover loss.
- Evidence [2] from science.nasa.gov (HIGH credibility) provides temperature data for the Coral Sea but does not address coral cover decline.
- Evidence [3] from aims.gov.au (HIGH credibility) mentions reef monitoring but provides no data about coral cover percentages.
- Evidence [4] from theguardian.com (HIGH credibility) discusses ocean acidification but not specific coral cover statistics.
- Despite all four sources being HIGH credibility, NONE provides data about the specific claim (50% loss between 1995 and 2017). HIGH credibility means the sources are generally reliable, but they do not address this particular assertion. Tangential evidence from credible sources is still insufficient.
Final Answer: NOT_ENOUGH_INFO
""".strip()

# --- Additional few-shot example for pmc_aware / combined modes ---
FEW_SHOT_PMC_3LABEL = """
Example 5 (MEDIUM-credibility research database with specific experimental findings):
Claim: "Inhibition of PI3K signaling reduces tumor growth in triple-negative breast cancer."
Evidence:
[1] (Domain: sciencedirect.com, Credibility: HIGH, Score: 0.70)
    "PI3K pathway alterations are frequently observed in breast cancer subtypes and represent potential therapeutic targets."
[2] (Domain: pmc.ncbi.nlm.nih.gov, Credibility: MEDIUM, Score: 0.46)
    "Treatment with the PI3K inhibitor BKM120 reduced tumor volume by 62% in TNBC xenograft models compared to vehicle control (p<0.001)."
[3] (Domain: nature.com, Credibility: HIGH, Score: 0.65)
    "Combination therapies targeting the PI3K/AKT/mTOR axis show promise in clinical trials for various solid tumors."

Reasoning:
- Evidence [1] from sciencedirect.com (HIGH credibility) discusses PI3K as a therapeutic target but provides no experimental data on tumor growth reduction.
- Evidence [2] from pmc.ncbi.nlm.nih.gov (MEDIUM credibility) provides specific experimental data: 62% tumor volume reduction with a PI3K inhibitor in TNBC models. Although the hosting platform (PubMed Central) has a MEDIUM credibility score, this is a peer-reviewed research article with concrete quantitative findings directly addressing the claim.
- Evidence [3] from nature.com (HIGH credibility) discusses combination therapies broadly but does not address TNBC specifically.
- The most directly relevant evidence is [2], which provides specific experimental results matching the claim, despite coming from a MEDIUM-credibility platform. A single piece of specific evidence with concrete data is more informative than general statements from higher-credibility sources.
Final Answer: SUPPORTS
""".strip()

FEW_SHOT_BLIND_3LABEL = """
Example 1:
Claim: "Arctic ice has been increasing since 2012."
Evidence:
[1] (Domain: nasa.gov)
    "NASA data shows Arctic sea ice extent has continued its long-term declining trend."
[2] (Domain: climatechangefacts.blog)
    "Arctic ice has been growing rapidly since 2012, disproving climate alarmists."
[3] (Domain: nature.com)
    "Multi-year ice volume remains well below the 1979-2000 average."

Reasoning:
- Evidence [1] from nasa.gov directly contradicts the claim with satellite data.
- Evidence [2] supports the claim but provides no supporting data or citations.
- Evidence [3] corroborates [1] with peer-reviewed ice volume measurements.
- Two sources with specific data agree that ice is declining. The one supporting source lacks data.
Final Answer: REFUTES

Example 2:
Claim: "Drinking green tea prevents cancer."
Evidence:
[1] (Domain: webmd.com)
    "Some studies suggest green tea polyphenols may have anti-cancer properties, but results are inconclusive."
[2] (Domain: healthline.com)
    "While lab studies show promise, clinical trials have not confirmed that green tea prevents cancer in humans."

Reasoning:
- Evidence [1] says results are inconclusive — neither clearly supporting nor refuting.
- Evidence [2] similarly notes lab promise but no clinical confirmation.
- Both indicate the evidence is insufficient to confirm the claim.
Final Answer: NOT_ENOUGH_INFO
""".strip()

FEW_SHOT_BLIND_2LABEL = """
Example 1:
Claim: "The WHO declared a global health emergency due to Ebola in 2019."
Evidence:
[1] (Domain: who.int)
    "On 17 July 2019, WHO declared the Ebola outbreak in the Democratic Republic of the Congo a Public Health Emergency of International Concern."
[2] (Domain: reuters.com)
    "The World Health Organization declared the Ebola epidemic a global emergency on Wednesday."

Reasoning:
- Evidence [1] is the primary source and directly confirms the claim.
- Evidence [2] corroborates with independent reporting.
- Both sources agree: the WHO did declare an emergency related to Ebola in 2019.
Final Answer: SUPPORTS

Example 2:
Claim: "Brain drain has no impact on healthcare in developing countries."
Evidence:
[1] (Domain: theguardian.com)
    "Africa struggles to stop brain drain of doctors and nurses, leaving healthcare systems critically understaffed."
[2] (Domain: lancet.com)
    "Migration of health workers from low-income countries significantly reduces healthcare access for local populations."

Reasoning:
- Evidence [1] describes serious healthcare staffing problems caused by brain drain.
- Evidence [2] provides peer-reviewed evidence that migration of health workers reduces healthcare access.
- Both sources directly contradict the claim that brain drain has "no impact."
Final Answer: REFUTES
""".strip()


def build_system_prompt(label_space: list[str], credibility_mode: str = "continuous_only") -> str:
    label_text = " / ".join(label_space)
    if credibility_mode == "blind":
        few_shot = FEW_SHOT_BLIND_2LABEL if len(label_space) == 2 else FEW_SHOT_BLIND_3LABEL
        return (
            "You are a fact-checking assistant.\n\n"
            "You will be given a claim and several pieces of evidence from web sources. "
            "Evaluate the evidence based on what it says — the specific facts, data, and "
            "reasoning presented.\n\n"
            "Think step by step about each piece of evidence and what it implies for the claim.\n"
            f"Then output exactly one label from: {label_text}\n\n"
            "Format your response as:\n"
            "Reasoning:\n<your step-by-step reasoning>\n"
            f"Final Answer: <one of {label_text}>\n\n"
            f"{few_shot}"
        )

    # Base credibility guidelines 
    base_guidelines = (
        "Guidelines for using credibility:\n"
        "1. Credibility reflects a domain's general editorial standards, not whether this specific text is accurate.\n"
        "2. HIGH-credibility sources are generally reliable but may not address the specific claim directly.\n"
        "3. UNRATED sources (no credibility data available) should be judged on their content alone — treat them neutrally.\n"
        "4. LOW-credibility sources may still contain correct specific facts — evaluate what they say, not just where they come from.\n"
        "5. When sources disagree, focus on which evidence is most specific and directly relevant to the claim, "
        "using credibility as a secondary tiebreaker.\n"
    )

    # Mode-specific additional guidelines
    extra_guidelines = ""
    if credibility_mode in ("relevance_gated", "combined"):
        extra_guidelines += (
            "6. Before your Final Answer, consider: how many evidence items DIRECTLY address the "
            "specific claim with concrete data or findings? If most evidence discusses the general "
            "topic without addressing the specific assertion, lean toward NOT_ENOUGH_INFO — even if "
            "sources are HIGH credibility. Tangential evidence from credible sources is still insufficient.\n"
        )
    if credibility_mode in ("pmc_aware", "combined"):
        extra_guidelines += (
            f"{'7' if credibility_mode == 'combined' else '6'}. Government research databases "
            "(pmc.ncbi.nlm.nih.gov, pubmed.ncbi.nlm.nih.gov) host peer-reviewed journal articles. "
            "Their MEDIUM credibility score reflects the hosting platform's web profile, not the "
            "scientific quality of the hosted research. A single peer-reviewed finding with specific "
            "experimental data from these platforms can be more informative than general statements "
            "from higher-credibility sources.\n"
        )

    # Select few-shot examples based on mode
    if len(label_space) == 2:
        few_shot = FEW_SHOT_2LABEL
    else:
        few_shot = FEW_SHOT_3LABEL
        if credibility_mode in ("relevance_gated",):
            few_shot += "\n\n" + FEW_SHOT_TANGENTIAL_3LABEL
        elif credibility_mode in ("pmc_aware",):
            few_shot += "\n\n" + FEW_SHOT_PMC_3LABEL
        elif credibility_mode == "combined":
            few_shot += "\n\n" + FEW_SHOT_TANGENTIAL_3LABEL + "\n\n" + FEW_SHOT_PMC_3LABEL

    return (
        "You are a claim verification assistant that uses source credibility to inform your judgment.\n\n"
        "Each piece of evidence is annotated with:\n"
        "- The source domain\n"
        "- A credibility label (HIGH, MEDIUM, LOW, or UNRATED)\n"
        "- A numeric credibility score (0 to 1)\n\n"
        + base_guidelines
        + extra_guidelines
        + "\nThink step by step about each piece of evidence, its credibility, and what it implies.\n"
        + f"Then output exactly one label from: {label_text}\n\n"
        + "Format your response as:\n"
        + "Reasoning:\n<your step-by-step reasoning considering credibility>\n"
        + f"Final Answer: <one of {label_text}>\n\n"
        + few_shot
    )


def build_user_prompt(
    claim: str,
    evidences: list[dict[str, Any]],
    credibility_mode: str = "continuous_only",
) -> str:
    if credibility_mode == "blind":
        blocks = []
        for i, ev in enumerate(evidences, start=1):
            text = evidence_text(ev)
            domain = ev.get("domain", "unknown")
            blocks.append(
                f"[{i}] (Domain: {domain})\n"
                f'    "{text}"'
            )
        evidence_block = "\n".join(blocks) if blocks else "(No evidence available)"
        return (
            f'Claim: "{claim}"\n\n'
            f"Evidence:\n{evidence_block}\n\n"
            "Now reason step by step about each evidence, "
            "then provide your final answer."
        )

    blocks = []
    for i, ev in enumerate(evidences, start=1):
        text = evidence_text(ev)
        domain = ev.get("domain", "unknown")
        cred_label = ev.get("credibility_label", "MEDIUM")
        cred_score = ev.get("credibility_score", 0.0)
        blocks.append(
            f"[{i}] (Domain: {domain}, Credibility: {cred_label}, Score: {cred_score:.2f})\n"
            f'    "{text}"'
        )
    evidence_block = "\n".join(blocks) if blocks else "(No evidence available)"
    return (
        f'Claim: "{claim}"\n\n'
        f"Evidence:\n{evidence_block}\n\n"
        "Now reason step by step about each evidence and its credibility, "
        "then provide your final answer."
    )


def openrouter_chat(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int = 512,
    retries: int = 2,
    json_mode: bool = False,
    reasoning_effort: str | None = None,
) -> str:
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
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    # Control reasoning/thinking tokens for models that support it (e.g., Gemini 3).
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                OPENROUTER_BASE_URL,
                headers=headers,
                json=payload,
                timeout=90,
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


def parse_final_label(text: str, label_space: list[str]) -> str | None:
    import re
    # Try JSON parsing first (for json_mode responses)
    try:
        data = json.loads(text)
        candidate = str(data.get("prediction", data.get("label", ""))).strip().upper().replace(" ", "_")
        if candidate in label_space:
            return candidate
    except (json.JSONDecodeError, AttributeError):
        pass
    # Try "Final Answer:" prefix
    m = re.search(r"Final Answer:\s*([A-Z_ ]+)", text, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip().upper().replace(" ", "_")
        if candidate in label_space:
            return candidate
    # Fallback: find the LAST mention of any valid label in the text.
    # Fallback for truncated responses
    label_pattern = r"\b(" + "|".join(re.escape(l) for l in label_space) + r")\b"
    matches = re.findall(label_pattern, text, re.IGNORECASE)
    if matches:
        candidate = matches[-1].strip().upper().replace(" ", "_")
        if candidate in label_space:
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generation-phase credibility integration: few-shot CoT with credibility labels."
    )
    parser.add_argument("--root-dir", type=Path, default=_PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS / "generation_phase")
    parser.add_argument(
        "--datasets", nargs="+",
        default=["climate_fever", "scifact", "confact"],
        choices=list(DATASETS.keys()),
    )
    parser.add_argument(
        "--evidence-source",
        type=str,
        default="no_credibility",
        choices=["no_credibility", "reranked", "filtered", "stratified"],
        help="Which retrieval mode's selected evidences to use.",
    )
    parser.add_argument("--mode-name", type=str, default="credibility_generation",
                        help="Mode tag for output files.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--openrouter-api-key", type=str,
                        default=os.environ.get("OPENROUTER_API_KEY", ""))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--num-claims", type=int, default=100, help="<=0 for all.")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--credibility-mode",
        type=str,
        default="continuous_only",
        choices=["continuous_only", "blind", "random",
                 "relevance_gated", "pmc_aware", "combined"],
        help=(
            "How to present credibility in prompts. "
            "continuous_only: HIGH/MEDIUM/LOW/UNRATED labels from continuous score. "
            "blind: no credibility info shown (Axis 2 baseline). "
            "random: shuffled credibility scores within each claim (ablation). "
            "relevance_gated: continuous + tangentiality-aware few-shot example and instructions. "
            "pmc_aware: continuous + PMC/PubMed peer-review annotation in instructions. "
            "combined: relevance_gated + pmc_aware together."
        ),
    )
    parser.add_argument(
        "--json-mode",
        action="store_true",
        default=False,
        help="Use JSON response_format for structured output parsing.",
    )
    parser.add_argument(
        "--credibility-corrections",
        type=Path,
        default=None,
        help="Path to JSON file mapping domain -> corrected credibility score.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="none",
        choices=["none", "minimal", "low", "medium", "high"],
        help=(
            "Control reasoning/thinking tokens for models that support it (Gemini 3, etc.). "
            "Thinking tokens count against max_tokens. 'none' disables thinking entirely. "
            "Default: 'none' to keep max_tokens fully available for the response."
        ),
    )
    args = parser.parse_args()

    if not args.openrouter_api_key:
        raise SystemExit("Missing --openrouter-api-key.")

    # Load credibility corrections 
    _cred_corrections: dict[str, float] = {}
    if args.credibility_corrections and args.credibility_corrections.exists():
        _cred_corrections = {
            k: float(v) for k, v in json.loads(args.credibility_corrections.read_text()).items()
        }
        print(f"Loaded {len(_cred_corrections)} credibility corrections from {args.credibility_corrections}")

    retrieval_base = args.root_dir / "outputs" / "retrieval_phase"

    run_output_base = args.output_dir / args.evidence_source / sanitize_model_for_path(args.model)
    run_output_base.mkdir(parents=True, exist_ok=True)

    global_summary: dict[str, Any] = {
        "evidence_source": args.evidence_source,
        "mode_name": args.mode_name,
        "model": args.model,
        "credibility_mode": args.credibility_mode,
        "high_threshold": HIGH_THRESHOLD,
        "low_threshold": LOW_THRESHOLD,
        "mode_summaries": {},
    }

    for dataset_name in args.datasets:
        ds_cfg = DATASETS[dataset_name]
        label_space = ds_cfg["labels"]
        default_pred = ds_cfg["default_prediction"]
        pred_path = retrieval_base / dataset_name / f"predictions_{args.evidence_source}.jsonl"
        if not pred_path.exists():
            print(f"[skip] {pred_path} not found")
            continue

        records = load_jsonl(pred_path)
        if args.offset > 0:
            records = records[args.offset:]
        if args.num_claims > 0:
            records = records[:args.num_claims]

        out_dir = run_output_base / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_jsonl = out_dir / f"predictions_{args.mode_name}.jsonl"
        out_csv = out_dir / f"predictions_{args.mode_name}.csv"

        if args.credibility_mode == "random":
            sys_mode = "continuous_only"
        else:
            sys_mode = args.credibility_mode
        system_prompt = build_system_prompt(label_space, credibility_mode=sys_mode)
        results: list[dict[str, Any]] = []
        y_true: list[str] = []
        y_pred: list[str] = []

        print(f"\n=== {dataset_name}: {len(records)} claims ===")

        for idx, rec in enumerate(records, start=1):
            claim_id = str(rec.get("claim_id", ""))
            claim = str(rec.get("claim", ""))
            claim_label = str(rec.get("claim_label", "")).upper().strip()
            print(f"[{idx}/{len(records)}] claim_id={claim_id}")

            selected = collect_selected_evidences(rec)
            enriched = normalize_credibility_fields(selected)

            if _cred_corrections:
                for ev in enriched:
                    domain = ev.get("domain", "")
                    if domain in _cred_corrections:
                        new_score = _cred_corrections[domain]
                        ev["credibility_score"] = new_score
                        is_default = abs(new_score - 0.35) < 0.01
                        ev["credibility_label"] = credibility_label(new_score, is_default=is_default)

            # Random ablation: shuffle credibility scores across evidence items
            if args.credibility_mode == "random" and len(enriched) > 1:
                rng = _random.Random(hash(claim_id))
                scores = [ev["credibility_score"] for ev in enriched]
                rng.shuffle(scores)
                for ev, new_score in zip(enriched, scores):
                    ev["credibility_score"] = new_score
                    is_default = abs(new_score - 0.35) < 0.01
                    ev["credibility_label"] = credibility_label(new_score, is_default=is_default)

            # random/relevance_gated/pmc_aware/combined all use continuous evidence formatting
            # (credibility labels shown); only blind uses the blind format
            if args.credibility_mode == "blind":
                prompt_mode = "blind"
            else:
                prompt_mode = "continuous_only"
            user_prompt = build_user_prompt(claim, enriched, credibility_mode=prompt_mode)

            try:
                response_text = openrouter_chat(
                    api_key=args.openrouter_api_key,
                    model=args.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    json_mode=args.json_mode,
                    reasoning_effort=args.reasoning_effort,
                )
                prediction = parse_final_label(response_text, label_space)
                if prediction is None:
                    prediction = default_pred
            except Exception as exc:
                response_text = f"Error: {exc}"
                prediction = default_pred

            row = {
                "claim_id": claim_id,
                "claim": claim,
                "claim_label": claim_label,
                "retrieval_mode": args.mode_name,
                "prediction": prediction,
                "num_hops": rec.get("num_hops", 0),
                "searches": [{
                    "hop": 1,
                    "query": claim,
                    "selected": enriched,
                    "summary": "",
                }],
                "final_response": response_text,
            }
            results.append(row)

            if claim_label in label_space:
                y_true.append(claim_label)
                y_pred.append(prediction)

            if args.sleep > 0:
                time.sleep(args.sleep)

        metrics = f1_stats(y_true, y_pred, label_space)

        with out_jsonl.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        with out_csv.open("w", encoding="utf-8", newline="") as f:
            cols = ["claim_id", "claim", "claim_label", "prediction", "retrieval_mode", "num_hops"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k) for k in cols})

        mode_summary = {
            "metrics": metrics,
            "num_claims": len(results),
            "num_labeled_claims": len(y_true),
            "output_jsonl": str(out_jsonl),
            "output_csv": str(out_csv),
        }
        global_summary["mode_summaries"][dataset_name] = mode_summary

        ds_run_summary = {
            "dataset": dataset_name,
            "evidence_source": args.evidence_source,
            "model": args.model,
            "credibility_mode": args.credibility_mode,
            "num_claims": len(results),
            "retrieval_modes": [args.mode_name],
            "high_threshold": HIGH_THRESHOLD,
            "low_threshold": LOW_THRESHOLD,
            "mode_summaries": {
                args.mode_name: mode_summary,
            },
        }
        (out_dir / "run_summary_verification.json").write_text(
            json.dumps(ds_run_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({"dataset": dataset_name, "mode": args.mode_name, "metrics": metrics}, indent=2))

    (run_output_base / "run_summary_verification.json").write_text(
        json.dumps(global_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved: {run_output_base / 'run_summary_verification.json'}")


if __name__ == "__main__":
    main()
