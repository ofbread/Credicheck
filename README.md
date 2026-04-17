# Domain-Level Credibility in Retrieval-Augmented Fact Verification


## Overview

This project investigates whether domain-level credibility scores (from [CrediGraph](https://huggingface.co/spaces/credi-net/credinet)) improve LLM-based fact verification when integrated into retrieval-augmented generation (RAG) pipelines. The pipeline retrieves web evidence for claims, scores source domains for credibility, and evaluates how credibility integration affects accuracy across three phases.

**Key findings:**
- Credibility labels improve generation-phase accuracy.
- The mechanism is an epistemic tiebreaker: HIGH-credibility labels resolve ambiguity toward SUPPORTS verdicts
- Accurate labels outperform random labels primarily on medium-difficulty claims (+12.6 pp)
- Credibility integration helps more on hard claims than on easy ones.

## Pipeline Architecture

```
                        Phase 1                    Phase 2                    Phase 3
                      RETRIEVAL                  GENERATION                AGGREGATION

  Claim ──> Agentic Multi-hop ──> Evidence ──> Few-shot CoT with ──> Credibility-weighted
            Web Search (Serper)   Selection     Credibility Labels     Voting over
            + CrediGraph Scoring  (4 modes)     (6 prompt modes)       Per-evidence Predictions

  Modes:    no_credibility        blind          uniform_vote
            reranked              continuous     cred_weighted (power sweep)
            filtered              random         adaptive / margin-gated
            stratified            combined       tiered consensus
                                  relevance_gated
                                  pmc_aware
```

**Models:** GPT-4o-mini, Gemini 3 Flash Preview, Llama-3.3-70B (via [OpenRouter](https://openrouter.ai))

**Datasets:** Climate-FEVER (n=300), SciFact (n=300), ConFact (n=287)

## Repository Structure

```
.
├── credibility_rag_pipeline.py        # Core library: CredibilityScorer, serper_search(), constants
├── project_paths.py                   # Path configuration (ROOT, DATA, OUTPUTS)
├── requirements.txt                   # Python dependencies (pinned)
├── pyproject.toml                     # Project metadata and Python version requirement
│
├── confact/                           # ConFact dataset loader
│   └── confact_loader.py             #   Typed loader for ConFact pickle files
│
├── src/
│   ├── common/
│   │   └── __init__.py               # Shared utilities: metrics, evidence helpers, credibility labels
│   │
│   ├── retrieval/                     # Phase 1: Evidence retrieval
│   │   ├── climate_fever_multihop_verify.py
│   │   ├── scifact_multihop_verify.py
│   │   └── confact_multihop_verify.py
│   │
│   ├── generation/                    # Phase 2: Credibility-aware generation
│   │   └── credibility_generation_verify.py
│   │
│   ├── aggregation/                   # Phase 3: Credibility-weighted aggregation
│   │   ├── generate_single_evidence_for_aggregation.py
│   │   └── credibility_weighted_aggregation_verify.py
│   │
│   └── analysis/                      # Experiment analysis and figure generation
│       ├── experiment_discovery.py               # Shared experiment file discovery
│       ├── comprehensive_analysis.py             # A1-A6: 2-class eval, McNemar, difficulty stratification
│       ├── generate_analysis_plots.py            # Publication figures (retrieval, calibration, etc.)
│       ├── generate_paper_figures.py             # Generation accuracy + difficulty stratification plots
│       ├── analyze_credibility_calibration.py    # Domain-document credibility calibration
│       ├── analyze_credibility_conflict_resolution.py  # Credibility as conflict tiebreaker
│       ├── analyze_evidence_informativeness.py   # NEI vs non-NEI evidence analysis
│       ├── analyze_compositional_informativeness.py    # Single vs multi-evidence composition gap
│       └── analyze_nei_mislabels.py              # Corpus-constrained NEI artifact quantification
│
├── paper.tex                          # Research paper (ACM SIGCONF format)
├── slides_short.tex                   # Presentation slides (Beamer)
│
├── data/                              # Datasets (not committed — see Data Setup below)
└── outputs/                           # Experiment results (not committed — see Reproducing below)
```

## Setup

### Requirements

- Python >= 3.10
- API keys for [OpenRouter](https://openrouter.ai), [Serper](https://serper.dev), and [CrediGraph](https://huggingface.co/spaces/credi-net/credinet)

### Installation

```bash
git clone https://github.com/<your-org>/TrustNetworkResearch.git
cd TrustNetworkResearch
pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Required variables:
| Variable | Service | Purpose |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | [OpenRouter](https://openrouter.ai) | LLM inference (GPT-4o-mini, Gemini, Llama) |
| `SERPER_API_KEY` | [Serper](https://serper.dev) | Web search for evidence retrieval |
| `CREDIGRAPH_TOKEN` | [CrediGraph](https://huggingface.co/spaces/credi-net/credinet) | Domain credibility scoring |

## Data Setup

Datasets are not included in the repository due to size. Place them as follows:

```
data/
├── climate_fever_subset/
│   └── clean_claims.csv              # 300 climate claims (CSV with claim_id, claim, claim_label)
├── scifact/
│   └── claims_dev.jsonl              # SciFact dev set (JSONL with id, claim, evidence)
├── confact/
│   └── HumC.pkl                      # ConFact HumC split (287 instances, pickle format)
└── credibility_corrections.json      # Manual domain score overrides (5 domains)
```

**Sources:**
- **Climate-FEVER:** Subset of [Climate-FEVER](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html) (Diggelmann et al., 2020)
- **SciFact:** [SciFact](https://github.com/allenai/scifact) dev split (Wadden et al., 2020)
- **ConFact:** [ConFact](https://arxiv.org/abs/2505.17762) HumC split (Ge et al., 2025)

## Reproducing Results

All scripts are standalone CLI tools. Run from the project root directory.

### Phase 1: Retrieval

```bash
# Climate-FEVER (300 claims, 4 retrieval modes, ~20 min)
python src/retrieval/climate_fever_multihop_verify.py \
    --num-claims 300 \
    --retrieval-modes no_credibility reranked stratified filtered

# SciFact
python src/retrieval/scifact_multihop_verify.py \
    --num-claims 300 \
    --retrieval-modes no_credibility reranked stratified filtered

# ConFact (uses provided evidence, no web search)
python src/retrieval/confact_multihop_verify.py \
    --num-claims 0 \
    --retrieval-modes no_credibility reranked filtered stratified
```

### Phase 2: Generation

```bash
# Example: GPT-4o-mini, blind baseline, no_credibility evidence
python src/generation/credibility_generation_verify.py \
    --model openai/gpt-4o-mini \
    --credibility-mode blind \
    --evidence-source no_credibility \
    --num-claims 300 \
    --max-tokens 1024

# Credibility-aware (continuous labels)
python src/generation/credibility_generation_verify.py \
    --model openai/gpt-4o-mini \
    --credibility-mode continuous_only \
    --evidence-source no_credibility \
    --num-claims 300 \
    --max-tokens 1024
```

Available `--credibility-mode` options: `blind`, `continuous_only`, `random`, `combined`, `relevance_gated`, `pmc_aware`

Available `--model` options: `openai/gpt-4o-mini`, `google/gemini-3-flash-preview`, `meta-llama/llama-3.3-70b-instruct`

### Phase 3: Aggregation

```bash
# Generate per-evidence predictions
python src/aggregation/generate_single_evidence_for_aggregation.py \
    --retrieval-mode no_credibility

# Run credibility-weighted aggregation
python src/aggregation/credibility_weighted_aggregation_verify.py \
    --evidence-source no_credibility \
    --credibility-power 1.0
```

### Analysis and Figures

```bash
# Comprehensive analysis (A1-A6: 2-class eval, McNemar, difficulty stratification)
python src/analysis/comprehensive_analysis.py

# Generate paper figures
python src/analysis/generate_paper_figures.py
python src/analysis/generate_analysis_plots.py
```

## Output Structure

Results are organized under `outputs/`:

```
outputs/
├── retrieval_phase/{dataset}/         # Retrieval predictions per mode
├── generation_phase/{evidence_source}/{model}/{dataset}/  # Generation predictions
├── aggregation_phase/                 # Single-evidence + aggregated predictions
├── comprehensive_analysis/            # A1-A6 JSON analysis results
├── plots/                             # Publication-quality PNG figures
└── {calibration,compositional,conflict_resolution,
     informativeness,nei_mislabel}_analysis/  # Detailed analysis JSONs
```


