# CrediCheck

Credibility-aware RAG pipeline for claim verification. Integrates domain-level credibility scores ([CrediGraph](https://huggingface.co/spaces/credi-net/credinet)) into LLM-based fact-checking at three pipeline phases: evidence retrieval, verdict generation, and multi-evidence aggregation.

Tested with GPT-4o-mini, Gemini 3 Flash, and Llama-3.3-70B on Climate-FEVER, SciFact, and ConFact.

## Repository Structure

```
.
├── credibility_rag_pipeline.py           # CredibilityScorer, web search, constants
├── project_paths.py                      # DATA / OUTPUTS path config
├── confact/
│   └── confact_loader.py                 # ConFact dataset loader
├── src/
│   ├── common/__init__.py                # Shared utilities (metrics, evidence helpers)
│   ├── retrieval/                        # Phase 1: multi-hop web search + evidence selection
│   │   ├── climate_fever_multihop_verify.py
│   │   ├── scifact_multihop_verify.py
│   │   └── confact_multihop_verify.py
│   ├── generation/                       # Phase 2: few-shot CoT with credibility labels
│   │   └── credibility_generation_verify.py
│   └── aggregation/                      # Phase 3: credibility-weighted voting
│       ├── generate_single_evidence_for_aggregation.py
│       └── credibility_weighted_aggregation_verify.py
├── requirements.txt
└── .env.example
```

## Setup

Requires Python >= 3.10.

```bash
git clone https://github.com/ofbread/Credicheck.git
cd Credicheck
pip install -r requirements.txt
cp .env.example .env   # then fill in API keys
```

### API Keys

Set these in `.env` or export as environment variables:

- `OPENROUTER_API_KEY` — LLM inference ([OpenRouter](https://openrouter.ai))
- `SERPER_API_KEY` — web search ([Serper](https://serper.dev))
- `CREDIGRAPH_TOKEN` — domain credibility scores ([CrediGraph](https://huggingface.co/spaces/credi-net/credinet))

### Data

Datasets are not included. Place them under `data/`:

- `data/climate_fever_subset/clean_claims.csv` — 300 climate claims ([Climate-FEVER](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html))
- `data/scifact/claims_dev.jsonl` — SciFact dev split ([SciFact](https://github.com/allenai/scifact))
- `data/confact/HumC.pkl` — ConFact HumC split ([ConFact](https://arxiv.org/abs/2505.17762))

## Usage

All scripts use argparse. Run `--help` on any script for full options.

### Phase 1: Retrieval

```bash
python src/retrieval/climate_fever_multihop_verify.py --num-claims 300 \
    --retrieval-modes no_credibility reranked stratified filtered

python src/retrieval/scifact_multihop_verify.py --num-claims 300 \
    --retrieval-modes no_credibility reranked stratified filtered

python src/retrieval/confact_multihop_verify.py --num-claims 0 \
    --retrieval-modes no_credibility reranked filtered stratified
```

### Phase 2: Generation

```bash
python src/generation/credibility_generation_verify.py \
    --model openai/gpt-4o-mini \
    --credibility-mode continuous_only \
    --evidence-source no_credibility \
    --num-claims 300 --max-tokens 1024
```

Credibility modes: `blind`, `continuous_only`, `random`, `combined`, `relevance_gated`, `pmc_aware`

Models: `openai/gpt-4o-mini`, `google/gemini-3-flash-preview`, `meta-llama/llama-3.3-70b-instruct`

### Phase 3: Aggregation

```bash
python src/aggregation/generate_single_evidence_for_aggregation.py \
    --retrieval-mode no_credibility

python src/aggregation/credibility_weighted_aggregation_verify.py \
    --evidence-source no_credibility --credibility-power 1.0
```
