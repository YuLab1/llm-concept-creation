# Beyond Noise: Characterizing Creative Potential in Unverifiable LLM Hallucinations

Official implementation for our **ACL 2026** paper.

> **Paper link** will be updated after publication.

## Overview

This repository provides the code and data for studying the creative potential of LLM hallucinations through a *conceptual creation* task. We propose a novelty-verifiability characterization that distinguishes **Creative Synthesis (Region A)** from **Groundless Fabrication (Region B)**, and empirically analyze 32,400 generated concepts across three technical domains (Communication, Electronics, AI) using multiple LLMs (GPT-4o, GPT-4o-mini, Qwen-max).

The codebase covers:
- **RQ1 & RQ2**: Concept generation with two complementary strategies (Prompt-Only Baseline vs. Structured Element-Based), multi-dimensional evaluation (novelty, coverage, diversity, plausibility), and supplementary analyses (temperature effects, semantic entropy, classification).
- **RQ3**: Retrospective recovery experiment — exhaustive 20×20 cross-domain concept combination, exact matching against real innovations, and quality scoring.

## Repository Structure

```
Code/
├── config.py                          # Unified configuration (API keys, model presets, paths)
├── llm_utils.py                       # Shared LLM utilities (client, JSON parsing, retry, cache)
├── .env.example                       # API key template
├── requirements.txt                   # Python dependencies
├── .gitignore
│
├── RQ1, 2/                            # Concept Creation & Evaluation
│   ├── create.py                      # Concept generation pipeline
│   ├── evaluation.py                  # Core metric computation (embedding-based)
│   ├── batch_evaluation.py            # Batch evaluation orchestration
│   ├── batch_eval_scibert.py           # SciBERT evaluation variant
│   ├── validate_format.py            # Output format validation
│   │
│   ├── data/
│   │   ├── generation concepts/       # 18 generation result files (150 concepts each)
│   │   ├── evaluation_all-MiniLM-L6-v2/  # Evaluation results (default embedding)
│   │   └── evaluation_SciBERT/        # Evaluation results (SciBERT embedding)
│   │
│   └── evaluation/                    # Analysis & visualization scripts
│       ├── Table 1/                   # Main metrics table (Table 1 in paper)
│       │   ├── table1_metrics.py      # Generate LaTeX table
│       │   └── visualize_metrics_single_column.py  # Faceted dot plot
│       ├── Table 5/                   # Plausibility breakdown (Table 5)
│       │   └── table5_feasibility.py
│       ├── Table 6/                   # Semantic deviation analysis (Table 6)
│       │   └── ana.py
│       ├── Figure 9/                  # Concept classification pie chart
│       │   └── figure9_classification.py
│       ├── Figure 10/                 # Temperature effect analysis
│       │   ├── eval_temperature.py    # Generate temperature experiment data
│       │   └── plot_temperature.py    # Plot temperature curves
│       └── Figure 11/                 # Semantic entropy visualization
│           └── entropy_analysis.py
│
└── RQ3/                               # Retrospective Recovery Experiment
    ├── step1_generate_combinations.py # Exhaustive 20×20 combination generation
    ├── step2_exact_match.py           # Exact match against real innovations
    ├── step3_evaluate_non_baseline.py # Quality scoring (1–4 scale)
    ├── step4_visualize.py             # Main figure + appendix heatmap
    └── data/                          # Source concepts & intermediate results
```

## Environment Setup

### Requirements

- Python 3.9+
- CUDA is **not** required (embedding models run on CPU)
- API access: OpenAI API (GPT-4o, GPT-4o-mini) and DashScope API (Qwen-max)

### Installation

```bash
pip install -r requirements.txt
```

### API Configuration

Copy the template and fill in your API keys:

```bash
cp .env.example .env
```

Required keys in `.env`:
- `OPENAI_API_KEY` — for GPT-4o / GPT-4o-mini
- `DASHSCOPE_API_KEY` — for Qwen-max

### Local Embedding Model (Optional)

The default embedding model (`all-MiniLM-L6-v2`) is auto-downloaded by `sentence-transformers`. For SciBERT, download the model to `models/scibert_scivocab_uncased/`:

```bash
git clone https://huggingface.co/allenai/scibert_scivocab_uncased models/scibert_scivocab_uncased
```

## Data Preparation

All evaluation data is included in the repository under `RQ1, 2/data/` and `RQ3/data/`. No additional downloads are required to reproduce the tables and figures.

**Data structure:**
- `generation concepts/` — 18 JSON files containing 150 generated concepts each (6 domains × 3 models)
- `evaluation_all-MiniLM-L6-v2/` — Pre-computed evaluation metrics (novelty, coverage, pairwise distance, feasibility, entropy, embeddings)
- `evaluation_SciBERT/` — Same metrics computed with SciBERT embeddings
- `RQ3/data/` — 20 Communication + 20 AI source concepts, 400 generated combinations, match results, and quality scores

## Quick Start

```bash
# Reproduce Table 1 (main evaluation metrics)
python "RQ1, 2/evaluation/Table 1/table1_metrics.py"

# Reproduce Table 5 (plausibility breakdown)
python "RQ1, 2/evaluation/Table 5/table5_feasibility.py"

# Reproduce Table 6 (semantic deviation)
python "RQ1, 2/evaluation/Table 6/ana.py"
```

## Reproducing Main Results

### Tables

| Paper Table | Script | Command |
|-------------|--------|---------|
| Table 1 (Main metrics, MiniLM) | `table1_metrics.py` | `python "RQ1, 2/evaluation/Table 1/table1_metrics.py"` |
| Table 1 (SciBERT variant) | `table1_metrics.py` | `python "RQ1, 2/evaluation/Table 1/table1_metrics.py" --embedding SciBERT` |
| Table 5 (Plausibility) | `table5_feasibility.py` | `python "RQ1, 2/evaluation/Table 5/table5_feasibility.py"` |
| Table 6 (Semantic deviation) | `ana.py` | `python "RQ1, 2/evaluation/Table 6/ana.py"` |

### Figures

| Paper Figure | Script | Command |
|--------------|--------|---------|
| Figure 9 (Classification pie) | `figure9_classification.py` | `python "RQ1, 2/evaluation/Figure 9/figure9_classification.py"` |
| Figure 10 (Temperature) | `plot_temperature.py` | `python "RQ1, 2/evaluation/Figure 10/plot_temperature.py"` |
| Figure 11 (Entropy) | `entropy_analysis.py` | `python "RQ1, 2/evaluation/Figure 11/entropy_analysis.py"` |

### RQ3 Pipeline

```bash
cd RQ3

# Step 1: Generate 20×20 = 400 cross-domain combinations (requires API)
python step1_generate_combinations.py

# Step 2: Exact match evaluation against 20 baseline innovations (requires API)
python step2_exact_match.py

# Step 3: Quality scoring for non-baseline concepts (requires API)
python step3_evaluate_non_baseline.py

# Step 4: Visualize results (no API needed)
python step4_visualize.py
```

> **Note:** Steps 1–3 require API calls. Pre-computed results are included in `RQ3/data/`, so Step 4 can be run directly.

### Full Generation Pipeline (RQ1 & RQ2)

```bash
# Generate concepts (example: Communication domain, single-domain, GPT-4o)
python "RQ1, 2/create.py" --model gpt-4o --input <concept_file> --mode single --domain Communication --output output.json

# Compute evaluation metrics
python "RQ1, 2/evaluation.py" --input output.json --output_dir eval_output/ --domain Communication
```

## Citation

```bibtex
@inproceedings{}
}
```

## License

This project is released under the [MIT License](LICENSE).
