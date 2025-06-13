# Python Benchmark Environment for BERTScore

This directory contains the Python reference implementation setup for comparing against the Rust BERTScore implementation.

## Setup

The environment is managed with `uv` for fast dependency management.

### Prerequisites
- Python 3.10+ 
- `uv` package manager (already installed)

### Installation

```bash
cd python-benchmark
uv sync  # Install dependencies from pyproject.toml
```

## Project Structure

```
python-benchmark/
├── pyproject.toml          # Python dependencies
├── scripts/                # Validation scripts
│   ├── make_direct_set.py  # Generate test sentence pairs
│   ├── run_direct_py.py    # Run Python BERTScore on test pairs
│   ├── run_wmt16_py.py     # Run Python BERTScore on WMT16 data
│   ├── compare_direct.py   # Compare Python vs Rust results
│   ├── compare_wmt16.py    # Compare WMT16 correlations
│   ├── dump_tokens_py.py   # Dump tokenization for comparison
│   └── fetch_wmt16.sh      # Create synthetic WMT16 data
├── data/                   # Test data
│   ├── direct_eval_pairs.tsv
│   └── wmt16/
│       ├── ref.txt
│       ├── sys/*.txt
│       └── human_sys_scores.tsv
└── reports/                # Output comparison reports
```

## Usage

### 1. Generate Test Data

```bash
# Create direct evaluation pairs
uv run python scripts/make_direct_set.py

# Create synthetic WMT16 data
bash scripts/fetch_wmt16.sh
```

### 2. Run Python BERTScore

```bash
# Score direct evaluation pairs
uv run python scripts/run_direct_py.py

# Score WMT16 systems (requires bert-score package)
uv run python scripts/run_wmt16_py.py
```

### 3. Compare with Rust Results

After the Rust implementation generates its results:

```bash
# Compare direct scores
uv run python scripts/compare_direct.py

# Compare WMT16 correlations
uv run python scripts/compare_wmt16.py
```

### 4. Tokenization Comparison

```bash
# Dump Python tokenization
uv run python scripts/dump_tokens_py.py
```

## Test Data

### Direct Evaluation Pairs (21 pairs)
- Identical sentences
- Paraphrases
- Word order variations
- Completely different sentences
- Edge cases (empty strings, special characters, unicode)
- Technical and domain-specific text

### WMT16 Synthetic Data
- 100 reference sentences
- 5 MT systems with varying quality
- Human judgment scores for correlation analysis

## Expected Outputs

### Reports Directory
- `direct_scores_python.csv` - BERTScore results from Python
- `direct_agreement.txt` - Comparison report with Rust
- `wmt16_sys_scores_py.csv` - System-level scores
- `wmt16_agreement.txt` - Correlation analysis
- `tokens_py.tsv` - Tokenization output

## Validation Criteria

### Direct Score Comparison
- Maximum absolute difference < 1e-4
- Pearson correlation > 0.99999

### WMT16 Correlation
- Correlation differences < ±0.002
- System ranking agreement (Kendall τ = 1)

## Notes

- The environment uses `uv` for fast package management
- All scripts should be run with `uv run python` to ensure correct environment
- The bert-score package will download model weights on first run
- Set `HF_HOME` environment variable to cache models in a specific location