# rust-bert-score

A high-performance Rust implementation of BERTScore, a metric for evaluating text generation quality using contextual embeddings from BERT-family models.

## Features

- 🚀 **High Performance**: Native Rust implementation with GPU support via PyTorch bindings
- 🤖 **Multiple Models**: Support for BERT, DistilBERT, RoBERTa, and DeBERTa
- 📊 **Complete Metrics**: Precision, Recall, and F1 scores with optional IDF weighting
- 🎯 **Baseline Rescaling**: Interpretable scores through baseline normalization
- 🔧 **Flexible API**: Both high-level pipeline and low-level module access
- 🧪 **Well Tested**: Comprehensive unit and integration tests
- 🛠️ **Command Line Interface**: Professional CLI with clap for batch processing
- 🤗 **HuggingFace Integration**: Direct model download from HuggingFace Hub
- ⚡ **Optimized Performance**: Iterator-based parallel processing and LTO compilation

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
rust-bert-score = "0.2.0"
```

**Note**: This crate requires PyTorch. Make sure you have libtorch installed on your system.

## Quick Start

### Library Usage

```rust
use rust_bert_score::{BERTScorerBuilder, BERTScoreResult};
use rust_bert::pipelines::common::ModelType;

// Create a scorer with HuggingFace model (automatically downloads)
let scorer = BERTScorerBuilder::new()
    .model(ModelType::Roberta, "roberta-large")
    .language("en")
    .vocab_paths(std::path::PathBuf::from("roberta-large"), None) // HF model name
    .use_idf(true)
    .rescale_with_baseline(true)
    .build()?;

// Score candidate sentences against references
let candidates = vec![
    "The cat sat on the mat.",
    "A dog ran in the park.",
];

let references = vec![
    "The cat was sitting on the mat.",
    "A dog was running in the park.",
];

let results: Vec<BERTScoreResult> = scorer.score(&candidates, &references)?;

for (i, result) in results.iter().enumerate() {
    println!("Pair {}: P={:.3}, R={:.3}, F1={:.3}", 
             i, result.precision, result.recall, result.f1);
}
```

### Command Line Interface

```bash
# Install the CLI tool
cargo install --path . --bin bert-score

# Score files using HuggingFace models
bert-score score \
    --candidates candidates.txt \
    --references references.txt \
    --pretrained roberta-large \
    --model-type roberta \
    --idf \
    --baseline

# Score using local vocabulary files
bert-score score \
    --candidates candidates.txt \
    --references references.txt \
    --vocab /path/to/vocab.json \
    --merges /path/to/merges.txt \
    --model-type roberta
```

## Architecture

The library follows a clean modular architecture organized into core components:

- **`core::tokenizer`**: Text preprocessing and tokenization with HuggingFace integration
- **`core::model`**: Pre-trained model loading and embedding extraction
- **`core::score`**: Cosine similarity computation and token matching (similarity module)
- **`core::idf`**: Inverse Document Frequency weighting
- **`core::baseline`**: Score rescaling for interpretability
- **`core::pipeline`**: High-level API orchestrating all components
- **`cli`**: Command-line interface with clap-based argument parsing
- **`python`**: Python bindings infrastructure (optional)

## Advanced Usage

### Custom Configuration

```rust
use rust_bert_score::{BERTScorerConfig, BERTScorer};
use rust_bert::pipelines::common::ModelType;
use tch::Device;
use std::path::PathBuf;

let config = BERTScorerConfig {
    model_type: ModelType::Bert,
    model_name: "bert-base-uncased".to_string(),
    language: "en".to_string(),
    vocab_path: PathBuf::from("/path/to/vocab.txt"),
    merges_path: None,
    lower_case: true,
    device: Device::cuda_if_available(),
    num_layers: Some(-2), // Use second-to-last layer
    max_length: 512,
    batch_size: 32,
    use_idf: true,
    rescale_with_baseline: true,
    custom_baseline: None,
};

let scorer = BERTScorer::new(config)?;
```

### HuggingFace Model Integration

```rust
use rust_bert_score::core::api::fetch_vocab_files;

// Automatically download vocabulary files from HuggingFace
let (vocab_path, merges_path) = fetch_vocab_files("roberta-large")?;

let scorer = BERTScorerBuilder::new()
    .model(ModelType::Roberta, "roberta-large")
    .vocab_paths(vocab_path, merges_path)
    .build()?;
```

### Multi-Reference Scoring

```rust
// Score each candidate against multiple references
let candidates = vec!["The cat sat on the mat."];
let references = vec![
    vec![
        "The cat was sitting on the mat.",
        "A cat sat on the mat.",
        "The feline rested on the rug.",
    ]
];

// Returns best F1 score among all references
let results = scorer.score_multi_refs(&candidates, &references)?;
```

### Low-Level API

For fine-grained control, you can use individual modules:

```rust
use rust_bert_score::core::score::compute_bertscore;
use tch::{Tensor, Device};

// Assume you have embeddings from your model
let candidate_embeddings = Tensor::randn(&[10, 768], (tch::Kind::Float, Device::Cpu));
let reference_embeddings = Tensor::randn(&[12, 768], (tch::Kind::Float, Device::Cpu));

// Create masks (1.0 for valid tokens, 0.0 for padding/special)
let cand_mask = Tensor::ones(&[10], (tch::Kind::Float, Device::Cpu));
let ref_mask = Tensor::ones(&[12], (tch::Kind::Float, Device::Cpu));

let result = compute_bertscore(
    &candidate_embeddings,
    &reference_embeddings,
    &cand_mask,
    &ref_mask,
    None, // No IDF weights
);
```

## Model Support

| Model Type | Example Models | Tokenizer |
|------------|---------------|-----------|
| BERT | bert-base-uncased, bert-base-multilingual | WordPiece |
| RoBERTa | roberta-base, roberta-large | BPE |
| DistilBERT | distilbert-base-uncased | WordPiece |
| DeBERTa | microsoft/deberta-base | SentencePiece |

## Performance

The Rust implementation provides significant performance improvements over Python:

- **Batching**: Efficient batch processing on GPU
- **Parallelization**: Iterator-based parallel tokenization with rayon
- **Memory**: Lower memory footprint through careful tensor management
- **Compilation**: Native code execution with LTO optimization
- **HuggingFace Integration**: Direct model downloads without Python dependencies
- **Benchmarking**: Criterion-based performance testing suite

## Development

### Building from Source

```bash
git clone https://github.com/glennmatlin/rust-bert-score
cd rust-bert-score
cargo build --release
```

### Running Tests

```bash
# Run all tests (includes unit and integration tests)
cargo test

# Run tests single-threaded if needed
cargo test -- --test-threads=1

# Run benchmarks
cargo bench
```

### CLI Development

```bash
# Build and install CLI locally
cargo install --path . --bin bert-score

# Test CLI functionality
bert-score score --help

# Run similarity computation
bert-score similarity --help
```

### Documentation

```bash
cargo doc --open
```

## Project Structure

This repository is organized as follows:

```
rust-bert-score/
├── src/                        # Main Rust library source
│   ├── core/                   # Core BERTScore implementation
│   ├── cli/                    # Command-line interface
│   ├── python/                 # PyO3 Python bindings (optional)
│   └── bin/                    # CLI binary entry point
├── examples/                   # Usage examples and demos
│   └── demo.rs                 # Comprehensive feature demonstration
├── benchmark/                  # Performance benchmarks (criterion)
│   └── performance_benchmarks.rs # CPU/GPU speed testing
├── tests/                      # Integration tests
│   ├── integration_test.rs     # Basic integration tests
│   └── integration_tests_full.rs # Comprehensive test suite
├── python/                     # Python package distribution
│   ├── rust_bert_score/        # Python API wrapper
│   └── test_basic.py           # Python bindings tests
├── python-benchmark/           # Validation environment
│   ├── scripts/                # Python/Rust comparison scripts
│   ├── data/                   # Test datasets
│   └── reports/                # Validation reports
└── .claude/                    # Project documentation and context
```

### Directory Purposes

- **`src/`**: Core Rust implementation with modular architecture
- **`examples/`**: Educational demos showing library features
- **`benchmark/`**: Performance testing and speed optimization
- **`tests/`**: Correctness validation and regression testing
- **`python/`**: Python distribution package (via maturin)
- **`python-benchmark/`**: Comprehensive validation against Python bert-score
- **`.claude/`**: Project documentation and development context

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original [BERTScore paper](https://arxiv.org/abs/1904.09675) by Zhang et al.
- [rust-bert](https://github.com/guillaume-be/rust-bert) for pre-trained model support
- [tch](https://github.com/LaurentMazare/tch) for PyTorch bindings

## Citation

If you use this library in your research, please cite:

```bibtex
@inproceedings{zhang2020bertscore,
  title={BERTScore: Evaluating Text Generation with BERT},
  author={Zhang, Tianyi and Kishore, Varsha and Wu, Felix and Weinberger, Kilian Q and Artzi, Yoav},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
```