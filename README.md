# rust-bert-score

A high-performance Rust implementation of BERTScore, a metric for evaluating text generation quality using contextual embeddings from BERT-family models.

## Features

- 🚀 **High Performance**: Native Rust implementation with GPU support via PyTorch bindings
- 🤖 **Multiple Models**: Support for BERT, DistilBERT, RoBERTa, and DeBERTa
- 📊 **Complete Metrics**: Precision, Recall, and F1 scores with optional IDF weighting
- 🎯 **Baseline Rescaling**: Interpretable scores through baseline normalization
- 🔧 **Flexible API**: Both high-level pipeline and low-level module access
- 🧪 **Well Tested**: Comprehensive unit and integration tests

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
rust-bert-score = "0.1.0"
```

**Note**: This crate requires PyTorch. Make sure you have libtorch installed on your system.

## Quick Start

```rust
use rust_bert_score::{BERTScorerBuilder, BERTScoreResult};
use rust_bert::pipelines::common::ModelType;

// Create a scorer with default settings
let scorer = BERTScorerBuilder::new()
    .model(ModelType::Roberta, "roberta-large")
    .language("en")
    .vocab_paths("/path/to/vocab.json", Some("/path/to/merges.txt"))
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

## Architecture

The library is organized into modular components:

- **`tokenizer`**: Text preprocessing and tokenization
- **`model`**: Pre-trained model loading and embedding extraction
- **`similarity`**: Cosine similarity computation and token matching
- **`idf`**: Inverse Document Frequency weighting
- **`baseline`**: Score rescaling for interpretability
- **`pipeline`**: High-level API orchestrating all components

## Advanced Usage

### Custom Configuration

```rust
use rust_bert_score::BERTScorerConfig;
use tch::Device;

let config = BERTScorerConfig {
    model_type: ModelType::Bert,
    model_name: "bert-base-uncased".to_string(),
    language: "en".to_string(),
    vocab_path: "/path/to/vocab.txt".to_string(),
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
use rust_bert_score::similarity::compute_bertscore;
use tch::Tensor;

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
)?;
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
- **Parallelization**: Multi-threaded tokenization
- **Memory**: Lower memory footprint through careful tensor management
- **Compilation**: Native code execution without Python overhead

## Development

### Building from Source

```bash
git clone https://github.com/yourusername/rust-bert-score
cd rust-bert-score
cargo build --release
```

### Running Tests

```bash
# Run all tests
cargo test

# Run with model download tests (requires internet)
cargo test -- --ignored
```

### Documentation

```bash
cargo doc --open
```

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