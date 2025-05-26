````markdown
# AGENT.md - BERTScore Implementation Plan Using rust-bert

## Introduction

This document provides a structured, step-by-step plan for implementing BERTScore using the `rust-bert` library. The AI agent will follow these steps to build the software, adhering to best practices, including test-driven development (TDD) and modular design.

## High-Level Overview

1. **Model and Tokenizer Setup**: Load a pre-trained BERT model and tokenizer using `rust-bert`.
2. **Sentence Preprocessing**: Tokenize candidate and reference sentences, ensuring compatibility with BERT model.
3. **Embedding Extraction**: Use `rust-bert` to extract token-level embeddings from the model.
4. **Cosine Similarity Calculation**: Compute cosine similarities between candidate and reference token embeddings.
5. **Precision, Recall, F1 Calculation**: Calculate BERTScore metrics: Precision, Recall, and F1 for each sentence pair.
6. **Optional IDF Weighting**: Implement IDF weighting to adjust token importance in the final scores.
7. **Baseline Rescaling**: Implement baseline rescaling to improve score interpretability.
8. **Testing and Validation**: Create unit tests to validate the correctness of the implementation.
9. **Python Bindings**: Provide Python bindings to allow easy integration into research projects.

## Step 1: Model and Tokenizer Setup

### 1.1 Load Pre-trained Model

- Load a pre-trained BERT model from `rust-bert` (e.g., `bert-base-uncased`, `roberta-large`).
- Use `rust-bert`'s model loading utilities (`BertModel` or `RobertaModel`).
- Ensure the model is placed on the appropriate device (`cuda` or `cpu`).

### 1.2 Load Tokenizer

- Use `rust-tokenizers` to load a compatible tokenizer (e.g., `BertTokenizer`).
- Ensure the tokenizer is configured for the correct model and language (e.g., lowercasing for uncased models).
- Verify that special tokens (`[CLS]`, `[SEP]`) are handled properly.

## Step 2: Sentence Preprocessing

### 2.1 Tokenize Sentences

- Implement tokenization of both candidate and reference sentences.
- Ensure that all sentences are tokenized into subword units and are padded to the same length within a batch.
- Generate attention masks (1 for real tokens, 0 for padding) and token type IDs (0 for single-sentence input).

### 2.2 Store Token IDs and Lengths

- Store the token IDs and attention masks as tensors.
- Track the true lengths of the sentences (ignoring padding) for later similarity computation.

## Step 3: Embedding Extraction

### 3.1 Forward Pass Through Model

- Run the model on tokenized inputs using the forward pass function.
- Ensure that the model outputs the hidden states (embeddings) for each token in each sentence.
- Store the embeddings from the layer specified by the user (default: last layer).

### 3.2 Select Embedding Layer

- Optionally, allow the user to specify which layer's embeddings to use.
- If no layer is specified, default to the last layer or a predefined layer (e.g., layer 9 for RoBERTa).

## Step 4: Cosine Similarity Calculation

### 4.1 Normalize Embeddings

- Normalize the embeddings of candidate and reference tokens to unit length (L2 normalization).
- This ensures that the cosine similarity calculation is valid (i.e., dot product of unit vectors).

### 4.2 Compute Cosine Similarity

- For each pair of candidate and reference sentences, calculate the pairwise cosine similarity between token embeddings.
- Ensure that padding tokens are excluded from the similarity calculation.

### 4.3 Store Pairwise Similarities

- Store the cosine similarity matrix for each candidate-reference pair in a tensor of shape `(T_c, T_r)`, where `T_c` is the number of candidate tokens and `T_r` is the number of reference tokens.
- If batching is used, compute similarity for each batch of candidate-reference pairs.

## Step 5: Precision, Recall, and F1 Calculation

### 5.1 Compute Precision

- For each candidate token, find the most similar reference token.
- Compute the average of the best matches (precision is the average similarity of candidate tokens to reference tokens).

### 5.2 Compute Recall

- For each reference token, find the most similar candidate token.
- Compute the average of the best matches (recall is the average similarity of reference tokens to candidate tokens).

### 5.3 Compute F1 Score

- Calculate the harmonic mean of precision and recall to compute the F1 score for each candidate-reference pair.

## Step 6: Optional IDF Weighting

### 6.1 Compute IDF Weights

- If IDF weighting is enabled, calculate the Inverse Document Frequency (IDF) for each token in the reference corpus.
- Use a simple formula for IDF:  
  $$ \text{idf}(w) = \log\frac{N + 1}{\text{df}(w) + 1} $$  
  where `N` is the total number of reference sentences and `df(w)` is the document frequency of token `w`.

### 6.2 Apply IDF Weighting

- Weight the precision and recall calculations by the IDF of the respective tokens.
- Update the precision and recall calculations to use weighted averages based on the IDF values.

## Step 7: Baseline Rescaling

### 7.1 Obtain Baseline Values

- If baseline rescaling is enabled, retrieve precomputed baseline scores for precision, recall, and F1.
- These baseline values are typically computed on random sentence pairs and are model-specific.

### 7.2 Apply Rescaling

- Rescale the scores by subtracting the baseline and normalizing:  
  $$ P' = \frac{P - P_b}{1 - P_b}, \quad R' = \frac{R - R_b}{1 - R_b}, \quad F1' = \frac{F1 - F1_b}{1 - F1_b} $$  
  where `P_b`, `R_b`, and `F1_b` are the baseline precision, recall, and F1.

## Step 8: Testing and Validation

### 8.1 Unit Tests

- Write unit tests for each component:
  - Tokenization: Ensure tokenization produces the expected tokens and IDs.
  - Embedding Extraction: Validate that embeddings are correctly extracted from the model.
  - Cosine Similarity: Ensure that cosine similarity is computed correctly for token embeddings.
  - Precision/Recall/F1: Verify that the precision, recall, and F1 calculations align with expectations.
  - IDF and Rescaling: Test that IDF weights and baseline rescaling are applied correctly.

### 8.2 Integration Tests

- Run the full pipeline on a few example pairs and compare the results with the original BERTScore Python implementation.
- Ensure that the Rust implementation produces similar precision, recall, and F1 scores.

## Step 9: Python Bindings

### 9.1 Expose Rust API to Python

- Use PyO3 to create Python bindings for the Rust implementation.
- Expose a Python class (`BERTScorer`) that allows users to initialize the scorer, provide candidate and reference sentences, and get back P, R, and F1 scores.

### 9.2 Python API Example

- Provide a simple Python API for the user to interact with:
  ```python
  import rust_bertscore
  scorer = rust_bertscore.BERTScorer(model_type="roberta-large", lang="en", num_layers=17, idf=True, rescale_with_baseline=True)
  P, R, F1 = scorer.score(candidates, references)
````

### 9.3 Performance Testing

* Test the Python bindings to ensure they are efficient and do not introduce significant overhead compared to the pure Rust implementation.
* Measure the performance improvements when using the Rust implementation over the original Python version.

## Best Practices

1. **Test-Driven Development (TDD):**

   * Write unit tests for each function before implementing the functionality.
   * Ensure that all tests pass before moving on to the next stage.

2. **Modular Design:**

   * Implement each step of the pipeline as a separate function or module. This ensures that the code is maintainable and extensible.

3. **Documentation:**

   * Document the functionality and expected inputs/outputs for each function.
   * Provide usage examples in the documentation for Python bindings.

4. **Performance Optimization:**

   * Use efficient algorithms for cosine similarity computation and batching to minimize memory usage and maximize speed.
   * Profile the code and optimize any slow sections using parallelization where appropriate.

5. **Extensibility:**

   * Design the implementation so that it can easily accommodate new models or changes to the BERTScore calculation (e.g., different layers, alternate similarity metrics, etc.).

6. **Code Quality:**

   * Follow Rust’s idiomatic coding practices, such as using `Result` for error handling, avoiding panics, and ensuring thread safety.

## Conclusion

This plan provides a step-by-step approach to implementing BERTScore in Rust using `rust-bert`. By following this detailed, hierarchical structure, the AI agent will be able to efficiently implement and test each component, ensuring that the final product is scientifically accurate, performant, and easy to use in a Python research setting.
```
