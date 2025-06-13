```markdown
# Implementing BERTScore in Rust: Detailed Design Plan

## 1. Sentence Preprocessing and Tokenization  
**Goal:** Convert each candidate and reference sentence into token IDs compatible with a chosen BERT-family model. This ensures the text is input to the model in the same way as the original BERTScore implementation.  

- **Tokenizer Selection:** Use the `rust-tokenizers` crate to load a tokenizer matching the pre-trained model (e.g. WordPiece for BERT, byte-pair encoding for RoBERTa, SentencePiece for XLM, etc.)  . The tokenizer must be identical to Hugging Face’s for that model to ensure the same subword segmentation and special tokens. For example:  
  - English default uses RoBERTa-large’s tokenizer (byte-level BPE).  
  - Chinese uses `bert-base-chinese` (WordPiece).  
  - If a model name or language code is provided, map it to the appropriate tokenizer (e.g. `"en"` → RoBERTa-large, `"zh"` → BERT Chinese)  .  
  
- **Text Normalization:** Apply any model-specific preprocessing (e.g. lowercasing and accent stripping for uncased models). For instance, for `bert-base-uncased`, enable the tokenizer’s lowercase mode so that “Apple” and “apple” map to the same token . This mirrors the original BERTScore which defers to the model’s own casing conventions.  

- **Tokenization & Special Tokens:** Tokenize each sentence into subword tokens. Then insert the required special tokens: e.g. prepend `[CLS]` and append `[SEP]` for BERT/RoBERTa, or `<s> ... </s>` for models that use those markers. Ensure the tokenizer’s vocabulary is loaded so that these special tokens get their correct IDs. (For example, `rust_tokenizers::BertTokenizer` automatically knows the `[CLS]` and `[SEP]` tokens from the vocab file .) Include only one separator at the end since we are handling single-sentence inputs.  

- **Output IDs and Masks:** Convert tokens to their integer IDs and record sequence lengths. Construct an attention mask for each sequence where real token positions (including [CLS]/[SEP]) are `1` and padding positions are `0`. In Rust, this means producing a `Tensor` of shape `(batch_size, seq_len)` for `input_ids` and another for `attention_mask`. If using BERT (which expects token type IDs), create a `token_type_ids` tensor (all zeros for single sequences). The data is now ready to feed into the model.  

- **Batch Preparation:** Group multiple sentences into batches for efficiency. The plan is to support a user-specified `batch_size` (e.g. 64 by default ) to trade off speed and memory. For each batch of N sentence pairs, prepare two batched inputs: one for the N candidates and one for the N references (or combine into one 2N batch if memory allows, then split outputs). Use padding so that within each batch all sequences have equal length (pad to the longest sentence in the batch). Record the true lengths to later ignore pad tokens in scoring.  

## 2. Contextual Embedding Extraction with `rust-bert`  
**Goal:** Run each sentence through a pre-trained model (BERT or similar) to get contextualized embeddings for each token. We will extract embeddings from a specific transformer layer as used by BERTScore.  

- **Model Loading:** Use the `rust-bert` library to load the chosen pre-trained model architecture and weights. For example, for RoBERTa-large (English), load `RobertaModel`; for XLM-R, load `XLMRobertaModel`; for DeBERTa, load the corresponding model if supported. The library provides ready resources and config for many models (e.g. `BertModelResources::BERT` for BERT base)  . Initialize the model’s `VarStore` on the desired device (`tch::Device::cuda_if_available()` or CPU) so that inference can run on GPU if available.  

- **Configure Output Layer:** We need the hidden states of a specific layer for each token. By default, BERTScore often uses the final layer or a tuned intermediate layer (e.g. RoBERTa-large layer 17 was found optimal for English ). In Rust, set the model’s config to output all hidden states: `config.output_hidden_states = Some(true)` . This ensures the forward pass returns a list of hidden state tensors for each layer. After a forward pass on a batch, we can index into this list to retrieve the desired layer’s embeddings. If a user specifies `num_layers=k`, we will take the *k*-th layer from the bottom (or a predefined default if not specified ). For example, `num_layers=17` for RoBERTa-large means we take the 17th layer’s output.  

- **Forward Pass (Batch Inference):** Feed the `input_ids`, `attention_mask`, and `token_type_ids` tensors into the model’s forward method. The model will produce hidden states of shape `(batch_size, seq_len, hidden_dim)` for each layer. We ignore any pooler or classifier outputs – we only need token-level embeddings. The output for each sentence *already includes context from surrounding words*, thanks to the self-attention mechanism . We will **exclude** embeddings for special tokens like `[CLS]` and `[SEP]` from scoring, since these are not actual words in the sentence. (The original BERTScore matches each token in the candidate/reference sentences – which do not include special markers – so we must omit them to align with that design.)  

- **Extract Chosen Layer Embeddings:** Select the hidden state tensor for the specified layer. For example, if using the final layer, pick `hidden_states[-1]`; if an intermediate layer *L*, pick that index. We obtain two sets of embeddings: one for the batch of candidate sentences (size `N x seq_len_c x hidden`) and one for references (`N x seq_len_r x hidden`). These are contextualized token representations. Optionally, for debugging or advanced use, allow an `all_layers=true` mode to output scores from all layers (the HuggingFace version has this feature for analysis ). However, for normal operation we use only one layer per the configuration.  

- **Memory and Batching:** Process batches iteratively. The model can handle a batch of sequences; after each batch forward pass, immediately move on to similarity computation (and discard the batch’s embeddings) to avoid holding all data in memory. This streaming/batching approach ensures we can handle large evaluation sets. Use multiple threads (controlled by an `nthreads` parameter ) for the Rust code if CPU-bound (e.g. tokenization or similarity steps), while the model inference itself can leverage internal parallelism or GPU.  

## 3. Pairwise Cosine Similarity Computation  
**Goal:** For each candidate-reference pair, compute the cosine similarity between every token in the candidate and every token in the reference. These similarity scores will form the basis for matching tokens.  

- **Embedding Normalization:** To compute cosine similarity, normalize each token’s embedding vector to unit length (L2 norm = 1). The BERTScore paper explicitly uses *pre-normalized vectors*, so cosine similarity reduces to a dot product . We will perform this normalization in Rust for numerical stability. For each tensor of shape `(seq_len, hidden_dim)`, compute the L2 norm of each token vector and divide the vector by its norm. This can be done efficiently using tensor operations (elementwise division) on the GPU or using vectorized BLAS calls on CPU.  

- **Similarity Matrix:** After normalization, compute the cosine similarity matrix **M** of size `(T_c × T_r)` for each pair (where T_c is the number of tokens in the candidate, T_r in the reference). Each entry `M[i,j]` is the dot product between the *i*-th token embedding of the candidate and the *j*-th token embedding of the reference. Using the `tch` library, we can leverage highly optimized routines: e.g. perform matrix multiplication between the candidate embedding matrix and the transpose of the reference embedding matrix to get all pairwise scores in one go. This yields the same result as computing similarities one pair at a time, but is much faster.  

- **Per-Pair Processing:** In practice, each sentence pair can have different lengths, so we cannot directly stack all similarity matrices into one tensor. Instead, iterate over each pair in the batch: slice out the candidate’s embedding sub-tensor (length `T_c` actual tokens) and reference’s sub-tensor (`T_r` tokens), then compute their `T_c×T_r` similarity matrix. Any padded positions (if batching was used) should be excluded – since we tracked each sequence’s true length, use only the first `T_c` tokens of the candidate and `T_r` of the reference. (Padded embeddings can be ignored or zeroed out beforehand to avoid any contribution.)  

- **Greedy Token Matching:** Following BERTScore’s algorithm, treat the similarity matrix as a bipartite graph of token similarities . We will **greedily match** tokens to maximize similarity, which in this context means: for each token in one sentence, find the single most similar token in the other sentence . (The paper notes this greedy approach is nearly as good as optimal matching for this task .) We implement this by taking row-wise and column-wise maxima of the matrix: 
  - For each candidate token *i*, find `max_j M[i,j]` – the highest similarity with any reference token. 
  - For each reference token *j*, find `max_i M[i,j]` – the highest similarity with any candidate token. 
  These values represent the best match for each token under a one-sided matching. We will use them to compute precision and recall.  

- **Implementation:** Use tensor reduction operations to get max values efficiently. For example, `M.max(dim=1)` gives a vector of length `T_c` of the max similarity for each candidate token (across all reference tokens). Similarly, `M.max(dim=0)` (or equivalently the max of `M^T` by row) gives a vector for reference tokens. These can also be done with explicit loops if needed, but with `tch` on GPU it’s highly optimized. We ensure to ignore special tokens: since we didn’t include `[CLS]`/`[SEP]` in the embeddings used, they’re already excluded. The result is two vectors: **max_candidate_sims** and **max_reference_sims**.  

## 4. Precision, Recall, and F1 Score Calculation  
**Goal:** Derive the BERTScore **Precision**, **Recall**, and **F1** for each candidate-reference pair from the token-level similarity matches. Precision (P) reflects how well each *candidate* token is matched by the reference, and Recall (R) reflects how well each *reference* token is matched by the candidate . F1 is the harmonic mean of P and R, providing an overall score.  

- **Precision:** Compute P as the average of the best-match similarities for each candidate token . If `max_candidate_sims[i]` is the similarity of candidate token *i* to its closest reference token, then:  
  $$ P = \frac{1}{T_c} \sum_{i=1}^{T_c} \max_j M[i,j] $$.  
  Intuitively, every token in the candidate finds its most similar token in the reference; we then average these similarity scores. This rewards candidate sentences whose words are all well-explained by the reference.  

- **Recall:** Compute R as the average of best-match similarities for each reference token :  
  $$ R = \frac{1}{T_r} \sum_{j=1}^{T_r} \max_i M[i,j] $$.  
  Here each reference token finds its closest match in the candidate. High recall means the candidate covers most of the content of the reference (each reference token is accounted for by some similar token in the candidate).  

- **F1 Score:** Compute the harmonic mean of P and R for the pair:  
  $$ F1 = \frac{2P \cdot R}{P + R} $$.  
  This F1 gives a balanced measure of overlap, valuing both precision and recall. As the BERTScore README notes, the tool outputs P, R, and F1 for completeness, though F1 is often recommended as the primary metric  . If either P or R is zero (edge cases with empty sentences), define F1 = 0 to avoid division by zero.  

- **Implementation Details:** Use double precision for accumulation if possible to reduce floating-point error when summing many values. However, since typical sentence lengths aren’t huge, single precision (f32) is generally sufficient and matches PyTorch’s default. The output scores for each pair can be stored as Rust floats. If batching, we will accumulate these in arrays of length N (for N pairs in the batch). The final result can be either aggregated (e.g. average F1 over dataset) or returned per pair depending on use-case. The design will support returning a vector of P, R, F1 for each input pair, just like the original library’s `score` function.  

- **Multi-Reference Handling:** BERTScore supports multiple reference sentences for a single candidate . We will implement this by computing a score for the candidate against each reference in the set and then taking the **maximum F1** as the overall score for that candidate (and reporting the P/R from the maximizing reference)  . In practice, this means extending the above steps: if references are given as a list for a candidate, loop over them (or batch them) to compute F1s, then choose the best. This yields one P/R/F1 triple per candidate. This design is easily extensible because the core similarity calculation is pairwise; we just repeat it for each reference.  

## 5. IDF Weighting (Importance Weighting) [*Optional*]  
**Goal:** Optionally reweight the contribution of each token by its Inverse Document Frequency, so that *rare tokens count more* and common tokens count less . This feature is turned on by an `idf` flag and is especially useful for tasks where function words (very common) should be down-weighted in the score .  

- **IDF Computation:** If IDF weighting is enabled, first build an IDF dictionary from the *entire set of reference sentences* . We treat each reference sentence as a document. Compute the document frequency *df* for each token *w*: the number of reference sentences in which *w* appears. In Rust, this involves iterating through all reference token lists (which we likely have from the tokenization step) and using a HashMap to count occurrences. Use **set semantics per sentence** (if a word appears twice in one reference, it still counts once toward df, since we care whether it appears at all in that sentence) . Then compute the IDF score as:  
  $$ \text{idf}(w) = \log\frac{N + 1}{\text{df}(w) + 1}, $$  
  where *N* is the number of reference sentences (add 1 to numerator and denominator for smoothing) . This plus-one smoothing avoids zero denominators for tokens not seen in any reference (they get idf = log(N+1)) and ensures very frequent tokens (df = N) get a low but non-negative weight. We will also explicitly set the IDF for special tokens like `[CLS]`, `[SEP]`, or padding to 0, so they contribute nothing . The resulting `idf_dict` (token→idf) remains fixed for a given evaluation run.  

- **Applying IDF to Scores:** Modify the precision/recall calculations to be weighted averages  . Specifically:  
  $$ P_{\text{idf}} = \frac{\sum_{i=1}^{T_c} \text{idf}(w_i^{(c)}) \cdot \max_j M[i,j]}{\sum_{i=1}^{T_c} \text{idf}(w_i^{(c)})}, $$  
  $$ R_{\text{idf}} = \frac{\sum_{j=1}^{T_r} \text{idf}(w_j^{(r)}) \cdot \max_i M[i,j]}{\sum_{j=1}^{T_r} \text{idf}(w_j^{(r)})}. $$  
  Here $w_i^{(c)}$ is the *i*-th candidate token and $w_j^{(r)}$ the *j*-th reference token. In practice, this means when summing the max similarities, weight each term by the token’s IDF, and normalize by the sum of IDF weights for that sentence. The F1 is then computed from these weighted P and R. If a token is very common (low idf), its similarity contributes very little. If a token is rare (high idf), it has a larger impact on the score . This mirrors the behavior of metrics like METEOR and CIDEr which emphasize rare n-grams .  

- **Implementation:** Once we have `max_candidate_sims` and `max_reference_sims` vectors, multiply each element by the corresponding token’s idf weight. These IDF weights can be retrieved by token *ID* or by the original string; since our tokenizer can convert IDs to tokens, we can use IDs as keys in the `idf_dict`. Compute the weighted sums and divides as above. The denominator (sum of IDFs) is just the sum of that sentence’s token idf values (which we could also compute once during tokenization to save time). Use f64 for the summation if available to avoid precision issues when summing many small weights.  

- **Custom IDF or Precomputation:** Allow the user to supply a precomputed `idf_dict` (the original library accepts a dict to avoid recomputing for the same corpus ).
```


We will expose this in the Python binding API: if provided, use the user’s values instead of computing afresh. Otherwise, compute from references as above. This is useful if the reference corpus is large or if users want IDF from a different background corpus.

* **Disable if Not Needed:** If `idf=False`, we skip all the above and simply treat all idf weights as 1.0 (so P and R reduce to the unweighted average formulas). The implementation will branch accordingly for efficiency.

## 6. Baseline Rescaling for Score Calibration \[*Optional*]

**Goal:** Adjust the scale of the P, R, F1 scores by subtracting a baseline and rescaling, so that scores become more interpretable . In the original BERTScore, this is used to make the scores roughly fall between 0 and 1 (instead of, say, 0.95–0.97 for decent translations). The baseline represents the *expected score for a pair of random sentences* in the given language, which serves as an empirical lower bound .

* **Baseline Data:** The BERTScore authors precomputed baseline mean scores for each supported model and language using large corpora (Common Crawl monolingual data) . For example, for English RoBERTa-large, the baseline might be around 0.85 (meaning two random English sentences have \~85% similarity on average due to common stopwords and structure). They found that subtracting this and scaling makes scores “more human-readable” . We will obtain these baseline values from the BERTScore repository’s data files. Each model×language pair has a baseline file (e.g. `en/roberta-large.tsv`) containing the average P, R, F1 on random sentence pairs . If available, our Rust package can include these as resources or download on the fly.

* **Applying Rescaling:** If `rescale_with_baseline=True`, adjust each score as:
  $P' = \frac{P - P_b}{1 - P_b}, \quad R' = \frac{R - R_b}{1 - R_b}, \quad F1' = \frac{F1 - F1_b}{1 - F1_b},$
  where \$P\_b, R\_b, F1\_b\$ are the baseline precision, recall, and F1 for this model and language . This linear transformation maps the baseline score to 0 and an F1 of 1.0 stays 1.0 (if a candidate exactly matches the reference, it remains at the max of 1). After rescaling, typical scores might move into a range like 0.0–0.8 instead of 0.8–0.95, which is easier to interpret . For instance, in the example, unscaled F1 \~0.959 became rescaled F1 \~0.759 , and what was a small numeric difference from the maximum became a mid-range score.

* **Implementation:** After computing the raw P, R, F1 for each pair (weighted or unweighted), we will subtract the corresponding baseline values and divide by (1 - baseline). The baseline values will be looked up by a key combining model and language (e.g. `"roberta-large_en"`). We will load a table of these values when the scorer is initialized. The `rust-bert` library doesn’t provide these, so we will either hard-code known values or parse bundled TSV files. (The Lightning AI documentation confirms these baseline files are accessible from the original package ). If the model or language is unknown (no baseline available), we will either (a) fall back to not rescaling (with a warning), or (b) require the user to provide a custom baseline file path (our API can accept `baseline_path`, similar to the Python API). Providing a custom file lets advanced users compute their own baseline for new models or languages and use it.

* **No Impact on Ranking:** Note that rescaling is purely monotonic and does not affect the relative ordering of scores . It’s only for readability. We will clarify in documentation that turning this on does not change which candidate is better, only the numeric scale.

* **Example:** Suppose for a given pair we got F1 = 0.60 raw, and the baseline F1 for the model is 0.40. Then \$F1' = (0.60 - 0.40)/(1 - 0.40) = 0.333...\$ (33.3%). If another pair had F1 = 0.50 raw, it becomes \$(0.50-0.40)/0.60 = 0.1667\$ (16.7%). The ordering 0.60 > 0.50 remains 33.3% > 16.7%, but now a random pair would score around 0%. We will apply this formula to P, R, and F1 individually so the triple of scores is all in a more intuitive range.

## 7. Integration with Rust Ecosystem and Python Bindings

**Goal:** Tie all the above components into a cohesive Rust implementation and expose a user-friendly Python API for it. We focus on data flow, compatibility, and performance.

* **Data Flow Overview:** The pipeline will be encapsulated in a Rust struct (e.g. `BertScorePipeline`) with methods to initialize the model+tokenizer and to score inputs. The end-to-end flow for scoring will be:
  **Input:** Lists of candidate strings and reference strings (or list of reference list for multi-ref).
  **Output:** For each candidate (or candidate-reference pair), a tuple of P, R, F1 scores.
  **Process:** Tokenize inputs → batch inputs → model forward pass → compute similarities → P/R/F1 → apply IDF weighting (if enabled) → apply baseline rescaling (if enabled) → return scores.
  All intermediate steps (token lists, tensors, similarity matrices) are handled inside Rust, invisible to the user.

* **Batching and Performance:** The implementation will make use of batching to maximize throughput, especially when a GPU is available. Batching is applied at two levels:

  1. **Model Inference:** Process up to `batch_size` pairs at once through the model, as described. This keeps the GPU (or CPU vector units) busy and amortizes overhead of loading the model weights. The `batch_size` is tunable based on memory – too high could cause out-of-memory on GPU, so the default (64) can be adjusted by the user .
  2. **Similarity & Scoring:** The cosine similarity calculation for each pair can also leverage parallelism. We can use Rust’s Rayon or other multithreading to compute similarity matrices for multiple pairs in parallel (especially if running on CPU). Alternatively, if using GPU, each batch’s similarity computation is already vectorized in the matrix multiplication. The combination of these ensures the Rust implementation is highly efficient.

* **Tokenizer and Model Compatibility:** We ensure that the Rust tokenization exactly matches Python’s. The `rust-tokenizers` crate has been validated against Hugging Face tokenizers for BERT, RoBERTa, DeBERTa, etc. . We will use the same vocab files as the original models. For example, to load a tokenizer, we might require the vocab file path (and merges file for BPE). These can be obtained if using `rust-bert`’s PretrainedResource mechanism or by asking the user to specify. The model’s config (e.g. max position embeddings) is also considered – extremely long sentences may need truncation; we can truncate to the model’s max length with a warning (or let the user choose a truncation strategy similar to `transformers` behavior).

* **Numerical Fidelity:** All computations (tokenization, embeddings, similarities) are designed to yield the same results as the reference Python implementation up to floating-point precision differences. We use the same cosine similarity formula and matching strategy as described in the BERTScore paper . By using pre-trained weights and identical tokenization, the embedding values are the same as one would get in Python (via Transformers). Any small differences (e.g. due to Rust using IEEE754 floats in the same way as PyTorch) should be negligible. We can verify this by testing the Rust output on a few example pairs against `bert_score.score` outputs. The use of `tch` (which wraps libtorch C++ code) means the core transformer computations are literally the same as PyTorch’s, ensuring scientific accuracy.

* **Extensibility to Different Models:** Our design abstracts over model and tokenizer so that new models can be plugged in. We will support any *encoder-based* model that outputs token embeddings. For instance, BERT variants (RoBERTa, DistilBERT, ALBERT, etc.), Multilingual models (XLM-R), and others like ELECTRA (the repo explicitly supports ELECTRA as well ). Thanks to `rust-tokenizers` and `rust-bert`, many of these are already available. In practice, we’ll likely implement a factory that given a model identifier (like a Hugging Face model name or a shorthand) will:

  * Load the appropriate tokenizer (vocab + merges or sentencepiece).
  * Load the model config and weights into a `TorchModel` instance.
  * Set the layer and other options.
    This could be done via an enum for model type or a trait object if we want a uniform interface. However, since all these models produce `Tensor` outputs in the same format, we can unify the processing after the forward pass. The main difference is just *which Rust struct to instantiate* (e.g. `BertModel` vs `RobertaModel`). We can hide that behind the factory. The user-facing API can accept model names like `"microsoft/deberta-xlarge-mnli"` just like the Python version . Under the hood, if a name contains "deberta", we instantiate the DeBERTa model class, etc. If the model is not natively supported by `rust-bert`, we could fall back to an ONNX Runtime path (since `rust-bert` can use ONNX for unsupported models ) – though this is an advanced extension if needed.

* **Python Bindings Design:** We will expose this functionality to Python using **PyO3** or a similar binding. The plan is to create a Python module (e.g. `rust_bertscore`) that mimics the usage of the original `bert-score` library. For example:

  ```python
  import rust_bertscore  
  scorer = rust_bertscore.BERTScorer(model_type="roberta-large", lang="en", num_layers=17, idf=True, rescale_with_baseline=True)  
  P, R, F1 = scorer.score(candidates, references)
  ```

This would internally call into our Rust code. The `BERTScorer` object can hold the model and tokenizer in memory (to avoid re-loading on multiple calls, as the original library does ). We will ensure that multi-threading in Rust doesn’t conflict with Python’s GIL by releasing the GIL during heavy computation sections (PyO3 allows marking functions as `#[pyo3(text_signature, ...) -> PyResult]` and using `Python::allow_threads` for blocking calls). The Python API will also allow simple one-off use: a function `rust_bertscore.score(cands, refs, ...)` can create a scorer internally and return scores.

* **Resource Management:** Loading a large model in Rust can consume hundreds of MB of RAM/VRAM. We will provide a way to free the model (e.g. `scorer.release()` or Python `__del__`) which drops the `VarStore`. In Python, this might be tied to object finalization. Also, we consider loading the model to GPU once and using it for all batches; switching devices mid-run is not needed. The user can specify the device string (like "cuda:0" or "cpu") and we will map that to `tch::Device`.

* **Testing and Validation:** Every component will be tested against known values. For example, tokenization of a sample sentence will be compared to Hugging Face tokenizer outputs. We will test the cosine similarity and matching on small examples where we can calculate by hand. Most importantly, we’ll run the entire pipeline on a few pair examples and compare P/R/F1 to those from the original Python implementation (within tolerance). This ensures our Rust pipeline’s **scientific accuracy** is on par. We will also measure performance on large inputs to confirm we meet the expectation of improved speed (Rust’s zero-cost abstractions and optional GPU usage should make it as fast or faster than Python).

* **Future Extensibility:** The design is modular – one could extend it to support *new embedding models* (for example, using a decoder or seq2seq model’s encoder for scoring). As long as the model provides token-level embeddings, the rest of the pipeline remains the same. We can also integrate this with the Hugging Face `evaluate` library by writing a small wrapper so that `evaluate.load("bertscore", module="rust_bertscore")` uses our implementation under the hood. This would give users the speed of Rust with the convenience of Python integration.

## 8. Summary of Planned Implementation Steps

To summarize, we break down the implementation into concrete steps reflecting the above design:

* **Step 1:** **Initialize** – Load model config & weights via `rust-bert`. Load or download tokenizer files. Initialize tokenizer and model, set `output_hidden_states=true`. Prepare IDF baseline data if needed (compute idf\_dict from references or load provided dict). Also load baseline rescaling values if `rescale_with_baseline` is true (from internal data or file).

* **Step 2:** **Tokenize Inputs** – Convert all candidate and reference sentences to token ID sequences using the tokenizer. Store token IDs and attention masks, along with lengths (and IDF weights per token if using IDF). Handle multi-reference by tokenizing each reference in the list.

* **Step 3:** **Batching** – Partition the tokenized pairs into batches of size B. For each batch, create Torch tensors for candidate IDs, reference IDs, masks, etc. Move tensors to the device (CPU/GPU).

* **Step 4:** **Model Forward** – For the current batch, run the model on candidates and references separately (or as a single combined batch, then split results) to get hidden states. Extract the hidden state tensor for the chosen layer for each sequence. (Dimensions: for candidates batch: B × max\_Tc × hidden; for references batch: B × max\_Tr × hidden.)

* **Step 5:** **Similarity Calculation** – For each pair in the batch, slice out the actual token embeddings (exclude pads and special tokens). Normalize them. Compute the cosine similarity matrix (using matrix multiplication or equivalent). From this, derive the `max_candidate_sims` and `max_reference_sims` vectors (greedy matching).

* **Step 6:** **Score Aggregation** – Compute Precision and Recall for each pair. If IDF weighting is enabled, use the precomputed idf weights for those tokens in a weighted average . Otherwise, use a simple average. Then compute F1. This yields P, R, F1 for each pair in the batch.

* **Step 7:** **Baseline Rescale** – If requested, apply the linear rescaling to each of the P, R, F1 values using the baseline constants for the model .

* **Step 8:** **Collect Results** – Append the scores for this batch to an output list. Continue with the next batch until all pairs are processed.

* **Step 9:** **Return/Output** – If using Python bindings, format the results as Python lists or NumPy arrays for P, R, F1. If a `hashcode` or configuration identifier is needed (the original outputs a hash string identifying the config), we can generate a similar hash (e.g. model name + layer + idf flag + baseline flag) so that users can report it for consistency.

Each of these steps corresponds to components in the original BERTScore algorithm (tokenization, embedding, matching, scoring, IDF, baseline) and leverages Rust crates for efficiency. By following this plan, we will recreate the BERTScore metric in Rust with full fidelity to the published method , while also providing the benefits of Rust’s speed and safety in a reusable library form.

**Sources:** The design above is informed by the BERTScore paper and repository, and maps those concepts onto Rust tools and libraries (e.g. `rust-tokenizers` for tokenization , `tch`/`rust-bert` for model inference). This ensures the implementation is scientifically accurate and efficient.

```
