#!/usr/bin/env python
"""
Dump tokenization results from HuggingFace transformers for comparison.
Reads sentences from tests/sentences.txt and outputs to reports/tokens_py.tsv
"""
from transformers import AutoTokenizer
import os
import sys
from pathlib import Path

# Change to project directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
os.chdir(project_dir)

# Input and output files
input_file = "tests/sentences.txt"
output_file = "reports/tokens_py.tsv"

# Create directories if needed
os.makedirs("tests", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Create test sentences if file doesn't exist
if not os.path.exists(input_file):
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Transformers are changing NLP.",
        "¿Dónde está la biblioteca?",
        "你好，世界！",
        "Hello, world!",
        "I love 🍕 and 🍔!",
        "",  # Empty string test
    ]
    
    with open(input_file, "w", encoding="utf-8") as f:
        for sent in test_sentences:
            f.write(sent + "\n")
    
    print(f"✓ Created test sentences file: {input_file}")

# Load tokenizer
model_name = "roberta-large"
print(f"Loading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Process sentences
results = []
with open(input_file, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        sentence = line.rstrip("\n")  # Remove newline but keep the sentence as-is
        
        # Tokenize
        encoding = tokenizer(sentence, add_special_tokens=True)
        token_ids = encoding["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        
        # Format as tab-separated: sentence TAB [id1, id2, ...] TAB [token1, token2, ...]
        ids_str = str(token_ids)
        tokens_str = str(tokens)
        
        results.append(f"{sentence}\t{ids_str}\t{tokens_str}")
        
        # Also print summary
        print(f"Line {line_num}: {len(token_ids)} tokens")

# Write results
with open(output_file, "w", encoding="utf-8") as f:
    for result in results:
        f.write(result + "\n")

print(f"\n✓ Tokenization results saved to: {output_file}")
print(f"Processed {len(results)} sentences")

# Also create a more detailed JSON version for easier parsing
import json

json_output = "reports/tokens_py.json"
json_results = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        sentence = line.rstrip("\n")
        encoding = tokenizer(sentence, add_special_tokens=True)
        
        json_results.append({
            "sentence": sentence,
            "token_ids": encoding["input_ids"],
            "tokens": tokenizer.convert_ids_to_tokens(encoding["input_ids"]),
            "length": len(encoding["input_ids"]),
        })

with open(json_output, "w", encoding="utf-8") as f:
    json.dump(json_results, f, indent=2, ensure_ascii=False)

print(f"✓ Also saved JSON format to: {json_output}")