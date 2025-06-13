#!/usr/bin/env python
"""
Generate direct evaluation pairs TSV file for testing.
Outputs: data/benchmark/direct_eval_pairs.tsv
"""
import csv
import os
import random

# Change to project directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))  # Go up two levels: scripts/benchmark -> scripts -> root
os.chdir(project_dir)

# Define test pairs (from the Rust test data generation)
pairs = [
    # Identical sentences
    ("The quick brown fox jumps over the lazy dog.", 
     "The quick brown fox jumps over the lazy dog."),
    
    # Paraphrases
    ("The cat sat on the mat.",
     "A feline was resting on the rug."),
    ("She completed the assignment yesterday.", 
     "Yesterday, she finished the task."),
    ("The weather is beautiful today.", 
     "Today's weather is lovely."),
    
    # Different word order
    ("Alice gave Bob a book.",
     "Bob was given a book by Alice."),
    
    # Partial overlap
    ("The weather is nice today.",
     "Today the weather seems pleasant."),
    
    # Completely different
    ("I love programming in Rust.",
     "The rain in Spain stays mainly in the plain."),
    
    # Long sentences
    ("The development of artificial intelligence has revolutionized many industries, from healthcare to finance, enabling more efficient processes and better decision-making through advanced algorithms and machine learning techniques.",
     "AI has transformed numerous sectors including medical care and banking by implementing sophisticated computational methods that enhance operational efficiency and improve the quality of decisions."),
    
    # Short sentences
    ("Yes.", "No."),
    ("OK", "Okay"),
    
    # Numbers and special characters
    ("The price is $29.99 (30% off)!",
     "Cost: twenty-nine dollars and ninety-nine cents with 30 percent discount."),
    
    # Code-like text
    ("def hello(): print('Hello, world!')",
     "function hello() { console.log('Hello, world!'); }"),
    
    # Empty vs non-empty
    ("", "This is some text."),
    
    # Punctuation differences
    ("Hello, world!", "Hello world"),
    
    # Case differences
    ("THIS IS UPPERCASE", "this is uppercase"),
    
    # Unicode and emoji
    ("I love 🍕 and 🍔!", "I enjoy pizza and hamburgers!"),
    
    # Technical jargon
    ("The transformer architecture uses self-attention mechanisms.",
     "Self-attention is a key component of transformers."),
    
    # Negation
    ("I like this movie.", "I don't like this movie."),
    
    # Additional edge cases
    ("   Leading and trailing spaces   ", "Leading and trailing spaces"),
    ("Multiple   spaces   between   words", "Multiple spaces between words"),
    ("Line\nbreak\nin\ntext", "Line break in text"),
]

# Shuffle for variety
random.shuffle(pairs)

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Write TSV file
output_file = "data/benchmark/direct_eval_pairs.tsv"
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["id", "candidate", "reference"])
    for i, (candidate, reference) in enumerate(pairs, 1):
        writer.writerow([f"S{i:04d}", candidate, reference])

print(f"✓ Wrote {len(pairs)} sentence pairs to {output_file}")