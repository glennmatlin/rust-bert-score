#!/usr/bin/env python3
"""
Generate test data for validating BERTScore implementations.
Creates diverse sentence pairs for comprehensive testing.
"""

import json
import random
import string
import pandas as pd
from typing import List, Tuple
import os

def generate_paraphrases() -> List[Tuple[str, str]]:
    """Generate paraphrase pairs."""
    return [
        ("The cat sat on the mat.", "A feline was resting on the rug."),
        ("She completed the assignment yesterday.", "Yesterday, she finished the task."),
        ("The weather is beautiful today.", "Today's weather is lovely."),
        ("He enjoys playing guitar.", "Playing guitar is his hobby."),
        ("The meeting was postponed.", "They delayed the meeting."),
        ("I need to buy groceries.", "I have to purchase food items."),
        ("The book was interesting.", "It was an engaging read."),
        ("They arrived early.", "They came ahead of time."),
        ("The solution works perfectly.", "This approach functions flawlessly."),
        ("She speaks three languages.", "She is trilingual."),
    ]

def generate_near_duplicates() -> List[Tuple[str, str]]:
    """Generate near-duplicate pairs with minor changes."""
    base_sentences = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning models require large datasets",
        "Python is a popular programming language",
        "Climate change affects global weather patterns",
        "Regular exercise improves mental health",
    ]
    
    pairs = []
    for sent in base_sentences:
        # Add punctuation
        pairs.append((sent, sent + "."))
        # Change case
        pairs.append((sent, sent.lower()))
        # Add extra space
        pairs.append((sent, sent.replace(" ", "  ", 1)))
    
    return pairs

def generate_adversarial_pairs() -> List[Tuple[str, str]]:
    """Generate adversarial pairs that might confuse simple metrics."""
    return [
        # Word order changes meaning
        ("The dog bit the man.", "The man bit the dog."),
        ("John gave Mary a book.", "Mary gave John a book."),
        
        # Negation
        ("I love this movie.", "I hate this movie."),
        ("The test passed.", "The test failed."),
        
        # Different numbers
        ("The price is $100.", "The price is $200."),
        ("Meet me at 3 PM.", "Meet me at 5 PM."),
        
        # Opposite meanings
        ("The temperature increased.", "The temperature decreased."),
        ("Sales went up.", "Sales went down."),
    ]

def generate_length_varied_pairs() -> List[Tuple[str, str]]:
    """Generate pairs with varying length differences."""
    pairs = []
    
    # Very short
    pairs.extend([
        ("Yes", "No"),
        ("OK", "Okay"),
        ("Hi", "Hello"),
    ])
    
    # Short to medium
    pairs.extend([
        ("Good", "This is very good indeed"),
        ("I agree", "I completely agree with your assessment of the situation"),
    ])
    
    # Long sentences
    long_sent1 = "The development of artificial intelligence has revolutionized many industries, from healthcare to finance, enabling more efficient processes and better decision-making through advanced algorithms and machine learning techniques that can analyze vast amounts of data in real-time."
    long_sent2 = "AI has transformed numerous sectors including medical care and banking by implementing sophisticated computational methods that enhance operational efficiency and improve the quality of decisions through rapid data analysis."
    pairs.append((long_sent1, long_sent2))
    
    return pairs

def generate_special_character_pairs() -> List[Tuple[str, str]]:
    """Generate pairs with special characters, numbers, and symbols."""
    return [
        # Emails and URLs
        ("Contact us at info@example.com", "Email: info@example.com"),
        ("Visit https://www.example.com", "Go to www.example.com"),
        
        # Code-like text
        ("if x > 0: return True", "return True if x > 0"),
        ("array[0] = 'hello'", "array[0] = \"hello\""),
        
        # Math and symbols
        ("2 + 2 = 4", "Two plus two equals four"),
        ("Temperature: 25°C", "Temperature: 77°F"),
        
        # Currency and percentages
        ("Save 50% today!", "Half price today!"),
        ("Cost: €100", "Cost: 100 euros"),
    ]

def generate_multilingual_pairs() -> List[Tuple[str, str]]:
    """Generate pairs with non-English text (for robustness testing)."""
    return [
        # Mixed languages
        ("Hello world", "Hola mundo"),
        ("Thank you", "Merci"),
        
        # Unicode
        ("Café", "Coffee shop"),
        ("naïve", "naive"),
        
        # Emoji
        ("I love pizza 🍕", "I love pizza"),
        ("Great job! 👍", "Great job!"),
    ]

def generate_edge_cases() -> List[Tuple[str, str]]:
    """Generate edge case pairs."""
    return [
        # Empty strings
        ("", ""),
        ("", "Something"),
        ("Something", ""),
        
        # Only punctuation
        ("...", "!!!"),
        ("?", "!"),
        
        # Repeated characters
        ("AAAAAA", "aaaaaa"),
        ("Hello!!!!!!", "Hello!"),
        
        # Only numbers
        ("123456", "654321"),
        ("1.5", "1.50"),
        
        # Whitespace variations
        ("  spaces  ", "spaces"),
        ("tab\ttab", "tab tab"),
        ("line\nbreak", "line break"),
    ]

def generate_domain_specific_pairs() -> List[Tuple[str, str]]:
    """Generate domain-specific pairs (medical, legal, technical)."""
    return [
        # Medical
        ("The patient has hypertension.", "The patient has high blood pressure."),
        ("Administer 5mg of medication.", "Give 5 milligrams of the drug."),
        
        # Legal
        ("The defendant pleaded not guilty.", "The accused denied the charges."),
        ("The contract is null and void.", "The agreement is invalid."),
        
        # Technical
        ("Restart the server.", "Reboot the system."),
        ("The API returned a 404 error.", "The interface gave a not found error."),
    ]

def save_test_data(output_dir: str = "data"):
    """Save all test data to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine all pairs
    all_pairs = []
    
    # Add pairs from each category with labels
    categories = [
        ("paraphrase", generate_paraphrases()),
        ("near_duplicate", generate_near_duplicates()),
        ("adversarial", generate_adversarial_pairs()),
        ("length_varied", generate_length_varied_pairs()),
        ("special_chars", generate_special_character_pairs()),
        ("multilingual", generate_multilingual_pairs()),
        ("edge_cases", generate_edge_cases()),
        ("domain_specific", generate_domain_specific_pairs()),
    ]
    
    for category, pairs in categories:
        for cand, ref in pairs:
            all_pairs.append({
                "candidate": cand,
                "reference": ref,
                "category": category,
                "id": f"{category}_{len(all_pairs)}"
            })
    
    # Save as TSV (as specified in EXPERIMENTS.md)
    df = pd.DataFrame(all_pairs)
    tsv_path = os.path.join(output_dir, "test_pairs.tsv")
    df[["id", "candidate", "reference"]].to_csv(tsv_path, sep="\t", index=False)
    print(f"Saved {len(all_pairs)} test pairs to {tsv_path}")
    
    # Also save with categories as JSON for analysis
    json_path = os.path.join(output_dir, "test_pairs_full.json")
    with open(json_path, "w") as f:
        json.dump(all_pairs, f, indent=2)
    print(f"Saved full data with categories to {json_path}")
    
    # Save category statistics
    stats = df["category"].value_counts().to_dict()
    stats_path = os.path.join(output_dir, "test_data_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "total_pairs": len(all_pairs),
            "categories": stats,
            "description": "Test data for BERTScore implementation validation"
        }, f, indent=2)
    print(f"Saved statistics to {stats_path}")
    
    return df

def generate_wmt_style_data(num_systems: int = 5, num_segments: int = 100):
    """Generate synthetic WMT-style evaluation data."""
    output_dir = "data/synthetic_wmt"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate reference translations
    references = []
    for i in range(num_segments):
        ref = f"This is reference sentence number {i} with some content about topic {random.choice(['science', 'technology', 'politics', 'sports'])}."
        references.append(ref)
    
    # Save references
    with open(os.path.join(output_dir, "ref.txt"), "w") as f:
        f.write("\n".join(references))
    
    # Generate system outputs with varying quality
    os.makedirs(os.path.join(output_dir, "sys"), exist_ok=True)
    
    systems_data = []
    for sys_id in range(num_systems):
        system_outputs = []
        quality = random.uniform(0.5, 1.0)  # System quality factor
        
        for ref in references:
            if random.random() < quality:
                # Good translation (paraphrase)
                words = ref.split()
                random.shuffle(words[2:-2])  # Shuffle middle words
                output = " ".join(words)
            else:
                # Poor translation
                output = f"This is a poor translation for system {sys_id}."
            
            system_outputs.append(output)
        
        # Save system outputs
        sys_path = os.path.join(output_dir, "sys", f"system{sys_id}.txt")
        with open(sys_path, "w") as f:
            f.write("\n".join(system_outputs))
        
        # Track system quality for synthetic scores
        systems_data.append({
            "system": f"system{sys_id}",
            "quality": quality,
            "human_score": quality * 100  # Synthetic human score
        })
    
    # Save synthetic human scores
    human_df = pd.DataFrame(systems_data)
    human_df.to_csv(os.path.join(output_dir, "human_sys_scores.tsv"), sep="\t", index=False)
    
    print(f"Generated synthetic WMT data:")
    print(f"  - {num_segments} segments")
    print(f"  - {num_systems} systems")
    print(f"  - Saved to {output_dir}/")

def main():
    """Generate all test data."""
    print("Generating BERTScore Test Data")
    print("=" * 50)
    
    # Generate main test pairs
    print("\nGenerating diverse test pairs...")
    df = save_test_data()
    
    print(f"\nTotal test pairs generated: {len(df)}")
    print("\nBreakdown by category:")
    print(df["category"].value_counts())
    
    # Generate synthetic WMT data
    print("\n" + "=" * 50)
    print("Generating synthetic WMT-style data...")
    generate_wmt_style_data()
    
    print("\n✅ Test data generation complete!")

if __name__ == "__main__":
    main()