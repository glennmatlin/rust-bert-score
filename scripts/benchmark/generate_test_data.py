#!/usr/bin/env python3
"""
Generate comprehensive test data for BERTScore validation.
Creates systematic test cases covering various scenarios.
"""

import pandas as pd
import random
import string
import os
from typing import List, Tuple
import itertools

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
os.chdir(project_dir)

# Word lists for generating sentences
NOUNS = ["cat", "dog", "house", "tree", "book", "computer", "phone", "car", "person", "city"]
VERBS = ["runs", "jumps", "reads", "writes", "builds", "drives", "calls", "sees", "helps", "makes"]
ADJECTIVES = ["big", "small", "red", "blue", "happy", "sad", "fast", "slow", "new", "old"]
ARTICLES = ["the", "a", "an"]

def generate_random_sentence(length: int = 5) -> str:
    """Generate a random sentence of approximately the given length."""
    words = []
    for i in range(length):
        if i % 3 == 0:
            words.append(random.choice(ARTICLES))
            words.append(random.choice(ADJECTIVES))
            words.append(random.choice(NOUNS))
        elif i % 3 == 1:
            words.append(random.choice(VERBS))
        else:
            words.append(random.choice(ARTICLES))
            words.append(random.choice(NOUNS))
    
    sentence = " ".join(words[:length])
    return sentence.capitalize() + "."

def generate_length_tests() -> List[Tuple[str, str, str]]:
    """Generate test cases for different text lengths."""
    tests = []
    
    # Very short texts
    for i in range(1, 6):
        text = " ".join(["word"] * i)
        tests.append((f"len_{i:03d}", text, text, f"Length {i} words"))
    
    # Medium length texts
    for length in [10, 20, 50, 100]:
        sent = generate_random_sentence(length)
        tests.append((f"len_{length:03d}", sent, sent, f"Length ~{length} words"))
    
    # Near max length (512 tokens ~ 350-400 words for RoBERTa)
    for length in [300, 350, 400, 450]:
        words = [random.choice(NOUNS) for _ in range(length)]
        text = " ".join(words)
        tests.append((f"len_{length:03d}", text, text, f"Length {length} words"))
    
    return tests

def generate_similarity_gradient() -> List[Tuple[str, str, str]]:
    """Generate pairs with gradually decreasing similarity."""
    tests = []
    
    base = "The quick brown fox jumps over the lazy dog"
    
    # Identical
    tests.append(("sim_100", base, base, "100% identical"))
    
    # One word changes
    variations = [
        "The quick brown fox jumps over the lazy cat",
        "The quick brown fox leaps over the lazy dog",
        "The fast brown fox jumps over the lazy dog",
        "A quick brown fox jumps over the lazy dog",
    ]
    
    for i, var in enumerate(variations):
        tests.append((f"sim_90_{i}", base, var, "One word different"))
    
    # Multiple word changes
    tests.append(("sim_70", base, "The brown fox quickly jumps over a lazy dog", "Word order + changes"))
    tests.append(("sim_50", base, "A fox jumps over a dog", "Simplified version"))
    tests.append(("sim_30", base, "Animals are jumping", "Highly abstracted"))
    tests.append(("sim_10", base, "The weather is nice today", "Completely different"))
    
    return tests

def generate_special_character_tests() -> List[Tuple[str, str, str]]:
    """Generate tests with special characters and formatting."""
    tests = []
    
    # Punctuation variations
    base = "Hello world"
    punctuations = [".", "!", "?", "...", "!?", ";", ":", ","]
    
    for i, punct in enumerate(punctuations):
        tests.append((f"punct_{i:02d}", base + punct, base, f"Added {punct}"))
    
    # Special symbols
    symbols = ["@", "#", "$", "%", "^", "&", "*", "()", "[]", "{}"]
    for i, symbol in enumerate(symbols):
        text = f"Test {symbol} text"
        tests.append((f"symbol_{i:02d}", text, text, f"Contains {symbol}"))
    
    # Unicode categories
    unicode_tests = [
        ("™", "trademark"),
        ("®", "registered"),
        ("©", "copyright"),
        ("€", "euro"),
        ("£", "pound"),
        ("¥", "yen"),
        ("°", "degree"),
        ("²", "squared"),
        ("½", "half"),
        ("…", "ellipsis"),
    ]
    
    for i, (char, name) in enumerate(unicode_tests):
        tests.append((f"unicode_{i:02d}", f"Price: 100{char}", f"Price: 100{name}", f"Unicode {name}"))
    
    return tests

def generate_multilingual_tests() -> List[Tuple[str, str, str]]:
    """Generate tests with multiple languages and scripts."""
    tests = []
    
    # Common phrases in different languages
    hello_phrases = [
        ("en", "Hello world", "Hello world"),
        ("es", "Hola mundo", "Hello world"),
        ("fr", "Bonjour le monde", "Hello world"),
        ("de", "Hallo Welt", "Hello world"),
        ("it", "Ciao mondo", "Hello world"),
        ("pt", "Olá mundo", "Hello world"),
        ("ru", "Привет мир", "Hello world"),
        ("ja", "こんにちは世界", "Hello world"),
        ("zh", "你好世界", "Hello world"),
        ("ar", "مرحبا بالعالم", "Hello world"),
        ("hi", "नमस्ते दुनिया", "Hello world"),
        ("ko", "안녕하세요 세계", "Hello world"),
    ]
    
    for lang, native, english in hello_phrases:
        tests.append((f"lang_{lang}_same", native, native, f"{lang} identical"))
        tests.append((f"lang_{lang}_trans", native, english, f"{lang} vs English"))
    
    # Mixed language texts
    tests.append(("mixed_01", "Hello 世界", "Hello world", "English + Chinese"))
    tests.append(("mixed_02", "Bonjour мир", "Hello world", "French + Russian"))
    tests.append(("mixed_03", "1234 こんにちは ABC", "1234 hello ABC", "Numbers + Japanese + Latin"))
    
    return tests

def generate_adversarial_tests() -> List[Tuple[str, str, str]]:
    """Generate adversarial test cases designed to find edge cases."""
    tests = []
    
    # Repeated characters
    for char in ['a', ' ', '.', '!', '?', '\n', '\t']:
        for count in [2, 5, 10, 50]:
            text = char * count
            tests.append((f"repeat_{ord(char):03d}_{count:02d}", text, char, f"Repeat '{char}' {count}x"))
    
    # Mixed whitespace
    whitespaces = [' ', '\t', '\n', '\r', '\u00A0', '\u2003']
    for i, ws_combo in enumerate(itertools.combinations(whitespaces, 2)):
        text = "Hello" + ws_combo[0] + "world" + ws_combo[1]
        tests.append((f"ws_mix_{i:02d}", text, "Hello world", f"Mixed whitespace"))
    
    # Case variations
    text = "The Quick BROWN fox"
    variations = [
        text.lower(),
        text.upper(),
        text.title(),
        text.swapcase(),
    ]
    
    for i, var in enumerate(variations):
        tests.append((f"case_{i:02d}", text, var, f"Case variation {i}"))
    
    # Near-empty texts
    tests.append(("near_empty_01", " ", "", "Single space vs empty"))
    tests.append(("near_empty_02", "\n", "", "Newline vs empty"))
    tests.append(("near_empty_03", "\t", "", "Tab vs empty"))
    tests.append(("near_empty_04", ".", "", "Single punct vs empty"))
    
    # Tokenization edge cases
    tests.append(("token_01", "don't", "do not", "Contraction"))
    tests.append(("token_02", "U.S.A.", "USA", "Abbreviation"))
    tests.append(("token_03", "e-mail", "email", "Hyphenation"))
    tests.append(("token_04", "co-operate", "cooperate", "Hyphen vs compound"))
    
    return tests

def generate_numerical_tests() -> List[Tuple[str, str, str]]:
    """Generate tests focusing on numbers and numerical content."""
    tests = []
    
    # Number formats
    numbers = [
        ("123", "123", "Integer"),
        ("123.45", "123.45", "Decimal"),
        ("1,234", "1234", "Thousands separator"),
        ("1.234,56", "1234.56", "European format"),
        ("1e6", "1000000", "Scientific notation"),
        ("1st", "first", "Ordinal"),
        ("2nd", "second", "Ordinal"),
        ("3rd", "third", "Ordinal"),
        ("10%", "10 percent", "Percentage"),
        ("$100", "100 dollars", "Currency"),
    ]
    
    for i, (num1, num2, desc) in enumerate(numbers):
        tests.append((f"num_{i:02d}", f"The value is {num1}", f"The value is {num2}", desc))
    
    # Math expressions
    expressions = [
        ("2+2=4", "2 + 2 = 4", "Math with spaces"),
        ("x^2", "x squared", "Exponent"),
        ("√16", "square root of 16", "Root symbol"),
        ("π≈3.14", "pi is approximately 3.14", "Greek letter"),
        ("∑x", "sum of x", "Sum symbol"),
    ]
    
    for i, (expr1, expr2, desc) in enumerate(expressions):
        tests.append((f"math_{i:02d}", expr1, expr2, desc))
    
    return tests

def main():
    """Generate all test data files."""
    print("🏗️  Generating Comprehensive Test Data")
    print("=" * 60)
    
    # Create output directory
    output_dir = "data/benchmark/generated"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all test categories
    test_generators = [
        ("length_tests", generate_length_tests),
        ("similarity_gradient", generate_similarity_gradient),
        ("special_characters", generate_special_character_tests),
        ("multilingual", generate_multilingual_tests),
        ("adversarial", generate_adversarial_tests),
        ("numerical", generate_numerical_tests),
    ]
    
    all_tests = []
    
    for name, generator in test_generators:
        print(f"\nGenerating {name}...")
        tests = generator()
        
        # Create individual file
        df = pd.DataFrame(tests, columns=["id", "candidate", "reference", "description"])
        output_file = os.path.join(output_dir, f"{name}.tsv")
        df.to_csv(output_file, sep="\t", index=False)
        print(f"  ✓ Generated {len(tests)} test cases → {output_file}")
        
        # Add to combined dataset
        df["category"] = name
        all_tests.append(df)
    
    # Create combined dataset
    combined_df = pd.concat(all_tests, ignore_index=True)
    combined_file = os.path.join(output_dir, "all_tests.tsv")
    combined_df.to_csv(combined_file, sep="\t", index=False)
    print(f"\n✓ Combined dataset: {len(combined_df)} test cases → {combined_file}")
    
    # Create a golden dataset with known good scores
    # (This would need to be manually verified or computed once and saved)
    golden_tests = [
        ("gold_001", "Hello world", "Hello world", 1.0, "Identical simple"),
        ("gold_002", "The cat sat on the mat", "The cat sat on the mat", 1.0, "Identical sentence"),
        ("gold_003", "", "", 1.0, "Both empty"),
        ("gold_004", "   Hello world   ", "Hello world", 1.0, "Whitespace trimmed"),
        # Add more golden tests with known expected scores
    ]
    
    golden_df = pd.DataFrame(golden_tests, columns=["id", "candidate", "reference", "expected_f1", "description"])
    golden_file = os.path.join(output_dir, "golden_tests.tsv")
    golden_df.to_csv(golden_file, sep="\t", index=False)
    print(f"\n✓ Golden test set: {len(golden_df)} test cases → {golden_file}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total test cases generated: {len(combined_df)}")
    print(f"Output directory: {output_dir}")
    print("\nTest categories:")
    for category, count in combined_df["category"].value_counts().items():
        print(f"  - {category}: {count} tests")
    
    print("\nNext steps:")
    print("1. Run validation on generated test data")
    print("2. Analyze results for patterns")
    print("3. Create regression tests from failures")

if __name__ == "__main__":
    main()