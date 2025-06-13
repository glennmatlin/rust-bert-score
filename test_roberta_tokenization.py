#!/usr/bin/env python3
"""Test RoBERTa tokenization of 'OK' and 'Okay' with and without prefix spaces."""

from transformers import RobertaTokenizer

def test_roberta_tokenization():
    # Load RoBERTa tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    
    # Test cases
    test_strings = [
        "OK",
        " OK",
        "Okay",
        " Okay",
        "ok",
        " ok",
        "okay",
        " okay"
    ]
    
    print("RoBERTa Tokenization Test")
    print("=" * 60)
    print(f"{'Input String':<15} {'Token IDs':<20} {'Tokens':<25}")
    print("-" * 60)
    
    for test_str in test_strings:
        # Tokenize
        token_ids = tokenizer.encode(test_str, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        
        # Format output
        input_display = repr(test_str)  # Use repr to show spaces clearly
        ids_display = str(token_ids)
        tokens_display = str(tokens)
        
        print(f"{input_display:<15} {ids_display:<20} {tokens_display:<25}")
    
    print("\n" + "=" * 60)
    print("\nAdditional Analysis:")
    print("-" * 60)
    
    # Check vocabulary directly
    print("\nDirect vocabulary lookups:")
    vocab_checks = ["OK", "ĠOK", "Okay", "ĠOkay", "ok", "Ġok", "okay", "Ġokay"]
    
    for token in vocab_checks:
        if token in tokenizer.vocab:
            token_id = tokenizer.vocab[token]
            print(f"  '{token}' -> ID: {token_id}")
        else:
            print(f"  '{token}' -> NOT IN VOCABULARY")
    
    print("\nNote: 'Ġ' represents a space in RoBERTa's vocabulary")
    
    # Test in context
    print("\n" + "=" * 60)
    print("\nTokenization in context:")
    print("-" * 60)
    
    context_examples = [
        "That's OK",
        "That's Okay",
        "OK then",
        "Okay then",
        "It's ok",
        "It's okay"
    ]
    
    for example in context_examples:
        token_ids = tokenizer.encode(example, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        print(f"\n'{example}':")
        print(f"  IDs: {token_ids}")
        print(f"  Tokens: {tokens}")

if __name__ == "__main__":
    test_roberta_tokenization()