#!/usr/bin/env python3
"""
Extract all baseline rescaling values from Python bert_score package.
Outputs a JSON file with all baseline data for integration into Rust.
"""

import os
import json
import pandas as pd
from pathlib import Path
import sys

def extract_all_baselines():
    """Extract baseline values from all TSV files in bert_score package."""
    # Find bert_score installation
    import bert_score
    bert_score_path = Path(bert_score.__file__).parent
    baseline_dir = bert_score_path / "rescale_baseline"
    
    if not baseline_dir.exists():
        print(f"Error: Baseline directory not found at {baseline_dir}")
        sys.exit(1)
    
    print(f"Found baseline directory: {baseline_dir}")
    
    all_baselines = {}
    
    # Walk through all language directories
    for lang_dir in sorted(baseline_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
            
        lang = lang_dir.name
        print(f"\nProcessing language: {lang}")
        all_baselines[lang] = {}
        
        # Process all TSV files in this language directory
        for tsv_file in sorted(lang_dir.glob("**/*.tsv")):
            # Get model name (handle subdirectories like microsoft/)
            relative_path = tsv_file.relative_to(lang_dir)
            model_name = str(relative_path).replace(".tsv", "")
            
            print(f"  - {model_name}")
            
            # Read TSV file
            try:
                df = pd.read_csv(tsv_file)
                
                # Convert to list of layer baselines
                baselines = []
                for _, row in df.iterrows():
                    baselines.append({
                        "layer": int(row["LAYER"]),
                        "precision": float(row["P"]),
                        "recall": float(row["R"]),
                        "f1": float(row["F"])
                    })
                
                all_baselines[lang][model_name] = baselines
                
            except Exception as e:
                print(f"    ERROR reading {tsv_file}: {e}")
    
    return all_baselines

def generate_summary(baselines):
    """Generate summary statistics about the baseline data."""
    total_models = 0
    models_by_lang = {}
    
    for lang, models in baselines.items():
        models_by_lang[lang] = len(models)
        total_models += len(models)
    
    print("\n" + "="*60)
    print("BASELINE DATA SUMMARY")
    print("="*60)
    print(f"Total languages: {len(baselines)}")
    print(f"Total model configurations: {total_models}")
    print("\nModels per language:")
    for lang, count in sorted(models_by_lang.items()):
        print(f"  {lang}: {count} models")
    
    # Find common models across languages
    all_model_names = set()
    for lang, models in baselines.items():
        all_model_names.update(models.keys())
    
    print(f"\nUnique model names: {len(all_model_names)}")

def main():
    """Extract baselines and save to JSON."""
    print("🔍 Extracting BERTScore Baseline Values")
    print("="*60)
    
    # Extract all baselines
    baselines = extract_all_baselines()
    
    # Generate summary
    generate_summary(baselines)
    
    # Save to JSON
    output_file = "data/baselines.json"
    os.makedirs("data", exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(baselines, f, indent=2, sort_keys=True)
    
    print(f"\n✓ Baselines saved to: {output_file}")
    
    # Also save a compact version without indentation
    compact_file = "data/baselines_compact.json"
    with open(compact_file, "w") as f:
        json.dump(baselines, f, separators=(",", ":"), sort_keys=True)
    
    print(f"✓ Compact version saved to: {compact_file}")
    
    # Print example usage
    print("\nExample baseline lookup:")
    print("  Language: en")
    print("  Model: roberta-large")
    print("  Layer 17 baselines:")
    if "en" in baselines and "roberta-large" in baselines["en"]:
        layer_17 = next(b for b in baselines["en"]["roberta-large"] if b["layer"] == 17)
        print(f"    Precision: {layer_17['precision']}")
        print(f"    Recall: {layer_17['recall']}")
        print(f"    F1: {layer_17['f1']}")

if __name__ == "__main__":
    main()