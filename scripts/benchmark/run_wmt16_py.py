#!/usr/bin/env python
"""
Compute system-level mean BERTScore F1 for each MT system (WMT16 en-de).
Requires:
  data/wmt16/ref.txt
  data/wmt16/sys/<system>.txt
Outputs CSV in reports/wmt16_sys_scores_py.csv
"""
from pathlib import Path
import pandas as pd
import bert_score as bs
import tqdm
import os
import sys

# Change to project directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
os.chdir(project_dir)

# Check if WMT16 data exists
ref_file = Path("data/benchmark/wmt16/ref.txt")
sys_dir = Path("data/benchmark/wmt16/sys")

if not ref_file.exists():
    print(f"✗ Reference file not found: {ref_file}")
    print("  Please run fetch_wmt16.sh to download WMT16 data")
    sys.exit(1)

if not sys_dir.exists() or not list(sys_dir.glob("*.txt")):
    print(f"✗ System outputs not found in: {sys_dir}")
    print("  Please run fetch_wmt16.sh to download WMT16 data")
    sys.exit(1)

# Create reports directory if needed
os.makedirs("reports", exist_ok=True)

# Load reference translations
ref_lines = ref_file.read_text(encoding="utf8").splitlines()
print(f"Loaded {len(ref_lines)} reference sentences")

# Process each system
records = []
sys_files = sorted(sys_dir.glob("*.txt"))
print(f"Found {len(sys_files)} MT systems to evaluate")

for sys_file in tqdm.tqdm(sys_files, desc="Evaluating systems"):
    cand_lines = sys_file.read_text(encoding="utf8").splitlines()
    
    # Ensure same number of lines
    if len(cand_lines) != len(ref_lines):
        print(f"⚠ Warning: {sys_file.name} has {len(cand_lines)} lines, expected {len(ref_lines)}")
        # Truncate or pad as needed
        if len(cand_lines) > len(ref_lines):
            cand_lines = cand_lines[:len(ref_lines)]
        else:
            cand_lines.extend([""] * (len(ref_lines) - len(cand_lines)))
    
    # Compute BERTScore
    P, R, F1 = bs.score(
        cand_lines, ref_lines,
        model_type="roberta-large",
        lang="en",
        idf=True,
        rescale_with_baseline=True,
        batch_size=64,
        verbose=False,
    )
    
    # Record system-level score
    records.append({
        "system": sys_file.stem,
        "mean_P_py": float(P.mean()),
        "mean_R_py": float(R.mean()),
        "mean_F1_py": float(F1.mean()),
        "std_F1_py": float(F1.std()),
    })

# Save results
df = pd.DataFrame(records)
df = df.sort_values("mean_F1_py", ascending=False)
output_file = "reports/wmt16_sys_scores_py.csv"
df.to_csv(output_file, index=False)

print(f"\n✓ WMT16 Python scoring done - saved to {output_file}")
print(f"\nTop 5 systems by F1:")
print(df[["system", "mean_F1_py"]].head())

# Also save segment-level scores for the best system
print(f"\n Computing segment-level scores for top system: {df.iloc[0]['system']}")
best_sys_file = sys_dir / f"{df.iloc[0]['system']}.txt"
best_cand_lines = best_sys_file.read_text(encoding="utf8").splitlines()[:len(ref_lines)]

P, R, F1 = bs.score(
    best_cand_lines, ref_lines,
    model_type="roberta-large",
    lang="en",
    idf=True,
    rescale_with_baseline=True,
    batch_size=64,
    verbose=False,
)

seg_records = []
for i, (p, r, f1) in enumerate(zip(P, R, F1)):
    seg_records.append({
        "system": df.iloc[0]['system'],
        "seg_id": i,
        "P_py": float(p),
        "R_py": float(r),
        "F1_py": float(f1),
    })

seg_df = pd.DataFrame(seg_records)
seg_output_file = "reports/wmt16_seg_scores_py.csv"
seg_df.to_csv(seg_output_file, index=False)
print(f"✓ Segment-level scores saved to {seg_output_file}")