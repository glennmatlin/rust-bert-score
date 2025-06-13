````markdown
# PYTHON.md  
**Automated Instructions for the Codex CLI Agent**  
*(Setting up the Python side of the BERTScore-comparison project with `uv`)*  

---

## 0  Prerequisites  

| Requirement | Notes |
|-------------|-------|
| **OS**      | Ubuntu 22.04 LTS or newer (container or bare-metal) |
| **Python**  | CPython ≥ 3.10 (managed by `uv`) |
| **CUDA**    | Optional. If GPU is present, install NVIDIA drivers + CUDA 11.8 runtime before step 2. |

---

## 1  Install `uv` and Bootstrap a Virtual Environment  

```bash
# 1-A. Install uv globally (recommended via pipx)
pipx install uv

# 1-B. Create a project-local virtual env (folder .venv)
uv venv .venv
. .venv/bin/activate           # activate for the current shell
````

> **Why `uv`?**
>
> * `uv` provides fast resolver + installer, lock-file generation, and Python version management out-of-the-box.

---

## 2  Create a Precise `pyproject.toml`

Save the following as **`pyproject.toml`** in project root:

```toml
[project]
name = "bertscore-python-benchmark"
version = "0.1.0"
description = "Python reference side for Rust vs Python BERTScore comparison"

[tool.uv]                 # uv-specific section
python = "3.11"           # ensure consistent interpreter

[project.dependencies]
bert-score = "*"
torch = {version = ">=2.2,<3.0"}
transformers = ">=4.41"
pandas = "*"
numpy = "*"
scipy = "*"
tqdm = "*"
seaborn = "*"
```

Lock and install:

```bash
uv pip install -r requirements.txt    # ← legacy syntax; with pyproject use:
uv pip install -e .                   # read deps from pyproject, editable install
uv pip freeze > reports/pip_freeze.txt
```

*If you prefer full lock-file:*

```bash
uv pip compile -o requirements.lock
uv pip sync              # installs exactly the lock
```

---

## 3  Project Directory Skeleton

```text
project_root/
├── pyproject.toml
├── PYTHON.md                # ← this file
├── scripts/
│   ├── run_direct_py.py
│   ├── run_wmt16_py.py
│   ├── compare_direct.py
│   ├── compare_wmt16.py
│   └── dump_tokens_py.py
├── data/                    # datasets live here
│   └── direct_eval_pairs.tsv
└── reports/
```

> The Rust side (crate + bindings) will live in `rust_score/`, but that’s out-of-scope for this file.

---

## 4  Script Templates

### 4-A  `scripts/run_direct_py.py`

```python
#!/usr/bin/env python
"""
Compute BERTScore on a small ad-hoc test suite using official Python package.
Outputs: reports/direct_scores_python.csv
"""
import pandas as pd, bert_score as bs

df = pd.read_csv("data/direct_eval_pairs.tsv", sep="\t")
P, R, F1 = bs.score(
    df["candidate"].tolist(),
    df["reference"].tolist(),
    model_type="roberta-large",
    lang="en",
    rescale_with_baseline=True,
    batch_size=32,
)
df["P_py"] = P.numpy()
df["R_py"] = R.numpy()
df["F1_py"] = F1.numpy()
df.to_csv("reports/direct_scores_python.csv", index=False)
print("✓ Python direct-set scoring complete — wrote reports/direct_scores_python.csv")
```

> **Run:**
> `python scripts/run_direct_py.py`

### 4-B  `scripts/run_wmt16_py.py`

```python
#!/usr/bin/env python
"""
Compute system-level mean BERTScore F1 for each MT system (WMT16 en-de).
Requires:
  data/wmt16/ref.txt
  data/wmt16/sys/<system>.txt
Outputs CSV in reports/wmt16_sys_scores_py.csv
"""
from pathlib import Path
import pandas as pd, bert_score as bs, tqdm

ref_lines = Path("data/wmt16/ref.txt").read_text(encoding="utf8").splitlines()
records = []

for sys_file in sorted(Path("data/wmt16/sys").glob("*.txt")):
    cand_lines = sys_file.read_text(encoding="utf8").splitlines()
    P, R, F1 = bs.score(
        cand_lines, ref_lines,
        model_type="roberta-large",
        lang="en",
        idf=True,
        rescale_with_baseline=True,
        batch_size=64,
        verbose=True,
    )
    records.append({"system": sys_file.stem, "mean_F1_py": float(F1.mean())})

pd.DataFrame(records).to_csv("reports/wmt16_sys_scores_py.csv", index=False)
print("✓ WMT16 Python scoring done")
```

### 4-C  `scripts/compare_direct.py`

```python
#!/usr/bin/env python
"""
Compare direct-set results between Python and Rust CSV outputs.
Outputs simple stats report to stdout + writes reports/direct_agreement.txt
"""
import pandas as pd, numpy as np, scipy.stats as ss, sys, textwrap

p = pd.read_csv("reports/direct_scores_python.csv")
r = pd.read_csv("reports/direct_scores_rust.csv")

if not (p.id == r.id).all():
    sys.exit("✗ ID mismatch between CSVs")

delta = p[["F1_py"]] - r[["F1_rust"]]
max_abs = delta.abs().max().item()
pearson = ss.pearsonr(p["F1_py"], r["F1_rust"])[0]

report = textwrap.dedent(f"""
    DIRECT-SET AGREEMENT
    --------------------
    samples        : {len(p)}
    max |ΔF1|      : {max_abs:.6g}
    Pearson r(F1)  : {pearson:.6f}
""").strip()

print(report)
Path("reports/direct_agreement.txt").write_text(report)
```

*(The Rust CSV is produced by `scripts/run_direct_rust.py` in the Rust docs.)*

---

## 5  Data Acquisition Helpers

### 5-A  Download WMT16 Metrics-Task Corpus

Add **`scripts/fetch_wmt16.sh`**:

```bash
#!/usr/bin/env bash
set -e
mkdir -p data/wmt16
cd data/wmt16
curl -O https://www.statmt.org/wmt16/metric-task.tgz
tar -xzf metric-task.tgz
# Example extraction commands ...
# Convert into ref.txt and sys/*.txt expected by run_wmt16_py.py
```

*(Implement exact conversion steps as needed — see original WMT16 folder structure.)*

### 5-B  Generate Direct-Set TSV

If you don’t have the TSV yet:

```python
# scripts/make_direct_set.py
import csv, random, pathlib
pairs = [
    ("The quick brown fox jumps over the lazy dog .", "The quick brown fox jumps over the lazy dog ."),
    ("Hello world !", "Hi world ."),
    # add more pairs or pull from PAWS/WMT/edge cases...
]
random.shuffle(pairs)
with open("data/direct_eval_pairs.tsv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["id","candidate","reference"])
    for i,(c,r) in enumerate(pairs,1):
        w.writerow([f"S{i:04d}", c, r])
print("✓ Wrote data/direct_eval_pairs.tsv")
```

---

## 6  CI Recipe Snippet (GitHub Actions)

```yaml
jobs:
  python-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        run: pipx install uv
      - name: Set up env
        run: |
          uv venv .venv
          source .venv/bin/activate
          uv pip install -e .
      - name: Direct set score (Python)
        run: |
          source .venv/bin/activate
          python scripts/run_direct_py.py
      # rust side runs separately → uploads its CSV
      - name: Compare direct
        run: |
          source .venv/bin/activate
          python scripts/compare_direct.py
```

*(Extend with WMT16 steps once Rust outputs are present.)*

---

## 7  Best-Practice Notes for the Agent

1. **Lock dependencies** with `uv pip compile` once versions stabilise; commit `requirements.lock`.
2. **Cache models**: set `export HF_HOME=$PWD/.hf_cache` to avoid re-downloads.
3. **GPU vs CPU**: expose env `USE_GPU=1` to Python scripts; pass `device="cuda"` to `bert_score.score` when `torch.cuda.is_available()`.
4. **Repro seeds**: set `torch.manual_seed(42)` in each script (though BERTScore has deterministic forward pass).
5. **Artifact paths**: keep all CSV/plots in `reports/` so Rust side can reference them for comparison.
6. **Logging**: each script should `print` key stats and write a log file in `reports/`.

---

## 8  Execution Order Checklist

1. `uv venv .venv && . .venv/bin/activate`
2. `uv pip install -e .`
3. `bash scripts/fetch_wmt16.sh`  *(if not present)*
4. `python scripts/make_direct_set.py` *(optional)*
5. `python scripts/run_direct_py.py`
6. `python scripts/run_wmt16_py.py`
7. Wait for Rust side to produce corresponding CSVs.
8. `python scripts/compare_direct.py`
9. `python scripts/compare_wmt16.py`

---

### End of `PYTHON.md`

```
```
