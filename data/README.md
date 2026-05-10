# Data

This directory contains the datasets used for training and evaluating our Sudoku-solving model.

---

## Dataset Overview

We use the **[`sapientinc/sudoku-extreme`](https://huggingface.co/datasets/sapientinc/sudoku-extreme)** dataset, hosted on Hugging Face. It consists of 9×9 Sudoku puzzles paired with their unique solutions.

| Split | Format |
|-------|--------|
| `train` | 262,144 puzzles | CSV |
| `test` | 32,768 puzzles | CSV |

> The full dataset is significantly larger. The sizes above reflect the defaults used in our experiments; these are configurable via `get_loaders()` in `Sudoku_DataLoader.py`.

---

## Data Format

Each CSV file (`train.csv`, `test.csv`) contains two columns:

| Column | Type | Description |
|--------|------|-------------|
| `question` | string (81 chars) | The puzzle, represented as a flat string of 81 characters. Known digits are represented as `'1'`–`'9'`; empty cells are represented as `'.'`. |
| `answer` | string (81 chars) | The complete solution, represented as a flat string of 81 digits (`'1'`–`'9'`), with no empty cells. |

### Example

```
question: ..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..
answer:   483921657967345821251876493548132976729564138136798245372689514814253769695417382
```

Each string encodes the board in **row-major order**: positions 0–8 are the first row, 9–17 the second, and so on.

---

## Encoding

During loading, puzzles are encoded into integer tensors:

- **Input (`x`)**: Each character is converted to an integer. Digits `'1'`–`'9'` map to `1`–`9`; empty cells (`'.'`) map to `0`. Shape: `(81,)`, dtype `torch.long`.
- **Target (`y`)**: Each digit in the solution maps to its integer value `1`–`9`. Shape: `(81,)`, dtype `torch.long`.

This encoding is handled by the `encode()` function in `Datasets/Sudoku_DataLoader.py`.

---

## Obtaining the Data

### Option 1 — Automatic download via the dataloader (recommended)

The dataloader downloads data automatically from Hugging Face at runtime. No manual steps required (recommended batch size is 2^7 but can be adjusted):

```python

from Datasets.Sudoku_DataLoader import get_loaders

train_loader, test_loader = get_loaders(train_size=2**18, test_size=2**15, batch_size=2**7)
```

Note that the import statement assumes you are in the data folder and may need to be adjusted. Internally, `get_loaders()` calls `hf_hub_download()` to fetch `train.csv` and `test.csv` from the `sapientinc/sudoku-extreme` repository and caches them locally via the Hugging Face Hub cache (`~/.cache/huggingface/`).


### Option 2 — Manual download from Hugging Face

```bash
import csv
from huggingface_hub import hf_hub_download
from datasets import Dataset

def load_split(split: str, size: int) -> Dataset:
    path = hf_hub_download(
        repo_id="sapientinc/sudoku-extreme",
        filename=f"{split}.csv",   # adjust if the filename on the Hub differs
        repo_type="dataset"
    )

    records = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)   # pure Python, no PyArrow involved
        for i, row in enumerate(reader):
            if i >= size:
                break
            records.append(row)      # all values are already strings

    return Dataset.from_list(records)

# Example: load the first 1000 train rows and 200 test rows
train_ds = load_split("train", 1000)
test_ds  = load_split("test",  200)
```

---

## Dependencies

The following packages are required to use the dataloader:

```
torch
datasets
huggingface_hub
```

Install them with:

```bash
pip install torch datasets huggingface_hub
```
