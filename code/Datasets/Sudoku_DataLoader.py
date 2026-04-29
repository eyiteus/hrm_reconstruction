import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Features, Value



# Encoding function to encode Sudokus
def encode(example):
    puzzle = [int(c) if c != '.' else 0 for c in example["question"]]
    solution = [int(c) for c in example["answer"]]
    return {"x": puzzle, "y": solution}


#Dataset Wrapper
class SudokuDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        x = torch.tensor(item["x"], dtype=torch.long)
        y = torch.tensor(item["y"], dtype=torch.long)
        return x, y

# # Main loader function
# def get_loaders(train_size=1000, test_size=200, batch_size=64):
#     #Will load min(train_size, train_dataset_size) examples
#     ds = load_dataset(
#         "sapientinc/sudoku-extreme",
#         split=f"train[:{train_size}]",
#     )
#     #Will load min(test_size, test_dataset_size) examples
#     ds_test = load_dataset(
#         "sapientinc/sudoku-extreme",
#         split=f"test[:{test_size}]",
#     )

#     ds = ds.map(encode)
#     ds_test = ds_test.map(encode)

#     train_dataset = SudokuDataset(ds)
#     test_dataset = SudokuDataset(ds_test)

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False
#     )

#     return train_loader, test_loader

import csv
from huggingface_hub import hf_hub_download
from datasets import Dataset

def load_split(split: str, size: int) -> Dataset:
    # First check actual filenames with:
    # from huggingface_hub import list_repo_files
    # print(list(list_repo_files("sapientinc/sudoku-extreme", repo_type="dataset")))
    
    path = hf_hub_download(
        repo_id="sapientinc/sudoku-extreme",
        filename=f"{split}.csv",  # adjust if filename differs
        repo_type="dataset"
    )

    records = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)  # pure Python, no PyArrow involved
        for i, row in enumerate(reader):
            if i >= size:
                break
            records.append(row)  # all values are already strings

    return Dataset.from_list(records)

def get_loaders(train_size=1000, test_size=200, batch_size=64):
    ds      = load_split("train", train_size)
    ds_test = load_split("test",  test_size)

    ds      = ds.map(encode)
    ds_test = ds_test.map(encode)

    train_loader = DataLoader(SudokuDataset(ds),      batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(SudokuDataset(ds_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def collect_puzzles_set(loader):
    puzzles = set()

    for _, (x, _) in enumerate(loader):
        for xi in x:
           puzzles.add(xi.cpu().numpy().tobytes())

    return puzzles