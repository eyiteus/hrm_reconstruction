# HRM Re-Implementation

A re-implementation of the **Hierarchical Reasoning Model (HRM)** for Sudoku reasoning, based on Wang et al.’s *Hierarchical Reasoning Model*. The project reproduces HRM’s Sudoku behavior at student-compute scale and explores whether extra inference-time reasoning improves accuracy.

## Introduction

This repository attempts to re-implement HRM, a latent-space recurrent reasoning architecture that uses fast low-level and slower high-level transformer modules instead of chain-of-thought decoding.

<img width="1007" height="433" alt="Screenshot 2026-05-10 at 10 21 27 AM" src="https://github.com/user-attachments/assets/f6c28ea1-7bf6-4ae3-b038-2ac1846391b1" />

## Chosen Result

We targeted the paper’s Sudoku benchmark, where HRM reports about **99.10% token accuracy at M = 8**. Sudoku was chosen because it is deterministic, structured, and strongly highlights HRM’s reasoning advantage over standard baselines.

## GitHub Contents

```text
code/      HRM, baseline, training, evaluation, and plotting code
data/      Sudoku dataset files
results/   Figures
poster/    Final poster PDF
report/    Final written report PDF
```

## Re-implementation Details

HRM was implemented with shared recurrent **L-module** and **H-module** transformer blocks, deep supervision at each segment, and one-step gradient approximation. We trained on a practical subset of the Sudoku dataset with ~33.6M parameters, 20% dropout, LR 1e-4 with warmup/cosine decay, and evaluated token accuracy plus full-board accuracy.

## Reproduction Steps

```bash
git clone https://github.com/Eyiteus/HRM_Reconstruction.git
cd HRM_Reconstruction
```

Then run the HRM or baseline notebooks under `code/`. You may need to download some python libraries. A GPU is strongly recommended.

## Results/Insights

| Model / Setting     |             Result | Takeaway                                |
| ------------------- | -----------------: | --------------------------------------- |
| Paper HRM, M=8      | ~99.10% token acc. | Original Sudoku benchmark               |
| Our HRM, M=16, T=1  |  92.12% token acc. | Smaller-scale reproduction              |
| Our HRM, M=128, T=8 |  98.04% token acc. | Longer inference closes most of the gap |

HRM generalized better than the BiLSTM and encoder-only transformer baselines, which overfit at the same dataset size and converged slowly even with 8× more data. Hidden-state residuals also dropped after solution convergence, suggesting possible fixed-point-style stopping criteria beyond ACT.

## Conclusion

Our re-implementation supports HRM’s main claim: latent recurrent reasoning is highly effective for structured tasks like Sudoku. The key insight is that HRM can improve when run for more inference steps than it was trained with, suggesting it learns an iterative refinement process.

## References

* Wang, G. et al. *Hierarchical Reasoning Model*. arXiv:2506.21734.
* D. Kahneman, Thinking, Fast and Slow (2011).
* PyTorch, Hugging Face, NumPy, Matplotlib.

## Acknowledgements

Completed as a final project for **CS 4782: Introduction to Deep Learning** at Cornell University. Thanks to the course staff and peer reviewers for feedback during the poster and project process.
