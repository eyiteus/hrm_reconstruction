# HRM Re-Implementation

A re-implementation of the **Hierarchical Reasoning Model (HRM)** for Sudoku reasoning, based on Wang et al.’s *Hierarchical Reasoning Model*. The project reproduces HRM’s Sudoku behavior at student-compute scale and explores whether extra inference-time reasoning improves accuracy.

## Introduction

This repository attempts to re-implement HRM, a latent-space recurrent reasoning architecture that uses fast low-level and slower high-level transformer modules instead of chain-of-thought decoding.

<img width="1007" height="433" alt="Screenshot 2026-05-10 at 10 21 27 AM" src="https://github.com/user-attachments/assets/f6c28ea1-7bf6-4ae3-b038-2ac1846391b1" />

## Chosen Result

We targeted the paper’s Sudoku benchmark, where HRM reports about **99.10% token accuracy at M = 8**. Sudoku was chosen because it is deterministic, structured, and strongly highlights HRM’s reasoning advantage over standard baselines.

There were also a few open questions and unjustified claims we wanted to address. In particular, we wanted to empirically show the local fixed-point contraction (an assumption barely justified in the paper), and determine if model size can be increased after training (implied by the architecture but not mentioned in the paper).

## GitHub Contents

```text
code/      HRM, baseline, training, evaluation, and plotting code
data/      Sudoku dataset files
results/   Figures
poster/    Final poster PDF
report/    Final written report PDF
```

## Re-implementation Details

We re-implemented HRM as a recurrent latent-space model with shared L-module and H-module transformer blocks: each segment runs repeated low-level updates, periodic high-level updates, then applies a prediction head to the final latent state.

We trained on 2^18 Sudoku examples instead of the paper’s 3M-example dataset due to compute constraints, using cross-entropy loss only on unhinted board positions and evaluating with token accuracy and full-board accuracy.

We also implemented Adaptive Computation Time (ACT) using a learned Q-head for halting decisions, but trained the ACT Q-head after the base HRM was already trained rather than jointly training it with the main model as in the paper.

Our main implementation differences from the paper were using AdamW, adding dropout, and running smaller-scale training. For comparison, we also trained a Bidirectional LSTM and an encoder-only transformer on the same dataset with the same optimizer/dropout setup, tuned to approximately match HRM’s parameter count; unlike the paper’s large-LLM comparisons, these baselines tested similarly sized non-HRM architectures directly.

## Reproduction Steps

To re-implement our re-implementation from scratch, first build a Sudoku dataloader that returns the puzzle input, solved board labels, and a mask for unhinted cells; train with cross-entropy only on masked positions and report token plus full-board accuracy.

Next implement HRM with shared recurrent L and H transformer modules and freeze the best trained base HRM and train an ACT Q-head for halting decisions. Finally, reproduce our comparisons by training a similarly sized Bidirectional LSTM and encoder-only transformer on the same data, optimizer, dropout, and evaluation pipeline.

If you don't want to do all that, just clone our repo:

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
