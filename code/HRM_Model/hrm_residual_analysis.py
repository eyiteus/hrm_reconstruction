import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


# Core recording logic
def collect_residuals(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor = None,
    device=None,
) -> dict:
    if not isinstance(device, (torch.device, str)):
        device = next(model.parameters()).device

    M, N, T = model.M, model.N, model.T

    x = x.long().to(device)
    if x.dim() == 1:
        x = x.unsqueeze(0)


    y = y.long().to(device)
    if y.dim() == 1:
        y = y.unsqueeze(0)

    was_training = model.training
    model.eval()

    zL_residuals:   list[float] = []
    zH_residuals:   list[float] = []
    zH_at_step:     list[int] = []
    seg_boundaries: list[int] = []
    seg_accuracies: list[float] = []

    global_L_step = 0

    with torch.no_grad():
        x_embed = model.encode(x)
        z_H, z_L = model.init_states(x_embed)

        for _ in range(M):
            seg_boundaries.append(len(zL_residuals))

            prev_z_L = z_L.clone()
            prev_z_H = z_H.clone()
            nt_step  = 0

            for _ in range(N * T):
                #Low-level
                z_L_new = model.low_level(x_embed, z_H, z_L)
                global_L_step += 1

                zL_residuals.append(
                    (z_L_new - prev_z_L).norm(dim=(1, 2)).mean().item()
                )
                prev_z_L = z_L_new.clone()
                z_L = z_L_new

                #High-level (every T low-level steps)
                nt_step += 1
                if nt_step % T == 0:
                    z_H_new = model.high_level(z_H, z_L)
                    zH_residuals.append(
                        (z_H_new - prev_z_H).norm(dim=(1, 2)).mean().item()
                    )
                    zH_at_step.append(global_L_step)
                    prev_z_H = z_H_new.clone()
                    z_H = z_H_new

            #Per-segment accuracy
            logits = model.head(z_H)   # (B, L, vocab)

            preds = logits.argmax(dim=-1)     # (B, L)
            mask  = (x != y)                  # positions being predicted
            if mask.any():
                correct = ((preds == y) & mask).sum().item()
                total   = mask.sum().item()
                seg_accuracies.append(correct / total)
            else:
                seg_accuracies.append(float("nan"))

            z_H = z_H.detach()
            z_L = z_L.detach()

    if was_training:
        model.train()

    return dict(
        zL_residuals=zL_residuals,
        zH_residuals=zH_residuals,
        zH_at_step=zH_at_step,
        seg_boundaries=seg_boundaries,
        seg_accuracies=seg_accuracies,
        M=M, N=N, T=T,
    )


#Plotting Code
def plot_residuals(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor = None,
    device=None,
    log_scale: bool = True,
):
    data = collect_residuals(
        model, x, y=y, device=device
    )

    zL = np.array(data["zL_residuals"])
    zH = np.array(data["zH_residuals"])
    zH_steps = np.array(data["zH_at_step"])
    seg_bds = data["seg_boundaries"]
    seg_accs = data["seg_accuracies"]

    total_L = len(zL)
    steps = np.arange(1, total_L + 1)


    _, (ax, ax2) = plt.subplots(
        2, 1, figsize=(12, 8),
        gridspec_kw={"height_ratios": [3, 1]}
    )

    ax.plot(
        steps,
        zL,
        label="z_L",
        linewidth=1.8,
        marker="o",
        markersize=3,
        alpha=0.9
    )

    ax.plot(
        zH_steps,
        zH,
        linestyle="--",
        marker="D",
        markersize=5,
        linewidth=1.5,
        label="z_H")

    for bd in seg_bds:
        ax.axvline(bd + 0.5, linestyle=":", alpha=0.8)

    if log_scale and (zL > 0).all():
        ax.set_yscale("log")

    ax.set_title("HRM Residual Convergence")
    ax.set_xlabel("Low-level step")
    ax.set_ylabel("Residual Frobenius Norm")
    ax.grid(True, alpha=0.3)
    ax.legend()

    #Accuracy (bar chart)
    seg_nums = np.arange(1, len(seg_accs) + 1)

    bars = ax2.bar(
        seg_nums,
        seg_accs,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
        label="Segment accuracy"
    )

    # Add value labels
    for bar, acc in zip(bars, seg_accs):
        if not np.isnan(acc):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                acc + 0.01,
                f"{acc:.1%}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Segment")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return data