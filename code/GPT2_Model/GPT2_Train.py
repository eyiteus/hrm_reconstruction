import torch
from tqdm import tqdm

from Utils.checkpointing import save_checkpoint
from Utils.metrics import (
    sudoku_metrics_from_logits,
    empty_sudoku_metrics,
    add_sudoku_metrics,
    sudoku_metric_accuracies,
)


@torch.no_grad()
def evaluate_gpt2(model, val_loader, device):
    model.eval()

    total_loss = 0.0
    metrics = empty_sudoku_metrics()

    for x, y in tqdm(val_loader, desc="Validation"):
        x = x.to(device)
        y = y.to(device)

        logits, loss = model(x, y)
        total_loss += loss.item()

        batch_metrics = sudoku_metrics_from_logits(x, y, logits)
        add_sudoku_metrics(metrics, batch_metrics)

    avg_loss = total_loss / len(val_loader)
    token_acc, board_acc = sudoku_metric_accuracies(metrics)

    return avg_loss, token_acc, board_acc


def train_gpt2(
    model,
    train_loader,
    optimizer,
    device,
    scheduler=None,
    num_epochs=10,
    checkpoint_dir="checkpoints",
    checkpoint_every=5,
    validate_every=1,
    val_loader=None,
    start_epoch=0,
    best_metric=0.0,
):
    print(
        "Number of trainable parameters:",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    )

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()

        epoch_loss = 0.0
        metrics = empty_sudoku_metrics()

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits, loss = model(x, y)
            loss.backward()

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()

            batch_metrics = sudoku_metrics_from_logits(x, y, logits.detach())
            add_sudoku_metrics(metrics, batch_metrics)

        avg_loss = epoch_loss / len(train_loader)
        token_acc, board_acc = sudoku_metric_accuracies(metrics)

        print(
            f"Epoch {epoch + 1}: "
            f"Loss={avg_loss:.4f}, "
            f"TokenAcc={token_acc:.4f}, "
            f"BoardAcc={board_acc:.4f}"
        )

        save_checkpoint(
            model, optimizer, scheduler, epoch,
            f"{checkpoint_dir}/gpt2_last.pt",
            best_board_acc=best_metric,
        )

    return model, best_metric