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
def evaluate_bert(model, val_loader, device):
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


def train_bert(
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
    grad_clip=None,
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

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()

            batch_metrics = sudoku_metrics_from_logits(x, y, logits.detach())
            add_sudoku_metrics(metrics, batch_metrics)

        avg_loss = epoch_loss / len(train_loader)
        token_acc, board_acc = sudoku_metric_accuracies(metrics)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}: "
            f"Avg Train Loss = {avg_loss:.4f}, "
            f"Train Token Accuracy = {token_acc * 100:.2f}%, "
            f"Train Board Accuracy = {board_acc * 100:.2f}%, "
            f"LR = {current_lr:.2e}"
        )

        should_validate = (
            val_loader is not None
            and validate_every is not None
            and (epoch + 1) % validate_every == 0
        )

        if should_validate:
            val_loss, val_token_acc, val_board_acc = evaluate_bert(
                model, val_loader, device
            )

            print(
                f"Val Loss = {val_loss:.4f}, "
                f"Val Token Accuracy = {val_token_acc * 100:.2f}%, "
                f"Val Board Accuracy = {val_board_acc * 100:.2f}%\n"
            )

        metric_for_best = val_board_acc if should_validate else board_acc

        save_checkpoint(
            model, optimizer, scheduler, epoch,
            f"{checkpoint_dir}/bert_last.pt",
            best_board_acc=best_metric,
        )

        if metric_for_best > best_metric:
            best_metric = metric_for_best

            save_checkpoint(
                model, optimizer, scheduler, epoch,
                f"{checkpoint_dir}/bert_best.pt",
                best_board_acc=best_metric,
            )

        if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                f"{checkpoint_dir}/bert_epoch_{epoch + 1}.pt",
                best_board_acc=best_metric,
            )

    return model, best_metric