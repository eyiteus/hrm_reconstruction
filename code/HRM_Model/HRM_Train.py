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
def evaluate_hrm(
    model,
    val_loader,
    loss_fn,
    device,
):
    model.eval()

    total_loss = 0.0
    metrics = empty_sudoku_metrics()

    for x, y in tqdm(val_loader, desc="Validation"):
        x = x.to(device)
        y = y.to(device)

        z_H = None
        z_L = None
        logits = None

        for _ in range(model.M):
            z_H, z_L, logits = model.segment(x, z_H, z_L)
            z_H = z_H.detach()
            z_L = z_L.detach()

        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)

        mask = x_flat != y_flat

        targets = y_flat.clone()
        targets[~mask] = -100

        pred = logits.reshape(-1, logits.size(-1))
        loss = loss_fn(pred, targets)

        total_loss += loss.item()

        batch_metrics = sudoku_metrics_from_logits(x, y, logits)
        add_sudoku_metrics(metrics, batch_metrics)

    avg_loss = total_loss / len(val_loader)
    token_acc, board_acc = sudoku_metric_accuracies(metrics)

    return avg_loss, token_acc, board_acc


def train_hrm_deepsup(
    model,
    train_loader,
    optimizer,
    loss_fn,
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
        "Number of trainable parameters: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()

        epoch_loss = 0.0
        metrics = empty_sudoku_metrics()

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            x = x.to(device)
            y = y.to(device)

            z_H = None
            z_L = None

            batch_loss = 0.0
            last_logits = None

            x_flat = x.reshape(-1)
            y_flat = y.reshape(-1)

            mask = x_flat != y_flat

            targets = y_flat.clone()
            targets[~mask] = -100

            for _ in range(model.M):
                optimizer.zero_grad(set_to_none=True)

                z_H, z_L, logits = model.segment(x, z_H, z_L)

                pred = logits.reshape(-1, logits.size(-1))
                loss = loss_fn(pred, targets)

                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                z_H = z_H.detach()
                z_L = z_L.detach()

                batch_loss += loss.item()
                last_logits = logits.detach()

            epoch_loss += batch_loss / model.M

            batch_metrics = sudoku_metrics_from_logits(x, y, last_logits)
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

        val_loss = None
        val_token_acc = None
        val_board_acc = None

        should_validate = (
            val_loader is not None
            and validate_every is not None
            and (epoch + 1) % validate_every == 0
        )

        if should_validate:
            val_loss, val_token_acc, val_board_acc = evaluate_hrm(
                model,
                val_loader,
                loss_fn,
                device,
            )

            print(
                f"Val Loss = {val_loss:.4f}, "
                f"Val Token Accuracy = {val_token_acc * 100:.2f}%, "
                f"Val Board Accuracy = {val_board_acc * 100:.2f}%\n"
            )

        metric_for_best = val_board_acc if val_board_acc is not None else board_acc

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            f"{checkpoint_dir}/hrm_last.pt",
            best_board_acc=best_metric,
        )

        if metric_for_best > best_metric:
            best_metric = metric_for_best

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                f"{checkpoint_dir}/hrm_best.pt",
                best_board_acc=best_metric,
            )

        if checkpoint_every is not None and (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                f"{checkpoint_dir}/hrm_epoch_{epoch + 1}.pt",
                best_board_acc=best_metric,
            )

    return model, best_metric