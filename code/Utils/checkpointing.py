import os
import torch


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    path: str,
    best_board_acc=None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if best_board_acc is not None:
        checkpoint["best_board_acc"] = best_board_acc

    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, scheduler, path: str, device):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_board_acc = checkpoint.get("best_board_acc", 0.0)

    return model, optimizer, scheduler, start_epoch, best_board_acc