import torch


@torch.no_grad()
def sudoku_metrics_from_logits(x, y, logits):
    pred = logits.argmax(dim=-1)

    unknown_mask = x != y

    correct_tokens = (pred[unknown_mask] == y[unknown_mask]).sum().item()
    total_tokens = unknown_mask.sum().item()

    filled_board = torch.where(x != 0, x, pred)
    board_correct = (filled_board == y).all(dim=1)

    correct_boards = board_correct.sum().item()
    total_boards = y.size(0)

    return {
        "correct_tokens": correct_tokens,
        "total_tokens": total_tokens,
        "correct_boards": correct_boards,
        "total_boards": total_boards,
    }


def empty_sudoku_metrics():
    return {
        "correct_tokens": 0,
        "total_tokens": 0,
        "correct_boards": 0,
        "total_boards": 0,
    }


def add_sudoku_metrics(total, batch):
    total["correct_tokens"] += batch["correct_tokens"]
    total["total_tokens"] += batch["total_tokens"]
    total["correct_boards"] += batch["correct_boards"]
    total["total_boards"] += batch["total_boards"]
    return total


def sudoku_metric_accuracies(metrics):
    token_acc = (
        metrics["correct_tokens"] / metrics["total_tokens"]
        if metrics["total_tokens"] > 0
        else 0.0
    )

    board_acc = (
        metrics["correct_boards"] / metrics["total_boards"]
        if metrics["total_boards"] > 0
        else 0.0
    )

    return token_acc, board_acc