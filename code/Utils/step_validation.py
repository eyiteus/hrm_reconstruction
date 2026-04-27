import torch


@torch.no_grad()
def small_validation_loss(model, val_loader, device, max_batches=1):
    model.eval()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, (x, y) in enumerate(val_loader):
        if batch_idx >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        _, loss = model(x, y)

        total_loss += loss.item()
        num_batches += 1

    model.train()

    return total_loss / max(1, num_batches)