import torch


@torch.no_grad()
def show_sudoku_predictions(model, val_loader, device, print_fn, num_examples=10):
    model.eval()

    x_batch, y_batch = next(iter(val_loader))

    for i in range(num_examples):
        x = x_batch[i].to(device).long()
        y = y_batch[i].to(device).long()

        pred = model.predict(x).squeeze(0).cpu()

        x_cpu = x.cpu()
        y_cpu = y.cpu()

        print_fn(x_cpu, pred, y_cpu)

        unknown_mask = x_cpu != y_cpu
        token_acc = (pred[unknown_mask] == y_cpu[unknown_mask]).float().mean()

        print("Unknown-cell token accuracy:", token_acc.item())
        print()