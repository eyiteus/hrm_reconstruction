import torch

def print_sudoku_comparison(x, pred, y):
    x = x.view(9, 9)
    pred = pred.view(9, 9)
    y = y.view(9, 9)

    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    print("Input / Prediction")
    
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("-" * 33)

        row = ""
        for j in range(9):
            if j % 3 == 0 and j != 0:
                row += "| "

            if x[i, j] != 0:
                # given clue
                val = str(x[i, j].item())
            else:
                if pred[i, j] == y[i, j]:
                    # correct prediction green
                    val = f"{GREEN}{pred[i, j].item()}{RESET}"
                else:
                    # incorrect prediction red
                    val = f"{RED}{pred[i, j].item()}{RESET}"

            row += val + " "

        print(row)
    print()

@torch.no_grad()
def show_sudoku_predictions(model, val_loader, device, print_fn=print_sudoku_comparison, num_examples=10):
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