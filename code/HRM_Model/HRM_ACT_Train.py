import torch
import torch.nn.functional as F
from tqdm import tqdm

from Utils.checkpointing import save_checkpoint
from Utils.metrics import (
    empty_sudoku_metrics,
    add_sudoku_metrics,
    sudoku_metric_accuracies,
)


class SampleStream:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.iter = iter(loader)
        self.buf_x = None
        self.buf_y = None
        self.buf_idx = 0

    def _refill(self):
        try:
            x, y = next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            x, y = next(self.iter)

        self.buf_x = x.to(self.device)
        self.buf_y = y.to(self.device)
        self.buf_idx = 0

    def draw(self, n: int):
        xs, ys = [], []
        remaining = n

        while remaining > 0:
            if self.buf_x is None or self.buf_idx >= self.buf_x.shape[0]:
                self._refill()

            avail = self.buf_x.shape[0] - self.buf_idx
            take = min(avail, remaining)

            xs.append(self.buf_x[self.buf_idx:self.buf_idx + take])
            ys.append(self.buf_y[self.buf_idx:self.buf_idx + take])

            self.buf_idx += take
            remaining -= take

        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


@torch.no_grad()
def evaluate_hrm_act(model, val_loader, device, M_max=None):
    if M_max is None:
        M_max = model.M

    model.eval()

    metrics = empty_sudoku_metrics()
    total_m_used = 0
    num_samples = 0

    for x, y in tqdm(val_loader, desc="ACT Validation"):
        x = x.to(device)
        y = y.to(device)

        filled, m_used = model.predict_act(x, M_max=M_max)

        total_m_used += m_used.sum().item()
        num_samples += x.size(0)

        unknown_mask = x != y
        correct_tokens = (filled[unknown_mask] == y[unknown_mask]).sum().item()
        total_tokens = unknown_mask.sum().item()
        board_correct = (filled == y).all(dim=1)
        correct_boards = board_correct.sum().item()
        total_boards = y.size(0)

        add_sudoku_metrics(metrics, {
            "correct_tokens": correct_tokens,
            "total_tokens": total_tokens,
            "correct_boards": correct_boards,
            "total_boards": total_boards,
        })

    token_acc, board_acc = sudoku_metric_accuracies(metrics)
    mean_m = total_m_used / max(num_samples, 1)

    return token_acc, board_acc, mean_m


def train_hrm_act(
    model,
    train_loader,
    optimizer,
    loss_fn,
    device,
    scheduler=None,
    num_epochs=5,
    M_max=None,
    checkpoint_dir="checkpoints_act",
    checkpoint_every=1,
    validate_every=1,
    val_loader=None,
    start_epoch=0,
    best_metric=0.0,
):
    if M_max is None:
        M_max = model.M

    batch_size = train_loader.batch_size

    print(
        "Number of trainable parameters:",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    )
    print(f"M_max={M_max}  epsilon={model.epsilon}  batch_size={batch_size}")

    history = {
        "step": [],
        "epoch": [],
        "train_loss": [],
        "ce": [],
        "bce": [],
        "halts": [],
        "mean_m_at_halt": [],
        "halt_acc": [],
    }

    global_step = 0

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        stream = SampleStream(train_loader, device)

        x_buf, y_buf = stream.draw(batch_size)
        x_embed = model.encode(x_buf)
        z_H_buf, z_L_buf = model.init_states(x_embed)
        z_H_buf = z_H_buf.detach()
        z_L_buf = z_L_buf.detach()
        m_buf = torch.zeros(batch_size, dtype=torch.long, device=device)
        m_min_buf = model.sample_m_min(batch_size, device=device)

        steps_per_epoch = len(train_loader) * M_max

        run_loss = 0.0
        run_ce = 0.0
        run_bce = 0.0
        halt_m_sum = 0.0
        halt_count = 0
        halt_correct = 0

        pbar = tqdm(range(steps_per_epoch), desc=f"ACT Epoch {epoch + 1}")

        for step in pbar:
            optimizer.zero_grad(set_to_none=True)

            z_H_buf, z_L_buf, logits, q = model.segment_act(x_buf, z_H_buf, z_L_buf)
            m_buf = m_buf + 1

            with torch.no_grad():
                pred_labels = logits.argmax(dim=-1)
                unknown = x_buf == 0
                per_sample_correct = (~unknown | (pred_labels == y_buf)).all(dim=-1).float()
                G_halt = per_sample_correct

                _, _, _, q_next = model.segment_act(x_buf, z_H_buf, z_L_buf)
                is_last = (m_buf + 1) >= M_max
                G_continue = torch.where(
                    is_last,
                    q_next[:, 0],
                    torch.maximum(q_next[:, 0], q_next[:, 1]),
                )

            G = torch.stack([G_halt, G_continue], dim=-1)

            x_flat = x_buf.reshape(-1)
            y_flat = y_buf.reshape(-1)

            sup_mask = x_flat != y_flat
            targets = y_flat.clone()
            targets[~sup_mask] = -100

            pred = logits.reshape(-1, logits.size(-1))
            ce = loss_fn(pred, targets)
            bce = F.binary_cross_entropy(q, G.detach())
            loss = ce + bce

            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            q_det = q.detach()
            halt = (m_buf >= M_max) | ((q_det[:, 0] > q_det[:, 1]) & (m_buf >= m_min_buf))

            z_H_buf = z_H_buf.detach()
            z_L_buf = z_L_buf.detach()

            run_loss += loss.item()
            run_ce += ce.item()
            run_bce += bce.item()

            if halt.any():
                halt_m_sum += m_buf[halt].float().sum().item()
                halt_correct += per_sample_correct[halt].sum().item()
                halt_count += int(halt.sum().item())

                n_halt = int(halt.sum().item())
                x_new, y_new = stream.draw(n_halt)
                halt_idx = halt.nonzero(as_tuple=True)[0]

                x_buf[halt_idx] = x_new
                y_buf[halt_idx] = y_new

                x_embed_new = model.encode(x_new)
                z_H_new, z_L_new = model.init_states(x_embed_new)

                z_H_buf = z_H_buf.clone()
                z_L_buf = z_L_buf.clone()
                z_H_buf[halt_idx] = z_H_new.detach()
                z_L_buf[halt_idx] = z_L_new.detach()

                m_buf[halt_idx] = 0
                m_min_buf[halt_idx] = model.sample_m_min(n_halt, device=device)

            global_step += 1

            history["step"].append(global_step)
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(loss.item())
            history["ce"].append(ce.item())
            history["bce"].append(bce.item())
            history["halts"].append(int(halt.sum().item()))
            history["mean_m_at_halt"].append(
                halt_m_sum / max(halt_count, 1)
            )
            history["halt_acc"].append(
                halt_correct / max(halt_count, 1)
            )

            if (step + 1) % 50 == 0:
                pbar.set_postfix({
                    "loss": f"{run_loss / (step + 1):.3f}",
                    "ce": f"{run_ce / (step + 1):.3f}",
                    "bce": f"{run_bce / (step + 1):.3f}",
                    "mean_m_halt": f"{halt_m_sum / max(halt_count, 1):.2f}",
                    "halt_acc": f"{halt_correct / max(halt_count, 1) * 100:.1f}%",
                })

        avg_loss = run_loss / steps_per_epoch
        avg_ce = run_ce / steps_per_epoch
        avg_bce = run_bce / steps_per_epoch
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}: "
            f"avg_loss={avg_loss:.4f}  "
            f"ce={avg_ce:.4f}  "
            f"bce={avg_bce:.4f}  "
            f"halts={halt_count}  "
            f"mean_m_at_halt={halt_m_sum / max(halt_count, 1):.2f}  "
            f"halt_acc={halt_correct / max(halt_count, 1) * 100:.2f}%  "
            f"LR={current_lr:.2e}"
        )

        val_token_acc = None
        val_board_acc = None
        val_mean_m = None

        should_validate = (
            val_loader is not None
            and validate_every is not None
            and (epoch + 1) % validate_every == 0
        )

        if should_validate:
            val_token_acc, val_board_acc, val_mean_m = evaluate_hrm_act(
                model,
                val_loader,
                device,
                M_max=M_max,
            )

            print(
                f"Val: token_acc={val_token_acc * 100:.2f}%  "
                f"board_acc={val_board_acc * 100:.2f}%  "
                f"mean_m_used={val_mean_m:.2f}/{M_max}\n"
            )

        metric_for_best = (
            val_board_acc if val_board_acc is not None else
            halt_correct / max(halt_count, 1)
        )

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            f"{checkpoint_dir}/hrm_act_last.pt",
            best_board_acc=best_metric,
        )

        if metric_for_best > best_metric:
            best_metric = metric_for_best

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                f"{checkpoint_dir}/hrm_act_best.pt",
                best_board_acc=best_metric,
            )

        if checkpoint_every is not None and (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                f"{checkpoint_dir}/hrm_act_epoch_{epoch + 1}.pt",
                best_board_acc=best_metric,
            )

    return model, best_metric, history
