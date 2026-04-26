from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BiLSTMConfig:
    vocab_size: int = 10
    num_classes: int = 10
    embedding_dim: int = 256
    hidden_size: int = 560
    num_layers: int = 5
    dropout: float = 0.10


class SudokuBiLSTM(nn.Module):
    def __init__(self, config: BiLSTMConfig):
        super().__init__()

        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
        )

        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        self.dropout = nn.Dropout(config.dropout)

        self.head = nn.Linear(
            config.hidden_size * 2,
            config.num_classes,
        )

    def forward(self, x, y=None):
        V = self.config.num_classes

        x_emb = self.embedding(x)
        h, _ = self.lstm(x_emb)
        h = self.dropout(h)

        logits = self.head(h)

        loss = None

        if y is not None:
            x_flat = x.reshape(-1)
            y_flat = y.reshape(-1)

            mask = x_flat != y_flat

            targets = y_flat.clone()
            targets[~mask] = -100

            loss = F.cross_entropy(
                logits.reshape(-1, V),
                targets,
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def predict(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = x.long().to(next(self.parameters()).device)

        logits, _ = self.forward(x)
        pred = logits.argmax(dim=-1)

        return torch.where(x != 0, x, pred)