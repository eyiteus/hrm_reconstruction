import torch
import torch.nn as nn
import transformers


class Encoder(nn.Module):
    def __init__(self, vocab_size=10, d_model=32, max_len=81):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x):
        # x: (B, L)
        x = self.embedding(x) + self.pos_embedding[:, :x.size(1), :]
        return x  # (B, L, d_model)



class HighLevel(nn.Module):
    def __init__(self, d_model=32, n_layers=2, n_heads=4, intermediate_size=64):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=intermediate_size,
            batch_first=True,  # keep (B, L, d_model)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x, z_H,z_L):
        # x: (B, L, d_model)
        return self.encoder(x+z_H + z_L)

class LowLevel(nn.Module):
    def __init__(self, d_model=32, n_layers=2, n_heads=4, intermediate_size=64):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=intermediate_size,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x, z_H, z_L):
        return self.encoder(x+z_H+z_L)  # (B, L, d_model) -> latent -> latent


class Head(nn.Module):
    def __init__(self, d_model=32, vocab_size=10):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.linear(x)  # (B, L, vocab_size)