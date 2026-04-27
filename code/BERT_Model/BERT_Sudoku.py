import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class BaselineConfig:
    num_heads: int = 8
    num_layers: int = 12
    vocab_size: int = 10
    embedding_dim: int = 512
    block_size: int = 81
    dropout: float = 0.2
    pad_token_id: int = 0


class BERT_Baseline(nn.Module):
    def __init__(self, config: BaselineConfig):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.embedding_dim),
                wpe=nn.Embedding(config.block_size, config.embedding_dim),
                h=nn.ModuleList(
                    [LayerBlock(config) for _ in range(config.num_layers)]
                ),
                ln_f=nn.LayerNorm(config.embedding_dim),
                drop=nn.Dropout(config.dropout),
            )
        )

        self.lm_head = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False,
        )

        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, x, y=None):
        B, N = x.size()
        V = self.config.vocab_size

        pos = torch.arange(0, N, dtype=torch.long, device=x.device)
        pos_emb = self.transformer.wpe(pos)

        x_emb = self.transformer.wte(x) + pos_emb
        x_emb = self.transformer.drop(x_emb)

        for layer_block in self.transformer.h:
            x_emb = layer_block(x_emb)

        x_emb = self.transformer.ln_f(x_emb)
        logits = self.lm_head(x_emb)

        loss = None

        if y is not None:
            x_flat = x.reshape(-1)
            y_flat = y.reshape(-1)

            # supervise only unknown cells
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


class LayerBlock(nn.Module):
    def __init__(self, config: BaselineConfig):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.embedding_dim)
        self.attn = AttentionMultiHeadFused(config)

        self.ln_2 = nn.LayerNorm(config.embedding_dim)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MLP(nn.Module):
    def __init__(self, config: BaselineConfig):
        super().__init__()

        self.c_fc = nn.Linear(config.embedding_dim, 4 * config.embedding_dim)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.embedding_dim, config.embedding_dim)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.drop(x)
        return x


class AttentionMultiHeadFused(nn.Module):
    def __init__(self, config: BaselineConfig):
        super().__init__()

        self.config = config

        self.w_qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.output = nn.Linear(config.embedding_dim, config.embedding_dim)

        self.attn_drop = config.dropout
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x):
        B, N, ED = x.size()

        qkv = self.w_qkv(x)
        q, k, v = qkv.split(self.config.embedding_dim, dim=2)

        q = q.view(B, N, self.config.num_heads, ED // self.config.num_heads).transpose(1, 2)
        k = k.view(B, N, self.config.num_heads, ED // self.config.num_heads).transpose(1, 2)
        v = v.view(B, N, self.config.num_heads, ED // self.config.num_heads).transpose(1, 2)

        dropout_p = self.attn_drop if self.training else 0.0

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, ED)

        y = self.output(attn_out)
        y = self.resid_drop(y)

        return y