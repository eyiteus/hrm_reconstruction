import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = torch.stack((-x2, x1), dim=-1)
    return out.flatten(start_dim=-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (L, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)            # (L, dim)
        cos = emb.cos()[None, None, :, :]                  # (1,1,L,dim)
        sin = emb.sin()[None, None, :, :]                  # (1,1,L,dim)
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, L, D)
    return (x * cos) + (rotate_half(x) * sin)


class MultiHeadSelfAttentionRoPE(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # (B,H,L,Dh)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(L, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,L,L)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # (B,H,L,Dh)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class PostNormTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        intermediate_size: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttentionRoPE(d_model, n_heads, dropout=dropout)
        self.norm1 = RMSNorm(d_model)
        self.ff = SwiGLUFeedForward(d_model, intermediate_size)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attn(x)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size: int = 10, d_model: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class HighLevel(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        intermediate_size: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            PostNormTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

    def forward(self, z_H: torch.Tensor, z_L: torch.Tensor) -> torch.Tensor:
        h = z_H + z_L
        for layer in self.layers:
            h = layer(h)
        return h


class LowLevel(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        intermediate_size: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            PostNormTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

    def forward(self, x_embed: torch.Tensor, z_H: torch.Tensor, z_L: torch.Tensor) -> torch.Tensor:
        h = x_embed + z_H + z_L
        for layer in self.layers:
            h = layer(h)
        return h


class Head(nn.Module):
    def __init__(self, d_model: int = 256, vocab_size: int = 10):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)