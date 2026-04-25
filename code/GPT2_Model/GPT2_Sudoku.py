"""
Note that this file is mostly copied from code written for 
a project written in CS 6787 this semester
"""
import torch
import torch.nn as nn
from torch import dropout, embedding
from dataclasses import dataclass
import torch.nn.functional as F
import sys
from dataclasses import dataclass
import math

#Configuration for model hyperparameters
@dataclass
class BaselineConfig:
  num_heads: int = 12
  num_layers: int = 12
  vocab_size: int = 10
  embedding_dim: int = 768
  block_size: int = 256
  dropout : float = .1
  weight_decay : float = .1
  pad_token_id : int = 0

#Configuration fr training hyperparameters
@dataclass
class TrainConfig:
  batch_per_iter : int = 32
  grad_acc_factor : int = 16
  warmup_steps : int = 150
  num_steps_train : int = 3623
  num_steps_val : int = 32
  max_lr : float = 3e-4
  min_lr : float = 3e-5
  weight_decay : float = .1
  betas : tuple[float, float] = (.9,.95)

  def get_lr(self, step):
    # warmup
    if step < self.warmup_steps:
        return self.max_lr * step / self.warmup_steps
    # cosine decay to min_lr
    progress = (step - self.warmup_steps) / \
               (self.num_steps_train - self.warmup_steps)
    cosine   = 0.5 * (1 + math.cos(math.pi * progress))
    return self.min_lr + cosine * (self.max_lr - self.min_lr)

  def make_optimizer(self, model):
    return torch.optim.AdamW(
      model.parameters(),
      betas=self.betas,
      lr=self.max_lr,
      weight_decay=self.weight_decay
    )
  
  def make_scheduler(self, optimizer):
    return torch.optim.lr_scheduler.LambdaLR(
      optimizer, lambda step: self.get_lr(step) / self.max_lr
    )
  
  def make_scaler(self):
    return torch.amp.GradScaler('cuda')



class GPT2_Baseline(nn.Module):
  #create a ModuleDict with wte, wpe, hidden layers, weight and bias
  def __init__(self, config: BaselineConfig, device):
    super().__init__()
    self.config = config
    self.device = device
    self.transformer = nn.ModuleDict(
      dict(
        wte = nn.Embedding(config.vocab_size, config.embedding_dim),
        wpe = nn.Embedding(config.block_size, config.embedding_dim),
        h = nn.ModuleList([LayerBlock(config) for _ in range(config.num_layers)]),
        ln_f = nn.LayerNorm(config.embedding_dim),
        drop = nn.Dropout(config.dropout)
      )
    )
    #Note that sice our vocab size is very small, we are not doing weight tying
    self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)



  def forward(self, x, y):
    _, N = x.size()
    V = self.config.vocab_size


    #Get the learned positional embeddings
    pos = torch.arange(0, N, dtype=torch.long, device=x.device)
    pos_emb = self.transformer.wpe(pos)

    #Push input through the embedding matrix
    x = self.transformer.wte(x) + pos_emb
    x = self.transformer.drop(x)

    #Push input through the transformer block
    for layer_block in self.transformer.h:
      x, _ = layer_block(x)

    #Do a final layer norm and push through the head
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)

    #Calculate Loss
    loss = F.cross_entropy(
        logits.view(-1, V), 
        y.view(-1),
        ignore_index=self.config.pad_token_id)

    return logits, loss

  #Top-k sampling with temperature
  def _sample(self, logits, temperature, top_k=None):
    logits = logits / temperature
    if top_k is not None:
      v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
      logits[logits < v[:, [-1]]] = float('-inf')
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)


class LayerBlock(nn.Module):
    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embedding_dim)
        self.attn = AttentionMultiHeadFused(config)
        self.ln_2 = nn.LayerNorm(config.embedding_dim)
        self.mlp = MLP(config)
    

    def forward(self, x, kv_cache = None):
        attn_out, new_cache = self.attn(self.ln_1(x), kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x)) #why do we layer-norm before passing mlp?
        return x, new_cache


class MLP(nn.Module):
    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.embedding_dim, 4*config.embedding_dim)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.embedding_dim, config.embedding_dim)
        self.drop   = nn.Dropout(config.dropout)

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


    def forward(self, x, kv_cache=None):
        B, N, d = x.size()
        h = self.config.num_heads
        d_eff = d // h

        q, k, v = self.w_qkv(x).split(d, dim=2)
        q = q.view(B, N, h, d_eff).transpose(1, 2)
        k = k.view(B, N, h, d_eff).transpose(1, 2)
        v = v.view(B, N, h, d_eff).transpose(1, 2)

        if not self.training and kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_cache = (k, v) if not self.training else None

        dropout_p = self.attn_drop if self.training else 0.0
        is_causal = kv_cache is None  # full causal mask during prefill and training
        attn_out  = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=dropout_p)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, d)
        out = self.output(attn_out)
        out = self.resid_drop(out)
        return out, new_cache