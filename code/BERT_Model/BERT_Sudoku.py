"""
Note that this file is mostly copied from code written for 
a project written in CS 6787 this semester
"""
import torch
import torch.nn as nn
from torch import dropout, embedding
from dataclasses import dataclass
import torch.nn.functional as F
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


class BERT_Baseline(nn.Module):
  #create a ModuleDict with wte, wpe, hidden layers, weight and bias
  def __init__(self, config: BaselineConfig, device):
    super().__init__()
    self.config = config
    self.device = device
    self.transformer = nn.ModuleDict(
        dict(
            wte = nn.Embedding(config.vocab_size, config.embedding_dim), # how do we determine the dimensions of this?
            wpe = nn.Embedding(config.block_size, config.embedding_dim),
            h = nn.ModuleList([LayerBlock(config, device) for _ in range(config.num_layers)]),
            ln_f = nn.LayerNorm(config.embedding_dim),
            drop = nn.Dropout(config.dropout)
        )
    )
    self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False) #why is bias set as False?

    self.transformer.wte.weight = self.lm_head.weight  # weight tying

  def forward(self, x, y):
    #Define some useful constants for the forward pass
    B, N = x.size()
    V = self.config.vocab_size


    #Get the learned positional embeddings
    pos = torch.arange(0, N, dtype=torch.long, device=x.device)
    pos_emb = self.transformer.wpe(pos)

    #Before getting token embeddings, create the mask for tokens to predict
    loss_mask = (x == 0)

    x = self.transformer.wte(x) + pos_emb
    x = self.transformer.drop(x)


    for layer_block in self.transformer.h:
      x = layer_block(x)

    x = self.transformer.ln_f(x)
    logits = self.lm_head(x[loss_mask])

    loss = F.cross_entropy(
      logits.view(-1, V), 
      y[loss_mask].view(-1),
      ignore_index=self.config.pad_token_id)

    return logits, loss
  


class LayerBlock(nn.Module):
  def __init__(self, config: BaselineConfig, device):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.embedding_dim)

    self.attn = AttentionMultiHeadFused(config, device)
    self.ln_2 = nn.LayerNorm(config.embedding_dim)

    self.mlp = MLP(config)
    

  def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x)) 
    return x


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
  def __init__(self, config: BaselineConfig, device):
    super().__init__()
    self.config = config
    self.device = device
    self.w_qkv = nn.Linear(config.embedding_dim, 3*config.embedding_dim)
    self.output = nn.Linear(config.embedding_dim, config.embedding_dim)
    self.attn_drop = config.dropout  # passed to scaled_dot_product_attention
    self.resid_drop = nn.Dropout(config.dropout)  # add this

  def forward(self, x):
    B,N,ED = x.size()
    qkv = self.w_qkv(x)
    q, k, v = qkv.split(self.config.embedding_dim, dim=2)
    q = q.view(B, N, self.config.num_heads, ED // self.config.num_heads).transpose(1,2)
    k = k.view(B, N, self.config.num_heads, ED // self.config.num_heads).transpose(1,2)
    v = v.view(B, N, self.config.num_heads, ED // self.config.num_heads).transpose(1,2)

    dropout_p=self.attn_drop if self.training else 0.0
    attn_out = F.scaled_dot_product_attention(q, k, v,
            attn_mask=None, dropout_p=dropout_p)

    attn_out = attn_out.transpose(1,2).contiguous().view(B, N, ED)
    y = self.output(attn_out)
    y = self.resid_drop(y)
    return y