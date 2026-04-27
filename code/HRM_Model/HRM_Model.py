import torch
import torch.nn as nn


class HRM(nn.Module):
    def __init__(
        self,
        L_module: nn.Module,
        H_module: nn.Module,
        encoder: nn.Module,
        head: nn.Module,
        M: int,
        N: int,
        T: int,
        max_len: int = 81,
        d_model: int = 256,
    ):
        super().__init__()

        assert M > 0
        assert N > 0
        assert T > 0

        self.low_level = L_module
        self.high_level = H_module
        self.encoder = encoder
        self.head = head

        self.M = M
        self.N = N
        self.T = T

        self.max_len = max_len
        self.d_model = d_model

        z_H0 = torch.empty(1, max_len, d_model)
        z_L0 = torch.empty(1, max_len, d_model)

        nn.init.trunc_normal_(z_H0, mean=0.0, std=1.0, a=-2.0, b=2.0)
        nn.init.trunc_normal_(z_L0, mean=0.0, std=1.0, a=-2.0, b=2.0)

        self.register_buffer("z_H0", z_H0)
        self.register_buffer("z_L0", z_L0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def init_states(self, x_embed: torch.Tensor):
        B, L, _ = x_embed.shape

        z_H = self.z_H0[:, :L, :].expand(B, -1, -1).clone()
        z_L = self.z_L0[:, :L, :].expand(B, -1, -1).clone()

        return z_H, z_L

    def step(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        x_embed: torch.Tensor,
    ):
        # NT - 1 no-grad updates
        with torch.no_grad():
            for i in range(self.N * self.T - 1):
                z_L = self.low_level(x_embed, z_H, z_L)

                if (i + 1) % self.T == 0:
                    z_H = self.high_level(z_H, z_L)

        # Final gradient-carrying updates
        z_L = self.low_level(x_embed, z_H, z_L)
        z_H = self.high_level(z_H, z_L)

        logits = self.head(z_H)

        return z_H, z_L, logits

    def segment(self, x: torch.Tensor, z_H=None, z_L=None):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = x.long().to(next(self.parameters()).device)
        x_embed = self.encode(x)

        if z_H is None or z_L is None:
            z_H, z_L = self.init_states(x_embed)

        return self.step(z_H, z_L, x_embed)

    
    def forward(self, x: torch.Tensor, y: torch.Tensor = None, loss_fn=None, M=None):
        if M is None:
            M = self.M

        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = x.long().to(next(self.parameters()).device)

        if y is not None:
            y = y.long().to(x.device)

        z_H = None
        z_L = None
        logits = None

        for _ in range(M):
            z_H, z_L, logits = self.segment(x, z_H, z_L)
            z_H = z_H.detach()
            z_L = z_L.detach()

        loss = None

        if y is not None:
            if loss_fn is None:
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

            x_flat = x.reshape(-1)
            y_flat = y.reshape(-1)

            mask = x_flat != y_flat
            targets = y_flat.clone()
            targets[~mask] = -100

            pred = logits.reshape(-1, logits.size(-1))
            loss = loss_fn(pred, targets)

        return logits, loss
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor, M=None):
        logits, _ = self.forward(x, M=M)
        x = x.long().to(logits.device)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        pred = logits.argmax(dim=-1)
        return torch.where(x != 0, x, pred)