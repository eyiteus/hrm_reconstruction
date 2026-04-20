import torch
import torch.nn as nn


class HRM(nn.Module):
    def __init__(
        self,
        L_module,
        H_module,
        encoder,
        head,
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

        # Fixed initial hidden states
        zH0 = torch.empty(1, max_len, d_model)
        zL0 = torch.empty(1, max_len, d_model)
        nn.init.trunc_normal_(zH0, mean=0.0, std=1.0, a=-2.0, b=2.0)
        nn.init.trunc_normal_(zL0, mean=0.0, std=1.0, a=-2.0, b=2.0)

        self.register_buffer("z_H0", zH0)
        self.register_buffer("z_L0", zL0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def init_states(self, x_embed: torch.Tensor):
        B, L, _ = x_embed.shape
        z_H = self.z_H0[:, :L, :].expand(B, -1, -1).clone()
        z_L = self.z_L0[:, :L, :].expand(B, -1, -1).clone()
        return z_H, z_L

    def step(self, z_H: torch.Tensor, z_L: torch.Tensor, x_embed: torch.Tensor):
        # NT - 1 no-grad updates
        with torch.no_grad():
            for i in range(self.N * self.T - 1):
                z_L = self.low_level(x_embed, z_H, z_L)
                if (i + 1) % self.T == 0:
                    z_H = self.high_level(x_embed, z_H, z_L)

        # final gradient-carrying updates
        z_L = self.low_level(x_embed, z_H, z_L)
        z_H = self.high_level(x_embed, z_H, z_L)
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

    @torch.no_grad()
    def predict(self, x: torch.Tensor, M=None):
        if M is None:
            M = self.M

        z_H, z_L = None, None
        logits = None

        for _ in range(M):
            z_H, z_L, logits = self.segment(x, z_H, z_L)
            z_H = z_H.detach()
            z_L = z_L.detach()

        return logits.argmax(dim=-1)