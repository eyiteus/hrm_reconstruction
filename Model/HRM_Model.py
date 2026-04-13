import torch
import torch.nn as nn
import transformers

#The module to combine the 4 networks above into the HRM
class HRM(nn.Module):
  def __init__(self, L_module, H_module, encoder, head, M, N, T):
    super().__init__()
    assert M > 0
    assert T > 0
    self.low_level = L_module
    self.high_level = H_module
    self.M = M
    self.N = N
    self.T = T
    self.head = head
    self.encoder = encoder


  def encode(self, x):
      return self.encoder(x)

  def step(self, z_H, z_L, x_embed):
    """One full segment — T low passes per each othe N high passes.
    Input states should already be detached by the caller."""
    with torch.no_grad():
      #Do N-1 High Passes
      for n in range(self.N - 1):
        for t in range(self.T):
          z_L = self.low_level(x_embed, z_H, z_L)
        z_H = self.high_level(x_embed, z_H, z_L)  # grad flows only here

      #Do the final High pass. Only take gradient at the end
      for t in range(self.T - 1):
        z_L = self.low_level(x_embed, z_H, z_L)
    z_L = self.low_level(x_embed, z_H, z_L)     # last L — grad flows
    z_H = self.high_level(x_embed, z_H, z_L)  # grad flows only here
    return z_H, z_L

  def predict(self, x):
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
    x = x.long()  # ensure indices are integer
    x = x.to(next(self.parameters()).device)  # move to same device as model
    out = self.forward(x)
    labels = out.argmax(dim=-1)
    return labels


  def forward(self, x):
    x_embed = self.encode(x)
    B, L, d = x_embed.shape
    z_H = torch.zeros(B, L, d, device=x.device)
    z_L = torch.zeros(B, L, d, device=x.device)

    for m in range(self.M):
      z_H, z_L = self.step(z_H, z_L.detach(), x_embed)


    return self.head(z_H)