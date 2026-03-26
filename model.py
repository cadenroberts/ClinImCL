import torch
import torch.nn as nn
import torch.nn.functional as F

IMG = 96


class Block(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(ci, co, 3, padding=1),
                                 nn.BatchNorm3d(co), nn.ReLU(True))
    def forward(self, x): return self.net(x)


class Encoder(nn.Module):
    def __init__(self, base=32, out=256):
        super().__init__()
        ch = [1, base, base*2, base*4, base*8]
        self.blocks = nn.ModuleList(Block(ch[i], ch[i+1]) for i in range(4))
        self.head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                                  nn.Linear(ch[-1], out))
    def forward(self, x):
        for b in self.blocks:
            x = b(F.max_pool3d(x, 2))
        return self.head(x)


class ClinImCL(nn.Module):
    def __init__(self, proj=128):
        super().__init__()
        self.enc = Encoder()
        self.proj = nn.Sequential(nn.Linear(256, 256), nn.ReLU(True),
                                  nn.Linear(256, proj))
    def forward(self, x):
        h = self.enc(x)
        return F.normalize(self.proj(h), dim=1), h


def info_nce(z1, z2, temp=0.07):
    logits = (z1 @ z2.t()) / temp
    tgt = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))


def augment(x):
    x = x.clone()
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[int(torch.randint(1, 4, (1,)).item())])
    if torch.rand(1).item() < 0.3:
        x = x * (0.9 + 0.2 * torch.rand(1).item()) + (-0.1 + 0.2 * torch.rand(1).item())
    if torch.rand(1).item() < 0.3:
        x = x + torch.randn_like(x) * (0.01 + 0.04 * torch.rand(1).item())
    return x.clamp(0.0, 1.0)
