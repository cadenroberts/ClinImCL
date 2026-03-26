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
