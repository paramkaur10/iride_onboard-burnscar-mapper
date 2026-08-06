# src/pynas/blocks/activations.py
import torch.nn as nn
from functools import partial

class GELU(nn.Module):
    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.op = nn.GELU(approximate=approximate)
    def forward(self, x):
        return self.op(x)

class ReLU(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.op = nn.ReLU(inplace=inplace)
    def forward(self, x):
        return self.op(x)

# If you want the inplace variant selectable via config:
ReLUInPlace = partial(ReLU, inplace=True)

class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.Sigmoid()
    def forward(self, x):
        return self.op(x)

class Softmax(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.op = nn.Softmax(dim=dim)
    def forward(self, x):
        return self.op(x)
