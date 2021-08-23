import torch
from torch import nn


class LNCMLP(nn.Module):
    def __init__(self, num_infeatures, num_layers, num_classes=2, d=None):
        super().__init__()

        if num_layers == 1:
            self.layers = nn.Linear(num_infeatures, num_classes)
        elif num_classes >= 2:
            if d is None:
                d = num_infeatures
            self.layers = nn.Sequential(
                nn.Linear(num_infeatures, d),
                nn.Sequential(*([nn.ReLU(), nn.Linear(d, d)] * (num_layers - 2))),
                nn.ReLU(), nn.Linear(d, num_classes)
            )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.layers(x)
        return out

    def infer(self, x):
        out = self.forward(x)
        return torch.nn.functional.softmax(out, dim=-1)


class LNCLSTM(nn.Module):
    def __init__(self):
        pass


class LNCTransformer(nn.Module):
    def __init__(self):
        pass
