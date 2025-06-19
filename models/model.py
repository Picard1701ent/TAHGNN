from types import SimpleNamespace

import torch
from torch import nn

import config

from .TAHGNN import TAHGConv
from .utils import Mlp

cfg = SimpleNamespace(**vars(config))


class TAHGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        in_features = cfg.in_features
        for i in range(cfg.layers - 1):
            self.layers.append(TAHGConv(in_features, cfg.hidden_features, cfg.dropout))
            in_features = cfg.hidden_features
        self.layers.append(TAHGConv(in_features, cfg.out_features, cfg.dropout))
        self.classifier = Mlp(cfg.node_num * cfg.out_features, 256, 2, cfg.dropout)

    def forward(self, x, H, node_strength, gloable_efficiency, dist_matrix):
        out = x
        for i in range(cfg.layers - 1):
            out, _ = self.layers[i](out, H, node_strength, gloable_efficiency)
            out = out + x
        out, e_fts = self.layers[-1](out, H, node_strength, gloable_efficiency)
        out = out + x

        out = torch.flatten(out, start_dim=-2, end_dim=-1)

        out = self.classifier(out)
        out = nn.functional.log_softmax(out, dim=-1)

        return out, e_fts

