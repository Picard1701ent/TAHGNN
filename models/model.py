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
        self.ahgnn0 = TAHGConv(cfg.in_features, cfg.hidden_features, cfg.dropout)
        self.ahgnn1 = TAHGConv(cfg.hidden_features, cfg.out_features, cfg.dropout)
        self.classifier = Mlp(cfg.node_num * cfg.out_features, 256, 2, cfg.dropout)

    def forward(self, x, H, node_strength, gloable_efficiency, dist_matrix):
        ahgnn_out0, _ = self.ahgnn0(x, H, node_strength, gloable_efficiency)
        ahgnn_out0 = ahgnn_out0 + x
        ahgnn_out1, e_fts = self.ahgnn1(
            ahgnn_out0, H, node_strength, gloable_efficiency
        )
        ahgnn_out1 = ahgnn_out0 + ahgnn_out1

        ahgnn_out1 = torch.flatten(ahgnn_out1, start_dim=-2, end_dim=-1)

        out = self.classifier(ahgnn_out1)
        out = nn.functional.log_softmax(out, dim=-1)

        return out, e_fts
