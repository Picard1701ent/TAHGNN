from types import SimpleNamespace

import torch
from torch import einsum, nn
from torch.nn import Parameter

import config

cfg = SimpleNamespace(**vars(config))


class TAHGConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout, bias=True) -> None:
        super().__init__()
        self.drop_out = dropout
        # self.kmeans = KMeans(n_clusters=n_clusters,max_iter=40,verbose=False)
        self.theta = Parameter(torch.Tensor(in_ch, out_ch))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ch))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()
        # self.bn = nn.BatchNorm1d(out_ch)
        self.bn = nn.LayerNorm(out_ch)
        self.gelu = nn.GELU()  # nn.ReLU(inplace=True)
        # self.H_attn = SelfAttention(in_ch,in_ch//4)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, H, node_strength, gloable_efficiency):
        assert len(x.shape) == 3, "the input of HyperConv should be N * V * C"
        # H = H * gaussian_distance
        y = einsum("nvc,co->nvo", x, self.theta)  # y: [batcj, node, target_features]
        Hv = H * gloable_efficiency.unsqueeze(1)  # N,V,E, N,E

        Dv = torch.diag_embed(
            1.0 / Hv.sum(2), dim1=-2, dim2=-1
        )  # H: [batch, node, edge] Dv: [batch, node, node]
        HDv = einsum("nkv,nve->nke", Dv, Hv)  # HDv = [batch, node, edge]
        He = H * node_strength
        De = torch.diag_embed(
            1.0 / He.sum(1), dim1=-2, dim2=-1
        )  # De: [batch, edge, edge]
        HDe = einsum("nve,nek->nvk", He, De)  # HDe: [batch, node, edge]
        e_fts = einsum(
            "nvc,nve->nec", y, HDe
        )  # y: [batch, node, target_features] HDe: [batch, node, edge] -> e_fts: [batch, edge, target_features]
        y = einsum("nec,nve->nvc", e_fts, HDv)  # y: [batch, node, target_features]
        y = y + self.bias.unsqueeze(0).unsqueeze(0)

        # y = self.pooling(y)
        return self.gelu(y), e_fts
