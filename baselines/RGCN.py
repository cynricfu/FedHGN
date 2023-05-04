from typing import Optional, Union
from collections.abc import Callable

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn

from utils import get_data_dict


class RGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, etypes: list[str], num_bases: int, *, use_weight: bool = True,
                 use_bias: bool = True, activation: Optional[Callable] = None, use_self_loop: bool = False,
                 dropout: float = 0.0) -> None:
        super(RGCNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.etypes = etypes
        self.num_bases = num_bases
        self.use_weight = use_weight
        self.use_bias = use_bias
        self.activation = activation
        self.use_self_loop = use_self_loop

        self.conv = dglnn.HeteroGraphConv(
            {etype: dglnn.GraphConv(in_dim, out_dim, norm='right', weight=False, bias=False) for etype in etypes})

        # basis coefficients are defined in class HGNModel
        if self.use_weight:
            if self.num_bases > 0:
                self.weight_basis = dglnn.WeightBasis((in_dim, out_dim), self.num_bases, len(etypes))
            else:
                self.weight = nn.Parameter(th.Tensor(len(etypes), in_dim, out_dim))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if use_bias:
            self.h_bias = nn.Parameter(th.Tensor(out_dim))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if use_self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_dim, out_dim))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g: dgl.DGLHeteroGraph, inputs: dict[str, th.FloatTensor]) -> dict[str, th.FloatTensor]:
        with g.local_scope():
            if self.use_weight:
                if self.num_bases > 0:
                    weight = self.weight_basis()
                else:
                    weight = self.weight
                w_dict = {etype: {"weight": weight[i]} for i, etype in enumerate(self.etypes)}
            else:
                w_dict = {}

            if g.is_block:
                inputs_src = inputs
                inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
            else:
                inputs_src = inputs_dst = inputs

            hs = self.conv(g, inputs, mod_kwargs=w_dict)

            def _apply(ntype, h):
                if self.use_self_loop:
                    h = h + th.matmul(inputs_dst[ntype], self.loop_weight)
                if self.use_bias:
                    h = h + self.h_bias
                if self.activation:
                    h = self.activation(h)
                return self.dropout(h)

            return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class RGCN(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, etypes: list[str], num_nodes_dict: dict[str, int], num_bases: int,
                 *, num_hidden_layers: int = 1, dropout: float = 0.0, use_self_loop: bool = False) -> None:
        super(RGCN, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.etypes = etypes
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop

        self.embed_layer = dglnn.HeteroEmbedding(num_nodes_dict, hidden_dim)

        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(RGCNLayer(self.hidden_dim, self.hidden_dim, etypes, self.num_bases, activation=F.relu,
                                     use_self_loop=self.use_self_loop, dropout=self.dropout, use_weight=False))
        # h2h
        for i in range(self.num_hidden_layers):
            self.layers.append(RGCNLayer(self.hidden_dim, self.hidden_dim, etypes, self.num_bases, activation=F.relu,
                                         use_self_loop=self.use_self_loop, dropout=self.dropout))
        # h2o
        self.layers.append(RGCNLayer(self.hidden_dim, self.out_dim, etypes, self.num_bases, activation=None,
                                     use_self_loop=self.use_self_loop))

    def forward(self, g: Union[dgl.DGLHeteroGraph, list], inputs: dict[str, th.FloatTensor]) -> dict[
        str, th.FloatTensor]:
        if isinstance(g, dgl.DGLHeteroGraph):
            # full graph
            nids_dict = {ntype: g.nodes(ntype) for ntype in g.ntypes}
            h_dict = self.embed_layer(nids_dict)
            for layer in self.layers:
                h_dict = layer(g, h_dict)
        else:
            # minibatch
            blocks = g
            nids_dict = get_data_dict(blocks[0].srcdata[dgl.NID], blocks[0].srctypes)
            h_dict = self.embed_layer(nids_dict)
            for layer, block in zip(self.layers, blocks):
                h_dict = layer(block, h_dict)
        return h_dict
