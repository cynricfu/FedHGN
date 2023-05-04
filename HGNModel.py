from argparse import Namespace
from collections.abc import Callable
from itertools import chain
from typing import Optional, Union

import dgl
import dgl.nn as dglnn
import torch as th
import torch.nn as nn
import torch.nn.functional as F

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

        # always use bases
        # basis coefficients are defined in class HGNModel
        if self.use_weight:
            self.bases = nn.Parameter(th.Tensor(num_bases, in_dim, out_dim))
            nn.init.xavier_uniform_(self.bases, gain=nn.init.calculate_gain('relu'))

        # bias
        if use_bias:
            self.h_bias = nn.Parameter(th.Tensor(out_dim))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if use_self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_dim, out_dim))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g: dgl.DGLHeteroGraph, inputs: dict[str, th.FloatTensor], basis_coeffs: nn.ParameterDict) -> dict[
        str, th.FloatTensor]:
        with g.local_scope():
            if self.use_weight:
                # compute the weight matrices from bases and basis coefficients
                w_dict = {}
                for etype in self.etypes:
                    w_dict[etype] = {
                        "weight": th.matmul(basis_coeffs[etype], self.bases.view(self.num_bases, -1)).view(self.in_dim,
                                                                                                           self.out_dim)}
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
    def __init__(self, hidden_dim: int, out_dim: int, etypes: list[str], num_bases: int, *, num_hidden_layers: int = 1,
                 dropout: float = 0.0, use_self_loop: bool = False) -> None:
        super(RGCN, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.etypes = etypes
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop

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

    def forward(self, g: Union[dgl.DGLHeteroGraph, list], inputs: dict[str, th.FloatTensor],
                basis_coeffs_encoder: nn.ModuleList) -> dict[str, th.FloatTensor]:
        h = inputs
        if isinstance(g, dgl.DGLHeteroGraph):
            # full graph
            for layer, basis_coeffs in zip(self.layers, chain([None], basis_coeffs_encoder)):
                h = layer(g, h, basis_coeffs)
        else:
            # minibatch
            blocks = g
            for layer, block, basis_coeffs in zip(self.layers, blocks, chain([None], basis_coeffs_encoder)):
                h = layer(block, h, basis_coeffs)
        return h


class HGNModel(nn.Module):
    def __init__(self, args: Namespace, out_dim: int, ntypes: list[str], etypes: list[str],
                 canonical_etypes: list[tuple[str, str, str]], num_nodes_dict: dict[str, int]) -> None:
        super(HGNModel, self).__init__()
        self.model_name = args.model
        self.num_bases = args.num_bases
        self.ntypes = ntypes
        self.etypes = etypes
        self.canonical_etypes = canonical_etypes

        # embedding layer: private
        self.embed_layer = dglnn.HeteroEmbedding(num_nodes_dict, args.hidden_dim)
        # self.linear_layer = dglnnHeteroLinear(in_dim_dict, args.hidden_dim)

        # HGNN model: shared
        if self.model_name == "RGCN":
            assert args.num_layers > 1
            # basis coefficients for relations: private
            self.basis_coeffs_encoder = nn.ModuleList()
            for _ in range(args.num_layers - 1):
                param_dict = nn.ParameterDict()
                for etype in self.etypes:
                    param_dict[etype] = nn.Parameter(th.Tensor(self.num_bases))
                    nn.init.xavier_uniform_(param_dict[etype].view(1, -1), gain=nn.init.calculate_gain('relu'))
                self.basis_coeffs_encoder.append(param_dict)
            self.model = RGCN(args.hidden_dim, out_dim, etypes, self.num_bases, num_hidden_layers=args.num_layers - 2,
                              dropout=args.dropout, use_self_loop=args.use_self_loop)
        else:
            raise ValueError(f"Unknown model name {self.model_name}")

    def forward(self, g: Union[dgl.DGLHeteroGraph, list], inputs: dict[str, th.FloatTensor]) -> dict[
        str, th.FloatTensor]:
        # ntype-specific embedding/projection
        if isinstance(g, dgl.DGLHeteroGraph):
            # full graph
            nids_dict = {ntype: g.nodes(ntype) for ntype in g.ntypes}
        else:
            # minibatch
            # g is a list of DGLBlock
            nids_dict = get_data_dict(g[0].srcdata[dgl.NID], g[0].srctypes)
        h_embed_dict = self.embed_layer(nids_dict)
        # h_linear_dict = self.linear_layer(inputs)
        # h_dict = h_embed_dict | h_linear_dict
        h_dict = h_embed_dict

        # HGNN model forward
        h_dict = self.model(g, h_dict, self.basis_coeffs_encoder)

        return h_dict
