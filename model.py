from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter, ReLU, Linear

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm


class SARMAConv(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int,
                 num_stacks: int = 1, num_layers: int = 1,
                 dropout: float = 0.,
                 bias: bool = True, KSGC: int = 1, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_stacks = num_stacks
        self.num_layers = num_layers

        self.dropout = dropout
        self.K_SGC = KSGC

        K, T, F_in, F_out = num_stacks, num_layers, in_channels, out_channels
        self.weight = Parameter(torch.Tensor(K, F_out, F_out))
        if in_channels > 0:
            self.SGC_weight = Parameter(torch.Tensor(1,F_in, F_out))
            self.init_weight = Parameter(torch.Tensor(K, F_in, F_out))
            self.root_weight = Parameter(torch.Tensor(K, F_in, F_out))
        else:
            self.SGC_weight = torch.nn.parameter.UninitializedParameter()
            self.init_weight = torch.nn.parameter.UninitializedParameter()
            self.root_weight = torch.nn.parameter.UninitializedParameter()
            self._hook = self.register_forward_pre_hook(self.initialize_parameters)

        if bias:
            self.bias = Parameter(torch.Tensor(K, 1, F_out))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        if not isinstance(self.init_weight, torch.nn.UninitializedParameter):
            glorot(self.init_weight)
            glorot(self.root_weight)
            glorot(self.SGC_weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        # if isinstance(edge_index, Tensor):
        edge_index, edge_weight = gcn_norm(  # yapf: disable
            edge_index, edge_weight, x.size(self.node_dim),
            add_self_loops=True, flow=self.flow, dtype=x.dtype)

        # elif isinstance(edge_index, SparseTensor):
        #     edge_index = gcn_norm(  # yapf: disable
        #         edge_index, edge_weight, x.size(self.node_dim),
        #         add_self_loops=False, flow=self.flow, dtype=x.dtype)


        # outs = x
        outs = x if self.dropout == 0 else  F.dropout(x, p = self.dropout, training=self.training)
        out_s = outs @ self.SGC_weight

        for k in range(self.K_SGC):
            out_s = self.propagate(edge_index, x=out_s, edge_weight=edge_weight,size=None)

        edge_index, edge_weight = gcn_norm(  # yapf: disable
            edge_index, edge_weight, x.size(self.node_dim),
            add_self_loops=False, flow=self.flow, dtype=x.dtype)

        x = x.unsqueeze(-3)
        out = x

        for t in range(self.num_layers):
            if t == 0:
                out = out @ self.init_weight
            else:
                out = out @ self.weight

            out = self.propagate(edge_index, x=out, edge_weight=edge_weight, size=None)
            root =  outs
            root = root @ self.root_weight
            out = out + root

            if self.bias is not None:
                out = out + self.bias

        out = out.mean(dim=-3) + out_s
        return out.mean(dim=-3)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    @torch.no_grad()
    def initialize_parameters(self, module, input):
        if isinstance(self.init_weight, nn.parameter.UninitializedParameter):
            F_in, F_out = input[0].size(-1), self.out_channels
            T, K = self.weight.size(0) + 1, self.weight.size(1)
            self.init_weight.materialize((K, F_in, F_out))
            self.root_weight.materialize((K, F_in, F_out))
            self.SGC_weight.materialize((1, F_in, F_out))
            glorot(self.init_weight)
            glorot(self.root_weight)
            glorot(self.SGC_weight)
        module._hook.remove()
        delattr(module, '_hook')

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_stacks={self.num_stacks}, '
                f'num_layers={self.num_layers})')
