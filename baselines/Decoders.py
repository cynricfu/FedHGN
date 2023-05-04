import torch as th
import torch.nn as nn
import torch.nn.functional as F


class NodeClassifier(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, pre_activation=F.relu) -> None:
        super(NodeClassifier, self).__init__()
        self.pre_activation = pre_activation
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: th.FloatTensor) -> th.FloatTensor:
        return self.linear(self.pre_activation(x)) if self.pre_activation else self.linear(x)
