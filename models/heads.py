import torch
import torch.nn as nn

class SlimViTHead(nn.Module):
    def __init__(self, in_features: int=768,
                 out_features: int=100,
                 bias: bool=True) -> None:
        """

        :param in_features:
        :param out_features:
        :param bias:
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.head = nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        x = self.head(x)
        return x