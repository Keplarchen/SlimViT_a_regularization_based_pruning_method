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
        self.head = nn.Linear(in_features, out_features, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        x = self.head(x)
        return x
