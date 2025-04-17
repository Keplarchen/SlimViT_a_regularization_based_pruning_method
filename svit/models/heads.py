import torch
import torch.nn as nn

class SlimViTHead(nn.Module):
    def __init__(self, in_features: int,
                 out_features: int,
                 bias: bool=True) -> None:
        """
        Initializes an instance of the class that represents a linear layer with
        defined input and output features, and an optional bias term.

        :param in_features: Number of input features for the linear transformation.
        :type in_features: int
        :param out_features: Number of output features for the linear transformation.
        :type out_features: int
        :param bias: Indicates whether the linear transformation includes
            the bias term. Defaults to True.
        :type bias: bool
        """
        super().__init__()
        self.head = nn.Linear(in_features, out_features, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input tensor through a specified transformation defined
        by the `head` module and returns the result.

        This method represents the forward pass of the network, where the input
        data is passed through layers or operations encapsulated in the `head`.

        :param x: A tensor input that is processed by the `head` transformation.
        :type x: torch.Tensor
        :return: A tensor output after being processed through the `head`.
        :rtype: torch.Tensor
        """
        x = self.head(x)
        return x
