import torch
import torch.nn as nn

from patch_scaler import PatchScaler
from heads import SlimViTHead

head_num = {
    "cifar10": 10,
    "cifar100": 100,
}

class SlimViT(nn.Module):
    def __init__(self, dataset: str,
                 base_model: nn.Module,
                 fine_tune: bool=True) -> None:
        """

        :param dataset:
        :param base_model:
        :param fine_tune:
        """
        super().__init__()
        self.vit = base_model(weights="DEFAULT" if fine_tune else None)
        self.scaler = PatchScaler(patch_size=self.vit.encoder.pos_embedding.shape[1] - 1)
        self.head = SlimViTHead(out_features=head_num[dataset])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        x = self.vit._process_input(x)

        cls = self.vit.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.vit.encoder.pos_embedding

        x = self.scaler(x)

        x = self.vit.encoder(x)
        x = x[:, 0, :]
        x = self.head(x)
        return x

