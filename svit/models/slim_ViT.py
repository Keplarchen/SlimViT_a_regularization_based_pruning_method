import torch
import torch.nn as nn

from svit import config
from svit.models.patch_scaler import PatchScaler
from svit.models.heads import SlimViTHead

from torchvision.models import vit_b_16

head_num = {
    "cifar10": 10,
    "cifar100": 100,
}

class SlimViT(nn.Module):
    def __init__(self, dataset: str=config["data"]["train_dataset"],
                 fine_tune: bool=config["models"]["fine_tune"],
                 multi_prune: bool=config["models"]["multi_prune"]) -> None:
        """

        :param dataset:
        :param fine_tune:
        :param multi_prune:
        """
        super().__init__()
        self.multi_prune = multi_prune
        self.vit = vit_b_16(weights="DEFAULT" if fine_tune else None)
        self.head = SlimViTHead(out_features=head_num[dataset])
        if self.multi_prune:
            self.scaler = nn.ModuleList([PatchScaler(patch_size=self.vit.encoder.pos_embedding.shape[1] - 1,
                                                     patch_dim=self.vit.encoder.pos_embedding.shape[2])
                                        for _ in range(len(self.vit.encoder.layers))])
        else:
            self.scaler = PatchScaler(patch_size=self.vit.encoder.pos_embedding.shape[1] - 1,
                                      patch_dim=self.vit.encoder.pos_embedding.shape[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        x = self.vit._process_input(x)

        cls = self.vit.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.vit.encoder.pos_embedding

        if self.multi_prune:
            for i, layer in enumerate(self.vit.encoder.layers):
                x = self.scaler[i](x)
                x = layer(x)
            x = self.vit.encoder.ln(x)
        else:
            x = self.scaler(x)
            x = self.vit.encoder(x)
        x = x[:, 0, :]
        x = self.head(x)
        return x
