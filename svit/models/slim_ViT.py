import torch
import torch.nn as nn

from svit import config
from svit.models.patch_scaler import PatchScaler

from torchvision.models import vit_b_16, ViT_B_16_Weights

head_num = {
    "cifar10": 10,
    "cifar100": 100,
    "imagenette": 10,
    "imagenet": 1000
}

class SlimViT(nn.Module):
    def __init__(self, config_var: dict = config,
                 is_base_model: bool=False) -> None:
        """

        :param config_var:
        :param is_base_model:
        """
        super().__init__()
        self.is_base_model = is_base_model
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT if config_var["models"]["fine_tune"] else None)
        self.head = nn.Linear(in_features=self.vit.encoder.pos_embedding.shape[-1],
                              out_features=head_num[config_var["data"]["train_dataset"]])
        self.scaler = nn.ModuleList([PatchScaler(patch_size=self.vit.encoder.pos_embedding.shape[1] - 1,
                                                 patch_dim=self.vit.encoder.pos_embedding.shape[2],
                                                 is_base_model=self.is_base_model,
                                                 init_scale=config_var["models"]["init_scale"],
                                                 init_scale_threshold=config_var["models"]["init_scale_threshold"],
                                                 init_sparsity_threshold=config_var["models"]["init_sparsity_threshold"],
                                                 granularity=config_var["models"]["granularity"])
                                     for _ in range(len(self.vit.encoder.layers) - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        x = self.vit._process_input(x)
        n = x.shape[0]

        cls = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat((cls, x), dim=1)

        x + self.vit.encoder.pos_embedding
        x = self.vit.encoder.dropout(x)

        for i, layer in enumerate(self.vit.encoder.layers):
            if i !=  len(self.vit.encoder.layers) - 1:
                x = layer(x)
                x = self.scaler[i](x)
            else:
                x = layer(x)
        self.vit.encoder.ln(x)

        x = x[:, 0]
        x = self.head(x)

        return x
