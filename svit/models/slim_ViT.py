import torch
import torch.nn as nn

from svit import config
from svit.models.patch_scaler import PatchScaler
from svit.models.heads import SlimViTHead

from torchvision.models import vit_b_16, ViT_B_16_Weights

head_num = {
    "cifar10": 10,
    "cifar100": 100,
    "imagenette": 10,
}

class SlimViT(nn.Module):
    def __init__(self, dataset: str=config["data"]["train_dataset"],
                 fine_tune: bool=config["models"]["fine_tune"],
                 multi_prune: bool=config["models"]["multi_prune"],
                 is_base_model: bool=False) -> None:
        """
        Initializes the model with specified options for fine-tuning, multi-step pruning,
        and dataset-specific configuration. The class leverages a Vision Transformer (ViT)
        model as the backbone and optionally applies multi-step pruning via `PatchScaler`.

        The class dynamically adapts to the input dataset and configures model components
        accordingly, including the positional embedding, layer structure, and pruning logic.

        Options for fine-tuning and multi-step pruning are provided for flexibility in model
        training and optimization.

        :param dataset: The name of the dataset used for training or evaluation.
        :type dataset: str
        :param fine_tune: Indicates if the Vision Transformer backbone should be fine-tuned.
        :type fine_tune: bool
        :param multi_prune: Flag for enabling multi-step pruning of patches using `PatchScaler`.
        :type multi_prune: bool
        """
        super().__init__()
        self.is_base_model = is_base_model
        self.multi_prune = multi_prune
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT if fine_tune else None)
        self.head = SlimViTHead(in_features=self.vit.encoder.pos_embedding.shape[-1], out_features=head_num[dataset])
        if self.multi_prune:
            self.scaler = nn.ModuleList([PatchScaler(patch_size=self.vit.encoder.pos_embedding.shape[1] - 1,
                                                     patch_dim=self.vit.encoder.pos_embedding.shape[2],
                                                     is_base_model=self.is_base_model)
                                        for _ in range(len(self.vit.encoder.layers) - 1)])
        else:
            self.scaler = PatchScaler(patch_size=self.vit.encoder.pos_embedding.shape[1] - 1,
                                      patch_dim=self.vit.encoder.pos_embedding.shape[2], is_base_model=is_base_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes an input tensor through a vision transformer-based model with added
        functionality for multi-layer or single-layer scaling and returns the output
        tensor. The process includes input transformation, token addition, positional
        embedding application, layer-specific transformations, and final classification
        adjustments.

        :param x: Input tensor expected to be in a format suitable for processing
            through the Vision Transformer (ViT) model.
        :type x: torch.Tensor

        :return: Output tensor after being processed through the model, particularly
            suitable for classification or other prediction tasks.
        :rtype: torch.Tensor
        """
        x = self.vit._process_input(x)

        cls = self.vit.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.vit.encoder.pos_embedding

        if self.multi_prune:
            for i, layer in enumerate(self.vit.encoder.layers):
                if i !=  len(self.vit.encoder.layers) - 1:
                    x = layer(x)
                    x = self.scaler[i](x)
                else:
                    x = layer(x)
            x = self.vit.encoder.ln(x)
        else:
            x = self.scaler(x)
            x = self.vit.encoder(x)
        x = x[:, 0, :]
        x = self.head(x)
        return x
