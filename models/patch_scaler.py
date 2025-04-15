import torch
import torch.nn as nn

from code import config

class PatchScaler(nn.Module):
    def __init__(self, patch_size: int,
                 init_scale: float=config["models"]["init_scale"],
                 init_threshold: float=config["models"]["init_threshold"],
                 hard: bool=config["models"]["hard_prune"]) -> None:
        """

        :param patch_size:
        :param init_scale:
        :param init_threshold:
        :param hard:
        """
        super().__init__()
        self.patch_size = patch_size
        self.init_scale = init_scale
        self.init_threshold = init_threshold
        self.hard = hard
        self.scaler = nn.Parameter(torch.full((self.patch_size,), self.init_scale))
        self.threshold = nn.Parameter(torch.tensor(self.init_threshold))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        cls = x[:, :1, :]
        patch = x[:, 1:, :]
        if self.hard:
            with torch.no_grad():
                hard_mask = (self.scaler > self.threshold).float()
            soft_mask = torch.sigmoid(self.scaler - self.threshold)
            gate = hard_mask + soft_mask - soft_mask.detach()
        else:
            gate = torch.sigmoid(self.scaler - self.threshold)
            if not self.training:
                gate = (gate > self.threshold).float()
        scaled_patch = patch * gate.view(1, -1, 1)
        pos_emb = torch.cat([cls, scaled_patch], dim=1)
        return pos_emb
