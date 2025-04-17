import torch
import torch.nn as nn

from svit import config

class PatchScaler(nn.Module):
    def __init__(self, patch_size: int,
                 patch_dim: int,
                 init_scale: float=config["models"]["init_scale"],
                 init_scale_threshold: float=config["models"]["init_scale_threshold"],
                 init_sparsity_threshold: float=config["models"]["init_sparsity_threshold"],
                 granularity: str=config["models"]["granularity"]) -> None:
        """

        :param patch_size:
        :param init_scale:
        :param init_scale_threshold:
        :param init_sparsity_threshold:
        :param granularity:
        """
        super().__init__()
        self.granularity = granularity
        if granularity == "patch":
            self.scaler = nn.Parameter(torch.full((patch_size,), init_scale))
        elif granularity == "embedding":
            self.scaler = nn.Parameter(torch.full((patch_size, patch_dim), init_scale))
        else:
            raise ValueError(f"Unknown granularity: {granularity}")
        self.scale_threshold = nn.Parameter(torch.tensor(init_scale_threshold))
        self.sparsity_threshold = nn.Parameter(torch.tensor(init_sparsity_threshold))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        cls = x[:, :1, :]
        patch = x[:, 1:, :]
        if self.granularity == "patch":
            scaled_patch = patch * self.scaler.view(1, -1, 1)
        elif self.granularity == "embedding":
            scaled_patch = patch * self.scaler.view(1, patch.shape[1], patch.shape[2])
        else:
            raise ValueError(f"Unknown granularity: {self.granularity}")

        with torch.no_grad():
            scale_hard_mask = scaled_patch > self.scale_threshold
        scale_soft_mask = torch.sigmoid(scaled_patch - self.scale_threshold)
        scale_gate = scale_hard_mask + scale_soft_mask - scale_soft_mask.detach()
        scale_gated_patch = scaled_patch * scale_gate

        with torch.no_grad():
            patch_sparsity = (scale_gated_patch.abs() < 1e-4).float().mean(dim=-1, keepdim=True)
            sparsity_hard_mask = patch_sparsity < self.sparsity_threshold
        sparsity_soft_mask = torch.sigmoid(self.sparsity_threshold - patch_sparsity)
        sparsity_gate = sparsity_hard_mask + sparsity_soft_mask - sparsity_soft_mask.detach()
        sparsity_gated_patch = scale_gated_patch * sparsity_gate

        # TODO: patch padding

        x = torch.cat((cls, sparsity_gated_patch), dim=1)
        return x
