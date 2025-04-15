import torch
import torch.nn as nn

from svit import config

class PatchScaler(nn.Module):
    def __init__(self, patch_size: int,
                 init_scale: float=config["models"]["init_scale"],
                 init_scale_threshold: float=config["models"]["init_scale_threshold"],
                 init_sparsity_threshold: float=config["models"]["init_sparsity_threshold"]) -> None:
        """

        :param patch_size:
        :param init_scale:
        :param init_scale_threshold:
        :param init_sparsity_threshold:
        """
        super().__init__()
        self.scaler = nn.Parameter(torch.full((patch_size,), init_scale))
        self.scale_threshold = nn.Parameter(torch.tensor(init_scale_threshold))
        self.sparsity_threshold = nn.Parameter(torch.tensor(init_sparsity_threshold))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        cls = x[:, :1, :]
        patch = x[:, 1:, :]
        scaled_patch = patch * self.scaler.view(1, -1, 1)

        with torch.no_grad():
            scale_hard_mask = scaled_patch > self.scale_threshold
        scale_soft_mask = torch.sigmoid(scaled_patch - self.scale_threshold)
        scale_gate = scale_hard_mask + scale_soft_mask - scale_soft_mask.detach()
        scale_gated_patch = scaled_patch * scale_gate

        with torch.no_grad():
            patch_sparsity = torch.sum((scale_gated_patch < self.sparsity_threshold).float(), dim=-1, keepdim=True)/patch.shape[-1]
            sparsity_hard_mask = patch_sparsity > self.sparsity_threshold
        # TODO: change soft sparsity gate
        sparsity_gate = sparsity_hard_mask + scale_gated_patch - scale_gated_patch.detach()

        # TODO: patch padding


        pos_emb = torch.cat([cls, gated_patch], dim=1)
        return pos_emb
