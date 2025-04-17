import torch
import torch.nn as nn

from svit import config

class PatchScaler(nn.Module):
    def __init__(self, patch_size: int,
                 patch_dim: int,
                 is_base_model: bool,
                 init_scale: float=config["models"]["init_scale"],
                 init_scale_threshold: float=config["models"]["init_scale_threshold"],
                 init_sparsity_threshold: float=config["models"]["init_sparsity_threshold"],
                 granularity: str=config["models"]["granularity"]) -> None:
        """
        Initializes a model component with parameters for scaling and sparsity threshold
        based on specified granularity. The initialization supports per-patch and
        per-embedding configurations, constructing appropriate scaling parameters for
        the given granularity.

        :param patch_size: The size of each patch for which the scaling or embedding
                           parameters are applied.
        :param patch_dim: The dimensionality of individual patches, relevant for the
                          "embedding" granularity.
        :param init_scale: Initial scaling factor applied to the patches or embeddings.
        :param init_scale_threshold: Threshold for managing the scale values.
        :param init_sparsity_threshold: Threshold for managing sparsity values.
        :param granularity: Specifies the granularity mode, which can be either "patch"
                            or "embedding".

        :raises ValueError: If the provided granularity is not recognized.
        """
        super().__init__()
        self.granularity = granularity
        self.is_base_model = is_base_model
        if not self.is_base_model:
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
        Processes an input tensor by applying scaling and sparsity gating mechanisms
        based on predefined thresholds and granularity settings.

        The forward method splits the input tensor into class (cls) and patch components,
        applies scaling to the patch segment depending on a granularity setting, and
        performs gated sparsity and scaling adjustments. These adjustments are used
        to restrict and modify the patch values according to specified thresholds
        and conditions.

        :param x: Input tensor with the first dimension representing batch size,
                  second dimension (split into class and patch components), and
                  third dimension representing feature dimensions.
        :type x: torch.Tensor
        :return: Tensor combining the class segment and sparsity-gated patch segment
                 after processing.
        :rtype: torch.Tensor
        """
        if not self.is_base_model:
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
        else:
            return x
