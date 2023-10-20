from typing import Optional

import nerfacc
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

class FeaturesRenderer(nn.Module):
    """Calculate features along the ray."""

    @classmethod
    def forward(
        cls,
        features: Float[Tensor, "*bs num_samples num_features"],
        weights: Float[Tensor, "*bs num_samples 1"],
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*bs num_classes"]:
        """Calculate features along the ray."""
        if ray_indices is not None and num_rays is not None:
            # Necessary for packed samples from volumetric ray sampler
            return nerfacc.accumulate_along_rays(
                weights[..., 0], values=features, ray_indices=ray_indices, n_rays=num_rays
            )
        else:
            return torch.sum(weights * features, dim=-2)

