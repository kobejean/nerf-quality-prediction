from __future__ import annotations

import torch

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.tensorf import TensoRFModel

class NQPTensoRFModel(TensoRFModel):
    
    def get_densities(self, ray_samples: RaySamples) -> torch.Tensor:
        assert self.field is not None
        field_outputs = self.field(ray_samples)
        return field_outputs[FieldHeadNames.DENSITY]
