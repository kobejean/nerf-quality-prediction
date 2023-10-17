# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Base Model implementation which takes in RayBundles
"""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model as _Model, ModelConfig as _ModelConfig


# Model related configs
@dataclass
class ModelConfig(_ModelConfig):
    """Configuration for model instantiation"""
    pass


class Model(_Model):
    """Model class
    Where everything (Fields, Optimizers, Samplers, Visualization, etc) is linked together. This should be
    subclassed for custom NeRF model.

    Args:
        config: configuration for instantiating model
        scene_box: dataset scene box
    """

    config: ModelConfig

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                if not torch.is_tensor(output):
                    # TODO: handle lists of tensors as well
                    continue
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            if output_name == "weight_hist":
                outputs[output_name] = torch.stack(outputs_list).sum(dim=0)
            elif output_name == "weight_hist_edges":
                outputs[output_name] = outputs_list[0]
            else:
                outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs


    @abstractmethod
    def get_densities(self, ray_samples: RaySamples) -> torch.Tensor:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """