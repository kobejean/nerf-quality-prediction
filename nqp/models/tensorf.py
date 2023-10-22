from __future__ import annotations

import torch

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.tensorf import TensoRFModel, TensoRFModelConfig


from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type, cast

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.encodings import (
    NeRFEncoding,
    TensorCPEncoding,
    TensorVMEncoding,
    TriplaneEncoding,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import MSELoss, tv_loss, scale_gradients_by_distance_squared
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import AABBBoxCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, misc
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig

from nqp.fields.tensorf_field import NQPTensoRFField

class NQPTensoRFModel(TensoRFModel):

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # setting up fields
        if self.config.tensorf_encoding == "vm":
            density_encoding = TensorVMEncoding(
                resolution=self.init_resolution,
                num_components=self.num_den_components,
            )
            color_encoding = TensorVMEncoding(
                resolution=self.init_resolution,
                num_components=self.num_color_components,
            )
        elif self.config.tensorf_encoding == "cp":
            density_encoding = TensorCPEncoding(
                resolution=self.init_resolution,
                num_components=self.num_den_components,
            )
            color_encoding = TensorCPEncoding(
                resolution=self.init_resolution,
                num_components=self.num_color_components,
            )
        elif self.config.tensorf_encoding == "triplane":
            density_encoding = TriplaneEncoding(
                resolution=self.init_resolution,
                num_components=self.num_den_components,
            )
            color_encoding = TriplaneEncoding(
                resolution=self.init_resolution,
                num_components=self.num_color_components,
            )
        else:
            raise ValueError(f"Encoding {self.config.tensorf_encoding} not supported")

        feature_encoding = NeRFEncoding(in_dim=self.appearance_dim, num_frequencies=2, min_freq_exp=0, max_freq_exp=2)
        direction_encoding = NeRFEncoding(in_dim=3, num_frequencies=2, min_freq_exp=0, max_freq_exp=2)

        self.field = NQPTensoRFField(
            self.scene_box.aabb,
            feature_encoding=feature_encoding,
            direction_encoding=direction_encoding,
            density_encoding=density_encoding,
            color_encoding=color_encoding,
            appearance_dim=self.appearance_dim,
            head_mlp_num_layers=2,
            head_mlp_layer_width=128,
            use_sh=False,
        )

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_uniform_samples, single_jitter=True)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_samples, single_jitter=True, include_original=False)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # colliders
        if self.config.enable_collider:
            self.collider = AABBBoxCollider(scene_box=self.scene_box)

        # regularizations
        if self.config.tensorf_encoding == "cp" and self.config.regularization == "tv":
            raise RuntimeError("TV reg not supported for CP decomposition")

    def get_outputs(self, ray_bundle: RayBundle):
        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        dens = self.field.get_density(ray_samples_uniform)
        weights = ray_samples_uniform.get_weights(dens)
        coarse_accumulation = self.renderer_accumulation(weights)
        acc_mask = torch.where(coarse_accumulation < 0.0001, False, True).reshape(-1)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights)

        # fine field:
        field_outputs_fine = self.field.forward(
            ray_samples_pdf, mask=acc_mask, bg_color=colors.WHITE.to(weights.device)
        )

        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])

        accumulation = self.renderer_accumulation(weights_fine)
        depth = self.renderer_depth(weights_fine, ray_samples_pdf)

        rgb = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )

        rgb = torch.where(accumulation < 0, colors.WHITE.to(rgb.device), rgb)
        accumulation = torch.clamp(accumulation, min=0)

        outputs = {"rgb": rgb, "accumulation": accumulation, "depth": depth}
        return outputs
    
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb"].device)
        image = self.renderer_rgb.blend_background(image)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        assert self.config.collider_params is not None
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)

        metrics_dict = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim.item()),
            "lpips": float(lpips.item()),
        }
        images_dict = {"img": combined_rgb, "rgb_gt": image, "rgb_pred": rgb, "accumulation": acc, "depth": depth}
        return metrics_dict, images_dict
