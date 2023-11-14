from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type

import torch

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.instant_ngp import NGPModel
from nqp.utils.post_processing import post_process_render
from nerfstudio.utils import colormaps


class NQPInstantNGPModel(NGPModel):


    def get_metrics_dict(self, outputs, batch):
        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)
        rgb = outputs["rgb"]
        if image.shape != rgb.shape:
            print("WARNING! render and ground truth shape doesn't match. Assuming post processing is required.")
            rgb = post_process_render(rgb)
        metrics_dict = {}
        metrics_dict["psnr"] = self.psnr(rgb, image)
        metrics_dict["num_samples_per_batch"] = outputs["num_samples_per_ray"].sum()
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"][..., :3].to(self.device)
        rgb = outputs["rgb"]
        if image.shape != rgb.shape:
            print("WARNING! render and ground truth shape doesn't match. Assuming post processing is required.")
            rgb = post_process_render(rgb)
        pred_rgb, image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=rgb,
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )
        rgb_loss = self.rgb_loss(image, pred_rgb)
        loss_dict = {"rgb_loss": rgb_loss}
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)
        rgb = outputs["rgb"]
        if image.shape != rgb.shape:
            print("WARNING! render and ground truth shape doesn't match. Assuming post processing is required.")
            print(rgb.shape)
            rgb = post_process_render(rgb)
            print(rgb.shape)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )
        
        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips)}  # type: ignore
        # TODO(ethan): return an image dictionary

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
        }

        return metrics_dict, images_dict

