"""
Nerfstudio Template Pipeline
"""

from matplotlib.animation import FuncAnimation
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import typing
import torch
import glob
from dataclasses import dataclass, field
from typing import Literal, Optional, Type
from pathlib import Path
import subprocess

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nqp.utils.utils import (
    load_config,
    save_as_image, read_depth_map, compute_errors,
    load_metadata
)
from nqp.utils.raybundle import contour_eval_ray_bundle, plane_eval_ray_bundle, sphere_eval_ray_bundle, cube_eval_ray_bundle
from nqp.utils.spacial import convert_from_transformed_space
# from nqp.template_datamanager import TemplateDataManagerConfig
# from nqp.template_model import TemplateModel, TemplateModelConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nqp.models.base_model import ModelConfig
from nqp.utils.eval_vis import (
    save_contour_renders,
    save_depth_vis,
    save_weight_distribution_plot,
    save_geometry_surface_eval,
    eval_set_renders_and_metrics,
)
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.utils import colormaps

import typing
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast

import torch
import torch.distributed as dist
from PIL import Image
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from torch import nn
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.configs import base_config as cfg
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
    VanillaDataManager,
)
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler



class NQPPipeline(VanillaPipeline):
    """Template Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """ 

    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        assert isinstance(self.datamanager, VanillaDataManager)
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for camera_ray_bundle, batch in self.datamanager.fixed_indices_eval_dataloader:
                # time this the following line
                inner_start = time()
                height, width = camera_ray_bundle.shape
                num_rays = height * width
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)

                if output_path is not None:
                    camera_indices = camera_ray_bundle.camera_indices
                    assert camera_indices is not None
                    for key, val in images_dict.items():
                        if val.shape[-1] in [1, 3, 4]:
                            # if greyscale, rgb, or rgba
                            Image.fromarray((val * 255).byte().cpu().numpy()).save(
                                output_path / "{0:06d}-{1}.jpg".format(int(camera_indices[0, 0, 0]), key)
                            )
                        else:
                            np.savez_compressed(
                                output_path / "{0:06d}-{1}.npz".format(int(camera_indices[0, 0, 0]), key),
                                arr=val.cpu().numpy()
                            )
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )
        self.train()
        return metrics_dict


    # def get_average_eval_image_metrics(
    #     self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    # ):
    #     """Iterate over all the images in the eval dataset and get the average.

    #     Args:
    #         step: current training step
    #         output_path: optional path to save rendered images to
    #         get_std: Set True if you want to return std with the mean metric.

    #     Returns:
    #         metrics_dict: dictionary of metrics
    #     """
    #     metrics_dict = super().get_average_eval_image_metrics(step, output_path, get_std)
    #     self.eval()

    #     self.train()
    #     return metrics_dict