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



class NQPPipeline(VanillaPipeline):
    """Template Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """ 

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