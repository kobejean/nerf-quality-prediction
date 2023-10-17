# from nerfstudio.engine.trainer import TrainerConfig
# from nerfstudio.configs.method_configs import method_configs
# from nerfstudio.configs.dataparser_configs import dataparsers as dataparser_configs
# from nerfstudio.utils.eval_utils import eval_load_checkpoint
from nerfstudio.cameras.rays import RayBundle
from nqp.utils.spacial import convert_to_transformed_space

from PIL import Image
from pathlib import Path
import torch
from torch.nn.functional import normalize
import numpy as np


def plane_eval_ray_bundle(dataparser_outputs, sampling_depth, dimensions=(1.0,1.0,1.0), padded=False, n = 1000):
    dataparser_scale = dataparser_outputs.dataparser_scale
    if padded:
        x_scale = 1 + 2*sampling_depth / dimensions[0]
        y_scale = 1 + 2*sampling_depth / dimensions[1]
        x = torch.linspace(-0.5*dimensions[0]*x_scale, 0.5*dimensions[0]*x_scale, int(n*x_scale)+1)
        y = torch.linspace(-0.5*dimensions[1]*y_scale, 0.5*dimensions[1]*y_scale, int(n*y_scale)+1)
    else:
        x = torch.linspace(-0.5*dimensions[0], 0.5*dimensions[0], n+1)
        y = torch.linspace(-0.5*dimensions[1], 0.5*dimensions[1], n+1)
    grid_x, grid_y = torch.meshgrid(x, y)
    z = torch.full_like(grid_x, sampling_depth)
    origins = torch.stack([grid_x, grid_y, z], dim=-1)
    directions = torch.zeros_like(origins)
    directions[:, :, 2] = -1.0
    origins = convert_to_transformed_space(origins, dataparser_outputs)
    directions = convert_to_transformed_space(directions, dataparser_outputs, is_direction=True)
    shape_1chan = (*origins.shape[:-1], 1)
    pixel_area = torch.ones(shape_1chan)
    nears = torch.zeros(shape_1chan)
    fars = torch.full(shape_1chan, 2 * sampling_depth * dataparser_scale)
    camera_indices = torch.zeros(shape_1chan)

    ray_bundle = RayBundle(
        origins=origins, directions=directions, pixel_area=pixel_area,
        camera_indices=camera_indices,
        nears=nears,
        fars=fars, 
    )
    return ray_bundle

def cube_eval_ray_bundle(dataparser_outputs, sampling_depth, dimensions=(1.,1.,1.), n = 1001):
    dataparser_scale = dataparser_outputs.dataparser_scale
    linspace = torch.linspace(-0.5, 0.5, n)
    grid_row, grid_col = torch.meshgrid(linspace, linspace)
    max_dim = max(dimensions)
    max_depth = 2*sampling_depth# + max_dim
    # max_depth *= 2

    origins = []
    directions = []
    pixel_area = []
    nears = []
    fars = []
    for axis in range(3):
        # positive face
        const = torch.full([n, n], 0.5)
        stack = [grid_row, grid_col]
        stack.insert(axis, const)
        p_origins = torch.stack(stack, dim=-1)
        p_origins = p_origins * torch.tensor(dimensions).view(1, 1, 3)
        p_origins[..., axis] += sampling_depth
        p_directions = torch.zeros_like(p_origins)
        p_directions[..., axis] = -1.0
        p_origins = convert_to_transformed_space(p_origins, dataparser_outputs)
        p_directions = convert_to_transformed_space(p_directions, dataparser_outputs, is_direction=True)
        # negative faces
        n_origins = torch.flip(-p_origins, dims=[0])
        n_directions = torch.flip(-p_directions, dims=[0])
        # axis
        axis_origins = torch.concat([p_origins, n_origins], dim=0)
        axis_directions = torch.concat([p_directions, n_directions], dim=0)
        axis_nears = torch.zeros((2*n, n, 1))
        axis_fars = torch.full((2*n, n, 1), max_depth * dataparser_scale)

        origins.append(axis_origins)
        directions.append(axis_directions)
        nears.append(axis_nears)
        fars.append(axis_fars)

    origins = torch.concat(origins, dim=1)
    directions = torch.concat(directions, dim=1)
    pixel_area = torch.ones((2*n, 3*n, 1))
    # pixel_area = (dataparser_scale ** 2) * torch.ones((2*n, 3*n, 1)) / (6 * n ** 2)
    nears = torch.concat(nears, dim=1)
    fars = torch.concat(fars, dim=1)
    camera_indices = torch.zeros_like(nears)

    ray_bundle = RayBundle(
        origins=origins, directions=directions, pixel_area=pixel_area,
        camera_indices=camera_indices,
        nears=nears,
        fars=fars, 
    )
    return ray_bundle


def sphere_eval_ray_bundle(dataparser_outputs, sampling_depth, radius=0.5, n = 1001, m = 2001):
    dataparser_scale = dataparser_outputs.dataparser_scale
    phi = torch.linspace(0, np.pi, n)
    theta = torch.linspace(-np.pi, np.pi, m)
    
    grid_phi, grid_theta = torch.meshgrid(phi, theta)
    r = radius + sampling_depth
    x = r * torch.cos(grid_theta) * torch.sin(grid_phi)
    y = r * torch.sin(grid_theta) * torch.sin(grid_phi)
    z = r * torch.cos(grid_phi)
    origins = torch.stack([x, y, z], dim=-1)
    origins = convert_to_transformed_space(origins, dataparser_outputs)
    directions = -normalize(origins, dim=-1)
    pixel_area = torch.ones((n, m, 1)) # (dataparser_scale ** 2) * torch.ones((n, m, 1)) / (n * m)
    nears = torch.zeros((n, m, 1))
    fars = torch.ones((n, m, 1)) * 2 * sampling_depth * dataparser_scale
    camera_indices = torch.zeros((n, m, 1))

    ray_bundle = RayBundle(
        origins=origins, directions=directions, pixel_area=pixel_area,
        camera_indices=camera_indices,
        nears=nears,
        fars=fars, 
    )
    return ray_bundle


def contour_eval_ray_bundle(dataparser_outputs, i, slice_count=10, dimensions=(1.0,1.0,1.0), n = 1001):
    dataparser_scale = dataparser_outputs.dataparser_scale
    x = torch.linspace(-0.5*dimensions[0], 0.5*dimensions[0], n)
    y = torch.linspace(-0.5*dimensions[1], 0.5*dimensions[1], n)
    sampling_depth = dimensions[2] / (slice_count)
    z = sampling_depth * ((slice_count)/2-i)
    grid_x, grid_y = torch.meshgrid(x, y)
    origins = torch.stack([grid_x, grid_y, z * torch.ones([n, n])], dim=-1)
    origins = convert_to_transformed_space(origins, dataparser_outputs)
    directions = torch.zeros_like(origins)
    directions[:, :, 2] = -1.0
    pixel_area = torch.ones((n, n, 1)) # (dataparser_scale ** 2) * torch.ones((n, n, 1)) / (n ** 2)
    nears = torch.zeros((n, n, 1))
    fars = torch.ones((n, n, 1)) * sampling_depth * dataparser_scale
    camera_indices = torch.zeros((n, n, 1))

    ray_bundle = RayBundle(
        origins=origins, directions=directions, pixel_area=pixel_area,
        camera_indices=camera_indices,
        nears=nears,
        fars=fars, 
    )
    return ray_bundle


