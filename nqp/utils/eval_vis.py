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
from nqp.utils.spacial import convert_to_transformed_space

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
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.utils import colormaps
from nerfstudio.model_components.ray_samplers import SpacedSampler

from plotly.subplots import make_subplots
import plotly.graph_objects as go

class NearFarSettings:
    def __init__(self, config, near, far):
        self.config = config
        self.near = near
        self.far = far

    def __enter__(self):
        if hasattr(self.config, "near_plane") and hasattr(self.config, "far_plane"):
            self.orig_near = self.config.near_plane
            self.orig_far = self.config.far_plane
            self.config.near_plane = self.near
            self.config.far_plane = self.far
        
    def __exit__(self, exc_type, exc_value, exc_tb):
        if hasattr(self.config, "near_plane") and hasattr(self.config, "far_plane"):
            self.config.near_plane = self.orig_near
            self.config.far_plane = self.orig_far

def save_contour_renders(pipeline, output_path, slice_count):
    method_name = pipeline.model.__class__.__name__
    geometry_analysis_type = pipeline.datamanager.train_dataparser_outputs.metadata.get("geometry_analysis_type", "unspecified")
    geometry_analysis_dimensions = pipeline.datamanager.train_dataparser_outputs.metadata.get("geometry_analysis_dimensions", {}) 
    dimensions = geometry_analysis_dimensions.get("size", [1,1,1])

    near = 0
    far = dimensions[0] * pipeline.datamanager.train_dataparser_outputs.dataparser_scale / (slice_count * 2)
    with NearFarSettings(pipeline.model.config, near, far):
        for i in range(slice_count):

            dimensions = geometry_analysis_dimensions.get("size", [1,1,1])
            camera_ray_bundle = contour_eval_ray_bundle(pipeline.datamanager.train_dataparser_outputs, i, slice_count, dimensions).to(pipeline.device)
            outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            
            rgb = outputs["rgb"] if "rgb" in outputs else outputs["rgb_fine"]
            acc = outputs["accumulation"] if "accumulation" in outputs else outputs["accumulation_fine"]
            rgb = torch.concat([rgb, acc], dim=-1)

            save_as_image(rgb, output_path / f"contour_rgb_{i:04d}.png")


def save_depth_vis(file_path, method_name, depth_gt, depth_pred, depth_diff, mask):
    # Convert torch tensors to numpy arrays and apply mask
    depth_gt_np = depth_gt.to("cpu").numpy()
    depth_pred_np = depth_pred.to("cpu").numpy()
    depth_diff_np = depth_diff.to("cpu").numpy()

    mask_np = mask.to("cpu").numpy()

    depth_gt_masked = np.ma.masked_where(mask_np == 0, depth_gt_np)
    depth_pred_masked = depth_pred_np
    depth_diff_masked = np.ma.masked_where(mask_np == 0, depth_diff_np)
    # Find min and max depth values for a unified color scale
    vmin = min(np.min(depth_gt_masked), np.min(depth_pred_masked))
    vmax = max(np.max(depth_gt_masked), np.max(depth_pred_masked))

    # Create the subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(method_name)

    # Plot the heatmaps
    cax1 = axs[0].imshow(depth_gt_masked, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0].set_title('depth_gt')
    fig.colorbar(cax1, ax=axs[0])

    cax2 = axs[1].imshow(depth_pred_masked, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1].set_title('depth_pred')
    fig.colorbar(cax2, ax=axs[1])

    eps = 1e-5
    vmin, vmax = min(depth_diff_masked.min(),-eps), max(depth_diff_masked.max(), eps)
    vcenter = 0

    cax3 = axs[2].imshow(depth_diff_masked, cmap='coolwarm', norm=mpl.colors.TwoSlopeNorm(vcenter, vmin, vmax))

    # max_abs = np.abs(depth_diff_masked).max()
    # vmin, vmax = -max_abs, max_abs
    # cax3 = axs[2].imshow(depth_diff_masked, cmap='coolwarm', vmin=vmin, vmax=vmax)

    axs[2].set_title('depth_diff')
    fig.colorbar(cax3, ax=axs[2])

    fig.savefig(file_path)
    plt.close(fig)



def save_weight_distribution_plot(filepath, outputs, dataparser_scale = 1.0, plot_cdf = False):
    if "weight_hist" in outputs:
        assert "weight_hist_edges" in outputs

        # Create figure object
        fig, ax = plt.subplots()

        # Create bar chart
        dist = torch.cumsum(outputs["weight_hist"], 0) if plot_cdf else outputs["weight_hist"]
        edges = outputs["weight_hist_edges"] / dataparser_scale
        ax.stairs(dist, edges, fill=True)

        # Set axis labels and title
        ax.set_xlabel('Depth')
        ax.set_ylabel('Weight')
        ax.set_title('Weight CDF' if plot_cdf else 'Weight Histogram')
        # ax.set_xlim(xlim)


        # Save the figure
        fig.savefig(filepath)
        plt.close(fig)

def save_geometry_surface_eval(pipeline, output_path, padded=False):
    method_name = pipeline.model.__class__.__name__
    geometry_analysis_type = pipeline.datamanager.train_dataparser_outputs.metadata.get("geometry_analysis_type", "unspecified")
    geometry_analysis_dimensions = pipeline.datamanager.train_dataparser_outputs.metadata.get("geometry_analysis_dimensions", {}) 
    sampling_depth = 0.1
    max_depth = 2*sampling_depth

    near = 0
    far = max_depth*pipeline.datamanager.train_dataparser_outputs.dataparser_scale
    with NearFarSettings(pipeline.model.config, near, far):
        if geometry_analysis_type == "plane":
            plane_dimensions = geometry_analysis_dimensions.get("size", [1,1,1])
            camera_ray_bundle = plane_eval_ray_bundle(pipeline.datamanager.train_dataparser_outputs, sampling_depth, dimensions=plane_dimensions, padded=padded).to(pipeline.device)
        elif geometry_analysis_type == "sphere":
            if padded: return {}
            sphere_radius = geometry_analysis_dimensions.get("radius", 0.5)
            camera_ray_bundle = sphere_eval_ray_bundle(pipeline.datamanager.train_dataparser_outputs, sampling_depth, radius=sphere_radius).to(pipeline.device)
        elif geometry_analysis_type == "cube":
            if padded: return {}
            dimensions = geometry_analysis_dimensions.get("size", [1,1,1])
            camera_ray_bundle = cube_eval_ray_bundle(pipeline.datamanager.train_dataparser_outputs, sampling_depth, dimensions=dimensions).to(pipeline.device)
        elif geometry_analysis_type == "line":
            if padded: return {}
            line_radius = geometry_analysis_dimensions.get("radius", 0.01)
            dimensions = geometry_analysis_dimensions.get("size", [1,1,1])
            line_metrics = line_eval(pipeline, output_path, dimensions=dimensions, line_radius=line_radius)
            return line_metrics
        else:
            raise Exception(f"unknown geometry_analysis_type: {geometry_analysis_type}")
        outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)


    rgb = outputs["rgb"] if "rgb" in outputs else outputs["rgb_fine"]
    acc = outputs["accumulation"] if "accumulation" in outputs else outputs["accumulation_fine"]
    rgb = torch.concat([rgb, acc], dim=-1)
    depth = outputs["depth"] if "depth" in outputs else outputs["depth_fine"]
    depth /= pipeline.datamanager.train_dataparser_outputs.dataparser_scale
    mask = depth <= max_depth
    # mask = torch.abs(depth - torch.mean(depth)) < 1 * torch.std(depth)
    acc = colormaps.apply_colormap(acc)
    depth_vis = torch.clone(depth)
    depth_vis[torch.logical_not(mask)] = torch.min(depth[mask]) if depth[mask].numel() > 0 else 0
    depth_vis = colormaps.apply_depth_colormap(
        depth_vis,
        accumulation=acc,
    )
    depth_vis = torch.concat([depth_vis, mask], dim=-1)
    z = (sampling_depth - depth)
    depth_diff = -z
    depth_gt = torch.full_like(depth, sampling_depth)

    if output_path is not None:
        prefix = "padded_" if padded else ""
        save_as_image(rgb, output_path / (prefix + "rgb.png"))
        save_as_image(acc, output_path / (prefix + "acc.png"))
        save_as_image(depth_vis, output_path / (prefix + "depth.png"))
        # save_weight_distribution_plot(output_path / (prefix + f"weight_hist.png"), outputs, pipeline.datamanager.train_dataparser_outputs.dataparser_scale, plot_cdf = False)
        save_weight_distribution_plot(output_path / (prefix + f"weight_cfd.png"), outputs, pipeline.datamanager.train_dataparser_outputs.dataparser_scale, plot_cdf = True)
        save_depth_vis(output_path / (prefix + "depth_plot.png"), method_name, depth_gt, depth, depth_diff, mask)

        # Convert PyTorch tensor to NumPy array
        z_numpy = z.squeeze().cpu().numpy()
        np.save(output_path / (prefix + "z.npy"), z_numpy)

        
        origins = camera_ray_bundle.origins.cpu()
        directions = camera_ray_bundle.directions.cpu()
        surface = origins + directions * outputs["depth"].cpu()
        surface = convert_from_transformed_space(surface, pipeline.datamanager.train_dataparser_outputs)

        # Create the 3D plot
        fig = plt.figure()
        fig.suptitle(method_name)
        ax = fig.add_subplot(111, projection='3d')

        if not padded:
            # Setting the TwoSlopeNorm
            eps = 1e-5
            vmin, vmax = min(z_numpy.min(), -eps), max(z_numpy.max(), eps)
            vcenter = 0
            norm = mpl.colors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
            surface = ax.plot_surface(surface[:,:,0], surface[:,:,1], surface[:,:,2], rstride=10, cstride=10, linewidth=0, edgecolor='none', antialiased=False, facecolors=plt.cm.coolwarm(norm(z_numpy)))

            mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.coolwarm)
            mappable.set_array(z_numpy)
            plt.colorbar(mappable, ax=ax)


            # surface = ax.plot_surface(surface[:,:,0], surface[:,:,1], surface[:,:,2], rstride=10, cstride=10, cmap='viridis')

            # Labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('3D Surface Plot of Z')

            # Function to update the plot at each frame
            def update(frame):
                ax.view_init(elev=20., azim=3.6*frame)
                return surface,

            # Create animation
            ani = FuncAnimation(fig, update, frames=np.arange(0, 100), blit=False, repeat=False)

            # To save the animation
            ani.save(output_path / '3D_rotation.gif', writer='imagemagick')
            plt.close(fig)

    surface_diff = z
    metrics_dict = {}
    metrics_dict["max_surface_diff"] = float(torch.max(surface_diff).item())
    metrics_dict["min_surface_diff"] = float(torch.min(surface_diff).item())
    metrics_dict["std_surface_diff"] = float(torch.std(surface_diff).item())
    metrics_dict["mean_surface_diff"] = float(torch.mean(surface_diff).item())
    return metrics_dict

def eval_set_renders_and_metrics(pipeline, output_path, get_std):
    depth_filenames = pipeline.datamanager.eval_dataset.metadata["depth_filenames"]
    method_name = pipeline.model.__class__.__name__

    metrics_dict_list = []
    for camera_ray_bundle, batch in pipeline.datamanager.fixed_indices_eval_dataloader:
        image_idx = batch["image_idx"]
        depth_filepath = depth_filenames[image_idx]
        depth_gt = read_depth_map(str(depth_filepath), pipeline.device)
        mask = depth_gt <= 2 / pipeline.datamanager.train_dataparser_outputs.dataparser_scale
        depth_gt[torch.logical_not(mask)] = torch.min(depth_gt[mask])
        outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        metrics_dict, images_dict = pipeline.model.get_image_metrics_and_images(outputs, batch)
        rgb_pred = outputs["rgb"] if "rgb" in outputs else outputs["rgb_fine"]
        acc = outputs["accumulation"] if "accumulation" in outputs else outputs["accumulation_fine"]

        rgb_pred_loss, rgb_gt_loss = pipeline.model.renderer_rgb.blend_background_for_loss_computation(
            pred_image=rgb_pred,
            pred_accumulation=acc,
            gt_image=batch["image"],
        )
        rgb_pred_loss, rgb_gt_loss = rgb_pred_loss.cpu(), rgb_gt_loss.cpu()
        rgb_pred = rgb_pred.cpu()
        rgb_pred = torch.concat([rgb_pred, torch.ones((rgb_pred.shape[0], rgb_pred.shape[1], 1))], dim=-1)
        # rgb_pred = torch.concat([rgb_pred, 1 - acc.cpu()], dim=-1)
        rgb_gt = batch["image"].cpu()
        
        rgb_compare = torch.concat([rgb_gt, rgb_pred], dim=1)
        rgb_compare_loss = torch.concat([rgb_gt_loss, rgb_pred_loss], dim=1)
        acc = acc
        acc_vis = colormaps.apply_colormap(acc)

        depth_pred = outputs["depth"] if "depth" in outputs else outputs["depth_fine"]
        depth_pred /= pipeline.datamanager.train_dataparser_outputs.dataparser_scale

        depth_pred_vis = colormaps.apply_depth_colormap(
            depth_pred,
            accumulation=acc,
        )
        depth_gt_vis = colormaps.apply_depth_colormap(
            depth_gt,
        )
        depth_gt_vis = torch.concat([depth_gt_vis, mask], dim=-1)
        depth_diff = depth_pred - depth_gt
        depth_diff[depth_gt > 1000] = torch.min(depth_diff[mask])
        depth_diff_vis = colormaps.apply_depth_colormap(
            depth_diff,
        )
        depth_diff_vis = torch.concat([depth_diff_vis, mask], dim=-1)

        if output_path is not None:
            # save_as_image(rgb_pred, output_path / f"rgb_pred_{image_idx:04d}.png")
            # save_as_image(rgb_gt, output_path / f"rgb_gt_{image_idx:04d}.png")
            # save_as_image(rgb_pred_loss, output_path / f"rgb_pred_loss_{image_idx:04d}.png")
            # save_as_image(rgb_gt_loss, output_path / f"rgb_gt_loss_{image_idx:04d}.png")
            save_as_image(rgb_compare, output_path / f"rgb_compare_{image_idx:04d}.png")
            save_as_image(rgb_compare_loss, output_path / f"rgb_compare_loss_{image_idx:04d}.png")
            save_as_image(acc_vis, output_path / f"acc_{image_idx:04d}.png")
            # save_as_image(depth_pred_vis, output_path / f"depth_pred_{image_idx:04d}.png")
            # save_as_image(depth_gt_vis, output_path / f"depth_gt_{image_idx:04d}.png")
            # save_as_image(depth_diff_vis, output_path / f"depth_diff_{image_idx:04d}.png")
            save_depth_vis(output_path / f"depth_plot_{image_idx:04d}.png", method_name, depth_gt, depth_pred, depth_diff, mask)
            # save_weight_distribution_plot(output_path / f"weight_hist_{image_idx:04d}.png", outputs, pipeline.datamanager.train_dataparser_outputs.dataparser_scale, plot_cdf = False)

        metrics_dict["depth_silog"], metrics_dict["depth_log10"], metrics_dict["depth_abs_rel"], metrics_dict["depth_sq_rel"], metrics_dict["depth_rms"], metrics_dict["depth_log_rms"], metrics_dict["depth_d1"], metrics_dict["depth_d2"], metrics_dict["depth_d3"] = compute_errors(
                depth_gt[mask].to("cpu").numpy(), depth_pred[mask].to("cpu").numpy())
        metrics_dict_list.append(metrics_dict)


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
    
    return metrics_dict
    
    # metrics_dict["depth_silog"] = float(depth_silog.mean())
    # metrics_dict["depth_log10"] = float(depth_log10.mean())
    # metrics_dict["depth_abs_rel"] = float(depth_abs_rel.mean())
    # metrics_dict["depth_sq_rel"] = float(depth_sq_rel.mean())
    # metrics_dict["depth_rms"] = float(depth_rms.mean())
    # metrics_dict["depth_log_rms"] = float(depth_log_rms.mean())
    # metrics_dict["depth_d1"] = float(depth_d1.mean())
    # metrics_dict["depth_d2"] = float(depth_d2.mean())
    # metrics_dict["depth_d3"] = float(depth_d3.mean())

def line_eval(pipeline, output_path, dimensions=(1.0,1.0,1.0), line_radius=0.01, n = 100):
    dataparser_outputs = pipeline.datamanager.train_dataparser_outputs
    padding = 950
    pad_scale = (n + 2*padding) / n
    # n = int(n*pad_scale)

    dataparser_scale = dataparser_outputs.dataparser_scale
    dimensions = np.array(dimensions)
    y = torch.linspace(-0.1, 0.1, n+1)
    y = (y[1:] + y[:-1]) / 2
    z = torch.full_like(y, 0.1)
    near = 0
    far = dataparser_outputs.dataparser_scale * 0.2

    # Initialize the UniformSampler
    sampler = SpacedSampler(lambda x : x, lambda x : x, n, train_stratified=False)
    xn = 100
    line_pts = []
    line_solid_densities = []
    line_air_densities = []
    line_solid_alpha = []
    line_air_alpha = []
    for i in range(xn):
        x = ((i+0.5) / xn - 0.5)*dimensions[0]
        x = torch.full_like(y, x)
        origins=torch.stack([x, y, z], dim=-1)
        directions=torch.zeros_like(origins)
        directions[..., 2] = -1
        origins_trans = convert_to_transformed_space(origins, dataparser_outputs)
        directions_trans = convert_to_transformed_space(directions, dataparser_outputs, is_direction=True)
        ray_bundle = RayBundle(
            origins=origins_trans,
            directions=directions_trans,
            pixel_area=torch.full([n, 1], dimensions[1]*dimensions[2]/(n*n)),
            nears=torch.full([n, 1], near),
            fars=torch.full([n, 1], far),
            camera_indices=torch.zeros([n, 1]),
        )


        # Generate samples
        samples = sampler.generate_ray_samples(ray_bundle).to("cuda")

        densities = pipeline.model.get_densities(samples)
        delta = (samples.frustums.ends - samples.frustums.starts).cpu().squeeze().detach().numpy()
        # log_densities = torch.log1p(densities)
        densities = densities.cpu().squeeze().detach().numpy()
        log_densities = np.log1p(densities)
        alpha = 1 - np.exp(-densities*delta)


        depth = ((samples.frustums.starts + samples.frustums.ends) / 2).cpu() / dataparser_scale
        pts = origins.view(n, 1, 3) + directions.view(n, 1, 3) * depth

        lcs_percentage = np.pi * 0.05**2
        lcs_percentile = 100-100*lcs_percentage
        threshold = 3 # np.percentile(log_densities, lcs_percentile)
        solid = log_densities >= threshold
        gt_solid = np.linalg.norm(pts[...,1:], axis=-1) <= line_radius

        solid_center = np.mean(pts[solid, :].numpy(), axis=0)
        solid_density = np.mean(densities[gt_solid])
        air_density = np.mean(densities[np.logical_not(gt_solid)])
        solid_alpha = np.mean(alpha[gt_solid])
        air_alpha = np.mean(alpha[np.logical_not(gt_solid)])
        print("solid_alpha", solid_alpha, line_radius)
        print("air_alpha", air_alpha, delta[0,0])
        line_pts.append(solid_center)
        line_solid_densities.append(solid_density)
        line_air_densities.append(air_density)
        line_solid_alpha.append(solid_alpha)
        line_air_alpha.append(air_alpha)

        if output_path is not None:
            # Create a subplot layout
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=("Vertical Slice of Densities", "Solid Region"),
                                horizontal_spacing=0.15,
            )

            # Add the first heatmap to subplot (1, 1)
            trace1 = go.Heatmap(
                z=log_densities.transpose(),
                x=pts[:, 0, 1],
                y=pts[0, :, 2],
                colorscale='Viridis',
                colorbar=dict(
                    x=0.425,  # Position of colorbar for first subplot
                    title=f"Log Density"
                )
            )
            fig.add_trace(trace1, row=1, col=1)

            # Add the second heatmap to subplot (1, 2)
            trace2 = go.Heatmap(
                z=np.array(solid, np.float32).transpose(),
                x=pts[:, 0, 1],
                y=pts[0, :, 2],
                colorscale='Viridis',
                colorbar=dict(
                    x=1.0,  # Position of colorbar for second subplot
                    title="Solid Region"
                )
            )
            fig.add_trace(trace2, row=1, col=2)

            # Add a scatter plot on top of the second heatmap
            trace3 = go.Scatter(
                x=[solid_center[1]],
                y=[solid_center[2]],
                mode='markers',
                marker=dict(
                    color='red',
                    size=3
                )
            )
            fig.add_trace(trace3, row=1, col=2)

            # Update layout
            fig.update_layout(
                title=f"X = {pts[0,0,0]:.4f}",
                xaxis_title="Y",
                yaxis_title="Z",
                xaxis2_title="Y",
                yaxis2_title="Z",
                width=800,
                height=400,
                hovermode=False,
                showlegend=False,
                margin=dict(l=20, r=20, t=50, b=20)
            )

            # Show figure
            # fig.show()
            fig.write_image(output_path / f"line_contour_{i:04d}.jpeg")
    
    line_pts = np.array(line_pts)
    line_distance = np.linalg.norm(line_pts[:,1:], axis=-1)

    if output_path is not None:
        np.save(output_path / "line_pts.npy", line_pts)
        fig = go.Figure(data=[go.Scatter3d(
            x=line_pts[:,0],
            y=line_pts[:,0],
            z=line_pts[:,0],
            mode='markers',
            marker=dict(
                size=3,
                color='red',  # You can set color based on any coordinate or custom array
                colorscale='Viridis',
            )
        )])

        fig.update_layout(
            title="Line Points",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        fig.write_image(output_path / f"line_plot.jpeg")
    
    def summarize_metrics(metric_name, array):
        return {
            f"{metric_name}_max": float(np.max(array)),
            f"{metric_name}_min": float(np.min(array)),
            f"{metric_name}_mean": float(np.mean(array)),
            f"{metric_name}_std": float(np.std(array)),
        }
    return {
        **summarize_metrics("line_deviation", line_distance),
        **summarize_metrics("line_solid_density", line_solid_densities),
        **summarize_metrics("line_air_density", line_air_densities),
        **summarize_metrics("line_solid_alpha", line_solid_alpha),
        **summarize_metrics("line_air_alpha", line_air_alpha),
    }
