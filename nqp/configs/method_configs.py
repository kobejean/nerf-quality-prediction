from __future__ import annotations
import math
import copy 
# from nqp.models.depth_nerfacto import DepthNerfactoModelConfig
# from nqp.models.generfacto import GenerfactoModelConfig
from nqp.models.instant_ngp import NQPInstantNGPModel
from nqp.models.mipnerf import NQPMipNerfModel
from nqp.models.nerfacto import NQPNerfactoModel
# from nqp.models.neus import NeuSModelConfig
# from nqp.models.neus_facto import NeuSFactoModelConfig
# from nqp.models.semantic_nerfw import SemanticNerfWModelConfig
from nqp.models.tensorf import NQPTensoRFModel
from nqp.models.vanilla_nerf import NQPNeRFModel

from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.configs.method_configs import method_configs as ns_method_configs, descriptions as ns_descriptions

from nqp.data.dataparsers.nqp_dataparser import NQPDataParserConfig
from nqp.data.dataparsers.nqp_blender_dataparser import NQPBlenderDataParserConfig
from nqp.pipelines.nqp_pipeline import NQPPipeline

def hasattr_nested(obj, attribute_path):
    attributes = attribute_path.split('.')
    current_obj = obj

    for attr in attributes:
        if hasattr(current_obj, attr):
            current_obj = getattr(current_obj, attr)
        else:
            return False

    return True

def hasattrs(obj, attr_names):
    for attr_name in attr_names:
        if not hasattr_nested(obj, attr_name):
            return False
    return True

def setattr_if_exists(obj, attr_name, new_value):
    if hasattr(obj, attr_name):
        setattr(obj, attr_name, new_value)
    else:
        print("No attribute:", attr_name)

def make_nqp_config(method, nqp_method, NQPModel):
    # scene_scale = 1.34
    # near_plane, far_plane = 1/3, 7/3 * scene_scale
    nqp_config = copy.deepcopy(ns_method_configs[method])
    nqp_config.method_name= nqp_method
    nqp_config.project_name="nerf-quality-prediction"
    # nqp_config.steps_per_eval_all_images=5000
    nqp_config.pipeline._target=NQPPipeline
    nqp_config.pipeline.datamanager.dataparser=NQPBlenderDataParserConfig()
    nqp_config.pipeline.model._target=NQPModel

    # # nqp_config.pipeline.model.enable_collider = False
    # if nqp_config.pipeline.model.collider_params is not None:
    #     # TODO: investigate orig space or transformed space?
    #     nqp_config.pipeline.model.collider_params["near_plane"] = near_plane
    #     nqp_config.pipeline.model.collider_params["far_plane"] = far_plane
    # setattr_if_exists(nqp_config.pipeline.model, "near_plane", near_plane)
    # setattr_if_exists(nqp_config.pipeline.model, "far_plane", far_plane)
    # setattr_if_exists(nqp_config.pipeline.model, "disable_scene_contraction", True)
    # setattr_if_exists(nqp_config.pipeline.model, "use_gradient_scaling", True)
    # setattr_if_exists(nqp_config.pipeline.model, "background_color", "random")

    # nqp_config.experiment_name=f"np{near_plane:04f}_fp{far_plane:04f}_ss{scene_scale:04f}"

    return nqp_config


nqp_nerfacto_config = make_nqp_config("nerfacto", "nqp-nerfacto", NQPNerfactoModel)
nqp_nerfacto = MethodSpecification(
    config=nqp_nerfacto_config,
    description=ns_descriptions["nerfacto"],
)


nqp_instant_ngp_config = make_nqp_config("instant-ngp", "nqp-instant-ngp", NQPInstantNGPModel)
nqp_instant_ngp = MethodSpecification(
    config=nqp_instant_ngp_config,
    description=ns_descriptions["instant-ngp"],
)

nqp_mipnerf_config = make_nqp_config("mipnerf", "nqp-mipnerf", NQPMipNerfModel)
nqp_mipnerf_config.steps_per_eval_all_images=1000000  # set to a very large model so we don't eval with all images
nqp_mipnerf = MethodSpecification(
    config=nqp_mipnerf_config,
    description=ns_descriptions["mipnerf"],
)

nqp_vanilla_nerf_config = make_nqp_config("vanilla-nerf", "nqp-vanilla-nerf", NQPNeRFModel)
nqp_vanilla_nerf_config.steps_per_eval_all_images=1000000  # set to a very large model so we don't eval with all images
nqp_vanilla_nerf = MethodSpecification(
    config=nqp_vanilla_nerf_config,
    description=ns_descriptions["vanilla-nerf"],
)

# TensoRF

nqp_tensorf_config = make_nqp_config("tensorf", "nqp-tensorf", NQPTensoRFModel)
nqp_tensorf = MethodSpecification(
    config=nqp_tensorf_config,
    description=ns_descriptions["tensorf"]+ " nqp-tensorf",
)

nqp_tensorf_half_res_config = make_nqp_config("tensorf", "nqp-tensorf-half-res", NQPTensoRFModel)
nqp_tensorf_half_res_config.pipeline.model.final_resolution = 150 # default 300
nqp_tensorf_half_res_config.pipeline.model.init_resolution = 64 # default 128
nqp_tensorf_half_res = MethodSpecification(
    config=nqp_tensorf_half_res_config,
    description=ns_descriptions["tensorf"]+ " nqp-tensorf-half-res",
)

nqp_tensorf_half_samp_config = make_nqp_config("tensorf", "nqp-tensorf-half-samp", NQPTensoRFModel)
nqp_tensorf_half_samp_config.pipeline.model.num_samples = 25 # default 50
nqp_tensorf_half_samp_config.pipeline.model.num_uniform_samples = 100 # default 200
nqp_tensorf_half_samp = MethodSpecification(
    config=nqp_tensorf_half_samp_config,
    description=ns_descriptions["tensorf"] + " nqp-tensorf-half-samp",
)

nqp_tensorf_half_color_config = make_nqp_config("tensorf", "nqp-tensorf-half-color", NQPTensoRFModel)
nqp_tensorf_half_color_config.pipeline.model.num_color_components = 27 # default 48
nqp_tensorf_half_color = MethodSpecification(
    config=nqp_tensorf_half_color_config,
    description=ns_descriptions["tensorf"] + " nqp-tensorf-half-color",
)

nqp_tensorf_half_den_config = make_nqp_config("tensorf", "nqp-tensorf-half-den", NQPTensoRFModel)
nqp_tensorf_half_den_config.pipeline.model.num_den_components = 8 # default 16
nqp_tensorf_half_den = MethodSpecification(
    config=nqp_tensorf_half_den_config,
    description=ns_descriptions["tensorf"] + " nqp-tensorf-half-den",
)

nqp_tensorf_half_rgb_config = make_nqp_config("tensorf", "nqp-tensorf-half-den", NQPTensoRFModel)
nqp_tensorf_half_rgb_config.pipeline.model.num_den_components = 8 # default 16
nqp_tensorf_half_den = MethodSpecification(
    config=nqp_tensorf_half_den_config,
    description=ns_descriptions["tensorf"] + " nqp-tensorf-half-den",
)





#3/4
nqp_tensorf_third_res_config = make_nqp_config("tensorf", "nqp-tensorf-third-res", NQPTensoRFModel)
nqp_tensorf_third_res_config.pipeline.model.final_resolution = 225 # default 300
nqp_tensorf_third_res_config.pipeline.model.init_resolution = 96 # default 128
nqp_tensorf_third_res = MethodSpecification(
    config=nqp_tensorf_third_res_config,
    description=ns_descriptions["tensorf"]+ " nqp-tensorf-third-res",
)

nqp_tensorf_third_samp_config = make_nqp_config("tensorf", "nqp-tensorf-third-samp", NQPTensoRFModel)
nqp_tensorf_third_samp_config.pipeline.model.num_samples = 38 # default 50
nqp_tensorf_third_samp_config.pipeline.model.num_uniform_samples = 150 # default 200
nqp_tensorf_third_samp = MethodSpecification(
    config=nqp_tensorf_third_samp_config,
    description=ns_descriptions["tensorf"] + " nqp-tensorf-third-samp",
)

nqp_tensorf_third_color_config = make_nqp_config("tensorf", "nqp-tensorf-third-color", NQPTensoRFModel)
nqp_tensorf_third_color_config.pipeline.model.num_color_components = 16 # default 48
nqp_tensorf_third_color = MethodSpecification(
    config=nqp_tensorf_third_color_config,
    description=ns_descriptions["tensorf"] + " nqp-tensorf-third-color",
)

nqp_tensorf_third_den_config = make_nqp_config("tensorf", "nqp-tensorf-third-den", NQPTensoRFModel)
nqp_tensorf_third_den_config.pipeline.model.num_den_components = 12 # default 16
nqp_tensorf_third_den = MethodSpecification(
    config=nqp_tensorf_third_den_config,
    description=ns_descriptions["tensorf"] + " nqp-tensorf-third-den",
)


#1\4
nqp_tensorf_quarter_res_config = make_nqp_config("tensorf", "nqp-tensorf-quarter-res", NQPTensoRFModel)
nqp_tensorf_quarter_res_config.pipeline.model.final_resolution = 75 # default 300
nqp_tensorf_quarter_res_config.pipeline.model.init_resolution = 32 # default 128
nqp_tensorf_quarter_res = MethodSpecification(
    config=nqp_tensorf_quarter_res_config,
    description=ns_descriptions["tensorf"]+ " nqp-quarter-quarter-res",
)

nqp_tensorf_quarter_samp_config = make_nqp_config("tensorf", "nqp-tensorf-quarter-samp", NQPTensoRFModel)
nqp_tensorf_quarter_samp_config.pipeline.model.num_samples = 12.5 # default 50
nqp_tensorf_quarter_samp_config.pipeline.model.num_uniform_samples = 50 # default 200
nqp_tensorf_quarter_samp = MethodSpecification(
    config=nqp_tensorf_quarter_samp_config,
    description=ns_descriptions["tensorf"] + " nqp-tensorf-quarter-samp",
)

nqp_tensorf_quarter_color_config = make_nqp_config("tensorf", "nqp-tensorf-quarter-color", NQPTensoRFModel)
nqp_tensorf_quarter_color_config.pipeline.model.num_color_components = 9 # default 48
nqp_tensorf_quarter_color = MethodSpecification(
    config=nqp_tensorf_quarter_color_config,
    description=ns_descriptions["tensorf"] + " nqp-tensorf-quarter-color",
)


# Instant NGP 


nqp_instant_ngp_half_hashmap_reso_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-half-hashmap-reso", NQPInstantNGPModel)
nqp_instant_ngp_half_hashmap_reso_config.pipeline.model.max_res = 1024 # default 1024
nqp_instant_ngp_half_hashmap_reso = MethodSpecification(
    config=nqp_instant_ngp_half_hashmap_reso_config ,
    description=ns_descriptions["instant-ngp"] + " nqp-instant-ngp-half-hashmap-reso",
)

nqp_instant_ngp_half_hashmap_log_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-half-hashmap-log", NQPInstantNGPModel)
nqp_instant_ngp_half_hashmap_log_config.pipeline.model.log2_hashmap_size = 10 # default 19
nqp_instant_ngp_half_hashmap_log = MethodSpecification(
    config=nqp_instant_ngp_half_hashmap_log_config ,
    description=ns_descriptions["instant-ngp"] + " nqp-instant-ngp-half-hashmap-log",
)

#3/4

nqp_instant_ngp_grid_levels3_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-grid-levels3", NQPInstantNGPModel)
nqp_instant_ngp_grid_levels3_config.pipeline.model.grid_levels = 3 # default 4
nqp_instant_ngp_grid_levels3 = MethodSpecification(
    config=nqp_instant_ngp_grid_levels3_config ,
    description=ns_descriptions["instant-ngp"] + " nqp-instant-ngp-grid-levels3",
)

nqp_instant_ngp_grid_levels5_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-grid-levels5", NQPInstantNGPModel)
nqp_instant_ngp_grid_levels5_config.pipeline.model.grid_levels = 5 # default 4
nqp_instant_ngp_grid_levels5 = MethodSpecification(
    config=nqp_instant_ngp_grid_levels5_config ,
    description=ns_descriptions["instant-ngp"] + " nqp-instant-ngp-grid-levels5",
)

nqp_instant_ngp_grid_levels4_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-grid-levels4", NQPInstantNGPModel)
nqp_instant_ngp_grid_levels4_config.pipeline.model.grid_levels = 4 # default 4
nqp_instant_ngp_grid_levels4 = MethodSpecification(
    config=nqp_instant_ngp_grid_levels4_config ,
    description=ns_descriptions["instant-ngp"] + " nqp-instant-ngp-grid-levels4",
)

nqp_instant_ngp_grid_levels4_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-grid-levels4", NQPInstantNGPModel)
nqp_instant_ngp_grid_levels4_config.pipeline.model.grid_levels = 4 # default 4
nqp_instant_ngp_grid_levels4 = MethodSpecification(
    config=nqp_instant_ngp_grid_levels4_config ,
    description=ns_descriptions["instant-ngp"] + " nqp-instant-ngp-grid-levels4",
)

nqp_instant_ngp_half_max_res_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-half-max-res", NQPInstantNGPModel)
nqp_instant_ngp_half_max_res_config.pipeline.model.max_res = 1024 # default 2048
nqp_instant_ngp_half_max_res = MethodSpecification(
    config=nqp_instant_ngp_half_max_res_config ,
    description=ns_descriptions["instant-ngp"] + " nqp-instant-ngp-half-max-res",
)

nqp_instant_ngp_half_grid_reso_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-half-grid-reso", NQPInstantNGPModel)
nqp_instant_ngp_half_grid_reso_config.pipeline.model.grid_resolution = 64 # default 128
nqp_instant_ngp_half_grid_reso = MethodSpecification(
    config=nqp_instant_ngp_half_grid_reso_config,
    description=ns_descriptions["instant-ngp"] + " nqp-instant-ngp-half-grid-reso",
)

nqp_instant_ngp_double_max_res_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-double-max-res", NQPInstantNGPModel)
nqp_instant_ngp_double_max_res_config.pipeline.model.max_res = 3000 # default 2048
nqp_instant_ngp_double_max_res = MethodSpecification(
    config=nqp_instant_ngp_double_max_res_config ,
    description=ns_descriptions["instant-ngp"] + " nqp-instant-ngp-double-max-res",
)

#nqp_instant_ngp_double_grid_res_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-double-grid-res", NQPInstantNGPModel)
#nqp_instant_ngp_double_grid_res_config.pipeline.model.grid_resolution = 256 # default 128
#nqp_instant_ngp_double_grid_res = MethodSpecification(
#    config=nqp_instant_ngp_double_grid_res_config ,
#    description=ns_descriptions["instant-ngp"] + " nqp-instant-ngp-double-grid-res",
#)

nqp_instant_ngp_default_max_res_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-default-max-res", NQPInstantNGPModel)
nqp_instant_ngp_default_max_res_config.pipeline.model.max_res = 2048 # default 2048
nqp_instant_ngp_default_max_res = MethodSpecification(
    config=nqp_instant_ngp_default_max_res_config ,
    description=ns_descriptions["instant-ngp"] + " nqp-instant-ngp-default-max-res",
)

nqp_instant_ngp_synthetic_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-synthetic", NQPInstantNGPModel)
nqp_instant_ngp_synthetic_config.pipeline.model.disable_scene_contraction = True
nqp_instant_ngp_synthetic_config.pipeline.model.near_plane = 2
nqp_instant_ngp_synthetic_config.pipeline.model.far_plane = 6
nqp_instant_ngp_synthetic_config.pipeline.model.grid_levels = 16
nqp_instant_ngp_synthetic = MethodSpecification(
    config=nqp_instant_ngp_synthetic_config,
    description=ns_descriptions["instant-ngp"],
)



nqp_instant_ngp_synthetic_half_grid_res_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-synthetic-half-grid-res", NQPInstantNGPModel)
nqp_instant_ngp_synthetic_half_grid_res_config.pipeline.model.disable_scene_contraction = True
nqp_instant_ngp_synthetic_half_grid_res_config.pipeline.model.near_plane = 2
nqp_instant_ngp_synthetic_half_grid_res_config.pipeline.model.far_plane = 6
nqp_instant_ngp_synthetic_half_grid_res_config.pipeline.model.grid_resolution = 64 # default 128
nqp_instant_ngp_synthetic_half_grid_res = MethodSpecification(
    config=nqp_instant_ngp_synthetic_half_grid_res_config,
    description=ns_descriptions["instant-ngp"],
)

#hashmap log
nqp_instant_ngp_synthetic_hashmap_log_14_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-synthetic-hashmap-log-14", NQPInstantNGPModel)
nqp_instant_ngp_synthetic_hashmap_log_14_config.pipeline.model.disable_scene_contraction = True
nqp_instant_ngp_synthetic_hashmap_log_14_config.pipeline.model.near_plane = 2
nqp_instant_ngp_synthetic_hashmap_log_14_config.pipeline.model.far_plane = 6
nqp_instant_ngp_synthetic_hashmap_log_14_config.pipeline.model.log2_hashmap_size = 14 # default 19
nqp_instant_ngp_synthetic_hashmap_log_14_config.pipeline.model.grid_levels = 4
nqp_instant_ngp_synthetic_hashmap_log_14 = MethodSpecification(
    config=nqp_instant_ngp_synthetic_hashmap_log_14_config,
    description=ns_descriptions["instant-ngp"],
)

nqp_instant_ngp_synthetic_hashmap_log_15_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-synthetic-hashmap-log-15", NQPInstantNGPModel)
nqp_instant_ngp_synthetic_hashmap_log_15_config.pipeline.model.disable_scene_contraction = True
nqp_instant_ngp_synthetic_hashmap_log_15_config.pipeline.model.near_plane = 2
nqp_instant_ngp_synthetic_hashmap_log_15_config.pipeline.model.far_plane = 6
nqp_instant_ngp_synthetic_hashmap_log_15_config.pipeline.model.log2_hashmap_size = 15 # default 19
nqp_instant_ngp_synthetic_hashmap_log_15_config.pipeline.model.grid_levels = 4 #default 4? 
nqp_instant_ngp_synthetic_hashmap_log_15 = MethodSpecification(
    config=nqp_instant_ngp_synthetic_hashmap_log_15_config,
    description=ns_descriptions["instant-ngp"],
)

nqp_instant_ngp_synthetic_hashmap_log_16_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-synthetic-hashmap-log-16", NQPInstantNGPModel)
nqp_instant_ngp_synthetic_hashmap_log_16_config.pipeline.model.disable_scene_contraction = True
nqp_instant_ngp_synthetic_hashmap_log_16_config.pipeline.model.near_plane = 2
nqp_instant_ngp_synthetic_hashmap_log_16_config.pipeline.model.far_plane = 6
nqp_instant_ngp_synthetic_hashmap_log_16_config.pipeline.model.log2_hashmap_size = 16 # default 19
nqp_instant_ngp_synthetic_hashmap_log_16_config.pipeline.model.grid_levels = 4 #default 4? 
nqp_instant_ngp_synthetic_hashmap_log_16 = MethodSpecification(
    config=nqp_instant_ngp_synthetic_hashmap_log_16_config,
    description=ns_descriptions["instant-ngp"],
)

nqp_instant_ngp_synthetic_hashmap_log_17_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-synthetic-hashmap-log-17", NQPInstantNGPModel)
nqp_instant_ngp_synthetic_hashmap_log_17_config.pipeline.model.disable_scene_contraction = True
nqp_instant_ngp_synthetic_hashmap_log_17_config.pipeline.model.near_plane = 2
nqp_instant_ngp_synthetic_hashmap_log_17_config.pipeline.model.far_plane = 6
nqp_instant_ngp_synthetic_hashmap_log_17_config.pipeline.model.log2_hashmap_size = 17 # default 19
nqp_instant_ngp_synthetic_hashmap_log_17_config.pipeline.model.grid_levels = 4 #default 4? 
nqp_instant_ngp_synthetic_hashmap_log_17 = MethodSpecification(
    config=nqp_instant_ngp_synthetic_hashmap_log_17_config,
    description=ns_descriptions["instant-ngp"],
)

nqp_instant_ngp_synthetic_hashmap_log_18_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-synthetic-hashmap-log-18", NQPInstantNGPModel)
nqp_instant_ngp_synthetic_hashmap_log_18_config.pipeline.model.disable_scene_contraction = True
nqp_instant_ngp_synthetic_hashmap_log_18_config.pipeline.model.near_plane = 2
nqp_instant_ngp_synthetic_hashmap_log_18_config.pipeline.model.far_plane = 6
nqp_instant_ngp_synthetic_hashmap_log_18_config.pipeline.model.log2_hashmap_size = 18 # default 19
nqp_instant_ngp_synthetic_hashmap_log_18_config.pipeline.model.grid_levels = 4 #default 4? 
nqp_instant_ngp_synthetic_hashmap_log_18 = MethodSpecification(
    config=nqp_instant_ngp_synthetic_hashmap_log_18_config,
    description=ns_descriptions["instant-ngp"],
)

nqp_instant_ngp_synthetic_hashmap_log_20_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-synthetic-hashmap-log-20", NQPInstantNGPModel)
nqp_instant_ngp_synthetic_hashmap_log_20_config.pipeline.model.disable_scene_contraction = True
nqp_instant_ngp_synthetic_hashmap_log_20_config.pipeline.model.near_plane = 2
nqp_instant_ngp_synthetic_hashmap_log_20_config.pipeline.model.far_plane = 6
nqp_instant_ngp_synthetic_hashmap_log_20_config.pipeline.model.log2_hashmap_size = 20 # default 19
nqp_instant_ngp_synthetic_hashmap_log_20_config.pipeline.model.grid_levels = 4 #default 4? 
nqp_instant_ngp_synthetic_hashmap_log_20 = MethodSpecification(
    config=nqp_instant_ngp_synthetic_hashmap_log_20_config,
    description=ns_descriptions["instant-ngp"],
)

nqp_instant_ngp_synthetic_hashmap_log_21_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-synthetic-hashmap-log-21", NQPInstantNGPModel)
nqp_instant_ngp_synthetic_hashmap_log_21_config.pipeline.model.disable_scene_contraction = True
nqp_instant_ngp_synthetic_hashmap_log_21_config.pipeline.model.near_plane = 2
nqp_instant_ngp_synthetic_hashmap_log_21_config.pipeline.model.far_plane = 6
nqp_instant_ngp_synthetic_hashmap_log_21_config.pipeline.model.log2_hashmap_size = 21 # default 19
nqp_instant_ngp_synthetic_hashmap_log_21_config.pipeline.model.grid_levels = 4 #default 4? 
nqp_instant_ngp_synthetic_hashmap_log_21 = MethodSpecification(
    config=nqp_instant_ngp_synthetic_hashmap_log_21_config,
    description=ns_descriptions["instant-ngp"],
)

nqp_instant_ngp_synthetic_hashmap_log_21_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-synthetic-hashmap-log-21", NQPInstantNGPModel)
nqp_instant_ngp_synthetic_hashmap_log_21_config.pipeline.model.disable_scene_contraction = True
nqp_instant_ngp_synthetic_hashmap_log_21_config.pipeline.model.near_plane = 2
nqp_instant_ngp_synthetic_hashmap_log_21_config.pipeline.model.far_plane = 6
nqp_instant_ngp_synthetic_hashmap_log_21_config.pipeline.model.log2_hashmap_size = 21 # default 19
nqp_instant_ngp_synthetic_hashmap_log_21_config.pipeline.model.grid_levels = 4 #default 4? 
nqp_instant_ngp_synthetic_hashmap_log_21 = MethodSpecification(
    config=nqp_instant_ngp_synthetic_hashmap_log_21_config,
    description=ns_descriptions["instant-ngp"],
)

nqp_instant_ngp_synthetic_hashmap_log_6_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-synthetic-hashmap-log-6", NQPInstantNGPModel)
nqp_instant_ngp_synthetic_hashmap_log_6_config.pipeline.model.disable_scene_contraction = True
nqp_instant_ngp_synthetic_hashmap_log_6_config.pipeline.model.near_plane = 2
nqp_instant_ngp_synthetic_hashmap_log_6_config.pipeline.model.far_plane = 6
nqp_instant_ngp_synthetic_hashmap_log_6_config.pipeline.model.log2_hashmap_size = 6 # default 19
nqp_instant_ngp_synthetic_hashmap_log_6_config.pipeline.model.grid_levels = 4 #default 4? 
nqp_instant_ngp_synthetic_hashmap_log_6 = MethodSpecification(
    config=nqp_instant_ngp_synthetic_hashmap_log_6_config,
    description=ns_descriptions["instant-ngp"],
)

nqp_instant_ngp_synthetic_hashmap_log_4_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-synthetic-hashmap-log-4", NQPInstantNGPModel)
nqp_instant_ngp_synthetic_hashmap_log_4_config.pipeline.model.disable_scene_contraction = True
nqp_instant_ngp_synthetic_hashmap_log_4_config.pipeline.model.near_plane = 2
nqp_instant_ngp_synthetic_hashmap_log_4_config.pipeline.model.far_plane = 6
nqp_instant_ngp_synthetic_hashmap_log_4_config.pipeline.model.log2_hashmap_size = 4 # default 19
nqp_instant_ngp_synthetic_hashmap_log_4_config.pipeline.model.grid_levels = 4 #default 4? 
nqp_instant_ngp_synthetic_hashmap_log_4 = MethodSpecification(
    config=nqp_instant_ngp_synthetic_hashmap_log_4_config,
    description=ns_descriptions["instant-ngp"],
)

nqp_instant_ngp_synthetic_hashmap_log_2_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-synthetic-hashmap-log-2", NQPInstantNGPModel)
nqp_instant_ngp_synthetic_hashmap_log_2_config.pipeline.model.disable_scene_contraction = True
nqp_instant_ngp_synthetic_hashmap_log_2_config.pipeline.model.near_plane = 2
nqp_instant_ngp_synthetic_hashmap_log_2_config.pipeline.model.far_plane = 6
nqp_instant_ngp_synthetic_hashmap_log_2_config.pipeline.model.log2_hashmap_size = 2 # default 19
nqp_instant_ngp_synthetic_hashmap_log_2_config.pipeline.model.grid_levels = 4 #default 4? 
nqp_instant_ngp_synthetic_hashmap_log_2 = MethodSpecification(
    config=nqp_instant_ngp_synthetic_hashmap_log_2_config,
    description=ns_descriptions["instant-ngp"],
)
nqp_instant_ngp_synthetic_hashmap_log_1_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-synthetic-hashmap-log-1", NQPInstantNGPModel)
nqp_instant_ngp_synthetic_hashmap_log_1_config.pipeline.model.disable_scene_contraction = True
nqp_instant_ngp_synthetic_hashmap_log_1_config.pipeline.model.near_plane = 2
nqp_instant_ngp_synthetic_hashmap_log_1_config.pipeline.model.far_plane = 6
nqp_instant_ngp_synthetic_hashmap_log_1_config.pipeline.model.log2_hashmap_size = 1 # default 19
nqp_instant_ngp_synthetic_hashmap_log_1_config.pipeline.model.grid_levels = 4 #default 4? 
nqp_instant_ngp_synthetic_hashmap_log_1 = MethodSpecification(
    config=nqp_instant_ngp_synthetic_hashmap_log_1_config,
    description=ns_descriptions["instant-ngp"],
)

nqp_instant_ngp_grid_levels_3_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-grid-levels-3", NQPInstantNGPModel)
nqp_instant_ngp_grid_levels_3_config.pipeline.model.grid_levels = 3 # default 4
nqp_instant_ngp_grid_levels_3_config.pipeline.model.disable_scene_contraction = True
nqp_instant_ngp_grid_levels_3_config.pipeline.model.near_plane = 2
nqp_instant_ngp_grid_levels_3_config.pipeline.model.far_plane = 6
nqp_instant_ngp_grid_levels_3 = MethodSpecification(
    config=nqp_instant_ngp_grid_levels_3_config ,
    description=ns_descriptions["instant-ngp"] + " nqp-instant-ngp-grid-levels-3",
)


nqp_instant_ngp_grid_levels_6_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-grid-levels-6", NQPInstantNGPModel)
nqp_instant_ngp_grid_levels_6_config.pipeline.model.grid_levels = 6 # default 4
nqp_instant_ngp_grid_levels_6_config.pipeline.model.disable_scene_contraction = True
nqp_instant_ngp_grid_levels_6_config.pipeline.model.near_plane = 2
nqp_instant_ngp_grid_levels_6_config.pipeline.model.far_plane = 6
nqp_instant_ngp_grid_levels_6 = MethodSpecification(
    config=nqp_instant_ngp_grid_levels_6_config ,
    description=ns_descriptions["instant-ngp"] + " nqp-instant-ngp-grid-levels-6",
)

nqp_instant_ngp_grid_levels_8_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-grid-levels-8", NQPInstantNGPModel)
nqp_instant_ngp_grid_levels_8_config.pipeline.model.grid_levels = 8 # default 4
nqp_instant_ngp_grid_levels_8_config.pipeline.model.disable_scene_contraction = True
nqp_instant_ngp_grid_levels_8_config.pipeline.model.near_plane = 2
nqp_instant_ngp_grid_levels_8_config.pipeline.model.far_plane = 6
nqp_instant_ngp_grid_levels_8 = MethodSpecification(
    config=nqp_instant_ngp_grid_levels_8_config ,
    description=ns_descriptions["instant-ngp"] + " nqp-instant-ngp-grid-levels-8",
)

nqp_instant_ngp_grid_levels_10_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-grid-levels-10", NQPInstantNGPModel)
nqp_instant_ngp_grid_levels_10_config.pipeline.model.grid_levels = 10 # default 4
nqp_instant_ngp_grid_levels_10_config.pipeline.model.disable_scene_contraction = True
nqp_instant_ngp_grid_levels_10_config.pipeline.model.near_plane = 2
nqp_instant_ngp_grid_levels_10_config.pipeline.model.far_plane = 6
nqp_instant_ngp_grid_levels_10 = MethodSpecification(
    config=nqp_instant_ngp_grid_levels_10_config ,
    description=ns_descriptions["instant-ngp"] + " nqp-instant-ngp-grid-levels-10",
)
nqp_instant_ngp_grid_levels_12_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-grid-levels-12", NQPInstantNGPModel)
nqp_instant_ngp_grid_levels_12_config.pipeline.model.grid_levels = 12 # default 4
nqp_instant_ngp_grid_levels_12_config.pipeline.model.disable_scene_contraction = True
nqp_instant_ngp_grid_levels_12_config.pipeline.model.near_plane = 2
nqp_instant_ngp_grid_levels_12_config.pipeline.model.far_plane = 6
nqp_instant_ngp_grid_levels_12 = MethodSpecification(
    config=nqp_instant_ngp_grid_levels_12_config ,
    description=ns_descriptions["instant-ngp"] + " nqp-instant-ngp-grid-levels-12",
)

nqp_instant_ngp_hashmap_log_10_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-hashmap-log-10", NQPInstantNGPModel)
nqp_instant_ngp_hashmap_log_10_config.pipeline.model.log2_hashmap_size = 10 # default 19
nqp_instant_ngp_hashmap_log_10_config.pipeline.model.grid_levels = 4 #default 4? 
nqp_instant_ngp_hashmap_log_10 = MethodSpecification(
    config=nqp_instant_ngp_hashmap_log_10_config,
    description=ns_descriptions["instant-ngp"],
)
nqp_instant_ngp_hashmap_log_16_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-hashmap-log-16", NQPInstantNGPModel)
nqp_instant_ngp_hashmap_log_16_config.pipeline.model.log2_hashmap_size = 16 # default 19
nqp_instant_ngp_hashmap_log_16_config.pipeline.model.grid_levels = 4 #default 4? 
nqp_instant_ngp_hashmap_log_16 = MethodSpecification(
    config=nqp_instant_ngp_hashmap_log_16_config,
    description=ns_descriptions["instant-ngp"],
)

nqp_instant_ngp_hashmap_log_21_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-hashmap-log-21", NQPInstantNGPModel)
nqp_instant_ngp_hashmap_log_21_config.pipeline.model.log2_hashmap_size = 21 # default 19
nqp_instant_ngp_hashmap_log_21_config.pipeline.model.grid_levels = 4 #default 4? 
nqp_instant_ngp_hashmap_log_21 = MethodSpecification(
    config=nqp_instant_ngp_hashmap_log_21_config,
    description=ns_descriptions["instant-ngp"],
)

nqp_instant_ngp_hashmap_log_20_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-hashmap-log-20", NQPInstantNGPModel)
nqp_instant_ngp_hashmap_log_20_config.pipeline.model.log2_hashmap_size = 20 # default 19
nqp_instant_ngp_hashmap_log_20_config.pipeline.model.grid_levels = 4 #default 4? 
nqp_instant_ngp_hashmap_log_20 = MethodSpecification(
    config=nqp_instant_ngp_hashmap_log_20_config,
    description=ns_descriptions["instant-ngp"],
)


nqp_instant_ngp_hashmap_log_12_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-hashmap-log-12", NQPInstantNGPModel)
nqp_instant_ngp_hashmap_log_12_config.pipeline.model.log2_hashmap_size = 12 # default 19
nqp_instant_ngp_hashmap_log_12_config.pipeline.model.grid_levels = 4 #default 4? 
nqp_instant_ngp_hashmap_log_12 = MethodSpecification(
    config=nqp_instant_ngp_hashmap_log_12_config,
    description=ns_descriptions["instant-ngp"],
)

#grid resolution 
nqp_instant_ngp_grid_resolution_64_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-grid-resolution-64", NQPInstantNGPModel)
nqp_instant_ngp_grid_resolution_64_config.pipeline.model.log2_hashmap_size = 19 # default 19
nqp_instant_ngp_grid_resolution_64_config.pipeline.model.grid_levels = 16 #default 4? 
nqp_instant_ngp_grid_resolution_64_config.pipeline.model.grid_resolution = 64 
nqp_instant_ngp_grid_resolution_64 = MethodSpecification(
    config=nqp_instant_ngp_grid_resolution_64_config ,
    description=ns_descriptions["instant-ngp"],
)

nqp_instant_ngp_grid_resolution_32_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-grid-resolution-32", NQPInstantNGPModel)
nqp_instant_ngp_grid_resolution_32_config.pipeline.model.log2_hashmap_size = 19 # default 19
nqp_instant_ngp_grid_resolution_32_config.pipeline.model.grid_levels = 16 #default 4? 
nqp_instant_ngp_grid_resolution_32_config.pipeline.model.grid_resolution = 32
nqp_instant_ngp_grid_resolution_32 = MethodSpecification(
    config=nqp_instant_ngp_grid_resolution_32_config ,
    description=ns_descriptions["instant-ngp"],
)

nqp_instant_ngp_grid_resolution_256_config = make_nqp_config("instant-ngp", "nqp-instant-ngp-grid-resolution-256", NQPInstantNGPModel)
nqp_instant_ngp_grid_resolution_256_config.pipeline.model.log2_hashmap_size = 19 # default 19
nqp_instant_ngp_grid_resolution_256_config.pipeline.model.grid_levels = 16 #default 4? 
nqp_instant_ngp_grid_resolution_256_config.pipeline.model.grid_resolution = 192
nqp_instant_ngp_grid_resolution_256 = MethodSpecification(
    config=nqp_instant_ngp_grid_resolution_256_config ,
    description=ns_descriptions["instant-ngp"],
)