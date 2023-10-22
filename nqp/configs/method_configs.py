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


