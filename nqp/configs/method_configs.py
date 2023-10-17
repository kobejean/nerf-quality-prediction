from __future__ import annotations
import math 
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

def make_nqp_config(method, NQPModel):
    # scene_scale = 1.34
    # near_plane, far_plane = 1/3, 7/3 * scene_scale
    nqp_method = "nqp-"+method
    nqp_config = ns_method_configs[method]
    nqp_config.method_name= nqp_method
    nqp_config.project_name="nerf-quality-prediction"
    # nqp_config.steps_per_eval_all_images=5000
    nqp_config.pipeline._target=NQPPipeline
    nqp_config.pipeline.datamanager.dataparser=NQPDataParserConfig()
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


nqp_nerfacto_config = make_nqp_config("nerfacto", NQPNerfactoModel)
nqp_nerfacto = MethodSpecification(
    config=nqp_nerfacto_config,
    description=ns_descriptions["nerfacto"],
)


nqp_instant_ngp_config = make_nqp_config("instant-ngp", NQPInstantNGPModel)
nqp_instant_ngp = MethodSpecification(
    config=nqp_instant_ngp_config,
    description=ns_descriptions["instant-ngp"],
)

nqp_mipnerf_config = make_nqp_config("mipnerf", NQPMipNerfModel)
nqp_mipnerf_config.steps_per_eval_all_images=1000000  # set to a very large model so we don't eval with all images
nqp_mipnerf = MethodSpecification(
    config=nqp_mipnerf_config,
    description=ns_descriptions["mipnerf"],
)

nqp_vanilla_nerf_config = make_nqp_config("vanilla-nerf", NQPNeRFModel)
nqp_vanilla_nerf_config.steps_per_eval_all_images=1000000  # set to a very large model so we don't eval with all images
nqp_vanilla_nerf = MethodSpecification(
    config=nqp_vanilla_nerf_config,
    description=ns_descriptions["vanilla-nerf"],
)

nqp_tensorf_config = make_nqp_config("tensorf", NQPTensoRFModel)
nqp_tensorf = MethodSpecification(
    config=nqp_tensorf_config,
    description=ns_descriptions["tensorf"],
)
