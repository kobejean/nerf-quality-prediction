# from nerfstudio.engine.trainer import TrainerConfig
# from nerfstudio.configs.method_configs import method_configs
# from nerfstudio.configs.dataparser_configs import dataparsers as dataparser_configs
# from nerfstudio.utils.eval_utils import eval_load_checkpoint
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils import colormaps, colors, misc
from nqp.utils.spacial import convert_to_transformed_space

from PIL import Image
from pathlib import Path
import yaml
import json
import torch
from torch.nn.functional import normalize
import numpy as np
import OpenEXR, Imath

def load_config(config_path):
    with open(config_path, "r") as f:
        config_str = f.read()
    config = yaml.load(config_str, Loader=yaml.Loader)
    # config.print_to_terminal()
    return config


def save_as_image(x, path):
    x = torch.clamp(x, 0, 1.0)
    Image.fromarray((x * 255).byte().cpu().numpy()).save(path)


def read_depth_map(file_path, device):
    exr_file = OpenEXR.InputFile(file_path)
    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    depth_str = exr_file.channel('R', pt)
    depth = np.frombuffer(depth_str, dtype=np.float32)
    depth.shape = (height, width)  # reshape
    
    return torch.unsqueeze(torch.from_numpy(depth), dim=-1).to(device=device)


# from: https://github.com/kobejean/bts/blob/d4f1de8f9c8f253cc2bcd1ce0d92fce193a72bac/pytorch/bts_eval.py#L91-L112
def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()
    
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    
    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
    
    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)
    
    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3


def load_metadata(config):
    if config.data.suffix == ".json":
        meta = load_from_json(config.data)
        data_dir = config.data.parent
    else:
        meta = load_from_json(config.data / "transforms.json")
        data_dir = config.data
    return meta, data_dir