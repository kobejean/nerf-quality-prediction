import torch

def get_weight_hist(weights, ray_samples, range, ray_indices=None, num_rays=None, accumulation=None):
    values = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
    # if ray_indices is not None and num_rays is not None:
    #     weights = weights / (accumulation[ray_indices] + 1e-10)
    return torch.histogram(values[...,0].cpu(), 10000, range=range, weight=weights[...,0].cpu())