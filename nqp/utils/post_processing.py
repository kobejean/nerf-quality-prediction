import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

def post_process_render(input_tensor, resize_to=(540, 960), crop_to=(522, 928), antialias=True):
    """
    Apply post-processing to a rendered RGB PyTorch tensor.
    The tensor is first converted from [H, W, C] to [C, H, W], downscaled to 'resize_to', cropped to 'crop_to',
    converted back to [H, W, C], and then values are clamped to [0, 1].
    
    Args:
        input_tensor (torch.Tensor): A PyTorch tensor representing RGB images.
                                     Expected shape: [height, width, channels].
        resize_to (tuple): Target resolution for downsampling (height, width).
        crop_to (tuple): Target resolution for cropping (height, width).
        antialias (bool): Whether to apply anti-aliasing during resizing.

    Returns:
        torch.Tensor: The post-processed PyTorch tensor in [height, width, channels] format.
    """
    if input_tensor.shape[-1] != 3:
        raise ValueError("Input tensor must have shape [height, width, 3].")

    # Change shape from [H, W, C] to [C, H, W]
    tensor_c_h_w = input_tensor.permute(2, 0, 1)

    # Resize
    resized_img = TF.resize(tensor_c_h_w, resize_to, antialias=antialias)

    # Crop (center crop)
    cropped_img = TF.center_crop(resized_img, crop_to)

    # Change shape back to [H, W, C]
    tensor_h_w_c = cropped_img.permute(1, 2, 0)

    # Clamping the values to [0, 1]
    clamped_tensor = torch.clamp(tensor_h_w_c, 0, 1)

    return clamped_tensor
