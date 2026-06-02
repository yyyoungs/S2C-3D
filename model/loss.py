import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights

def integrated_multi_loss(
    pred_img: torch.Tensor, 
    mask: torch.Tensor, 
    target_img: torch.Tensor,
    lambda_l1_max: float = 1e-6,  
    lambda_boundary: float = 0.5,   
    lambda_tv: float = 1e-4        
) -> torch.Tensor:
    """Combine L1-max, boundary consistency, and masked inner total variation losses."""
    epsilon = 1e-6 

    if mask.dim() == 4 and mask.shape[1] == 1:
        mask_bc = mask.expand_as(pred_img)
    else:
        mask_bc = mask
        
    mask_float = mask_bc.float()
    missing_region_mask = 1.0 - mask_float

    pred_missing_l1 = torch.abs(pred_img) * missing_region_mask
    sum_l1_norm = torch.sum(pred_missing_l1)
    num_missing_pixels = torch.sum(missing_region_mask)
    
    mean_l1_norm = sum_l1_norm / (num_missing_pixels + epsilon)
    L_l1_max_final = 1.0 / (mean_l1_norm + epsilon)

    dilated_mask = F.max_pool2d(
        mask_float, 
        kernel_size=3, 
        stride=1, 
        padding=1
    )
    boundary_mask = dilated_mask * missing_region_mask
    
    
    l1_diff = torch.abs(pred_img - target_img)
    boundary_l1 = l1_diff * boundary_mask
    
    
    num_boundary_pixels = torch.sum(boundary_mask)
    
    if num_boundary_pixels.item() > 0:
        L_Boundary = torch.sum(boundary_l1) / (num_boundary_pixels + epsilon)
    else:
        L_Boundary = torch.tensor(0.0, device=pred_img.device, dtype=pred_img.dtype)

    diff_h = pred_img[:, :, :, 1:] - pred_img[:, :, :, :-1]
    diff_v = pred_img[:, :, 1:, :] - pred_img[:, :, :-1, :]
    
    
    mask_h = missing_region_mask[:, :, :, :-1] * missing_region_mask[:, :, :, 1:]
    mask_v = missing_region_mask[:, :, :-1, :] * missing_region_mask[:, :, 1:, :]
    
    
    tv_h = torch.sum(torch.abs(diff_h) * mask_h)
    tv_v = torch.sum(torch.abs(diff_v) * mask_v)
    
    
    num_tv_pixels = torch.sum(mask_h) + torch.sum(mask_v)
    
    if num_tv_pixels.item() > 0:
        L_TV = (tv_h + tv_v) / (num_tv_pixels + epsilon)
    else:
        L_TV = torch.tensor(0.0, device=pred_img.device, dtype=pred_img.dtype)

    L_integrated = (lambda_l1_max * L_l1_max_final) +\
                   (lambda_boundary * L_Boundary) +\
                   (lambda_tv * L_TV)
    
    return L_integrated


def integrated_multi_loss_with_local_color(
    pred_img: torch.Tensor, 
    mask: torch.Tensor, 
    target_img: torch.Tensor,
    lambda_l1_max: float = 1e-3,    
    lambda_boundary: float = 0.5,     
    lambda_tv: float = 1e-3,          
    lambda_color: float = 0.05        
) -> torch.Tensor:
    """Add a local color consistency term to the integrated inpainting loss."""
    epsilon = 1e-6 

    if mask.dim() == 4 and mask.shape[1] == 1:
        mask_bc = mask.expand_as(pred_img)
    else:
        mask_bc = mask
        
    mask_float = mask_bc.float()
    missing_region_mask = 1.0 - mask_float

    pred_missing_l1 = torch.abs(pred_img) * missing_region_mask
    sum_l1_norm = torch.sum(pred_missing_l1)
    num_missing_pixels = torch.sum(missing_region_mask)
    mean_l1_norm = sum_l1_norm / (num_missing_pixels + epsilon)
    L_l1_max_final = 1.0 / (mean_l1_norm + epsilon)

    dilated_mask = F.max_pool2d(mask_float, kernel_size=3, stride=1, padding=1)
    boundary_mask = dilated_mask * missing_region_mask
    l1_diff = torch.abs(pred_img - target_img)
    boundary_l1 = l1_diff * boundary_mask
    num_boundary_pixels = torch.sum(boundary_mask)
    if num_boundary_pixels.item() > 0:
        L_Boundary = torch.sum(boundary_l1) / (num_boundary_pixels + epsilon)
    else:
        L_Boundary = torch.tensor(0.0, device=pred_img.device, dtype=pred_img.dtype)

    diff_h = pred_img[:, :, :, 1:] - pred_img[:, :, :, :-1]
    diff_v = pred_img[:, :, 1:, :] - pred_img[:, :, :-1, :]
    mask_h = missing_region_mask[:, :, :, :-1] * missing_region_mask[:, :, :, 1:]
    mask_v = missing_region_mask[:, :, :-1, :] * missing_region_mask[:, :, 1:, :]
    tv_h = torch.sum(torch.abs(diff_h) * mask_h)
    tv_v = torch.sum(torch.abs(diff_v) * mask_v)
    num_tv_pixels = torch.sum(mask_h) + torch.sum(mask_v)
    if num_tv_pixels.item() > 0:
        L_TV = (tv_h + tv_v) / (num_tv_pixels + epsilon)
    else:
        L_TV = torch.tensor(0.0, device=pred_img.device, dtype=pred_img.dtype)

    dilated_missing_mask = F.max_pool2d(
        missing_region_mask, 
        kernel_size=5, 
        stride=1, 
        padding=2
    )
    local_sample_mask = dilated_missing_mask * mask_float 
    local_target_color = target_img * local_sample_mask 

    num_local_known_pixels = torch.sum(local_sample_mask, dim=[0, 2, 3], keepdim=True)
    mean_target_color = torch.sum(
        local_target_color,
        dim=[0, 2, 3],
        keepdim=True,
    ) / (num_local_known_pixels + epsilon)

    missing_region_pred = pred_img * missing_region_mask
    num_missing_pixels_per_channel = torch.sum(missing_region_mask, dim=[0, 2, 3], keepdim=True)
    mean_pred_color = torch.sum(
        missing_region_pred,
        dim=[0, 2, 3],
        keepdim=True,
    ) / (num_missing_pixels_per_channel + epsilon)

    L_Color = torch.mean(torch.abs(mean_pred_color - mean_target_color))

    L_integrated = (lambda_l1_max * L_l1_max_final) +\
                   (lambda_boundary * L_Boundary) +\
                   (lambda_tv * L_TV) +\
                   (lambda_color * L_Color)
    
    return L_integrated


def maximize_variance_loss_batch(pred_img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Maximize the mean L1 norm in the missing region via an inverse loss."""
    epsilon = 1e-6 
    
    if mask.dim() == 4 and mask.shape[1] == 1:
        mask = mask.expand_as(pred_img)

    missing_region_mask = 1.0 - mask 
    pred_missing_l1 = torch.abs(pred_img) * missing_region_mask
    sum_l1_norm = torch.sum(pred_missing_l1)
    num_missing_pixels = torch.sum(missing_region_mask)
    mean_l1_norm = sum_l1_norm / (num_missing_pixels + epsilon)
    L_l1_max_final = 1.0 / (mean_l1_norm + epsilon)
    return L_l1_max_final


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 intermediate features."""

    def __init__(self, feature_layer='relu4_3', use_l1=True):
        super(VGGPerceptualLoss, self).__init__()
        # relu1_2: 3, relu2_2: 8, relu3_3: 15, relu4_3: 22, relu5_3: 29
        layer_indices = {
            'relu1_2': 3, 'relu2_2': 8, 'relu3_3': 15, 'relu4_3': 22, 'relu5_3': 29
        }
        
        if feature_layer not in layer_indices:
            raise ValueError(
                f"Feature layer {feature_layer} not supported. "
                f"Must be one of {list(layer_indices.keys())}"
            )

        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        features = vgg.features
        target_index = layer_indices[feature_layer]
        self.vgg = nn.Sequential(*features[:target_index + 1])

        for param in self.vgg.parameters():
            param.requires_grad = False
        self.vgg.eval() 
        self.loss_fn = F.l1_loss if use_l1 else F.mse_loss
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _normalize(self, x):
        """Normalize an image tensor with ImageNet statistics."""
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        normalized_x = (x - self.mean) / self.std
        return normalized_x

    def forward(self, generated_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        """Compute perceptual distance between generated and target images."""
        normalized_gen = self._normalize(generated_img)
        normalized_target = self._normalize(target_img)

        with torch.no_grad():
            target_features = self.vgg(normalized_target).detach()
        
        generated_features = self.vgg(normalized_gen)
        loss = self.loss_fn(generated_features, target_features)
        return loss

# Define style weights for different layers
STYLE_WEIGHTS = {
    'relu1_2': 1.0 / 2.6,
    'relu2_2': 1.0 / 4.8,
    'relu3_3': 1.0 / 3.7,
    'relu4_3': 1.0 / 5.6,
    'relu5_3': 10.0 / 1.5
}


def get_features(image, model, layers=None):
    """
    Extract features from specific layers of a model for a given image.
    
    Args:
        image (torch.Tensor): Input image tensor.
        model (torch.nn.Module): Pretrained model (e.g., VGG).
        layers (dict): Mapping of layer indices to layer names.
    
    Returns:
        dict: A dictionary of features for the specified layers.
    """
    if layers is None:
        layers = {
            '3': 'relu1_2',
            '8': 'relu2_2',
            '15': 'relu3_3',
            '22': 'relu4_3',
            '29': 'relu5_3'
        }
    
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    """
    Compute the Gram matrix for a given tensor.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (batch_size, depth, height, width).
    
    Returns:
        torch.Tensor: Gram matrix of the input tensor.
    """
    b, d, h, w = tensor.size()
    tensor = tensor.reshape(b * d, h * w)  # Reshape tensor for matrix multiplication
    gram = torch.mm(tensor, tensor.t())  # Compute Gram matrix
    return gram

def gram_loss(style, target, model):
    """
    Compute the Gram loss (style loss) between a style image and a target image.
    
    Args:
        style (torch.Tensor): Style image tensor.
        target (torch.Tensor): Target image tensor.
        model (torch.nn.Module): Pretrained model (e.g., VGG).
    
    Returns:
        torch.Tensor: The computed Gram loss.
    """
    # Extract features for the style and target images
    style_features = get_features(style, model)
    target_features = get_features(target, model)
    
    # Compute Gram matrices for the style image
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    
    # Initialize total loss
    total_loss = 0
    
    # Compute the weighted Gram loss for each layer
    for layer, weight in STYLE_WEIGHTS.items():
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        
        # Compute the layer-specific Gram loss
        _, d, h, w = target_feature.shape
        layer_loss = weight * torch.mean((target_gram - style_gram) ** 2)
        total_loss += layer_loss / (d * h * w)
    
    return total_loss
