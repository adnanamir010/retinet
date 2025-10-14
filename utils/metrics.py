import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_mse(pred, target):
    """Mean Squared Error"""
    return F.mse_loss(pred, target).item()


def compute_lmse(pred, target, k=20):
    """
    Local Mean Squared Error (LMSE)
    Compute MSE over local patches
    
    Args:
        pred: (B, C, H, W) prediction in [0, 255]
        target: (B, C, H, W) ground truth in [0, 255]
        k: patch size (default 20 as per paper)
    """
    B, C, H, W = pred.shape
    stride = k // 2
    
    # Unfold into patches
    pred_patches = F.unfold(pred, kernel_size=k, stride=stride)
    target_patches = F.unfold(target, kernel_size=k, stride=stride)
    
    # Compute MSE for each patch
    mse_per_patch = ((pred_patches - target_patches) ** 2).mean(dim=1)
    
    # Average over all patches
    lmse = mse_per_patch.mean().item()
    
    # Normalize by max possible error (255^2)
    lmse_normalized = lmse / (255.0 ** 2)
    
    return lmse_normalized


def compute_dssim(pred, target):
    """
    Dissimilarity Structural Similarity Index (DSSIM)
    DSSIM = (1 - SSIM) / 2
    
    Args:
        pred: (B, C, H, W) prediction in [0, 255]
        target: (B, C, H, W) ground truth in [0, 255]
    """
    B = pred.shape[0]
    dssim_sum = 0.0
    
    # Convert to numpy
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    for i in range(B):
        # Normalize to [0, 1] for SSIM computation
        pred_img = np.transpose(pred_np[i] / 255.0, (1, 2, 0))
        target_img = np.transpose(target_np[i] / 255.0, (1, 2, 0))
        
        # Compute SSIM (use channel_axis instead of multichannel)
        ssim_val = ssim(target_img, pred_img, 
                       channel_axis=2,  # Channel is last dimension
                       data_range=1.0,
                       win_size=7)
        
        # Convert to DSSIM
        dssim = (1.0 - ssim_val) / 2.0
        dssim_sum += dssim
    
    return dssim_sum / B


def compute_all_metrics(pred_r, pred_s, target_r, target_s):
    """
    Compute all metrics for intrinsic decomposition
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Reflectance metrics
    metrics['mse_reflectance'] = compute_mse(pred_r, target_r)
    metrics['lmse_reflectance'] = compute_lmse(pred_r, target_r)
    metrics['dssim_reflectance'] = compute_dssim(pred_r, target_r)
    
    # Shading metrics (expand to 3 channels for DSSIM)
    pred_s_3ch = pred_s.expand(-1, 3, -1, -1)
    target_s_3ch = target_s.expand(-1, 3, -1, -1)
    
    metrics['mse_shading'] = compute_mse(pred_s, target_s)
    metrics['lmse_shading'] = compute_lmse(pred_s, target_s)
    metrics['dssim_shading'] = compute_dssim(pred_s_3ch, target_s_3ch)
    
    # Average metrics
    metrics['mse_avg'] = (metrics['mse_reflectance'] + metrics['mse_shading']) / 2
    metrics['lmse_avg'] = (metrics['lmse_reflectance'] + metrics['lmse_shading']) / 2
    metrics['dssim_avg'] = (metrics['dssim_reflectance'] + metrics['dssim_shading']) / 2
    
    return metrics


if __name__ == "__main__":
    print("Testing metrics...")
    
    pred_r = torch.rand(2, 3, 120, 160) * 255
    pred_s = torch.rand(2, 1, 120, 160) * 255
    target_r = torch.rand(2, 3, 120, 160) * 255
    target_s = torch.rand(2, 1, 120, 160) * 255
    
    metrics = compute_all_metrics(pred_r, pred_s, target_r, target_s)
    
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    
    print("\nMetrics test passed!")