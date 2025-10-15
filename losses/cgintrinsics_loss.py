import torch
import torch.nn as nn
import torch.nn.functional as F


class CGIntrinsicsLoss(nn.Module):
    """
    Loss function based on CGIntrinsics (Li & Snavely, CVPR 2018).
    
    Combines scale-invariant MSE with multi-scale gradient matching
    and image formation constraint.
    
    L = gamma_R * (L_R_SI + gamma_grad * L_R_grad) + 
        gamma_S * (L_S_SI + gamma_grad * L_S_grad) + 
        gamma_IMF * L_IMF
    """
    
    def __init__(self, gamma_r=1.0, gamma_s=1.0, gamma_imf=1.0, 
                 gamma_grad=1.0, num_scales=4):
        super(CGIntrinsicsLoss, self).__init__()
        self.gamma_r = gamma_r
        self.gamma_s = gamma_s
        self.gamma_imf = gamma_imf
        self.gamma_grad = gamma_grad
        self.num_scales = num_scales
    
    def scale_invariant_mse(self, pred, target, mask=None):
        """
        Scale-invariant MSE computed in linear domain.
        
        Args:
            pred: (B, C, H, W) predicted image in [0, 255]
            target: (B, C, H, W) ground truth in [0, 255]
            mask: (B, 1, H, W) optional validity mask
        
        Returns:
            Scalar loss value
        """
        if mask is None:
            mask = torch.ones_like(pred[:, 0:1])
        
        mask_expanded = mask.expand_as(pred)
        
        pred_valid = pred[mask_expanded > 0]
        target_valid = target[mask_expanded > 0]
        
        numerator = (pred_valid * target_valid).sum()
        denominator = (pred_valid * pred_valid).sum()
        scale = numerator / (denominator + 1e-8)
        
        diff = (target - scale * pred) ** 2
        loss = (diff * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
        
        return loss
    
    def compute_gradients(self, x):
        """
        Compute gradient magnitude using Sobel filters.
        
        Args:
            x: (B, C, H, W) input image
        
        Returns:
            (B, C, H, W) gradient magnitude
        """
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(x, sobel_x.expand(x.size(1), 1, 3, 3), 
                         padding=1, groups=x.size(1))
        grad_y = F.conv2d(x, sobel_y.expand(x.size(1), 1, 3, 3), 
                         padding=1, groups=x.size(1))
        
        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    
    def multi_scale_gradient_loss(self, pred, target, mask=None):
        """
        Multi-scale L1 gradient matching.
        
        Args:
            pred: (B, C, H, W) predicted image
            target: (B, C, H, W) ground truth
            mask: (B, 1, H, W) optional mask
        
        Returns:
            Scalar loss value
        """
        total_loss = 0.0
        
        for scale_idx in range(self.num_scales):
            if scale_idx > 0:
                scale_factor = 2 ** scale_idx
                pred_scaled = F.avg_pool2d(pred, scale_factor)
                target_scaled = F.avg_pool2d(target, scale_factor)
                mask_scaled = F.avg_pool2d(mask, scale_factor) if mask is not None else None
            else:
                pred_scaled = pred
                target_scaled = target
                mask_scaled = mask
            
            if mask_scaled is None:
                mask_scaled = torch.ones_like(pred_scaled[:, 0:1])
            
            mask_expanded = mask_scaled.expand_as(pred_scaled)
            pred_valid = pred_scaled[mask_expanded > 0]
            target_valid = target_scaled[mask_expanded > 0]
            
            scale = (pred_valid * target_valid).sum() / (pred_valid * pred_valid).sum() + 1e-8
            
            pred_grad = self.compute_gradients(scale * pred_scaled)
            target_grad = self.compute_gradients(target_scaled)
            
            grad_diff = torch.abs(pred_grad - target_grad)
            mask_grad = mask_scaled.expand_as(grad_diff)
            loss = (grad_diff * mask_grad).sum() / (mask_grad.sum() + 1e-8)
            
            total_loss += loss
        
        return total_loss / self.num_scales
    
    def image_formation_loss(self, r_pred, s_pred, input_image, mask=None):
        """
        Physics-based constraint: R x S should reconstruct input.
        
        Args:
            r_pred: (B, 3, H, W) predicted reflectance in [0, 255]
            s_pred: (B, 1, H, W) predicted shading in [0, 255]
            input_image: (B, 3, H, W) original input in [0, 255]
            mask: (B, 1, H, W) optional mask
        
        Returns:
            Scalar loss value
        """
        s_normalized = s_pred / 255.0
        reconstructed = r_pred * s_normalized
        
        return self.scale_invariant_mse(reconstructed, input_image, mask)
    
    def forward(self, r_pred, s_pred, r_gt, s_gt, input_image, mask=None):
        """
        Compute combined loss.
        
        Args:
            r_pred: (B, 3, H, W) predicted reflectance in [0, 255]
            s_pred: (B, 1, H, W) predicted shading in [0, 255]
            r_gt: (B, 3, H, W) ground truth reflectance in [0, 255]
            s_gt: (B, 1, H, W) ground truth shading in [0, 255]
            input_image: (B, 3, H, W) original input in [0, 255]
            mask: (B, 1, H, W) optional mask
        
        Returns:
            total_loss: Scalar tensor
            loss_dict: Dictionary with loss components
        """
        loss_r_si = self.scale_invariant_mse(r_pred, r_gt, mask)
        loss_s_si = self.scale_invariant_mse(s_pred, s_gt, mask)
        
        loss_r_grad = self.multi_scale_gradient_loss(r_pred, r_gt, mask)
        loss_s_grad = self.multi_scale_gradient_loss(s_pred, s_gt, mask)
        
        loss_imf = self.image_formation_loss(r_pred, s_pred, input_image, mask)
        
        total_loss = (
            self.gamma_r * (loss_r_si + self.gamma_grad * loss_r_grad) +
            self.gamma_s * (loss_s_si + self.gamma_grad * loss_s_grad) +
            self.gamma_imf * loss_imf
        )
        
        loss_dict = {
            'loss_r': loss_r_si.item(),
            'loss_s': loss_s_si.item(),
            'loss_r_grad': loss_r_grad.item(),
            'loss_s_grad': loss_s_grad.item(),
            'loss_imf': loss_imf.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


if __name__ == "__main__":
    print("Testing CGIntrinsicsLoss...")
    
    batch_size = 4
    h, w = 120, 160
    
    r_pred = torch.rand(batch_size, 3, h, w) * 255
    s_pred = torch.rand(batch_size, 1, h, w) * 255
    r_gt = torch.rand(batch_size, 3, h, w) * 255
    s_gt = torch.rand(batch_size, 1, h, w) * 255
    input_img = r_gt * (s_gt / 255.0)
    
    criterion = CGIntrinsicsLoss(gamma_r=1.0, gamma_s=1.0, gamma_imf=1.0, gamma_grad=1.0)
    loss, loss_dict = criterion(r_pred, s_pred, r_gt, s_gt, input_img)
    
    print("\nLoss components:")
    for key, val in loss_dict.items():
        print(f"  {key}: {val:.4f}")
    
    print(f"\nValue ranges:")
    print(f"  r_pred: [{r_pred.min():.1f}, {r_pred.max():.1f}]")
    print(f"  s_pred: [{s_pred.min():.1f}, {s_pred.max():.1f}]")
    print(f"  input: [{input_img.min():.1f}, {input_img.max():.1f}]")
    
    print("\nLoss computation successful!")