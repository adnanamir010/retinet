import torch
import torch.nn as nn
from .retinet_stage1 import RetiNet_Stage1
from .retinet_stage2 import RetiNet_Stage2

class RetiNet(nn.Module):
    """
    Complete RetiNet: Two-stage Retinex-inspired intrinsic decomposition.
    
    Stage 1: Separates image gradients into reflectance and shading gradients
    Stage 2: Reintegrates gradients with original image to produce final decomposition
    
    This is the full pipeline that combines both stages.
    """
    
    def __init__(self, use_dropout=True, compute_gradients=True):
        """
        Args:
            use_dropout: Whether to use dropout during training
            compute_gradients: If True, computes gradients internally.
                              If False, expects precomputed gradients as input.
        """
        super(RetiNet, self).__init__()
        
        self.stage1 = RetiNet_Stage1(use_dropout=use_dropout)
        self.stage2 = RetiNet_Stage2(use_dropout=use_dropout)
        self.compute_gradients = compute_gradients
    
    def _compute_gradient(self, image):
        """
        Compute image gradients (differentiable PyTorch version).
        
        Args:
            image: (B, 3, H, W) in [0, 255] range
        
        Returns:
            gradient: (B, 3, H, W) gradient magnitude per channel
        """
        # Scale to 16-bit range
        tmp_img = image * 257.0
        
        # Vertical gradient
        I_y = torch.zeros_like(tmp_img)
        I_y[:, :, 1:, :] = tmp_img[:, :, 1:, :] - tmp_img[:, :, :-1, :]
        
        # Horizontal gradient
        I_x = torch.zeros_like(tmp_img)
        I_x[:, :, :, 1:] = tmp_img[:, :, :, 1:] - tmp_img[:, :, :, :-1]
        
        # Gradient magnitude
        grad = torch.sqrt(I_x**2 + I_y**2)
        
        return grad
    
    def forward(self, rgb, rgb_grad=None):
        """
        Full forward pass through RetiNet.
        
        Args:
            rgb: (B, 3, H, W) RGB image in [0, 255]
            rgb_grad: (B, 3, H, W) Optional precomputed gradients.
                     If None and compute_gradients=True, will be computed.
        
        Returns:
            albedo: (B, 3, H, W) Final reflectance in [0, 255]
            shading: (B, 1, H, W) Final shading in [0, 255]
            albedo_grad: (B, 3, H, W) Intermediate albedo gradients
            shading_grad: (B, 1, H, W) Intermediate shading gradients
        """
        # Compute gradients if needed
        if rgb_grad is None:
            if self.compute_gradients:
                rgb_grad = self._compute_gradient(rgb)
            else:
                raise ValueError("rgb_grad must be provided when compute_gradients=False")
        
        # Stage 1: Gradient separation
        albedo_grad, shading_grad = self.stage1(rgb, rgb_grad)
        
        # Stage 2: Reintegration
        albedo, shading = self.stage2(rgb, albedo_grad, shading_grad)
        
        return albedo, shading, albedo_grad, shading_grad
    
    def forward_stage1_only(self, rgb, rgb_grad=None):
        """Run only Stage 1 (gradient separation)"""
        if rgb_grad is None and self.compute_gradients:
            rgb_grad = self._compute_gradient(rgb)
        return self.stage1(rgb, rgb_grad)
    
    def forward_stage2_only(self, rgb, albedo_grad, shading_grad):
        """Run only Stage 2 (reintegration)"""
        return self.stage2(rgb, albedo_grad, shading_grad)


if __name__ == "__main__":
    print("Testing Complete RetiNet Pipeline...")
    
    model = RetiNet(use_dropout=True, compute_gradients=True)
    model.eval()
    
    rgb = torch.rand(2, 3, 120, 160) * 255.0
    
    with torch.no_grad():
        albedo, shading, albedo_grad, shading_grad = model(rgb)
    
    print("\n=== Full Pipeline ===")
    print(f"Input RGB: {rgb.shape}, range: [{rgb.min():.1f}, {rgb.max():.1f}]")
    print(f"Output Albedo: {albedo.shape}, range: [{albedo.min():.1f}, {albedo.max():.1f}]")
    print(f"Output Shading: {shading.shape}, range: [{shading.min():.1f}, {shading.max():.1f}]")
    print(f"Intermediate Albedo Grad: {albedo_grad.shape}, range: [{albedo_grad.min():.1f}, {albedo_grad.max():.1f}]")
    print(f"Intermediate Shading Grad: {shading_grad.shape}, range: [{shading_grad.min():.1f}, {shading_grad.max():.1f}]")
    
    stage1_params = sum(p.numel() for p in model.stage1.parameters())
    stage2_params = sum(p.numel() for p in model.stage2.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n=== Parameters ===")
    print(f"Stage 1: {stage1_params:,}")
    print(f"Stage 2: {stage2_params:,}")
    print(f"Total: {total_params:,}")
    
    print("\nRetiNet pipeline test passed!")