import torch
import torch.nn as nn

class IntrinsicLoss(nn.Module):
    """
    Combined loss for IntrinsicNet (Equation 13):
    L = gamma_R * L_R + gamma_S * L_S + gamma_IMF * L_IMF
    """
    
    def __init__(self, gamma_r=1.0, gamma_s=1.0, gamma_imf=1.0):
        super(IntrinsicLoss, self).__init__()
        self.gamma_r = gamma_r
        self.gamma_s = gamma_s
        self.gamma_imf = gamma_imf
        
        self.mse_loss = nn.MSELoss()
    
    def image_formation_loss(self, r_pred, s_pred, input_image):
        """
        Physics-based constraint: R x S should reconstruct the input.
        
        Since values are in [0, 255]:
        - R is in [0, 255] (albedo values)
        - S is in [0, 255] but normalized to [0, 1] for multiplication
        
        Args:
            r_pred: (B, 3, H, W) Predicted reflectance in [0, 255]
            s_pred: (B, 1, H, W) Predicted shading in [0, 255]
            input_image: (B, 3, H, W) Original input in [0, 255]
        """
        # Normalize shading to [0, 1] for multiplication
        s_normalized = s_pred / 255.0
        
        # Element-wise multiplication with broadcasting
        reconstructed_image = r_pred * s_normalized
        
        return self.mse_loss(reconstructed_image, input_image)
    
    def forward(self, r_pred, s_pred, r_gt, s_gt, input_image):
        """
        Compute combined loss.
        All inputs should be in [0, 255] range.
        
        Returns:
            total_loss: Combined loss (scalar tensor)
            loss_dict: Dictionary with individual loss components
        """
        # Reconstruction losses (Equation 10-11)
        loss_r = self.mse_loss(r_pred, r_gt)
        loss_s = self.mse_loss(s_pred, s_gt)
        
        # Image formation loss (Equation 12)
        loss_imf = self.image_formation_loss(r_pred, s_pred, input_image)
        
        # Combined loss (Equation 13)
        total_loss = (self.gamma_r * loss_r + 
                     self.gamma_s * loss_s + 
                     self.gamma_imf * loss_imf)
        
        loss_dict = {
            'loss_r': loss_r.item(),
            'loss_s': loss_s.item(),
            'loss_imf': loss_imf.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


if __name__ == "__main__":
    print("Testing IntrinsicLoss with [0, 255] range...")
    
    batch_size = 4
    h, w = 120, 160
    
    r_pred = torch.rand(batch_size, 3, h, w) * 255
    s_pred = torch.rand(batch_size, 1, h, w) * 255
    r_gt = torch.rand(batch_size, 3, h, w) * 255
    s_gt = torch.rand(batch_size, 1, h, w) * 255
    
    # Create input as R_gt x (S_gt/255)
    input_img = r_gt * (s_gt / 255.0)
    
    criterion = IntrinsicLoss(gamma_r=1.0, gamma_s=1.0, gamma_imf=1.0)
    loss, loss_dict = criterion(r_pred, s_pred, r_gt, s_gt, input_img)
    
    print("\nLoss components:")
    for key, val in loss_dict.items():
        print(f"  {key}: {val:.4f}")
    
    print(f"\nValue ranges:")
    print(f"  r_pred: [{r_pred.min():.1f}, {r_pred.max():.1f}]")
    print(f"  s_pred: [{s_pred.min():.1f}, {s_pred.max():.1f}]")
    print(f"  input: [{input_img.min():.1f}, {input_img.max():.1f}]")
    
    print("\nLoss computation successful!")