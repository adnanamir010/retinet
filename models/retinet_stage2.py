import torch
import torch.nn as nn

class RetiNet_Stage2(nn.Module):
    """
    Stage 2: Gradient Reintegration Network
    
    Takes original RGB image + predicted gradients from Stage 1
    and reconstructs final albedo and shading images.
    
    Two separate shallow networks:
    - Albedo stream: RGB + Albedo_grad (6 channels) -> Albedo (3 channels)
    - Shading stream: RGB + Shading_grad (4 channels) -> Shading (1 channel)
    """
    
    def __init__(self, use_dropout=True):
        super(RetiNet_Stage2, self).__init__()
        self.use_dropout = use_dropout
        
        # Albedo reconstruction stream (RGBA = 6 channels)
        self.albedo_conv1 = self._conv_bn_lrelu(6, 64)
        self.albedo_conv2 = self._conv_bn_lrelu(64, 128)
        self.albedo_conv3 = self._conv_bn_lrelu(128, 128)
        self.albedo_dropout = nn.Dropout2d(p=0.5)
        self.albedo_conv4 = self._conv_bn_lrelu(128, 64)
        self.albedo_conv5 = nn.Conv2d(64, 3, 3, padding=1)
        
        # Shading reconstruction stream (RGBS = 4 channels)
        self.shading_conv1 = self._conv_bn_lrelu(4, 64)
        self.shading_conv2 = self._conv_bn_lrelu(64, 128)
        self.shading_conv3 = self._conv_bn_lrelu(128, 128)
        self.shading_dropout = nn.Dropout2d(p=0.5)
        self.shading_conv4 = self._conv_bn_lrelu(128, 64)
        self.shading_conv5 = nn.Conv2d(64, 1, 3, padding=1)
        
        self._init_weights()
    
    def _conv_bn_lrelu(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch, momentum=0.1, eps=1e-5),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, rgb, albedo_grad, shading_grad):
        # Albedo stream
        rgba = torch.cat([rgb, albedo_grad], dim=1)
        
        albedo = self.albedo_conv1(rgba)
        albedo = self.albedo_conv2(albedo)
        albedo = self.albedo_conv3(albedo)
        
        if self.use_dropout and self.training:
            albedo = self.albedo_dropout(albedo)
        
        albedo = self.albedo_conv4(albedo)
        albedo = self.albedo_conv5(albedo)
        
        # Shading stream
        rgbs = torch.cat([rgb, shading_grad], dim=1)
        
        shading = self.shading_conv1(rgbs)
        shading = self.shading_conv2(shading)
        shading = self.shading_conv3(shading)
        
        if self.use_dropout and self.training:
            shading = self.shading_dropout(shading)
        
        shading = self.shading_conv4(shading)
        shading = self.shading_conv5(shading)
        
        albedo = torch.clamp(albedo, 0.0, 255.0)
        shading = torch.clamp(shading, 0.0, 255.0)
        
        return albedo, shading


if __name__ == "__main__":
    print("Testing RetiNet_Stage2...")
    
    model = RetiNet_Stage2(use_dropout=True)
    model.eval()
    
    rgb = torch.rand(2, 3, 120, 160) * 255.0
    albedo_grad = torch.rand(2, 3, 120, 160) * 100.0
    shading_grad = torch.rand(2, 1, 120, 160) * 100.0
    
    albedo, shading = model(rgb, albedo_grad, shading_grad)
    
    print(f"RGB: {rgb.shape}")
    print(f"Albedo grad: {albedo_grad.shape}")
    print(f"Shading grad: {shading_grad.shape}")
    print(f"Albedo output: {albedo.shape}, range: [{albedo.min():.1f}, {albedo.max():.1f}]")
    print(f"Shading output: {shading.shape}, range: [{shading.min():.1f}, {shading.max():.1f}]")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print("Test passed!")