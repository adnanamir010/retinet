import torch
import torch.nn as nn
import torch.nn.functional as F

class RetiNet_Stage1(nn.Module):
    """
    Stage 1: Gradient Separation Network
    
    Input: RGB (3) + RGB gradients (3) = 6 channels
    Output: Albedo gradients (3) + Shading gradients (1)
    
    VGG16-style encoder-decoder with skip connections.
    Outputs are clipped to [0, 360.63] as per original implementation.
    """
    
    def __init__(self, use_dropout=True):
        super(RetiNet_Stage1, self).__init__()
        self.use_dropout = use_dropout
        
        # Encoder (6 input channels)
        self.conv1_1 = self._conv_bn_lrelu(6, 64)
        self.conv1_2 = self._conv_bn_lrelu(64, 64)
        self.pool1 = self._pool_layer(64, 64)
        
        self.conv2_1 = self._conv_bn_lrelu(64, 128)
        self.conv2_2 = self._conv_bn_lrelu(128, 128)
        self.pool2 = self._pool_layer(128, 128)
        
        self.conv3_1 = self._conv_bn_lrelu(128, 256)
        self.conv3_2 = self._conv_bn_lrelu(256, 256)
        self.conv3_3 = self._conv_bn_lrelu(256, 256)
        self.pool3 = self._pool_layer(256, 256)
        self.dropout3 = nn.Dropout2d(p=0.5)
        
        self.conv4_1 = self._conv_bn_lrelu(256, 512)
        self.conv4_2 = self._conv_bn_lrelu(512, 512)
        self.conv4_3 = self._conv_bn_lrelu(512, 512)
        self.pool4 = self._pool_layer(512, 512)
        self.dropout4 = nn.Dropout2d(p=0.5)
        
        self.conv5_1 = self._conv_bn_lrelu(512, 512)
        self.conv5_2 = self._conv_bn_lrelu(512, 512)
        self.conv5_3 = self._conv_bn_lrelu(512, 512)
        self.pool5 = self._pool_layer(512, 512)
        self.dropout5 = nn.Dropout2d(p=0.5)
        
        # Decoder - Albedo gradients
        self.unpool5_albedo = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1)
        self.deconv5_3_albedo = self._conv_bn_lrelu(512, 512)
        self.deconv5_2_albedo = self._conv_bn_lrelu(512, 512)
        self.deconv5_1_albedo = self._conv_bn_lrelu(512, 512)
        self.dropout5_albedo = nn.Dropout2d(p=0.5)
        
        self.unpool4_albedo = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1)
        self.deconv4_3_albedo = self._conv_bn_lrelu(1024, 512)
        self.deconv4_2_albedo = self._conv_bn_lrelu(512, 512)
        self.deconv4_1_albedo = self._conv_bn_lrelu(512, 256)
        self.dropout4_albedo = nn.Dropout2d(p=0.5)
        
        self.unpool3_albedo = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1)
        self.deconv3_3_albedo = self._conv_bn_lrelu(512, 256)
        self.deconv3_2_albedo = self._conv_bn_lrelu(256, 256)
        self.deconv3_1_albedo = self._conv_bn_lrelu(256, 128)
        self.dropout3_albedo = nn.Dropout2d(p=0.5)
        
        self.unpool2_albedo = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        self.deconv2_2_albedo = self._conv_bn_lrelu(256, 128)
        self.deconv2_1_albedo = self._conv_bn_lrelu(128, 64)
        
        self.unpool1_albedo = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.deconv1_2_albedo = self._conv_bn_lrelu(128, 64)
        self.deconv1_1_albedo = nn.Conv2d(64, 3, 3, padding=1)
        
        # Decoder - Shading gradients
        self.unpool5_shading = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1)
        self.deconv5_3_shading = self._conv_bn_lrelu(512, 512)
        self.deconv5_2_shading = self._conv_bn_lrelu(512, 512)
        self.deconv5_1_shading = self._conv_bn_lrelu(512, 512)
        self.dropout5_shading = nn.Dropout2d(p=0.5)
        
        self.unpool4_shading = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1)
        self.deconv4_3_shading = self._conv_bn_lrelu(1024, 512)
        self.deconv4_2_shading = self._conv_bn_lrelu(512, 512)
        self.deconv4_1_shading = self._conv_bn_lrelu(512, 256)
        self.dropout4_shading = nn.Dropout2d(p=0.5)
        
        self.unpool3_shading = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1)
        self.deconv3_3_shading = self._conv_bn_lrelu(512, 256)
        self.deconv3_2_shading = self._conv_bn_lrelu(256, 256)
        self.deconv3_1_shading = self._conv_bn_lrelu(256, 128)
        self.dropout3_shading = nn.Dropout2d(p=0.5)
        
        self.unpool2_shading = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        self.deconv2_2_shading = self._conv_bn_lrelu(256, 128)
        self.deconv2_1_shading = self._conv_bn_lrelu(128, 64)
        
        self.unpool1_shading = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.deconv1_2_shading = self._conv_bn_lrelu(128, 64)
        self.deconv1_1_shading = nn.Conv2d(64, 1, 3, padding=1)
        
        self._init_weights()
    
    def _conv_bn_lrelu(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch, momentum=0.1, eps=1e-5),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    def _pool_layer(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.1, eps=1e-5),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _match_size(self, x, target):
        if x.shape[2:] != target.shape[2:]:
            diff_h = target.shape[2] - x.shape[2]
            diff_w = target.shape[3] - x.shape[3]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                         diff_h // 2, diff_h - diff_h // 2])
        return x
    
    def forward(self, rgb, rgb_grad):
        # Concatenate RGB and gradients
        x = torch.cat([rgb, rgb_grad], dim=1)
        
        # Encoder
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        skip1 = x
        x = self.pool1(x)
        
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        skip2 = x
        x = self.pool2(x)
        
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        skip3 = x
        x = self.pool3(x)
        if self.use_dropout and self.training:
            x = self.dropout3(x)
        
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        skip4 = x
        x = self.pool4(x)
        if self.use_dropout and self.training:
            x = self.dropout4(x)
        
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)
        if self.use_dropout and self.training:
            x = self.dropout5(x)
        
        # Decoder - Albedo gradients
        albedo = self.unpool5_albedo(x)
        albedo = self.deconv5_3_albedo(albedo)
        albedo = self.deconv5_2_albedo(albedo)
        albedo = self.deconv5_1_albedo(albedo)
        if self.use_dropout and self.training:
            albedo = self.dropout5_albedo(albedo)
        
        albedo = self.unpool4_albedo(albedo)
        albedo = self._match_size(albedo, skip4)
        albedo = torch.cat([albedo, skip4], dim=1)
        albedo = self.deconv4_3_albedo(albedo)
        albedo = self.deconv4_2_albedo(albedo)
        albedo = self.deconv4_1_albedo(albedo)
        if self.use_dropout and self.training:
            albedo = self.dropout4_albedo(albedo)
        
        albedo = self.unpool3_albedo(albedo)
        albedo = self._match_size(albedo, skip3)
        albedo = torch.cat([albedo, skip3], dim=1)
        albedo = self.deconv3_3_albedo(albedo)
        albedo = self.deconv3_2_albedo(albedo)
        albedo = self.deconv3_1_albedo(albedo)
        if self.use_dropout and self.training:
            albedo = self.dropout3_albedo(albedo)
        
        albedo = self.unpool2_albedo(albedo)
        albedo = self._match_size(albedo, skip2)
        albedo = torch.cat([albedo, skip2], dim=1)
        albedo = self.deconv2_2_albedo(albedo)
        albedo = self.deconv2_1_albedo(albedo)
        
        albedo = self.unpool1_albedo(albedo)
        albedo = self._match_size(albedo, skip1)
        albedo = torch.cat([albedo, skip1], dim=1)
        albedo = self.deconv1_2_albedo(albedo)
        albedo = self.deconv1_1_albedo(albedo)
        
        # Decoder - Shading gradients
        shading = self.unpool5_shading(x)
        shading = self.deconv5_3_shading(shading)
        shading = self.deconv5_2_shading(shading)
        shading = self.deconv5_1_shading(shading)
        if self.use_dropout and self.training:
            shading = self.dropout5_shading(shading)
        
        shading = self.unpool4_shading(shading)
        shading = self._match_size(shading, skip4)
        shading = torch.cat([shading, skip4], dim=1)
        shading = self.deconv4_3_shading(shading)
        shading = self.deconv4_2_shading(shading)
        shading = self.deconv4_1_shading(shading)
        if self.use_dropout and self.training:
            shading = self.dropout4_shading(shading)
        
        shading = self.unpool3_shading(shading)
        shading = self._match_size(shading, skip3)
        shading = torch.cat([shading, skip3], dim=1)
        shading = self.deconv3_3_shading(shading)
        shading = self.deconv3_2_shading(shading)
        shading = self.deconv3_1_shading(shading)
        if self.use_dropout and self.training:
            shading = self.dropout3_shading(shading)
        
        shading = self.unpool2_shading(shading)
        shading = self._match_size(shading, skip2)
        shading = torch.cat([shading, skip2], dim=1)
        shading = self.deconv2_2_shading(shading)
        shading = self.deconv2_1_shading(shading)
        
        shading = self.unpool1_shading(shading)
        shading = self._match_size(shading, skip1)
        shading = torch.cat([shading, skip1], dim=1)
        shading = self.deconv1_2_shading(shading)
        shading = self.deconv1_1_shading(shading)
        
        albedo_grad = torch.clamp(albedo, 0.0, 360.63)
        shading_grad = torch.clamp(shading, 0.0, 360.63)
        
        return albedo_grad, shading_grad


if __name__ == "__main__":
    print("Testing RetiNet_Stage1...")
    
    model = RetiNet_Stage1(use_dropout=True)
    model.eval()
    
    rgb = torch.rand(2, 3, 120, 160) * 255.0
    rgb_grad = torch.rand(2, 3, 120, 160) * 100.0
    
    a_grad, s_grad = model(rgb, rgb_grad)
    
    print(f"RGB: {rgb.shape}")
    print(f"RGB grad: {rgb_grad.shape}")
    print(f"Albedo grad: {a_grad.shape}, range: [{a_grad.min():.1f}, {a_grad.max():.1f}]")
    print(f"Shading grad: {s_grad.shape}, range: [{s_grad.min():.1f}, {s_grad.max():.1f}]")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print("Test passed!")