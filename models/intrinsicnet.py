import torch
import torch.nn as nn
import torch.nn.functional as F

class IntrinsicNet(nn.Module):
    """
    IntrinsicNet pytorch implementation.
    Simple encoder-decoder without gradients.
    Input: RGB (3 channels) in [0, 255]
    Output: Reflectance (3 channels) + Shading (1 channel) in [0, 255]
    """
    
    def __init__(self, use_dropout=True):
        super(IntrinsicNet, self).__init__()
        self.use_dropout = use_dropout
        
        # Encoder
        self.conv1_1 = self._conv_bn_lrelu(3, 64)
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
        
        # Decoder - Albedo
        self.unpool5 = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1)
        self.deconv5_3 = self._conv_bn_lrelu(512, 512)
        self.deconv5_2 = self._conv_bn_lrelu(512, 512)
        self.deconv5_1 = self._conv_bn_lrelu(512, 512)
        self.dropout5_dec = nn.Dropout2d(p=0.5)
        
        self.unpool4 = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1)
        self.deconv4_3 = self._conv_bn_lrelu(1024, 512)
        self.deconv4_2 = self._conv_bn_lrelu(512, 512)
        self.deconv4_1 = self._conv_bn_lrelu(512, 256)
        self.dropout4_dec = nn.Dropout2d(p=0.5)
        
        self.unpool3 = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1)
        self.deconv3_3 = self._conv_bn_lrelu(512, 256)
        self.deconv3_2 = self._conv_bn_lrelu(256, 256)
        self.deconv3_1 = self._conv_bn_lrelu(256, 128)
        self.dropout3_dec = nn.Dropout2d(p=0.5)
        
        self.unpool2 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        self.deconv2_2 = self._conv_bn_lrelu(256, 128)
        self.deconv2_1 = self._conv_bn_lrelu(128, 64)
        
        self.unpool1 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.deconv1_2 = self._conv_bn_lrelu(128, 64)
        self.deconv1_1 = nn.Conv2d(64, 3, 3, padding=1)
        
        # Decoder - Shading
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
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _match_size(self, x, target):
        """Pad or crop x to match target spatial dimensions."""
        if x.shape[2:] != target.shape[2:]:
            diff_h = target.shape[2] - x.shape[2]
            diff_w = target.shape[3] - x.shape[3]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                         diff_h // 2, diff_h - diff_h // 2])
        return x
    
    def forward(self, x):
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
        
        # Decoder - Albedo
        albedo = self.unpool5(x)
        albedo = self.deconv5_3(albedo)
        albedo = self.deconv5_2(albedo)
        albedo = self.deconv5_1(albedo)
        if self.use_dropout and self.training:
            albedo = self.dropout5_dec(albedo)
        
        albedo = self.unpool4(albedo)
        albedo = self._match_size(albedo, skip4)
        albedo = torch.cat([albedo, skip4], dim=1)
        albedo = self.deconv4_3(albedo)
        albedo = self.deconv4_2(albedo)
        albedo = self.deconv4_1(albedo)
        if self.use_dropout and self.training:
            albedo = self.dropout4_dec(albedo)
        
        albedo = self.unpool3(albedo)
        albedo = self._match_size(albedo, skip3)
        albedo = torch.cat([albedo, skip3], dim=1)
        albedo = self.deconv3_3(albedo)
        albedo = self.deconv3_2(albedo)
        albedo = self.deconv3_1(albedo)
        if self.use_dropout and self.training:
            albedo = self.dropout3_dec(albedo)
        
        albedo = self.unpool2(albedo)
        albedo = self._match_size(albedo, skip2)
        albedo = torch.cat([albedo, skip2], dim=1)
        albedo = self.deconv2_2(albedo)
        albedo = self.deconv2_1(albedo)
        
        albedo = self.unpool1(albedo)
        albedo = self._match_size(albedo, skip1)
        albedo = torch.cat([albedo, skip1], dim=1)
        albedo = self.deconv1_2(albedo)
        albedo = self.deconv1_1(albedo)
        
        # Decoder - Shading
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
        
        albedo = torch.clamp(albedo, 0.0, 255.0)
        shading = torch.clamp(shading, 0.0, 255.0)
        
        return albedo, shading


if __name__ == "__main__":
    print("Testing IntrinsicNet...")
    
    model = IntrinsicNet(use_dropout=True)
    model.eval()
    
    x = torch.rand(2, 3, 120, 160) * 255.0
    albedo, shading = model(x)
    
    print(f"Input: {x.shape}, range: [{x.min():.1f}, {x.max():.1f}]")
    print(f"Albedo: {albedo.shape}, range: [{albedo.min():.1f}, {albedo.max():.1f}]")
    print(f"Shading: {shading.shape}, range: [{shading.min():.1f}, {shading.max():.1f}]")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print("Test passed!")