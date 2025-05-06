import torch
from torch import nn 

#CBAM paper: https://arxiv.org/pdf/1807.06521
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        # Channel attention
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Spatial attention
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False), #shape -> shape
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # Channel attention
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        channel_att = torch.sigmoid(self.mlp(avg_out) + self.mlp(max_out)).view(b, c, 1, 1)
        x = x * channel_att

        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = self.spatial(spatial_att)
        x = x * spatial_att

        return x


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size =3, stride=1, padding = 1, expand_ratio=6, reduction=16):
        super(MBConvBlock, self).__init__()
        mid_channels = in_channels * expand_ratio
        assert in_channels * expand_ratio > reduction
        self.use_residual = (in_channels == out_channels) and (stride == 1)

        self.expand_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False) #shape -> shape
        self.bn0 = nn.BatchNorm2d(mid_channels)
        self.dwconv = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False) 
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.cbam = CBAM(mid_channels, reduction=reduction)
        
        self.project_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        identity = x
        out = self.activation(self.bn0(self.expand_conv(x)))
        out = self.activation(self.bn1(self.dwconv(out)))
        out = self.cbam(out)
        out = self.bn2(self.project_conv(out))
        if self.use_residual:
            out += identity
        return out
    
if __name__ == '__main__':
    model = MBConvBlock(3, 64, 1)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print(out.shape) 