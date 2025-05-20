from Network.MBblock import MBConvBlock
import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAMEfficientNet(nn.Module):
    def __init__(self, num_classes=2, attack = False):
        super(CBAMEfficientNet, self).__init__()
        self.attack = attack
        # Stem
        self.shallow = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False), #299x299 -> 150x150
            nn.BatchNorm2d(32),
            nn.SiLU(),
            MBConvBlock(32, 16, kernel_size = 3, stride=1), #150x150 -> 75x75

            MBConvBlock(16, 24, kernel_size = 3, stride=2), #75x75 -> 75x75
            MBConvBlock(24, 24, kernel_size = 3, stride=1), #75x75 -> 75x75
        )
        
        self.deep = nn.Sequential(

            MBConvBlock(24, 40, kernel_size=5, stride=2, padding=2), #75x75 -> 38x38
            MBConvBlock(40, 40, kernel_size=5, stride=1, padding=2), #38x38 -> 38x38

            MBConvBlock(40, 80, kernel_size=3, stride=2), #38x38 -> 19x19
            MBConvBlock(80, 80, kernel_size=3, stride=1), #19x19 -> 19x19
            MBConvBlock(80, 80, kernel_size=3, stride=1), #19x19 -> 19x19

            MBConvBlock(80, 112, kernel_size=5, stride=2, padding=2), #19x19 -> 10x10
            MBConvBlock(112, 112, kernel_size=5, stride=1, padding=2), #10x10 -> 10x10
            MBConvBlock(112, 112, kernel_size=5, stride=1, padding=2), #10x10 -> 10x10

            MBConvBlock(112, 192, kernel_size=5, stride=2, padding=2), #10x10 -> 5x5
            MBConvBlock(192, 192, kernel_size=5, stride=1, padding=2), #5x5 -> 5x5
            MBConvBlock(192, 192, kernel_size=5, stride=1, padding=2), #5x5 -> 5x5
            MBConvBlock(192, 192, kernel_size=5, stride=1, padding=2), #5x5 -> 5x5

            MBConvBlock(192, 320, kernel_size=5, stride=1), #5x5 -> 1x1
            nn.Conv2d(320, 1280, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        
        # Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        shallow = self.shallow(x)
        x = self.deep(shallow)
        x = self.classifier(x)
        if self.attack:
            return x
        return x, shallow
    
if __name__ == '__main__':
    model = CBAMEfficientNet()
    x = torch.randn(4, 3, 299, 299)
    out = model(x)
    print(out)