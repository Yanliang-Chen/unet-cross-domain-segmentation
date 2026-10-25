"""Lightweight binary segmentation U-Net."""
import torch
from torch import nn
from torch.nn import functional as F

class DoubleConv(nn.Sequential):
    def __init__(self, ci, co):
        super().__init__(
            nn.Conv2d(ci,co,3,padding=1,bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.Conv2d(co,co,3,padding=1,bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))

class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=16):
        super().__init__()
        b=base_channels
        self.enc1=DoubleConv(in_channels,b)
        self.enc2=DoubleConv(b,2*b)
        self.enc3=DoubleConv(2*b,4*b)
        self.enc4=DoubleConv(4*b,8*b)
        self.bottleneck=DoubleConv(8*b,16*b)
        self.pool=nn.MaxPool2d(2)
        self.dec4=DoubleConv(24*b,8*b)
        self.dec3=DoubleConv(12*b,4*b)
        self.dec2=DoubleConv(6*b,2*b)
        self.dec1=DoubleConv(3*b,b)
        self.out=nn.Conv2d(b,1,1)

    @staticmethod
    def upcat(low,skip):
        return torch.cat([skip,F.interpolate(low,size=skip.shape[2:],mode="bilinear",align_corners=False)],dim=1)

    def forward(self,x):
        x1=self.enc1(x)
        x2=self.enc2(self.pool(x1))
        x3=self.enc3(self.pool(x2))
        x4=self.enc4(self.pool(x3))
        z=self.bottleneck(self.pool(x4))
        z=self.dec4(self.upcat(z,x4))
        z=self.dec3(self.upcat(z,x3))
        z=self.dec2(self.upcat(z,x2))
        z=self.dec1(self.upcat(z,x1))
        return self.out(z)
