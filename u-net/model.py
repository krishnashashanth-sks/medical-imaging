import torch.nn as nn
from layers import ConvBlock
import torch

class UNet(nn.Module):
  def __init__(self,in_channels,num_classes):
    super(UNet,self).__init__()
    self.down1=ConvBlock(in_channels,64)
    self.pool1=nn.MaxPool2d(kernel_size=2,stride=2)
    self.down2=ConvBlock(64,128)
    self.pool2=nn.MaxPool2d(kernel_size=2,stride=2)
    self.down3=ConvBlock(128,256)
    self.pool3=nn.MaxPool2d(kernel_size=2,stride=2)
    self.down4=ConvBlock(256,512)
    self.pool4=nn.MaxPool2d(kernel_size=2,stride=2) # Changed kernel_size from 3 to 2

    self.bottleneck=ConvBlock(512,1024)

    self.upconv4=nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
    self.up4=ConvBlock(1024,512)
    self.upconv3=nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
    self.up3=ConvBlock(512,256)
    self.upconv2=nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
    self.up2=ConvBlock(256,128)
    self.upconv1=nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
    self.up1=ConvBlock(128,64)
    self.out_conv=nn.Conv2d(64,num_classes,kernel_size=1)
  def forward(self,x):
    d1=self.down1(x)
    p1=self.pool1(d1)
    d2=self.down2(p1)
    p2=self.pool2(d2)
    d3=self.down3(p2)
    p3=self.pool3(d3)
    d4=self.down4(p3)
    p4=self.pool4(d4)

    bottleneck=self.bottleneck(p4) # Changed d4 to p4 here to align with original UNet design

    u4=self.upconv4(bottleneck)
    u4=torch.cat([u4,d4],dim=1)
    u4=self.up4(u4)
    u3=self.upconv3(u4)
    u3=torch.cat([u3,d3],dim=1)
    u3=self.up3(u3)
    u2=self.upconv2(u3)
    u2=torch.cat([u2,d2],dim=1)
    u2=self.up2(u2)
    u1=self.upconv1(u2)
    u1=torch.cat([u1,d1],dim=1)
    u1=self.up1(u1)

    output=self.out_conv(u1)
    return output