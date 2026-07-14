import torch.nn as nn

class ConvBlock(nn.Module):
  def __init__(self,in_channels,out_channels):
    super(ConvBlock,self).__init__()
    self.block=nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.3,inplace=True),
        nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.3,inplace=True)
    )
  def forward(self,x):
    return self.block(x)