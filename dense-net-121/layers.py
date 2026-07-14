import torch.nn as nn
import torch
from layers import DenseLayer

class DenseLayer(nn.Sequential):
  def __init__(self,num_input_features,growth_rate,bn_size,drop_rate):
    super().__init__()
    self.add_module('norm1',nn.BatchNorm2d(num_input_features))
    self.add_module('relu1',nn.ReLU(inplace=True))
    self.add_module('conv1',nn.Conv2d(num_input_features,bn_size*growth_rate,kernel_size=1,stride=1,bias=False))
    self.add_module('norm2',nn.BatchNorm2d(bn_size*growth_rate))
    self.add_module('relu2',nn.ReLU(inplace=True))
    self.add_module('conv2',nn.Conv2d(bn_size*growth_rate,growth_rate,kernel_size=3,stride=1,padding=1,bias=False)) # Changed stride from 2 to 1
    self.drop_rate=drop_rate
  def forward(self,x):
    new_features=super().forward(x)
    if self.drop_rate>0:
      new_features.F.dropout(new_features,p=self.drop_rate,training=self.training)
    return torch.cat([x,new_features],1)

class DenseBlock(nn.Sequential):
  def __init__(self,num_layers,num_input_features,bn_size,growth_rate,drop_rate):
    super().__init__()
    for i in range(num_layers):
      layer=DenseLayer(num_input_features+i*growth_rate,growth_rate,bn_size,drop_rate)
      self.add_module('denselayer%d'%(i+1),layer)

class Transition(nn.Sequential):
  def __init__(self,num_input_features,num_output_features):
    super().__init__()
    self.add_module('norm',nn.BatchNorm2d(num_input_features))
    self.add_module('relu',nn.ReLU(inplace=True))
    self.add_module('conv',nn.Conv2d(num_input_features,num_output_features,
                                     kernel_size=1,stride=1,bias=False)) # Changed to 1x1 convolution
    self.add_module('pool',nn.AvgPool2d(kernel_size=2,stride=2))