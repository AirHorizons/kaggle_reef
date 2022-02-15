import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import numpy as np

def conv_bn_relu(in_channels, out_channels, kernel_size):
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2))
    layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU())

    return layers

class Darknet19(nn.Module):
  layer_config = {
      'layers_0' : [32],
      'layers_1' : ['P', 64],
      'layers_2' : ['P', 128, 64, 128],
      'layers_3' : ['P', 256, 128, 256],
      'layers_4' : ['P', 512, 256, 512, 256, 512],
      'layers_5' : ['P', 1024, 512, 1024, 512, 1024],
  }

  def __init__(self):
    super(Darknet19, self).__init__()

    self.in_channels = 3

    self.layer_0 = self._create_layer(self.layer_config['layers_0'])
    self.layer_1 = self._create_layer(self.layer_config['layers_1'])
    self.layer_2 = self._create_layer(self.layer_config['layers_2'])
    self.layer_3 = self._create_layer(self.layer_config['layers_3'])
    self.layer_4 = self._create_layer(self.layer_config['layers_4'])
    self.layer_5 = self._create_layer(self.layer_config['layers_5'])

    self.conv_last = nn.Conv2d(self.in_channels, 1000, kernel_size=1, stride=1)
    self.pool_last = nn.AvgPool2d((7, 7))
    self.softmax = nn.Softmax()

  def forward(self, x):
    x = self.layer_0(x)
    x = self.layer_1(x)
    x = self.layer_2(x)
    x = self.layer_3(x)
    x = self.layer_4(x)
    x = self.layer_5(x)
    x = self.conv_last(x)
    x = self.pool_last(x)
    x = self.softmax(x)
    x = x.view(-1, 1000, 1)

    return x

  def _create_layer(self, config):
    layers = []
    kernel_size = 3

    for filters in config:
      if filters == 'P':
        layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
      else:
        layers += conv_bn_relu(self.in_channels, filters, kernel_size)
        kernel_size = 3 if kernel_size != 3 else 1
        self.in_channels = filters
    return nn.Sequential(*layers)

if __name__ == '__main__':
	your_model = Darknet19().cuda()
	summary(your_model, input_size=(3, 224, 224))