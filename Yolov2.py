import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.autograd import Variable

from Darknet import Darknet19, conv_bn_relu
import config as cfg

class Reorg(nn.Module):
	def __init__(self):
		super(Reorg, self).__init__()

	def forward(self, x):
		Batch, Channel, Height, Width = x.size()
		x = x.view(Batch, Channel, Height//2, 2, Width//2, 2).transpose(3, 4).contiguous()
		x = x.view(Batch, Channel, Height * Width //4, 4).transpose(2, 3).contiguous()
		x = x.view(Batch, Channel, 4, Height//2, Width//2).transpose(1, 2).contiguous()
		x = x.view(Batch, 4*Channel, Height//2, Width//2)

		return x

class Yolov2(nn.Module):
	def __init__(self, backbone_path='./e4_checkpoint.pth.tar'):
		super(Yolov2, self).__init__()
		# backbone: filename of loaded model, Darknet19
		self.backbone = Darknet19().cuda()
		self.backbone.load_state_dict(torch.load(backbone_path)['state_dict'])

		layers = [self.backbone.layer_0, self.backbone.layer_1, self.backbone.layer_2,\
			 self.backbone.layer_3, self.backbone.layer_4, self.backbone.layer_5]

		self.layer_0 = nn.Sequential(*layers[:-1])
		self.layer_1 = layers[-1]
		self.layer_2 = nn.Sequential(*(conv_bn_relu(1024, 1024, 3) + conv_bn_relu(1024, 1024, 3)))
		self.reorg = Reorg()
		self.downsample = nn.Sequential(*conv_bn_relu(512, 64, 1))
		self.layer_3 = nn.Sequential(*conv_bn_relu(1280, 1024, 3), nn.Conv2d(1024, 25, kernel_size=1, stride=1))

	def forward(self, x):
		x = self.layer_0(x) # 512 * 14 * 14
		shortcut = self.downsample(x) # 64 * 14 * 14
		shortcut = self.reorg(shortcut) # 256 * 7 * 7
		x = self.layer_1(x) # 1024 * 7 * 7
		x = self.layer_2(x) # 1024 * 7 * 7
		x = torch.cat((x, shortcut), 1) # 1280 * 7 * 7
		x = self.layer_3(x) # -1 * 25 * 7 * 7

		Batch, Box, Height, Width = x.size()

		x = x.permute(0, 2, 3, 1).contiguous() # Batch, Height, Width, Box
		x = x.view(Batch, Height * Width * 5, Box//5) # last dimension consist of box info

		# x, y, w, h, conf
		pos = torch.sigmoid(x[:,:,0:2])
		size = torch.exp(x[:,:,2:4])
		confidence = torch.sigmoid(x[:,:,4:5])

		return pos, size, confidence

if __name__ == '__main__':
	your_model = Yolov2().cuda()
	# summary(your_model, input_size=(3, 224, 224))
	rand_input = Variable(torch.from_numpy(np.random.randn(1, 3, 224, 224)).float()).cuda()
	out = your_model(rand_input)
	pos, size, confidence = out
	print(pos.size())
	print(size.size())
	print(confidence.size())