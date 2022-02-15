import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.io import read_image
from tqdm import tqdm
import os, sys
import pandas as pd
from glob import glob

from torch.utils.data import Dataset, DataLoader

from torch import optim
from Yolov2 import Yolov2
from loss import loss

from tensorboardX import SummaryWriter

import config as cfg

class YoloDataset(Dataset):
	def __init__(self, training=False):
		self.labels = pd.read_csv(cfg.reef_data_path + 'train_new.csv', names=['file_name', 'gt_bbox'])
		self.img_dir = cfg.reef_data_path + '/train_images_new/'
		if training:
			self.transform = transforms.Compose([
				transforms.Resize((224, 224)),
				transforms.RandomHorizontalFlip(),
				transforms.ConvertImageDtype(torch.float),
				transforms.Normalize(mean=[67.0/255, 145.0/255, 157.0/255],\
				std=[24.0/255, 8.31/255, 25.0/255]),
			])
		else:
			self.transform = transforms.Compose([
				transforms.Resize((224, 224)),
				transforms.ConvertImageDtype(torch.float),
				transforms.Normalize(mean=[67.0/255, 145.0/255, 157.0/255],\
				std=[24.0/255, 8.31/255, 25.0/255]),
			])

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
		img= read_image(img_path)
		img = self.transform(img)
		gt_bbox = self.labels.iloc[idx, 1]

		return img, gt_bbox


# Train with save?
def train():
	# Load data

	model = Yolov2().cuda()

	start_epoch = 0
	# load model parameters
	if False:
	#if len(glob('*'+cfg.save_dir)) != 0:
		checkpoint = torch.load(glob('*'+cfg.save_dir)[-1])
		model.load_state_dict(checkpoint['state_dict'])
		start_epoch = checkpoint['epoch']
	print(start_epoch)
	optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
	optimizer.zero_grad()

	train_dataset = YoloDataset(True)
	train_loader = DataLoader(train_dataset, batch_size = cfg.batch_size, shuffle=True)

	for epoch in range(start_epoch, cfg.epoch):
		
		# for batch_data in my_batch_loader():
		for i, (images, target) in tqdm(enumerate(train_loader)):
			images = images.cuda()

			output = model(images)
			loss_ = loss(output, target)

			optimizer.zero_grad()
			loss_.backward()
			optimizer.step()

		torch.save({
			'epoch': epoch,
			'state_dict': model.state_dict(),
			'optimizer' : optimizer.state_dict(),
		}, f'epoch{epoch}_'+cfg.save_dir)

	
if __name__ == '__main__':
	train()