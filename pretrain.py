import config as cfg
from Darknet import Darknet19

from tqdm import tqdm
import shutil
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim

import torchvision.transforms as transforms
import torchvision.datasets as datasets

def train(train_loader, model, criterion, epoch):
	model.train()

	for i, (images, target) in tqdm(enumerate(train_loader)):
		images = images.cuda()
		target = target.view(-1, 1).cuda()

		output = model(images)
		if i % 1000 == 0:
			print(f'output={output.shape}\ntarget={target.shape}')
		loss = criterion(output, target)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

def save_checkpoint(state, filename='checkpoint.pth.tar'):
	torch.save(state, filename)

train_path = cfg.data_path + 'train/'
val_path  = cfg.data_path + 'val/'
test_path = cfg.data_path + 'test/'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],\
	std=[0.229, 0.224, 0.225])


train_dataset = datasets.ImageFolder(train_path,\
	transforms.Compose([
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalize
	]))

train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
shuffle = True, pin_memory=True)

model = Darknet19().cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=cfg.lr,\
		 momentum=cfg.momentum, weight_decay=cfg.weight_decay)

for epoch in range(1, 1+cfg.epoch):
	train(train_loader, model, criterion, epoch)
	
	save_checkpoint({
			'epoch': epoch,
			'state_dict': model.state_dict(),
			'optimizer' : optimizer.state_dict(),
		})
