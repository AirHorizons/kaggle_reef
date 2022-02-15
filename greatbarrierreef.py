import torch
from glob import glob
import Yolov2

class Env:
	def __init__(self):
		if torch.cuda.is_available():
			device='cuda'
		else:
			device='cpu'
		model = Yolov2().to(device)
		checkpoint = torch.load(glob('*.pth')[-1])
		self.valid = checkpoint is not None
		if self.valid:
			model.load_state_dict(checkpoint['state_dict'])

	def iter_test():
		pass

make_env = lambda: Env()

