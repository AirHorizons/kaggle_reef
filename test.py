import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

def main():
  input_tensor = torch.tensor(np.arange(1, 37)).view(1, 1, 6, 6)
  print(input_tensor)
  B, C, H, W = input_tensor.size()
  print(B, C, H, W)

  input_tensor = input_tensor.view(B, C, H//2, 2, W//2, 2).transpose(3, 4).contiguous()
  print(input_tensor)
  input_tensor = input_tensor.view(B, C, H * W //4, 4).transpose(2, 3).contiguous()
  print(input_tensor) 
  input_tensor = input_tensor.view(B, C, 4, H//2, W//2).transpose(1, 2).contiguous()
  print(input_tensor)
  input_tensor = input_tensor.view(B, 4*C, H//2, W//2)
  print(input_tensor)

  t = torch.tensor(np.arange(1, 37))
  print(t.view(3, 3, 4))

  from PIL import Image
  img = Image.open('./snow_duck.jpg')
  print(transforms.ToTensor()(img).size())

# variable args

def f_args(*args):
  return sum(args)

print(f_args(*[1, 2, 3]))
print(f_args(1, 2, 3))
print(f_args(1, 2, 3, 4, 5))