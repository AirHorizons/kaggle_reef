import torch
import numpy as np
import config as cfg
from Yolov2 import Yolov2
from torch.autograd import Variable

# for debugging
from torchvision.io import read_image
import torchvision.transforms as transforms

def nms(out): # 25 * 7 * 7 tensor
	pos, size, conf = out
	xywh = torch.cat([pos, size], 2)
	lrud = xywh2lrud_tensor(xywh)
	lrud = torch.cat((lrud, conf), 2)
	conf_sort = torch.sort(lrud[:,:,4], dim=1, descending=True)

	finalist = []
	for img in range(cfg.batch_size):
		finalist.append([])
		lrud_img = lrud[img,:,:]
		conf_img = conf_sort[img,:]

		while conf_img.size() > 0:
			
			best_idx = conf_sort.indices[img,0]
			best_img = lrud_img[best_idx,0:3]
			finalist[img].append(best_img)
			rest_img = lrud_img[conf_sort.indices[img,1:],0:3]

			ious = iou_tensor(best_img, rest_img)
			indices = torch.nonzero(ious < 0.8)
			conf_img = conf_img[indices]

	return finalist


def xywh2lrud_tensor(box):
	return torch.stack((box[:,:,0]-box[:,:,2]/2, \
		box[:,:,0]+box[:,:,2]/2, \
		box[:,:,1]+box[:,:,3]/2, \
		box[:,:,1]-box[:,:,3]/2), 2)	

def xywh2lrud(box):
	return (box[0]-box[2]/2, box[0]+box[2]/2, box[1]+box[3]/2, box[1]-box[3]/2)
	
def iou_tensor(box1, box2):
	li = max(box1[0], box2[0])
	ri = min(box1[1], box2[1])
	ui = min(box1[2], box2[2])
	di = max(box1[3], box2[3])
	
	ai = 0 # area of intersection
	if li < ri and di < ui:
		ai = (ri - li) * (ui - di)
	a1 = (box1[1] - box1[0]) * (box1[2] - box1[3])
	a2 = (box2[1] - box2[0]) * (box2[2] - box2[3])
	au = a1 + a2 - ai # area of union

	return ai / au

def iou(box1, box2):
	# box1 and box2: (x, y, w, h)
	box1, box2 = map(xywh2lrud, [box1, box2]) # [xywh2lrud(box1), xywh2lrud(box2)]
	# box1 and box2: (l, r, u, d)
	li = max(box1[0], box2[0])
	ri = min(box1[1], box2[1])
	ui = min(box1[2], box2[2])
	di = max(box1[3], box2[3])
	
	ai = 0 # area of intersection
	if li < ri and di < ui:
		ai = (ri - li) * (ui - di)
	a1 = (box1[1] - box1[0]) * (box1[2] - box1[3])
	a2 = (box2[1] - box2[0]) * (box2[2] - box2[3])
	au = a1 + a2 - ai # area of union

	return ai / au

def loss(out, gt):
	gt = map(eval, gt)
	#print(list(gt))
	loss_val = torch.Tensor([0.0]).cuda()
	pos, size, confidence = out
	# pos = sig(tx), sig(ty), (B, 5*W*H, 2)
	# size = sig(tw), sig(th), (B, 5*W*H, 2)
	# confidence = sig(conf)=IOU, (B, 5*W*H, 1)
	batch, wh = pos.size(0), pos.size(1)
	object_pos = []
	for _ in range(cfg.batch_size):
		object_pos.append({})
	# gt = (list of {x, y, w, h}) -> (B, 5*w*h, 5[x, y, w, h, iou])
	gt_tensor = pos.new_zeros((batch, wh, 5))
	for idx, gt_item in enumerate(gt):
		for gt_box in gt_item:
			#print(gt_box)
			x, y, _, _ = gt_box['x'], gt_box['y'], gt_box['width'], gt_box['height']
			grid_pos = x//(1280//7), y//(720//7)
			if grid_pos not in object_pos:
				object_pos[idx][grid_pos] = [gt_box]
			else:
				object_pos[idx][grid_pos].append(gt_box)
	for b in range(batch):
		for row in range(7):
			for col in range(7):
				if (row, col) in object_pos[b]:
					if len(object_pos[b][(row, col)]) > 1: # 5*7*7 anchor boxes
						bbox_ratio = torch.div( \
							size[b,5*(7*row+col):5*(7*row+col)+5,0], \
							size[b,5*(7*row+col):5*(7*row+col)+5,1]).view(5) # width/height
						
						# map to the gt box which has minimal width/height ratio error with given output boxes
						min_ratio_error = [float("inf")]*5
						gt_box = [None]*5
						for grid in object_pos[(row, col)]:
							gt_ratio = grid['width']/grid['height']
							for i in range(5):
								if abs(gt_ratio-bbox_ratio[i].item()) < min_ratio_error[i]:
									min_ratio_error[i] = abs(gt_ratio-bbox_ratio[i].item())
									gt_box[i] = [grid['x'], grid['y'], grid['width'], grid['height']]
					else: # only one gt box
						grid = object_pos[b][(row, col)][0]
						gt_box = []
						for _ in range(5):
							gt_box.append([grid['x'], grid['y'], grid['width'], grid['height']])

					# compute xywh loss, conf loss
					out_x = (pos[b,5*(7*row+col):5*(7*row+col)+5,0] + col).view(5)
					out_y = (pos[b,5*(7*row+col):5*(7*row+col)+5,1] + row).view(5)
					size_grid = size[b,5*(7*row+col):5*(7*row+col)+5,:]
					out_w = size_grid[:,0].view(5)
					out_h = size_grid[:,1].view(5)
					
					gt_box = torch.Tensor(gt_box).cuda()
					
					gt_x = gt_box[:,0].view(5) / cfg.grid_size
					gt_y = gt_box[:,1].view(5) / cfg.grid_size
					gt_w = gt_box[:,2].view(5) / cfg.prior_size
					gt_h = gt_box[:,3].view(5) / cfg.prior_size
					loss_val += torch.sum(cfg.lambda_conf*(\
						(out_x-gt_x)**2 + \
						(out_y-gt_y)**2 + \
						(torch.sqrt(out_w) - torch.sqrt(gt_w))**2 + \
						(torch.sqrt(out_h) - torch.sqrt(gt_h))**2))

					out_conf = confidence[b,5*(7*row+col):5*(7*row+col)+5,:].view(5)
					gt_ious = []
					for i in range(5):
						box1 = tuple(map(lambda x:x[i].item()*32, (out_x, out_y, out_w, out_h)))
						box2 = tuple(map(lambda x:x[i].item()*32, (gt_x, gt_y, gt_w, gt_h)))
						
						gt_ious.append(iou(box1, box2))

					gt_ious = torch.Tensor(gt_ious).cuda()
					loss_val += torch.sum((out_conf - gt_ious)**2)
					

				else:
					out_conf = confidence[b,5*(7*row+col):5*(7*row+col)+5,:].view(5)
					loss_val += cfg.lambda_noobj*torch.sum((out_conf)**2)

				if cfg.debug:
					print(loss_val)

	return loss_val


if __name__ == '__main__':
	# t = torch.tensor(np.arange(1, 25*7*7 + 1)).view(25, 7, 7)
	your_model = Yolov2().cuda()
	'''
	# summary(your_model, input_size=(3, 224, 224))
	rand_input = Variable(torch.from_numpy(np.random.randn(1, 3, 224, 224)).float()).cuda()
	out = your_model(rand_input)
	gt = [{'x':100, 'y':150, 'width':50, 'height':30}, {'x':100, 'y':150, 'width':30, 'height':50}, {'x':32, 'y':64, 'width':20, 'height':25}]
	print(loss(out, gt))'''
	sample_image = read_image(cfg.reef_data_path + 'train_images_new\\' + '0-1.jpg')
	sample_image = transforms.Compose([
				transforms.Resize((224, 224)),
				transforms.ConvertImageDtype(torch.float),
				transforms.Normalize(mean=[67.0/255, 145.0/255, 157.0/255],\
				std=[24.0/255, 8.31/255, 25.0/255]),
			])(sample_image).view(-1, 3, 224, 224).cuda()
	checkpoint = torch.load('epoch0_Y2_checkpoint.pth')

	print(checkpoint['state_dict'])

	your_model.load_state_dict(checkpoint['state_dict'])

	#out = your_model(sample_image)
	#print(nms(out))
	