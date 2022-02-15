import os, sys, csv
import shutil
import config as cfg
from glob import glob
from PIL import Image
import numpy as np

def merge_folder():
	out_dir = cfg.reef_data_path + '/train_images_new/'
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	image_path = cfg.image_path
	for video_dir in glob(image_path + 'video*'):
		for i, image_dir in enumerate(glob(video_dir + '/*.jpg')):
			
			video_name = video_dir.split('_')[-1]
			image_name = image_dir.split('\\')[-1].split('.')[0]
			new_image_dir = out_dir + '-'.join([video_name, image_name]) + '.jpg'
			#print(new_image_dir)
			shutil.copy(image_dir, new_image_dir)

def trim_csv():
	new_annotation_path = cfg.reef_data_path + 'train_new.csv'
	with open(cfg.annotation_path) as fr:
		csv_reader = csv.reader(fr)
		next(fr)
		with open(new_annotation_path, 'w', newline='') as fw:
			csv_writer = csv.writer(fw)
			for i, line in enumerate(csv_reader):
				image_id = line[-2] + '.jpg'
				gt = line[-1]

				csv_writer.writerow([image_id, gt])

def measure_rgb():
	file_dir = cfg.reef_data_path + 'train_images_new/'
	r, g, b = [], [], []
	for i, image_dir in enumerate(glob(file_dir + '*.jpg')):
		if i%1000==999:
			print(f'{i+1} images processed')
		img = Image.open(image_dir)
		rp, gp, bp = np.array(img).mean(axis=(0, 1))
		r.append(rp)
		g.append(gp)
		b.append(bp)

	print(f'means: {np.mean(r)} {np.mean(g)} {np.mean(b)}')
	print(f'std: {np.std(r)} {np.std(g)} {np.std(b)}')
		

if __name__ == '__main__':
	#merge_folder()
	#trim_csv()
	measure_rgb()