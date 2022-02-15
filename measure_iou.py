import csv

file_path = 'D:/reef/tensorflow-great-barrier-reef/train.csv'
width_sum, height_sum, data_num = 0.0, 0.0, 0
with open(file_path) as fr:
	ious = []
	csvfile = csv.reader(fr)
	next(csvfile)
	for i, line in enumerate(csvfile):
		#if i == 300:
		#	break
		boxes = eval(line[-1])
		if len(boxes) > 0:
			data_num += len(boxes)
			for j in range(len(boxes)):
				width_sum += boxes[j]['width']
				height_sum += boxes[j]['height']
	
print(f'Average width: {width_sum/data_num:.3f}, Average height: {height_sum/data_num:.3f}')