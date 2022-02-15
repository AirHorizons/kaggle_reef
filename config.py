epoch = 3
lr = 0.0003
momentum = 0.9
weight_decay = 0.0005
batch_size = 32
data_size = 1281167
data_path = 'D:/ImageNet/imagenet_object_localization_patched2019/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/'
reef_data_path = 'D:/reef/tensorflow-great-barrier-reef/'
image_path = reef_data_path + 'train_images/'
annotation_path = reef_data_path + 'train.csv'
grid_size = 32
prior_size = 32
lambda_conf = 5.0
lambda_noobj = 0.01
debug = False

save_dir = 'Y2_checkpoint.pth'