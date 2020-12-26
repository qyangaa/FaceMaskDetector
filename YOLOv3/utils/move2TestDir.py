import random
import shutil
import os

split_ratio = 0.1
source_dir = '/home/arky/files/CV/data/face_mask/train'
target_dir = '/home/arky/files/CV/data/face_mask/test'

all_img_file_names = os.listdir(source_dir + "/JPEGImages")
length = len(all_img_file_names)
img_file_names = random.sample(all_img_file_names, int(length * split_ratio))
ann_file_names = [name.replace(".jpg", ".xml") for name in img_file_names]

for i in range(len(img_file_names)):
    source_image = source_dir+"/JPEGImages/"+img_file_names[i]
    shutil.move(source_image,
                target_dir+"/JPEGImages/")
    source_annotation = source_dir+"/Annotations/"+ann_file_names[i]
    shutil.move(source_annotation,
                target_dir+"/Annotations/")