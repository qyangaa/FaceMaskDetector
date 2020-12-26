import random
import shutil
import os

target_dir = '/home/arky/files/CV/data/face_mask/train'
source_dir = '/home/arky/files/CV/data/face_mask/test'

img_file_names = os.listdir(source_dir + "/JPEGImages")
ann_file_names = [name.replace(".jpg", ".xml") for name in img_file_names]

for i in range(len(img_file_names)):
    shutil.move(source_dir+"/JPEGImages/"+img_file_names[i],
                target_dir+"/JPEGImages/")
    shutil.move(source_dir+"/Annotations/"+ann_file_names[i],
                target_dir+"/Annotations/")