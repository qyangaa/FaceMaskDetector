import os
import xml.etree.ElementTree as ET

label2idx = dict({'face_mask': 1, 'face': 0})
idx2label = dict({'0': 'face', '1': 'face_mask'})

# Convert labels to COCO standard

def convertLabels(data_path,filename):
    labels_dir = data_path + '/' + "labels/"
    f = open(filename,"w")
    for rel_path in os.listdir(labels_dir):
        img_path = data_path + '/' + "images/"
        img_file_name = (img_path+rel_path).replace(".txt",".jpg")
        f.write(img_file_name + '\n')
    f.close()


if __name__ == '__main__':
    cwd = os.getcwd()
    data_dir = os.path.join(os.path.join(cwd, os.pardir), os.pardir)
    my_data_path = os.path.join(data_dir, "data/face_mask/valid")
    convertLabels(my_data_path, "valid.txt")
