import os
from PIL import Image
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
import numpy as np


def default_loader(path):
    return Image.open(path).convert('RGB')


label2idx = dict({'face_mask': 1, 'face': 0})
idx2label = dict({'0': 'face', '1': 'face_mask'})

cwd = os.getcwd()
my_data_path = os.path.join(cwd, "data/face_mask/")


class MyDataset(Dataset):
    def __init__(self, data_path=my_data_path, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        imgs = []
        annotations = []
        for rel_path in os.listdir(data_path + "Annotations"):
            annotation_path = data_path + "Annotations" + '/' + rel_path
            img_path = data_path + "JPEGImages" + '/' + rel_path.replace(".xml",".jpg")
            imgs.append(img_path)
            annotations.append(annotation_path)
        self.imgs = imgs
        self.annotations = annotations
        self.transform = transform
        self.traget_transform = target_transform
        self.loader = loader
        self.data_path = data_path

    def __getitem__(self, index):
        img_path = self.imgs[index]
        annotation_path = self.annotations[index]
        img = self.loader(img_path)
        annotation = ET.parse(annotation_path).getroot()
        if self.transform is not None:
            img = self.transform(img)
        text_label = annotation.find('object').find('name').text
        label = label2idx[text_label]
        bndbox = annotation.find('object').find('bndbox')
        width = float(annotation.find('size').find('width').text)
        height = float(annotation.find('size').find('height').text)
        xmin = float(bndbox.find('xmin').text) / width
        ymin = float(bndbox.find('ymin').text) / height
        xmax = float(bndbox.find('xmax').text) / width
        ymax = float(bndbox.find('ymax').text) / height
        bbox = torch.from_numpy(np.array([xmin, ymin, xmax, ymax]))
        return img, label, bbox

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    cwd = os.getcwd()
    data_dir = os.path.join(os.path.join(cwd, os.pardir),os.pardir)
    my_data_path = os.path.join(data_dir, "data/face_mask/")
    dataset = MyDataset(my_data_path)
    print(len(dataset))
    img, label, bbox = dataset[0]
    print(label)
    print(bbox)
    print(img)