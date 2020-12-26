import glob
import pdb
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)].rstrip()
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    label2idx = dict({'face_mask': 1, 'face': 0})
    idx2label = dict({'0': 'face', '1': 'face_mask'})
    def __init__(self, data_path, img_size=416, augment=True, multiscale=True, normalized_labels=False):
        self.img_files = []
        self.annotations = []
        for rel_path in os.listdir(data_path + "Annotations"):
            annotation_path = data_path + "Annotations" + '/' + rel_path
            img_path = data_path + "JPEGImages" + '/' + rel_path.replace(".xml", ".jpg")
            annotation = ET.parse(annotation_path).getroot()
            object = annotation.find('object')
            if object is not None:
                # some annotations are empty
                self.img_files.append(img_path)
                self.annotations.append(annotation_path)

        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        # print(self.annotations[715:716])
        # pdb.set_trace()

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        annotation = ET.parse(self.annotations[index]).getroot().find('object').find('name').text
        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.annotations[index]

        targets = None
        if os.path.exists(label_path):
            annotation = ET.parse(label_path).getroot()
            text_label = annotation.find('object').find('name').text
            label = ListDataset.label2idx[text_label]
            bndbox = annotation.find('object').find('bndbox')
            width = float(annotation.find('size').find('width').text)
            height = float(annotation.find('size').find('height').text)
            xmin = float(bndbox.find('xmin').text) / width
            ymin = float(bndbox.find('ymin').text) / height
            xmax = float(bndbox.find('xmax').text) / width
            ymax = float(bndbox.find('ymax').text) / height

            # label: [class, x1, y1, x2, y2]
            # e.g.: 16 0.606688 0.341381 0.544156 0.510000 (normalized)
            boxes = torch.from_numpy(np.array([label,xmin, ymin, xmax, ymax]).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)

if __name__ == '__main__':
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, os.pardir)
    data_dir = os.path.join(data_dir, os.pardir)
    data_dir = os.path.join(data_dir, os.pardir)
    my_data_path = os.path.join(data_dir, "data/face_mask/")
    dataset = ListDataset(my_data_path)
    print(len(dataset))
    img, label, bbox = dataset[0]
    print(label)
    print(bbox)
    print(img)