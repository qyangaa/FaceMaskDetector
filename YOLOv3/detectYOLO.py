import random

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
from PIL import Image

from YOLOv3.utils.models import Darknet
from YOLOv3.utils.utils import non_max_suppression, rescale_boxes

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_input(path):
    # reference: https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/
    img = cv2.imread(path)
    img = cv2.resize(img, (416, 416))  # shape = (416, 416, 3)
    img = img[:, :, ::-1].transpose((2, 0, 1))  # BGR-> RGB, (3,416,416)
    img = img[np.newaxis, :, :, :] / 255.0  # add batch channel, normalize
    img = torch.from_numpy(img).float()  # convert to float tensor
    img = Variable(img)
    return img


cwd = os.getcwd()
config_dir = os.path.join(cwd, "config/yolov3_face_mask.cfg")
weights_dir = os.path.join(cwd, "weights/yolov3.weights")
path = "utils/dog-cycle-car.png"

model = Darknet(config_dir).to(device)
model.load_darknet_weights(weights_dir)
model.eval()
input = get_input(path)
input = input.to(device)
pred = model(input)
detections = non_max_suppression(pred, 0.8)[0]

img = np.array(Image.open(path))
plt.figure()
fig, ax = plt.subplots(1)
ax.imshow(img)

detections = rescale_boxes(detections, 416, img.shape[:2])  # img.shape = (452, 602, 3)
unique_labels = detections[:, -1].cpu().unique()
n_cls_preds = len(unique_labels)

cmap = plt.get_cmap("tab20b")
colors = [cmap(i) for i in np.linspace(0, 1, 20)]
bbox_colors = random.sample(colors, n_cls_preds)
for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
    print("\t+ Label: %s, Conf: %.5f" % (int(cls_pred), cls_conf.item()))
    box_w = x2 - x1
    box_h = y2 - y1
    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
    ax.add_patch(bbox)
    plt.text(
        x1,
        y1,
        s = int(cls_pred),
        color="white",
        verticalalignment="top",
        bbox={"color": color, "pad": 0}
    )
plt.axis("off")
plt.gca().xaxis.set_major_locator(NullLocator())
plt.gca().yaxis.set_major_locator(NullLocator())
plt.show()
# filename = path.split("/")[-1].split(".")[0]
# plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
# plt.close()