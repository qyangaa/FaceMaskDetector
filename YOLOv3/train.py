from __future__ import division

import argparse
import datetime
import os
import time

import torch
from terminaltables import AsciiTable
from torch.autograd import Variable
from torch.utils.data import DataLoader

from YOLOv3.utils.models import Darknet
from test import evaluate
from utils.datasets import *
from utils.logger import *
from utils.utils import *
from utils.parse_config import *

if __name__ == "__main__":
    pretrained_weights = True
    batch_size = 4
    num_workers = 4
    epochs = 1
    gradient_accumulations = 2
    checkpoint_interval = 1


    logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    cwd = os.getcwd()
    config_dir = os.path.join(cwd, "config/yolov3_face_mask.cfg")
    data_config = os.path.join(cwd, "config/face_mask.data")
    weights_dir = os.path.join(cwd, "weights/yolov3.weights")
    data_dir = os.path.join(cwd, os.pardir)
    data_dir = os.path.join(data_dir, os.pardir)
    train_path = os.path.join(data_dir, "data/face_mask/")

    # Initiate model
    model = Darknet(config_dir).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if pretrained_weights:
        model.load_darknet_weights(weights_dir)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=False,normalized_labels=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

            if epoch % checkpoint_interval == 0:
                torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)