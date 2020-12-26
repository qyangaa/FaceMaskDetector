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
    epochs = 10
    gradient_accumulations = 2
    checkpoint_interval = 20
    evaluation_interval = 1
    split_ratio = 0.2
    img_size = 416

    logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    cwd = os.getcwd()
    config_dir = os.path.join(cwd, "config/yolov3_face_mask.cfg")
    data_config = os.path.join(cwd, "config/face_mask.data")
    # weights_dir = os.path.join(cwd, "weights/yolov3_ckpt_14.pth")
    weights_dir = os.path.join(cwd, "weights/yolov3.weights")
    data_dir = os.path.join(cwd, os.pardir)
    data_dir = os.path.join(data_dir, os.pardir)
    train_path = os.path.join(data_dir, "data/face_mask/train/")
    val_path = os.path.join(data_dir, "data/face_mask/test/")

    # Initiate model
    model = Darknet(config_dir).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if pretrained_weights:
        if weights_dir.endswith(".pth"):
            model.load_state_dict(torch.load(weights_dir))
        else:
            model.load_darknet_weights(weights_dir)

    # Get dataloader
    train_set = ListDataset(train_path, augment=True, multiscale=False, normalized_labels=False)
    dataloader_train = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=train_set.collate_fn,
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
        for batch_i, (_, imgs, targets) in enumerate(dataloader_train):
            batches_done = len(dataloader_train) * epoch + batch_i

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

            log_str = "[Epoch %d/%d, Batch %d/%d]" % (epoch, epochs, batch_i, len(dataloader_train))

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
                            tensorboard_log += [(f"{name}_{j + 1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            # log_str += AsciiTable(metric_table).table
            log_str += f"Total loss {loss.item()}; "

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader_train) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"ETA {time_left}; "

            print(log_str)

            model.seen += imgs.size(0)

            # if epochs % evaluation_interval == 0:
            #     print("\n---- Evaluating Model ----")
            #     # Evaluate the model on the validation set
            #     evaluation = evaluate(
            #         model,
            #         val_path,
            #         iou_thres=0.5,
            #         conf_thres=0.5,
            #         nms_thres=0.5,
            #         img_size=img_size,
            #         batch_size=8,
            #         device=device
            #     )
            #     if evaluation == 0:
            #         print("No results")
            #     else:
            #         precision, recall, AP, f1, ap_class = evaluation
            #         evaluation_metrics = [
            #             ("val_precision", precision.mean()),
            #             ("val_recall", recall.mean()),
            #             ("val_mAP", AP.mean()),
            #             ("val_f1", f1.mean()),
            #         ]
            #         logger.list_of_scalars_summary(evaluation_metrics, epoch)
            #
            #         # Print class APs and mAP
            #         ap_table = [["Index", "Class name", "AP"]]
            #         for i, c in enumerate(ap_class):
            #             ap_table += [[c, "%.5f" % AP[i]]]
            #         print(AsciiTable(ap_table).table)
            #         print(f"---- mAP {AP.mean()}")

            if batch_i % checkpoint_interval == 0:
                torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
