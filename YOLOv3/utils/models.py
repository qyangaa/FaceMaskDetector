import pdb

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from .parse_config import *
from .utils import build_targets, to_cpu, non_max_suppression

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_modules(module_defs):
    """
    :param module_defs:
    :return: module list of layer blocks
    """
    hyperparams = module_defs.pop(0)  # stored in [net] block of .cfg file
    output_filters = [int(hyperparams["channels"])]  # 3 channel input RGB
    module_list = nn.ModuleList()  # add all parameters to parameters of nn.Module object
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()  # initialize a sequence of layers

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,  # cannot use bias and batch normalization together because batch normalization
                    # already has bias: gamma * normalized(x) + bias
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
                # momentum: Constant for running mean / variance. Momentum is the “lag” in learning mean and variance,
                # so that noise due to mini-batch can be ignored.
                #    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
                # eps: Constant for numeric stability
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
                # LeakyReLU(negative_slope): LeakyReLU(x)=max(0,x)+negative_slope∗min(0,x)

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            # mode is interpolation method
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"updample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())
            # add EmptyLayer() as place holder
            # information of this layer is passed to output_filters
            # actual concatenation is implemented in nn.Module().forward later in Darknet

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())

        elif module_def["type"] == "yolo":
            # mask has the id of anchors to use in this YOLO layer:
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]  # Extract list from map
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]  # convert to tuple
            anchors = [anchors[i] for i in anchor_idxs]  # select the ones to use in this layer
            num_classes = int(module_def["classes"])
            img_size = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            # img_dim = img_size is not used in the YOLOLayer class
            modules.add_module(f"yolo_{module_i}", yolo_layer)

        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers """

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer
    YOLO layer is a head that deals with outputs of previous layers and generates loss based on anchors"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)  # = 3 by default
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.ignore_thres = 0.5
        # ignore_thres:
        #   if anchor_ious>ignore_thresh (large overlap with ground truth), noobj_mask is set to 0
        #   then the results won't be penalized in noobj_loss
        #   to encourage close result (not throwing out everything but the best)
        # noobj_loss: loss for false positives
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.grid_size = 0  # number of grids: we divide the image into number of grids to place anchors and regress

    def compute_grid_offsets(self, grid_size, cuda=True):
        """ Compute
        1. coordinate of each grid corresponding to (0,0) point of the image
        2. anchor dimension in grid-coordinates
        """
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        # torch.arange(g) = tensor([0,1,2,3..,g])
        # .. .repeat(g,1) = tensor([0,1,2..g], [0,1,2..g]....) total g rows, gives g*g matrix
        # .. .view([1,1,g,g]) = tensor([[[0,1,2..g], [0,1,2..g]....]]) added first 2 dimensions: batch size and
        # channel size
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        # .t() = tensor([0,0,0], [1,1,1]....) g*g matrix
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        # scale to grid-coordinate: a_w/stride = number of grids the anchor span in width
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        # only one set of anchors, not assigning different ones to each grid
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        # x: [batch_size, channel_numbers, x, y]
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)  # (N,3,C+(x,y,w,h,c),g,g)
                .permute(0, 1, 3, 4, 2)  # (N,3,g,g, C+(x,y,w,h,c))
                .contiguous()  # tensor.view() only adds conversion between indices without changing the actual value of
            # the tensor, .contiguous() changes the actual memory layout. Normally you don't need to worry about
            # this. If PyTorch expects contiguous tensor but if its not then you will get RuntimeError: input is not
            # contiguous and then you just add a call to contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Confidence
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x  # x= x location relative to current grid, grid_x = current grid
        # relative to the whole image (0,0)
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w  # force prediction to be log(fraction of anchor_w)
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,  # *stride brings back to image coordinate
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:  # not computing loss
            return output, 0
        else:
            #try:
                # convert ground truth to targets for later loss calculation
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )
            #except:
            #    pdb.set_trace()

            # Loss : only calculate loss for pixels with object, ignoring negative spaces
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])  # x: real center x, tx: normalized ground truth x
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            # loss_conf: confidence about object/ no object
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            # add back loss in negative spaces
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            # scale losses: to balance positive and negative samples
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics about current training results
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()  # prediction confidence
            iou50 = (iou_scores > 0.5).float()  # overlap between prediction and ground truth
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            # precision = TP/(TP+FP), TP = (iou>50)*detected_mask
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }
            return output, total_loss


class Darknet(nn.Module):
    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        # module_defs: dictionary of module parameter: value
        # module_list: actual nn.Modulelist
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        # only object YOLOLayer has attribute "metrics"
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size
        # TODO: self.seen is:
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        img_dim = x.shape[2]  # x[b,c,w,h]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (module_defs, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_defs["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_defs[
                "type"] == "route":  # remember we only passed dummy emptyLayer as route layer when constructing module_list
                # concat layers with indices given in def
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_defs["layers"].split(",")], 1)
            elif module_defs["type"] == "shortcut":
                # add last layer to layer to shortcut from
                layer_i = int(module_defs["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_defs["type"] == "yolo":
                # forward call of yolo layer
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                # yolo head: detection output
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def get_text_input():
        # reference: https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/
        img = cv2.imread("dog-cycle-car.png")
        img = cv2.resize(img, (416, 416))  # shape = (416, 416, 3)
        img = img[:, :, ::-1].transpose((2, 0, 1))  # BGR-> RGB, (3,416,416)
        img = img[np.newaxis, :, :, :] / 255.0  # add batch channel, normalize
        img = torch.from_numpy(img).float()  # convert to float tensor
        img = Variable(img)
        return img


    cwd = os.getcwd()
    parent_dir = os.path.join(cwd, os.pardir)
    config_dir = os.path.join(parent_dir, "config/yolov3_face_mask.cfg")
    weights_dir = os.path.join(parent_dir, "weights/yolov3.weights")

    # Test without preload weights
    model = Darknet(config_dir).to(device)
    input = get_text_input()
    input = input.to(device)
    pred = model(input)
    print("No preload weights: ", pred.shape)  # torch.Size([1, 10647, 85]), each of 10647 rows is a bounding box


    # Test with preload weights
    model.load_darknet_weights(weights_dir)
    pred2 = model(input)
    print("With preload weights: ", pred2.shape)
