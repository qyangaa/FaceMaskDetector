import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# from utils.parse_config import *
# from utils.utils import build_targets, to_cpu, non_max_suppression

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
                # momentum: Constant for running mean / variance.
                #    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
                # eps: Constant for numeric stability
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
