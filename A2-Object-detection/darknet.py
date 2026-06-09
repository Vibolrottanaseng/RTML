from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import predict_transform


class EmptyLayer(nn.Module):
    """
    Placeholder for route and shortcut layers.
    """
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    """
    YOLO detection layer.
    Stores anchors.
    """
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


class Mish(nn.Module):
    """
    Mish activation used in YOLOv4.
    Mish(x) = x * tanh(softplus(x))
    """
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def parse_cfg(cfgfile):
    """
    Takes a configuration file and returns a list of blocks.
    Each block describes one layer or net info.
    """
    file = open(cfgfile, "r")
    lines = file.read().split("\n")
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != "#"]
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()

    blocks.append(block)
    return blocks


def create_modules(blocks):
    """
    Creates PyTorch modules from YOLO config blocks.
    Supports YOLOv3 and YOLOv4 layers:
    - convolutional
    - upsample
    - route
    - shortcut
    - yolo
    - maxpool
    - mish activation
    """
    net_info = blocks[0]
    module_list = nn.ModuleList()

    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        if x["type"] == "convolutional":
            activation = x["activation"]

            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            conv = nn.Conv2d(
                prev_filters,
                filters,
                kernel_size,
                stride,
                pad,
                bias=bias
            )

            module.add_module("conv_{0}".format(index), conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

            elif activation == "mish":
                activn = Mish()
                module.add_module("mish_{0}".format(index), activn)

            elif activation == "linear":
                pass

        elif x["type"] == "upsample":
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=stride, mode="nearest")
            module.add_module("upsample_{}".format(index), upsample)

            filters = prev_filters

        elif x["type"] == "maxpool":
            size = int(x["size"])
            stride = int(x["stride"])

            if stride == 1:
                maxpool = nn.MaxPool2d(
                    kernel_size=size,
                    stride=stride,
                    padding=size // 2
                )
            else:
                maxpool = nn.MaxPool2d(
                    kernel_size=size,
                    stride=stride
                )

            module.add_module("maxpool_{}".format(index), maxpool)
            filters = prev_filters

        elif x["type"] == "route":
            x["layers"] = x["layers"].split(",")

            start = int(x["layers"][0])

            try:
                end = int(x["layers"][1])
            except:
                end = 0

            filters = 0

            for layer in x["layers"]:
                layer = int(layer)

                if layer > 0:
                    filters += output_filters[layer]
                else:
                    filters += output_filters[index + layer]

            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
            filters = prev_filters

        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(m) for m in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]

            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

            filters = prev_filters

        else:
            print("Unknown layer type:", x["type"])
            filters = prev_filters

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return net_info, module_list


class MyDarknet(nn.Module):
    """
    Main YOLO model.
    Works for YOLOv3 and YOLOv4 cfg files.
    """
    def __init__(self, cfgfile):
        super(MyDarknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA=False):
        modules = self.blocks[1:]
        outputs = {}
        write = 0

        for i, module in enumerate(modules):
            module_type = module["type"]

            if module_type in ["convolutional", "upsample", "maxpool"]:
                x = self.module_list[i](x)
                outputs[i] = x

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                maps = []

                for layer in layers:
                    if layer > 0:
                        maps.append(outputs[layer])
                    else:
                        maps.append(outputs[i + layer])

                if len(maps) == 1:
                    x = maps[0]
                else:
                    x = torch.cat(maps, dim=1)

                outputs[i] = x

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]
                outputs[i] = x

            elif module_type == "yolo":
                anchors = self.module_list[i][0].anchors

                inp_dim = int(self.net_info["height"])
                num_classes = int(module["classes"])

                x = predict_transform(
                    x,
                    inp_dim,
                    anchors,
                    num_classes,
                    CUDA
                )

                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

                outputs[i] = outputs[i - 1]

        return detections

    def load_weights(self, weightfile):
        """
        Loads official Darknet weights.
        Example:
        model.load_weights("weights/yolov4.weights")
        """
        fp = open(weightfile, "rb")

        header = torch.from_numpy(
            __import__("numpy").fromfile(fp, dtype=__import__("numpy").int32, count=5)
        )

        self.header = header
        self.seen = self.header[3]

        weights = __import__("numpy").fromfile(fp, dtype=__import__("numpy").float32)

        ptr = 0

        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            if module_type == "convolutional":
                model = self.module_list[i]

                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if batch_normalize:
                    bn = model[1]

                    num_bn_biases = bn.bias.numel()

                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    num_biases = conv.bias.numel()

                    conv_biases = torch.from_numpy(weights[ptr:ptr + num_biases])
                    ptr += num_biases

                    conv_biases = conv_biases.view_as(conv.bias.data)
                    conv.bias.data.copy_(conv_biases)

                num_weights = conv.weight.numel()

                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr += num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)