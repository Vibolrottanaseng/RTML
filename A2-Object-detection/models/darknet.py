import torch
import torch.nn as nn

from utils.parse_config import parse_model_config
from models.layers import Mish, EmptyLayer, YOLOLayer


def create_modules(module_defs):
    """
    Constructs module list from YOLO config.
    Supports YOLOv4 layers:
    - convolutional
    - maxpool
    - upsample
    - route
    - shortcut
    - yolo
    """

    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()

    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def.get("batch_normalize", 0))
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            pad = (kernel_size - 1) // 2 if int(module_def["pad"]) else 0

            conv = nn.Conv2d(
                in_channels=output_filters[-1],
                out_channels=filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                bias=not bn,
            )

            modules.add_module(f"conv_{module_i}", conv)

            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters))

            activation = module_def["activation"]

            if activation == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1, inplace=True))

            elif activation == "mish":
                modules.add_module(f"mish_{module_i}", Mish())

            elif activation == "linear":
                pass

            else:
                raise ValueError(f"Unsupported activation: {activation}")

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])

            if kernel_size == 2 and stride == 1:
                modules.add_module(f"zero_pad_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))

            modules.add_module(
                f"maxpool_{module_i}",
                nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2)
            )

            filters = output_filters[-1]

        elif module_def["type"] == "upsample":
            stride = int(module_def["stride"])
            modules.add_module(f"upsample_{module_i}", nn.Upsample(scale_factor=stride, mode="nearest"))
            filters = output_filters[-1]

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]

            # YOLOv4 can have route layers with more than 2 inputs
            filters = sum([output_filters[1:][i] for i in layers])

            modules.add_module(f"route_{module_i}", EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[-1]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())

        elif module_def["type"] == "yolo":
            mask = [int(x) for x in module_def["mask"].split(",")]
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            num_classes = int(module_def["classes"])
            img_dim = int(hyperparams["height"])

            modules.add_module(
                f"yolo_{module_i}",
                YOLOLayer(anchors, num_classes, img_dim)
            )

            filters = output_filters[-1]

        else:
            raise ValueError(f"Unsupported layer type: {module_def['type']}")

        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class Darknet(nn.Module):
    def __init__(self, config_path):
        super().__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)

    def load_darknet_weights(self, weights_path):
        """
        Loads pretrained Darknet weights.
        """
        import numpy as np

        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]
            weights = np.fromfile(f, dtype=np.float32)

        ptr = 0

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]

                if int(module_def.get("batch_normalize", 0)):
                    bn_layer = module[1]

                    num_b = bn_layer.bias.numel()

                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b

                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b

                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b

                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b

                else:
                    num_b = conv_layer.bias.numel()

                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b

                num_w = conv_layer.weight.numel()

                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

        print(f"Loaded Darknet weights from {weights_path}")

    def forward(self, x):
        layer_outputs = []
        yolo_outputs = []

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            module_type = module_def["type"]

            if module_type in ["convolutional", "upsample", "maxpool"]:
                x = module(x)

            elif module_type == "route":
                layers = [int(layer) for layer in module_def["layers"].split(",")]
                x = torch.cat([layer_outputs[layer] for layer in layers], dim=1)

            elif module_type == "shortcut":
                from_layer = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[from_layer]

            elif module_type == "yolo":
                x = module[0](x)
                yolo_outputs.append(x)

            layer_outputs.append(x)

        return yolo_outputs