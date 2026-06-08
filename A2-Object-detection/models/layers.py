import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    """
    Mish activation used in YOLOv4.
    mish(x) = x * tanh(softplus(x))
    """
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class EmptyLayer(nn.Module):
    """
    Placeholder for route and shortcut layers.
    These layers are handled manually in Darknet forward().
    """
    def __init__(self):
        super().__init__()


class YOLOLayer(nn.Module):
    """
    YOLO detection layer.

    Converts raw convolution output:
        [batch, 255, grid, grid]

    Into:
        [batch, 3, grid, grid, 85]

    For COCO:
        3 anchors
        80 classes
        85 = x, y, w, h, objectness + 80 class scores
    """

    def __init__(self, anchors, num_classes, img_dim=608):
        super().__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.bbox_attrs = 5 + num_classes

    def forward(self, x):
        batch_size = x.size(0)
        grid_size = x.size(2)

        prediction = x.view(
            batch_size,
            self.num_anchors,
            self.bbox_attrs,
            grid_size,
            grid_size
        )

        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()

        return prediction