import torch


def load_classes(path):
    """
    Loads class names from coco.names.
    """
    with open(path, "r") as fp:
        names = fp.read().split("\n")

    return [x for x in names if len(x) > 0]


def weights_init_normal(m):
    """
    Initializes convolution and batch norm layers.
    """
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Placeholder NMS.
    Later we can implement real NMS.
    """
    return prediction