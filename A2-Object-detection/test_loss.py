import torch
from torch.utils.data import DataLoader

from models.darknet import Darknet
from models.yolo_loss import SimpleYOLOLoss
from utils.datasets import ListDataset, collate_fn


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = ListDataset("data/train.txt", img_size=608)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = Darknet("configs/yolov4.cfg").to(device)
    model.load_darknet_weights("weights/yolov4.weights")
    model.train()

    criterion_normal = SimpleYOLOLoss(num_classes=80, loss_type="normal")
    criterion_ciou = SimpleYOLOLoss(num_classes=80, loss_type="ciou")

    paths, imgs, targets = next(iter(loader))

    imgs = imgs.to(device)
    targets = targets.to(device)

    outputs = model(imgs)

    normal_loss, normal_items = criterion_normal(outputs, targets)
    ciou_loss, ciou_items = criterion_ciou(outputs, targets)

    print("Normal loss:", normal_loss.item())
    print("Normal loss items:", normal_items)

    print("CIoU loss:", ciou_loss.item())
    print("CIoU loss items:", ciou_items)


if __name__ == "__main__":
    main()