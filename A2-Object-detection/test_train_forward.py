import torch
from torch.utils.data import DataLoader

from models.darknet import Darknet
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

    paths, imgs, targets = next(iter(loader))

    imgs = imgs.to(device)
    targets = targets.to(device)

    print("Images shape:", imgs.shape)
    print("Targets shape:", targets.shape)

    outputs = model(imgs)

    print("Number of YOLO outputs:", len(outputs))

    for i, out in enumerate(outputs):
        print(f"Output {i} shape:", out.shape)


if __name__ == "__main__":
    main()