import os
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ListDataset(Dataset):
    """
    YOLO dataset loader.

    It reads image paths from train.txt / valid.txt.
    For each image, it loads the matching YOLO label file.

    Image:
        data/images/train/xxx.jpg

    Label:
        data/labels/train/xxx.txt
    """

    def __init__(self, list_path, img_size=608):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.img_files = [path.strip() for path in self.img_files if path.strip()]
        self.img_size = img_size

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        img_path = self.img_files[index]

        # Load image
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # Convert image path to label path
        label_path = (
            img_path
            .replace("images", "labels")
            .replace(".jpg", ".txt")
            .replace(".jpeg", ".txt")
            .replace(".png", ".txt")
        )

        boxes = []

        if os.path.exists(label_path):
            with open(label_path, "r") as file:
                for line in file.readlines():
                    values = line.strip().split()

                    if len(values) == 5:
                        class_id, x, y, w, h = values
                        boxes.append([
                            int(class_id),
                            float(x),
                            float(y),
                            float(w),
                            float(h),
                        ])

        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 5), dtype=torch.float32)

        return img_path, img, boxes

    def __len__(self):
        return len(self.img_files)


def collate_fn(batch):
    """
    Custom collate function because each image can have different number of boxes.

    Returns:
        paths: list of image paths
        imgs: tensor [batch, 3, img_size, img_size]
        targets: tensor [num_boxes, 6]

    Target format:
        image_index, class_id, x, y, w, h
    """

    paths, imgs, boxes = list(zip(*batch))

    imgs = torch.stack(imgs, dim=0)

    targets = []

    for i, box in enumerate(boxes):
        if box.size(0) > 0:
            image_index = torch.full((box.size(0), 1), i)
            target = torch.cat((image_index, box), dim=1)
            targets.append(target)

    if len(targets) > 0:
        targets = torch.cat(targets, dim=0)
    else:
        targets = torch.zeros((0, 6))

    return paths, imgs, targets