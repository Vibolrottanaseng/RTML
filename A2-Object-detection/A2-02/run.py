import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import OxfordIIITPet
from tqdm import tqdm


class PetSegDataset(Dataset):
    def __init__(self, base, size=128):
        self.ds = base
        self.img_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.PILToTensor(),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, mask = self.ds[idx]
        img = self.img_tf(img)
        # Oxford-IIIT Pet segmentation labels are 1,2,3. Convert to 0,1,2.
        mask = (self.mask_tf(mask).squeeze(0).long() - 1).clamp(0, 2)
        return img, mask


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetResNet18(nn.Module):
    """Baseline: ResNet-18 encoder + U-Net decoder WITH skip connections."""
    def __init__(self, n_classes=3, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        self.stem_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # H/2, 64
        self.stem_pool = resnet.maxpool                                         # H/4
        self.enc1 = resnet.layer1                                               # H/4, 64
        self.enc2 = resnet.layer2                                               # H/8, 128
        self.enc3 = resnet.layer3                                               # H/16, 256
        self.enc4 = resnet.layer4                                               # H/32, 512

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(512 + 512, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(256 + 256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(128 + 128, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(64 + 64, 64)

        self.up0 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0 = DoubleConv(32 + 64, 32)

        self.output = nn.Conv2d(32, n_classes, kernel_size=1)

    def _cat(self, x, skip):
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([skip, x], dim=1)

    def forward(self, x):
        s0 = self.stem_conv(x)
        sp = self.stem_pool(s0)
        s1 = self.enc1(sp)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        x = self.bottleneck(s4)
        x = self.up4(x); x = self._cat(x, s4); x = self.dec4(x)
        x = self.up3(x); x = self._cat(x, s3); x = self.dec3(x)
        x = self.up2(x); x = self._cat(x, s2); x = self.dec2(x)
        x = self.up1(x); x = self._cat(x, s1); x = self.dec1(x)
        x = self.up0(x); x = self._cat(x, s0); x = self.dec0(x)
        return self.output(x)


class UNetResNet18NoSkip(nn.Module):
    """Ablation: same ResNet-18 encoder + decoder WITHOUT skip connections."""
    def __init__(self, n_classes=3, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        self.stem_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.stem_pool = resnet.maxpool
        self.enc1 = resnet.layer1
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(512, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(128, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(64, 64)

        self.up0 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0 = DoubleConv(32, 32)

        self.output = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        s0 = self.stem_conv(x)
        sp = self.stem_pool(s0)
        s1 = self.enc1(sp)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        x = self.bottleneck(s4)
        x = self.dec4(self.up4(x))
        x = self.dec3(self.up3(x))
        x = self.dec2(self.up2(x))
        x = self.dec1(self.up1(x))
        x = self.dec0(self.up0(x))

        if x.shape[2:] != (128, 128):
            x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        return self.output(x)


def build_model(model_name, device):
    if model_name == "unet_resnet18":
        return UNetResNet18(n_classes=3, pretrained=True).to(device)
    if model_name == "unet_resnet18_no_skip":
        return UNetResNet18NoSkip(n_classes=3, pretrained=True).to(device)
    raise ValueError(f"Unknown model: {model_name}")


def make_loaders(data_dir, img_size, batch_size, num_workers):
    train_raw = OxfordIIITPet(data_dir, split="trainval", target_types="segmentation", download=True)
    test_raw = OxfordIIITPet(data_dir, split="test", target_types="segmentation", download=True)

    train_data = PetSegDataset(train_raw, img_size)
    test_data = PetSegDataset(test_raw, img_size)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, test_loader


def compute_iou(logits, target, n_classes=3):
    pred = logits.argmax(dim=1)
    ious = []
    for cls in range(n_classes):
        inter = ((pred == cls) & (target == cls)).sum().float()
        union = ((pred == cls) | (target == cls)).sum().float()
        if union > 0:
            ious.append((inter / union).item())
    return float(np.mean(ious)) if ious else 0.0


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ious = []
    for imgs, masks in tqdm(loader, desc="Evaluate"):
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        ious.append(compute_iou(logits, masks))
    return float(np.mean(ious))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = make_loaders(args.data_dir, args.img_size, args.batch_size, args.num_workers)
    model = build_model(args.model, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    save_path = args.weights or f"{args.model}_pet.pt"
    metrics_path = f"{args.model}_metrics.json"

    history = []
    best_miou = -1.0

    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        losses = []

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            loss = criterion(model(imgs), masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val_miou = evaluate(model, test_loader, device)
        scheduler.step()
        seconds = time.time() - start

        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "val_miou": val_miou,
            "time_epoch_sec": seconds,
        }
        history.append(row)

        print(f"Epoch {epoch+1:02d} | Loss: {row['train_loss']:.4f} | "
              f"Val mIoU: {val_miou:.4f} | Time/epoch: {seconds:.1f}s")

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model -> {save_path}")

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"Done. Best Val mIoU: {best_miou:.4f}")
    print(f"Metrics saved -> {metrics_path}")


def eval_only(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, test_loader = make_loaders(args.data_dir, args.img_size, args.batch_size, args.num_workers)
    model = build_model(args.model, device)

    if not args.weights:
        raise ValueError("--weights is required for --evaluate")
    model.load_state_dict(torch.load(args.weights, map_location=device))

    val_miou = evaluate(model, test_loader, device)
    print(f"Val mIoU: {val_miou:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Oxford-IIIT Pet image segmentation exercise")
    parser.add_argument("--model", required=True, choices=["unet_resnet18", "unet_resnet18_no_skip"])
    parser.add_argument("--dataset", default="oxford_pet", choices=["oxford_pet"])
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    if args.train:
        train(args)
    elif args.evaluate:
        eval_only(args)
    else:
        raise ValueError("Choose either --train or --evaluate")


if __name__ == "__main__":
    main()
