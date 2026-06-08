import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.darknet import Darknet
from models.yolo_loss import SimpleYOLOLoss
from utils.datasets import ListDataset, collate_fn


def test_model(config_path, weights_path):
    print("Loading model...")
    model = Darknet(config_path)

    print("Loading weights...")
    model.load_darknet_weights(weights_path)

    print("Model loaded successfully.")

    dummy_input = torch.randn(1, 3, 608, 608)

    print("Running dummy forward pass...")
    outputs = model(dummy_input)

    print("Forward pass successful.")
    print(f"Number of YOLO outputs: {len(outputs)}")

    for i, output in enumerate(outputs):
        print(f"Output {i}: shape = {output.shape}")


def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = ListDataset(args.train_list, img_size=args.img_size)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = Darknet(args.config).to(device)

    if args.weights:
        model.load_darknet_weights(args.weights)

    model.train()

    criterion = SimpleYOLOLoss(
        num_classes=args.num_classes,
        loss_type=args.loss_type,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    print(f"Training with loss type: {args.loss_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_box_loss = 0.0
        epoch_obj_loss = 0.0
        epoch_cls_loss = 0.0

        progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch_i, (paths, imgs, targets) in enumerate(progress_bar):
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)

            loss, loss_items = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss_items["total_loss"]
            epoch_box_loss += loss_items["box_loss"]
            epoch_obj_loss += loss_items["obj_loss"]
            epoch_cls_loss += loss_items["cls_loss"]

            progress_bar.set_postfix({
                "loss": f"{loss_items['total_loss']:.4f}",
                "box": f"{loss_items['box_loss']:.4f}",
                "obj": f"{loss_items['obj_loss']:.4f}",
                "cls": f"{loss_items['cls_loss']:.4f}",
            })

        num_batches = len(loader)

        avg_loss = epoch_loss / num_batches
        avg_box_loss = epoch_box_loss / num_batches
        avg_obj_loss = epoch_obj_loss / num_batches
        avg_cls_loss = epoch_cls_loss / num_batches

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"loss={avg_loss:.4f}, "
            f"box={avg_box_loss:.4f}, "
            f"obj={avg_obj_loss:.4f}, "
            f"cls={avg_cls_loss:.4f}"
        )

        checkpoint_path = (
            f"outputs/checkpoints/"
            f"yolov4_{args.loss_type}_epoch_{epoch + 1}.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "loss_type": args.loss_type,
            },
            checkpoint_path,
        )

        print(f"Saved checkpoint: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="test", help="test or train")
    parser.add_argument("--config", type=str, default="configs/yolov4.cfg")
    parser.add_argument("--weights", type=str, default="weights/yolov4.weights")

    parser.add_argument("--train-list", type=str, default="data/train.txt")
    parser.add_argument("--valid-list", type=str, default="data/valid.txt")

    parser.add_argument("--img-size", type=int, default=608)
    parser.add_argument("--num-classes", type=int, default=80)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)

    parser.add_argument(
        "--loss-type",
        type=str,
        default="normal",
        choices=["normal", "ciou"],
        help="normal or ciou",
    )

    args = parser.parse_args()

    if args.mode == "test":
        test_model(args.config, args.weights)

    elif args.mode == "train":
        train_model(args)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()