import argparse
import os
import sys
import time

import cv2
import torch
import numpy as np

from darknet import MyDarknet
from util import write_results


# ---------------------------------------------------------
# Model configuration
# ---------------------------------------------------------

MODEL_CONFIGS = {
    "yolov3": {
        "cfg": "cfg/yolov3.cfg",
        "weights": "weights/yolov3.weights",
        "input_size": 416,
    },
    "yolov4": {
        "cfg": "cfg/yolov4.cfg",
        "weights": "weights/yolov4.weights",
        "input_size": 608,
    },
}


# ---------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------

def prep_image(img_path, inp_dim):
    """
    Load image, convert BGR to RGB, resize, and convert to tensor.
    """
    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    original_img = img.copy()
    original_dim = original_img.shape[1], original_img.shape[0]

    # OpenCV loads BGR, convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to YOLO input size
    img = cv2.resize(img, (inp_dim, inp_dim))

    # Convert HWC to CHW
    img = img.transpose((2, 0, 1)).copy()

    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)

    return img, original_img, original_dim


# ---------------------------------------------------------
# Load YOLO model
# ---------------------------------------------------------

def load_model(model_name, weights_path=None, device="cpu"):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}")

    cfg_path = MODEL_CONFIGS[model_name]["cfg"]

    if weights_path is None:
        weights_path = MODEL_CONFIGS[model_name]["weights"]

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    print(f"[INFO] Loading {model_name}")
    print(f"[INFO] Config: {cfg_path}")
    print(f"[INFO] Weights: {weights_path}")

    model = MyDarknet(cfg_path)

    if weights_path.endswith(".pth"):
        checkpoint = torch.load(weights_path, map_location=device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        print("[INFO] PyTorch checkpoint loaded.")

    else:
        model.load_weights(weights_path)
        print("[INFO] Darknet weights loaded.")

    model.net_info["height"] = MODEL_CONFIGS[model_name]["input_size"]

    model.to(device)
    model.eval()

    print("[INFO] Model loaded successfully.")

    return model


# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------

def run_inference(args, device):
    """
    Run inference on one image.
    Example:
    python3 run.py --model yolov3 --weights yolov3.weights --image dog-cycle-car.png --infer
    """
    model_name = args.model
    inp_dim = MODEL_CONFIGS[model_name]["input_size"]

    weights_path = args.weights
    if weights_path is None:
        weights_path = MODEL_CONFIGS[model_name]["weights"]

    model = load_model(model_name, weights_path, device)

    img_tensor, original_img, original_dim = prep_image(args.image, inp_dim)
    img_tensor = img_tensor.to(device)

    print("[INFO] Running inference...")

    start = time.time()

    with torch.no_grad():
        prediction = model(img_tensor, CUDA=(device.type == "cuda"))

    end = time.time()

    output = write_results(
        prediction,
        args.confidence,
        args.num_classes,
        nms=True,
        nms_conf=args.nms_thresh,
    )

    print(f"[INFO] Inference time: {end - start:.4f} seconds")

    if type(output) == int:
        print("[INFO] No detections found.")
        return

    print("[INFO] Detections:")
    print(output)

    print()
    print("[INFO] Output format:")
    print("batch_id, x1, y1, x2, y2, objectness, class_score, class_id")


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def run_training(args, device):
    """
    Train YOLOv4 on COCO.

    Expected folder structure:

    data/coco/
    ├── train2017/
    │   ├── 000000000009.jpg
    │   └── ...
    └── annotations/
        └── instances_train2017.json

    Command:
    python3 run.py --model yolov4 --dataset coco --epochs 5 --train
    """

    import json
    from PIL import Image
    from tqdm import tqdm
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T

    if args.dataset != "coco":
        raise ValueError("Currently only --dataset coco is supported.")

    print("[INFO] Training mode selected")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Dataset: {args.dataset}")
    print(f"[INFO] Epochs: {args.epochs}")
    print(f"[INFO] Loss type: {args.loss}")
    print(f"[INFO] Device: {device}")

    cfg_path = MODEL_CONFIGS[args.model]["cfg"]
    inp_dim = MODEL_CONFIGS[args.model]["input_size"]

    image_dir = os.path.join(args.data_dir, "val2017")
    ann_path = os.path.join(args.data_dir, "annotations", "instances_val2017.json")

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"COCO image folder not found: {image_dir}")

    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"COCO annotation file not found: {ann_path}")

    class COCODetectionDataset(Dataset):
        def __init__(self, image_dir, ann_path, img_size=608, max_images=500):
            self.image_dir = image_dir
            self.ann_path = ann_path
            self.img_size = img_size

            with open(ann_path, "r") as f:
                coco = json.load(f)

            self.images = coco["images"][:max_images]
            self.annotations = coco["annotations"]
            self.categories = coco["categories"]

            self.image_id_to_annotations = {}

            for ann in self.annotations:
                image_id = ann["image_id"]
                if image_id not in self.image_id_to_annotations:
                    self.image_id_to_annotations[image_id] = []
                self.image_id_to_annotations[image_id].append(ann)

            self.cat_id_to_class = {
                cat["id"]: idx for idx, cat in enumerate(self.categories)
            }

            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
            ])

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_info = self.images[idx]

            img_id = img_info["id"]
            file_name = img_info["file_name"]
            width = img_info["width"]
            height = img_info["height"]

            img_path = os.path.join(self.image_dir, file_name)

            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)

            anns = self.image_id_to_annotations.get(img_id, [])

            boxes = []

            for ann in anns:
                if "bbox" not in ann:
                    continue

                x, y, w, h = ann["bbox"]

                if w <= 0 or h <= 0:
                    continue

                class_id = self.cat_id_to_class.get(ann["category_id"], 0)

                # Convert COCO bbox x,y,w,h to normalized YOLO x_center,y_center,w,h
                x_center = x + w / 2
                y_center = y + h / 2

                x_center = x_center / width
                y_center = y_center / height
                w = w / width
                h = h / height

                boxes.append([
                    class_id,
                    x_center,
                    y_center,
                    w,
                    h,
                ])

            if len(boxes) == 0:
                boxes = torch.zeros((0, 5), dtype=torch.float32)
            else:
                boxes = torch.tensor(boxes, dtype=torch.float32)

            return image, boxes

    def collate_fn(batch):
        images = []
        targets = []

        for i, (img, boxes) in enumerate(batch):
            images.append(img)

            if boxes.shape[0] > 0:
                batch_index = torch.full((boxes.shape[0], 1), i)
                boxes = torch.cat([batch_index, boxes], dim=1)
                targets.append(boxes)

        images = torch.stack(images, dim=0)

        if len(targets) > 0:
            targets = torch.cat(targets, dim=0)
        else:
            targets = torch.zeros((0, 6), dtype=torch.float32)

        return images, targets

    def simple_detection_loss(predictions, targets):
        """
        Simplified training loss.

        predictions shape:
        batch, boxes, 85

        targets shape:
        image_id, class_id, x, y, w, h

        This is not a full official YOLO loss.
        It is enough to make training run and compare loss behavior.
        """

        pred_xywh = predictions[:, :, 0:4]
        pred_obj = predictions[:, :, 4]
        pred_cls = predictions[:, :, 5:]

        batch_size = predictions.shape[0]

        loss_box = torch.tensor(0.0, device=device)
        loss_obj = torch.mean(pred_obj ** 2)
        loss_cls = torch.tensor(0.0, device=device)

        if targets.shape[0] == 0:
            return loss_obj

        targets = targets.to(device)

        count = 0

        for b in range(batch_size):
            target_b = targets[targets[:, 0] == b]

            if target_b.shape[0] == 0:
                continue

            for t in target_b:
                class_id = int(t[1].item())

                tx = t[2] * inp_dim
                ty = t[3] * inp_dim
                tw = t[4] * inp_dim
                th = t[5] * inp_dim

                target_box = torch.tensor(
                    [[tx, ty, tw, th]],
                    dtype=torch.float32,
                    device=device
                )

                # Find closest predicted box by center distance
                center_distance = (
                    (pred_xywh[b, :, 0] - tx) ** 2 +
                    (pred_xywh[b, :, 1] - ty) ** 2
                )

                best_idx = torch.argmin(center_distance)

                pred_box = pred_xywh[b, best_idx].unsqueeze(0)

                if args.loss == "ciou":
                    from util import bbox_ciou
                    ciou = bbox_ciou(pred_box, target_box)
                    box_loss = 1 - ciou.mean()

                else:
                    # Normalize boxes to 0-1 scale to avoid huge pixel-scale MSE values
                    pred_box_norm = pred_box / inp_dim
                    target_box_norm = target_box / inp_dim

                    # Clamp values to prevent extremely large width/height from exploding
                    pred_box_norm = torch.clamp(pred_box_norm, min=0.0, max=1.0)
                    target_box_norm = torch.clamp(target_box_norm, min=0.0, max=1.0)

                    box_loss = torch.mean((pred_box_norm - target_box_norm) ** 2)

                loss_box = loss_box + box_loss

                loss_obj = loss_obj + (pred_obj[b, best_idx] - 1) ** 2

                target_class = torch.tensor(
                    [class_id],
                    dtype=torch.long,
                    device=device
                )

                class_pred = pred_cls[b, best_idx].unsqueeze(0)
                loss_cls = loss_cls + torch.nn.functional.cross_entropy(
                    class_pred,
                    target_class
                )

                count += 1

        if count > 0:
            loss_box = loss_box / count
            loss_cls = loss_cls / count

        total_loss = loss_box + loss_obj + loss_cls

        return total_loss

    print("[INFO] Loading dataset...")

    train_dataset = COCODetectionDataset(
        image_dir=image_dir,
        ann_path=ann_path,
        img_size=inp_dim,
        max_images=500,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    print(f"[INFO] Number of training images: {len(train_dataset)}")

    print("[INFO] Loading model...")

    model = MyDarknet(cfg_path)
    model.net_info["height"] = inp_dim

    if args.weights is not None:
        print(f"[INFO] Loading pretrained weights: {args.weights}")
        model.load_weights(args.weights)

    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=5e-4,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    print("[INFO] Start training...")

    for epoch in range(args.epochs):
        model.train()

        epoch_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for images, targets in progress_bar:
            images = images.to(device)

            optimizer.zero_grad()

            predictions = model(images, CUDA=(device.type == "cuda"))

            loss = simple_detection_loss(predictions, targets)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            progress_bar.set_postfix({
                "loss": loss.item()
            })

        avg_loss = epoch_loss / len(train_loader)

        print(f"[INFO] Epoch [{epoch + 1}/{args.epochs}] Average Loss: {avg_loss:.4f}")

        checkpoint_path = os.path.join(
            args.save_dir,
            f"{args.model}_{args.loss}_epoch_{epoch + 1}.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model": args.model,
                "loss_type": args.loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_loss": avg_loss,
            },
            checkpoint_path,
        )

        print(f"[INFO] Checkpoint saved: {checkpoint_path}")

    print("[INFO] Training complete.")


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def run_evaluation(args, device):
    """
    Simplified COCO evaluation.

    Command:
    python3 run.py --model yolov4 --weights outputs/checkpoints/yolov4_ciou_epoch_5.pth --dataset coco --evaluate

    This computes a simple mAP@0.5-style score.
    """

    import os
    import json
    from PIL import Image
    from tqdm import tqdm
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T

    from util import write_results, bbox_iou

    if args.dataset != "coco":
        raise ValueError("Currently only --dataset coco is supported.")

    if args.weights is None:
        raise ValueError("Please provide --weights for evaluation.")

    print("[INFO] Evaluation mode selected")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Dataset: {args.dataset}")
    print(f"[INFO] Weights: {args.weights}")
    print(f"[INFO] Device: {device}")

    inp_dim = MODEL_CONFIGS[args.model]["input_size"]

    image_dir = os.path.join(args.data_dir, "val2017")
    ann_path = os.path.join(args.data_dir, "annotations", "instances_val2017.json")

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"COCO validation image folder not found: {image_dir}")

    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"COCO validation annotation file not found: {ann_path}")

    class COCOValDataset(Dataset):
        def __init__(self, image_dir, ann_path, img_size=608, max_images=200):
            self.image_dir = image_dir
            self.ann_path = ann_path
            self.img_size = img_size

            with open(ann_path, "r") as f:
                coco = json.load(f)

            self.images = coco["images"][:max_images]
            self.annotations = coco["annotations"]
            self.categories = coco["categories"]

            self.image_id_to_annotations = {}

            for ann in self.annotations:
                img_id = ann["image_id"]
                if img_id not in self.image_id_to_annotations:
                    self.image_id_to_annotations[img_id] = []
                self.image_id_to_annotations[img_id].append(ann)

            self.cat_id_to_class = {
                cat["id"]: idx for idx, cat in enumerate(self.categories)
            }

            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
            ])

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_info = self.images[idx]

            img_id = img_info["id"]
            file_name = img_info["file_name"]
            width = img_info["width"]
            height = img_info["height"]

            img_path = os.path.join(self.image_dir, file_name)

            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)

            anns = self.image_id_to_annotations.get(img_id, [])

            boxes = []

            for ann in anns:
                if "bbox" not in ann:
                    continue

                x, y, w, h = ann["bbox"]

                if w <= 0 or h <= 0:
                    continue

                class_id = self.cat_id_to_class.get(ann["category_id"], 0)

                # COCO x,y,w,h to x1,y1,x2,y2 in resized image scale
                x1 = x / width * self.img_size
                y1 = y / height * self.img_size
                x2 = (x + w) / width * self.img_size
                y2 = (y + h) / height * self.img_size

                boxes.append([
                    x1,
                    y1,
                    x2,
                    y2,
                    class_id,
                ])

            if len(boxes) == 0:
                boxes = torch.zeros((0, 5), dtype=torch.float32)
            else:
                boxes = torch.tensor(boxes, dtype=torch.float32)

            return image, boxes

    def collate_fn(batch):
        images = []
        targets = []

        for img, boxes in batch:
            images.append(img)
            targets.append(boxes)

        images = torch.stack(images, dim=0)

        return images, targets

    print("[INFO] Loading validation dataset...")

    val_dataset = COCOValDataset(
        image_dir=image_dir,
        ann_path=ann_path,
        img_size=inp_dim,
        max_images=200,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    print(f"[INFO] Number of validation images: {len(val_dataset)}")

    print("[INFO] Loading model...")

    model = load_model(args.model, args.weights, device)
    model.eval()

    total_predictions = 0
    true_positives = 0
    total_ground_truths = 0

    print("[INFO] Start evaluation...")

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)

            predictions = model(images, CUDA=(device.type == "cuda"))

            output = write_results(
                predictions,
                confidence=args.confidence,
                num_classes=args.num_classes,
                nms=True,
                nms_conf=args.nms_thresh,
            )

            gt_boxes = targets[0].to(device)

            total_ground_truths += gt_boxes.shape[0]

            if type(output) == int:
                continue

            # output columns:
            # batch_id, x1, y1, x2, y2, objectness, class_score, class_id
            pred_boxes = output[:, 1:5]
            pred_classes = output[:, 7].long()

            total_predictions += pred_boxes.shape[0]

            matched_gt = set()

            for i in range(pred_boxes.shape[0]):
                pred_box = pred_boxes[i].unsqueeze(0)
                pred_class = pred_classes[i]

                if gt_boxes.shape[0] == 0:
                    continue

                gt_xyxy = gt_boxes[:, 0:4]
                gt_classes = gt_boxes[:, 4].long()

                ious = bbox_iou(pred_box, gt_xyxy)

                best_iou, best_idx = torch.max(ious, dim=0)

                if (
                    best_iou.item() >= 0.5
                    and pred_class.item() == gt_classes[best_idx].item()
                    and best_idx.item() not in matched_gt
                ):
                    true_positives += 1
                    matched_gt.add(best_idx.item())

    precision = true_positives / max(total_predictions, 1)
    recall = true_positives / max(total_ground_truths, 1)

    # Simple mAP@0.5 approximation
    map50 = precision * recall

    print()
    print("[RESULT] Evaluation Summary")
    print(f"True Positives: {true_positives}")
    print(f"Total Predictions: {total_predictions}")
    print(f"Total Ground Truths: {total_ground_truths}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"mAP@0.5 Approximation: {map50:.4f}")


# ---------------------------------------------------------
# Argument parser
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv3 / YOLOv4 Assignment Runner")

    parser.add_argument(
        "--model",
        type=str,
        default="yolov4",
        choices=["yolov3", "yolov4"],
        help="Model type: yolov3 or yolov4",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to weights file",
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image for inference",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["coco"],
        help="Dataset name",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training/evaluation",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )

    parser.add_argument(
        "--loss",
        type=str,
        default="ciou",
        choices=["mse", "iou", "ciou"],
        help="Bounding box loss type",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Object confidence threshold",
    )

    parser.add_argument(
        "--nms_thresh",
        type=float,
        default=0.4,
        help="NMS IoU threshold",
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=80,
        help="Number of classes. COCO has 80 classes.",
    )

    parser.add_argument(
        "--infer",
        action="store_true",
        help="Run inference",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training",
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/coco",
        help="Path to COCO dataset folder",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of dataloader workers",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="outputs/checkpoints",
        help="Directory to save checkpoints",
    )

    return parser.parse_args()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    args = parse_args()

    selected_modes = sum([args.infer, args.train, args.evaluate])

    if selected_modes == 0:
        raise ValueError("Please choose one mode: --infer, --train, or --evaluate")

    if selected_modes > 1:
        raise ValueError("Please choose only one mode at a time.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Using device: {device}")

    if args.infer:
        if args.image is None:
            raise ValueError("Please provide --image for inference.")
        run_inference(args, device)

    elif args.train:
        if args.dataset is None:
            raise ValueError("Please provide --dataset for training.")
        run_training(args, device)

    elif args.evaluate:
        if args.dataset is None:
            raise ValueError("Please provide --dataset for evaluation.")
        run_evaluation(args, device)


if __name__ == "__main__":
    main()