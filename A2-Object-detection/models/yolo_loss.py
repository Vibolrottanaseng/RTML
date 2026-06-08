import torch
import torch.nn as nn
import torch.nn.functional as F


def xywh_to_xyxy(boxes):
    """
    Convert boxes from:
        x_center, y_center, width, height

    To:
        x1, y1, x2, y2
    """
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]

    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    return torch.stack([x1, y1, x2, y2], dim=-1)


def bbox_iou_xyxy(box1, box2, eps=1e-7):
    """
    IoU for boxes in x1, y1, x2, y2 format.
    """

    b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h

    area1 = torch.clamp(b1_x2 - b1_x1, min=0) * torch.clamp(b1_y2 - b1_y1, min=0)
    area2 = torch.clamp(b2_x2 - b2_x1, min=0) * torch.clamp(b2_y2 - b2_y1, min=0)

    union = area1 + area2 - inter_area + eps

    return inter_area / union


def bbox_ciou(box1, box2, eps=1e-7):
    """
    CIoU for boxes in x_center, y_center, width, height format.
    Coordinates should be normalized between 0 and 1.
    """

    box1_xyxy = xywh_to_xyxy(box1)
    box2_xyxy = xywh_to_xyxy(box2)

    iou = bbox_iou_xyxy(box1_xyxy, box2_xyxy, eps)

    # Center distance
    b1_x, b1_y = box1[..., 0], box1[..., 1]
    b2_x, b2_y = box2[..., 0], box2[..., 1]

    center_dist = (b1_x - b2_x) ** 2 + (b1_y - b2_y) ** 2

    # Enclosing box diagonal distance
    c_x1 = torch.min(box1_xyxy[..., 0], box2_xyxy[..., 0])
    c_y1 = torch.min(box1_xyxy[..., 1], box2_xyxy[..., 1])
    c_x2 = torch.max(box1_xyxy[..., 2], box2_xyxy[..., 2])
    c_y2 = torch.max(box1_xyxy[..., 3], box2_xyxy[..., 3])

    c_diag = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps

    # Aspect ratio term
    w1 = torch.clamp(box1[..., 2], min=eps)
    h1 = torch.clamp(box1[..., 3], min=eps)
    w2 = torch.clamp(box2[..., 2], min=eps)
    h2 = torch.clamp(box2[..., 3], min=eps)

    v = (4 / (torch.pi ** 2)) * torch.pow(
        torch.atan(w2 / h2) - torch.atan(w1 / h1),
        2
    )

    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (center_dist / c_diag + alpha * v)

    return ciou


class SimpleYOLOLoss(nn.Module):
    """
    Simplified YOLO training loss for this assignment.

    loss_type:
        "normal" = MSE bbox loss
        "ciou"   = CIoU bbox loss
    """

    def __init__(self, num_classes=80, loss_type="normal"):
        super().__init__()
        self.num_classes = num_classes
        self.loss_type = loss_type

        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        """
        outputs:
            list of 3 tensors:
            [B, 3, G, G, 85]

        targets:
            [num_boxes, 6]
            columns: image_index, class_id, x, y, w, h
        """

        device = outputs[0].device

        total_box_loss = torch.tensor(0.0, device=device)
        total_obj_loss = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)

        # Use the highest resolution output for simplified training
        pred = outputs[0]

        batch_size, num_anchors, grid_size, _, attrs = pred.shape

        # Prediction components
        pred_xy = torch.sigmoid(pred[..., 0:2])
        pred_wh = torch.sigmoid(pred[..., 2:4])
        pred_obj = pred[..., 4]
        pred_cls = pred[..., 5:]

        obj_target = torch.zeros_like(pred_obj)

        if targets.size(0) == 0:
            obj_loss = self.bce(pred_obj, obj_target)
            return obj_loss, {
                "box_loss": 0.0,
                "obj_loss": obj_loss.item(),
                "cls_loss": 0.0,
                "total_loss": obj_loss.item(),
            }

        box_losses = []
        cls_losses = []

        for target in targets:
            img_idx = int(target[0].item())
            class_id = int(target[1].item())
            tx, ty, tw, th = target[2], target[3], target[4], target[5]

            # Convert normalized center to grid cell
            gx = tx * grid_size
            gy = ty * grid_size

            gi = int(torch.clamp(gx.long(), 0, grid_size - 1).item())
            gj = int(torch.clamp(gy.long(), 0, grid_size - 1).item())

            # Use anchor 0 for simple version
            anchor_idx = 0

            obj_target[img_idx, anchor_idx, gj, gi] = 1.0

            # Local cell coordinates
            target_xy = torch.stack([gx - gi, gy - gj]).to(device)
            target_wh = torch.stack([tw, th]).to(device)

            pred_box = torch.cat([
                pred_xy[img_idx, anchor_idx, gj, gi],
                pred_wh[img_idx, anchor_idx, gj, gi],
            ])

            target_box = torch.cat([target_xy, target_wh])

            if self.loss_type == "normal":
                box_loss = self.mse(pred_box, target_box)

            elif self.loss_type == "ciou":
                # For CIoU use normalized global center
                pred_global_x = (gi + pred_xy[img_idx, anchor_idx, gj, gi, 0]) / grid_size
                pred_global_y = (gj + pred_xy[img_idx, anchor_idx, gj, gi, 1]) / grid_size

                pred_global_box = torch.stack([
                    pred_global_x,
                    pred_global_y,
                    pred_wh[img_idx, anchor_idx, gj, gi, 0],
                    pred_wh[img_idx, anchor_idx, gj, gi, 1],
                ])

                target_global_box = torch.stack([tx, ty, tw, th]).to(device)

                ciou = bbox_ciou(
                    pred_global_box.unsqueeze(0),
                    target_global_box.unsqueeze(0)
                )

                box_loss = 1.0 - ciou.mean()

            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

            class_target = torch.zeros(self.num_classes, device=device)
            class_target[class_id] = 1.0

            cls_loss = self.bce(
                pred_cls[img_idx, anchor_idx, gj, gi],
                class_target
            )

            box_losses.append(box_loss)
            cls_losses.append(cls_loss)

        if len(box_losses) > 0:
            total_box_loss = torch.stack(box_losses).mean()

        if len(cls_losses) > 0:
            total_cls_loss = torch.stack(cls_losses).mean()

        total_obj_loss = self.bce(pred_obj, obj_target)

        total_loss = total_box_loss + total_obj_loss + total_cls_loss

        return total_loss, {
            "box_loss": float(total_box_loss.item()),
            "obj_loss": float(total_obj_loss.item()),
            "cls_loss": float(total_cls_loss.item()),
            "total_loss": float(total_loss.item()),
        }