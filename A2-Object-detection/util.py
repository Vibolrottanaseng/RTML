from __future__ import division

import torch
import numpy as np


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)

    return tensor_res


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    """
    Converts raw YOLO output feature map into detection predictions.

    Output shape:
    batch_size x num_boxes x bbox_attrs

    bbox_attrs = 5 + num_classes
    where 5 = x, y, w, h, objectness
    """
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride

    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(
        batch_size,
        bbox_attrs * num_anchors,
        grid_size * grid_size
    )

    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(
        batch_size,
        grid_size * grid_size * num_anchors,
        bbox_attrs
    )

    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # Sigmoid x, y, objectness
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add grid offsets
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    device = prediction.device

    x_offset = torch.FloatTensor(a).view(-1, 1).to(device)
    y_offset = torch.FloatTensor(b).view(-1, 1).to(device)

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors)
    x_y_offset = x_y_offset.view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # Width and height
    anchors = torch.FloatTensor(anchors).to(device)

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)

    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # Class scores
    prediction[:, :, 5:5 + num_classes] = torch.sigmoid(
        prediction[:, :, 5:5 + num_classes]
    )

    # Resize boxes back to input image scale
    prediction[:, :, :4] *= stride

    return prediction


def bbox_iou(box1, box2):
    """
    Calculates IoU between two sets of boxes.

    Boxes should be in corner format:
    x1, y1, x2, y2
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = (
        box1[:, 0],
        box1[:, 1],
        box1[:, 2],
        box1[:, 3],
    )

    b2_x1, b2_y1, b2_x2, b2_y2 = (
        box2[:, 0],
        box2[:, 1],
        box2[:, 2],
        box2[:, 3],
    )

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def bbox_ciou(box1, box2):
    """
    Complete IoU, CIoU, used in YOLOv4.

    Boxes should be in center format:
    x, y, w, h

    Returns CIoU score.
    Loss can be:
    ciou_loss = 1 - bbox_ciou(pred_box, target_box)
    """
    b1_x, b1_y, b1_w, b1_h = (
        box1[:, 0],
        box1[:, 1],
        box1[:, 2],
        box1[:, 3],
    )

    b2_x, b2_y, b2_w, b2_h = (
        box2[:, 0],
        box2[:, 1],
        box2[:, 2],
        box2[:, 3],
    )

    b1_x1 = b1_x - b1_w / 2
    b1_y1 = b1_y - b1_h / 2
    b1_x2 = b1_x + b1_w / 2
    b1_y2 = b1_y + b1_h / 2

    b2_x1 = b2_x - b2_w / 2
    b2_y1 = b2_y - b2_h / 2
    b2_x2 = b2_x + b2_w / 2
    b2_y2 = b2_y + b2_h / 2

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                 torch.clamp(inter_y2 - inter_y1, min=0)

    b1_area = torch.clamp(b1_x2 - b1_x1, min=0) * \
              torch.clamp(b1_y2 - b1_y1, min=0)

    b2_area = torch.clamp(b2_x2 - b2_x1, min=0) * \
              torch.clamp(b2_y2 - b2_y1, min=0)

    union_area = b1_area + b2_area - inter_area + 1e-16

    iou = inter_area / union_area

    # Center distance
    center_distance = (b1_x - b2_x) ** 2 + (b1_y - b2_y) ** 2

    # Enclosing box diagonal
    enclose_x1 = torch.min(b1_x1, b2_x1)
    enclose_y1 = torch.min(b1_y1, b2_y1)
    enclose_x2 = torch.max(b1_x2, b2_x2)
    enclose_y2 = torch.max(b1_y2, b2_y2)

    c2 = (enclose_x2 - enclose_x1) ** 2 + \
         (enclose_y2 - enclose_y1) ** 2 + 1e-16

    # Aspect ratio penalty
    v = (4 / (np.pi ** 2)) * torch.pow(
        torch.atan(b1_w / (b1_h + 1e-16)) -
        torch.atan(b2_w / (b2_h + 1e-16)),
        2
    )

    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-16)

    ciou = iou - (center_distance / c2 + alpha * v)

    return ciou


def write_results(prediction, confidence, num_classes, nms=True, nms_conf=0.4):
    """
    Applies confidence threshold and Non-Max Suppression.

    Output columns:
    batch_id, x1, y1, x2, y2, objectness, max_class_score, class_id
    """
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    box_corner = prediction.new(prediction.shape)

    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2

    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)

    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]

        max_conf, max_conf_score = torch.max(
            image_pred[:, 5:5 + num_classes],
            1
        )

        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)

        seq = (
            image_pred[:, :5],
            max_conf,
            max_conf_score
        )

        image_pred = torch.cat(seq, 1)

        non_zero_ind = torch.nonzero(image_pred[:, 4])

        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        except:
            continue

        if image_pred_.shape[0] == 0:
            continue

        img_classes = unique(image_pred_[:, -1])

        for cls in img_classes:
            cls_mask = image_pred_ * (
                image_pred_[:, -1] == cls
            ).float().unsqueeze(1)

            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()

            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            conf_sort_index = torch.sort(
                image_pred_class[:, 4],
                descending=True
            )[1]

            image_pred_class = image_pred_class[conf_sort_index]

            idx = image_pred_class.size(0)

            if nms:
                for i in range(idx):
                    try:
                        ious = bbox_iou(
                            image_pred_class[i].unsqueeze(0),
                            image_pred_class[i + 1:]
                        )
                    except ValueError:
                        break
                    except IndexError:
                        break

                    iou_mask = (ious < nms_conf).float().unsqueeze(1)

                    image_pred_class[i + 1:] *= iou_mask

                    non_zero_ind = torch.nonzero(
                        image_pred_class[:, 4]
                    ).squeeze()

                    image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            batch_ind = image_pred_class.new(
                image_pred_class.size(0), 1
            ).fill_(ind)

            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    try:
        return output
    except:
        return 0