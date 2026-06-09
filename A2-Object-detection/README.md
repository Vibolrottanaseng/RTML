# A2-01-Object-Detection
## Overview
This assignment focuses on extending a YOLOv3 Darknet-based object detection implementation to support YOLOv4 in PyTorch. The main work involved parsing the YOLOv4 configuration file, adding Mish activation, supporting maxpool layers, handling route layers with multiple inputs, and loading pretrained YOLOv4 weights. The model was tested using COCO images, with input images resized to 608×608 and processed in RGB format.

## Summmary
The training experiment compared YOLOv4 using standard MSE/IoU-style bounding box loss with YOLOv4 using CIoU loss. Both models were trained for 5 epochs on a COCO subset, and their performance was evaluated using a simplified mAP calculation. The assignment also compared the trained YOLOv4 models with a pretrained YOLOv3 model. YOLOv3 achieved the best mAP because it used fully pretrained COCO weights, while the YOLOv4 models were trained only for a small number of epochs. Finally, the assignment explains why YOLOv3 is faster than Faster R-CNN.

## Traning Script

```bash
#Download COCO dataset
wget -nc http://images.cocodataset.org/zips/val2017.zip
wget -nc http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip -q val2017.zip
unzip -q annotations_trainval2017.zip

#YOLOv3 Inference
python3 run.py --model yolov3 --weights weights/yolov3.weights --image dog-cycle-car.png --infer

#YOLOv3 Evaluation
python3 run.py --model yolov3 --weights weights/yolov3.weights --dataset coco --evaluate

#YOLOv4 training
python3 run.py --model yolov4 --dataset coco --epochs 5 --loss ciou --batch_size 4 --train

#YOLOv4 Evaluation
python3 run.py --model yolov4 --weights outputs/checkpoints/yolov4_ciou_epoch_5.pth --dataset coco --loss ciou --evaluate

#YOLOv4 MSE training
python3 run.py --model yolov4 --dataset coco --epochs 5 --loss mse --batch_size 4 --lr 1e-5 --train

#YOLOv4 MSE Evaluation
python3 run.py --model yolov4 --weights outputs/checkpoints/yolov4_mse_epoch_5.pth --dataset coco --loss mse --evaluate

```

## Results

