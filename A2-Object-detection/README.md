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
| Model | Dataset | mAP | Time/epoch | Notes |
|---|---|---:|---:|---|
| YOLOv3 (pretrained) | COCO | 0.4444 | — | inference/evaluation only |
| YOLOv4 (IoU / MSE loss) | COCO | 0.0000 | ~41 sec | trained from scratch |
| YOLOv4 (CIoU loss) | COCO | 0.0103 | ~59 sec | loss comparison |

## Discussion 
CIoU loss improved the YOLOv4 model compared with the standard MSE/IoU loss because it produced a higher mAP result in the experiment. The standard MSE loss decreased during training, but it still generated many false positive detections, resulting in very low precision and near-zero mAP. CIoU was more effective because it considers not only bounding box overlap, but also center distance and aspect ratio, giving the model a better localization signal. The main challenges in training on COCO were the large dataset size, long training time, GPU memory limits, and unstable loss behavior when using standard MSE for bounding box regression.

# A20-02 Image Segmentation with U-Net
## Overview
This assignment focuses on image segmentation, where the goal is to classify each pixel in an image into a meaningful category. In this exercise, I use the Oxford-IIIT Pet dataset and train a U-Net model with a ResNet-18 encoder to generate segmentation masks for pet images. The main experiment compares the standard U-Net with skip connections against a modified version without skip connections to understand how skip connections affect segmentation quality. The models are evaluated using mean Intersection over Union (mIoU) and training time per epoch. Overall, the assignment demonstrates why encoder-decoder architectures are useful for pixel-level prediction and how skip connections help recover fine spatial details such as object boundaries.


## Traning Script
I use this script to run inside the notebook and just creating ``run.py``
```bash
# 1. Baseline — ResNet-18 encoder + skip connections
python3 run.py --model unet_resnet18         --dataset oxford_pet --epochs 20 --train

# 2. Ablation — same ResNet-18 encoder, skip connections REMOVED
python3 run.py --model unet_resnet18_no_skip --dataset oxford_pet --epochs 20 --train

# Evaluate saved model
python3 run.py --model unet_resnet18 --weights unet_resnet18_pet.pt --dataset oxford_pet --evaluate

```

## Results

| Model | Encoder | Skip connections | Val mIoU | Time/epoch |
|---|---|---|---|---|
| `unet_resnet18` | ResNet-18 (ImageNet) | ✅ | 0.7570 | ~20 |
| `unet_resnet18_no_skip` | ResNet-18 (ImageNet) | ❌ | 0.6875 | ~22 |

## Discussion

Skip connections improved the segmentation performance because they allow the decoder to recover fine spatial details from the encoder. This is especially important in image segmentation because the model must classify every pixel, including object boundaries. Without skip connections, the decoder relies mostly on compressed low-resolution features, so the predicted masks can become blurry or less accurate.

Choose **U-Net** when the task is semantic segmentation, when we only need to classify each pixel into a class, such as pet/background, road/sky etc. 

Choose **Mask R-CNN** when the task is instance segmentation, like when we need to separate each individual object, such as detecting and segmenting each dog, car, or person separately.