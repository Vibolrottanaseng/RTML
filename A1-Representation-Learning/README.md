# A1: Representation Learning

## Overview
This assignment focuses on representation learning for image classification using the CIFAR-10 dataset. The main objective is to implement, train, evaluate, and compare different deep learning architectures, including convolutional neural networks and Vision Transformers.

The project includes training several models from scratch, such as AlexNet, GoogLeNet, ResNet-18, and ViT-Small. Each model is implemented in a separate file to keep the code organized. A main training script, ```run.py```, is used to select the model, train it on CIFAR-10, test saved weights, and save the best-performing model checkpoints.

The assignment also explores the effect of architectural design choices. For AlexNet, the comparison includes training with and without Local Response Normalization. For GoogLeNet, the model uses Inception modules and auxiliary classifiers. ResNet-18 is used to demonstrate the benefit of residual connections, which help reduce the vanishing gradient problem and make deeper networks easier to train. ViT-Small introduces a transformer-based approach for image classification by splitting images into patches and using self-attention to learn visual representations.

In addition to training models from scratch, the assignment requires fine-tuning pretrained models such as ResNet-18 and ViT-B/16. These pretrained models are initialized with weights learned from large-scale datasets and then adapted to CIFAR-10. This allows comparison between models trained from scratch and models using transfer learning.

The final goal is to compare all models based on training performance, test accuracy, number of parameters, and training time. The results help show the strengths and weaknesses of CNN-based models and transformer-based models for image classification tasks.

## Traning Script

```bash
# Train from scratch
python3 run.py --model alexnet      --dataset cifar10 --epochs 10 --batch_size 64 --train
python3 run.py --model googlenet    --dataset cifar10 --epochs 25 --batch_size 64 --train
python3 run.py --model resnet18     --dataset cifar10 --epochs 20 --batch_size 64 --train
python3 run.py --model vit_small    --dataset cifar10 --epochs 20 --batch_size 64 --train

# Fine-tune pretrained models
python3 run.py --model resnet18_pretrained  --dataset cifar10 --epochs 15 --batch_size 64 --train
python3 run.py --model vit_b16_pretrained   --dataset cifar10 --epochs 15 --batch_size 64 --train

# Test saved weights
python3 run.py --model resnet18 --dataset cifar10 --teset --weights resnet18_cifar10.pth
```
## Results

| Model | # Params | Test Accuracy | Time/epoch | Architecture Type |
|---|---|---|---|---|
| AlexNet (from scratch) | 57,044,810 | 65.38% | ~53s | CNN |
| GoogLeNet (from scratch) | 10,326,350 | 83.81% | ~40s | CNN + Inception |
| ResNet-18 (from scratch) | 11,173,962 | 87.93% | ~22s | CNN + Skip connections |
| ResNet-18 (pretrained) | 11,181,642 | 91.76% | ~77s| CNN + Skip connections |
| ViT-Small (from scratch) | 3,195,146 | 61.31% | ~33s | Transformer |
| ViT-B/16 (pretrained, fine-tuned) | ? | ? | ? | Transformer |

## Discussions
