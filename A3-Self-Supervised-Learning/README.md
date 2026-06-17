# A3 Self-Supervised Learning Project

## Overview
This assignment focuses on self-supervised learning for image representation learning using the CIFAR-10 dataset. The main objective is to study how deep learning models can learn useful visual features without relying on labeled data during pretraining. Instead of training directly with class labels, the models learn from image transformations, reconstruction tasks, and teacher-student learning strategies. After pretraining, the quality of the learned representations is evaluated using linear classification and visualization techniques.

## Model
In this assignment, two self-supervised learning methods are studied: DINO and Masked Autoencoder (MAE). DINO is a self-distillation method that trains a student network to match the output of a teacher network under different augmented views of the same image. It uses global and local crops, teacher momentum updates, and centering to prevent representation collapse while MAE, learns by masking a large portion of image patches and training the model to reconstruct the missing parts. This helps the model to understand image structure and visual patterns.

## Ablation
The assignment consist of ablation studies to analyze the effect of important design choices. For DINO, the experiments compare the default setting, a version without centering, and a version without local crops. These experiments help show how centering and multi-crop augmentation affect representation learning. For MAE, different mask ratios are tested to observe how the amount of hidden image information affects reconstruction quality and downstream classification performance.

The learned representations are evaluated using linear evaluation accuracy, training loss curves, and visualizations. 

## Reslts
| Model | Linear Eval Acc | Time/epoch | Notes |
|---|---|---|---|
| DINO (Default) | 69.14% | 139.6s | (2 global + 4 local, with centering)  |
| DINO (no centering) | 37.19% | 138.0s |No centering (```- self.center``` removed)  |
| DINO (no local crops) | 62.82% | 55.6s | ablation ```n_local = 0``` |
| MAE mask=0.75 | 45.63% | 11.8s | Default reconstruction |
| MAE mask=0.50 | 82.83% | 11.9s | masking ablation |
| MAE mask=0.25 | 41.38% | 12.1s | masking ablation |

## Visualization 
#### - Loss curves for DINO and MAE
<img src=".\figures\dino_mae_training_curves.png" alt="dino_and_mae" width="700">

#### - MAE reconstruction grid (original / masked / reconstructed)
<img src=".\figures\mae_reconstruction_grid.png" alt="mae_recon_grid" width="700">

#### - DINO attention map grid (10 images × all heads, from original exercises)
<img src=".\figures\dino_attention_grid.png" alt="mae_recon_grid" width="700">

#### - SNE comparison: DINO vs MAE
<img src=".\figures\tsne_dino_vs_mae.png" alt="mae_recon_grid" width="700">

## Training Script
```bash
# Train
python3 run.py --model dino --epochs 50 --train
python3 run.py --model mae --epochs 50 --train

# Linear evaluation
python3 run.py --model dino --weights saved/dino.pt --evaluate --linear
python3 run.py --model mae --weights saved/mae_encoder_mask075.pt --evaluate --linear

# Ablations
python3 run.py --model dino --no-centering --epochs 50 --train
python3 run.py --model dino --n-local 0 --epochs 50 --train
python3 run.py --model mae --mask-ratio 0.25 --epochs 50 --train
python3 run.py --model mae --mask-ratio 0.50 --epochs 50 --train
```

## Dicussion 
### 1. DINO (Self-distillation on ViT + centering trick)
Removing centering can cause collapse because DINO relies on the teacher network to Removing centering can cause collapse because the teacher outputs may become biased toward the same prediction for many images. Centering balances the teacher distribution, so without it the student may learn similar features for different images instead of meaningful representations.

Removing local crops hurts representation quality because DINO loses local-to-global learning. Local crops force the model to match small image regions with the full image meaning. Without them, the model sees less diverse views and may learn weaker, less robust features.

### 2. MAE (Reconstruct 75%-masked patches)

Very low masking, for example 0.25, can give lower reconstruction loss because the task is too easy. Most image patches are still visible, so the model can reconstruct missing parts using nearby pixels without learning strong semantic features.

However, this can produce worse representations because the model is not forced to understand the whole image structure or object-level meaning. A higher mask ratio, such as 0.75, makes reconstruction harder and encourages the encoder to learn more useful visual representations for downstream tasks like classification.

3. Three-way comparison 
   | Metric | DINO | MAE |
   |---|---|---|
   | Backbone | ViT-Tiny | ViT |
   | Needs negative pairs? | No | No |
   | Needs EMA teacher? | Yes | No |
   | Linear Eval Accuracy | 69.14% | 45.63% |
   | Training time/epoch | 139.6s | 11.8s |
   | t-SNE cluster quality (1–5) | 4 | 3 |
   | Has interpretable attention maps? | Yes | No |

   DINO learns by matching teacher and student views of the same image. This encourages the model to focus on meaningful object regions, so its attention maps often highlight objects clearly. While MAE learns by reconstructing masked patches. Its attention is mainly used to help rebuild missing pixels, not to localize semantic objects. Because of this, MAE attention maps are usually less clear and less useful for interpretation.

a) MAE became more popular for large-scale general pre-training for two main reasons. 
    1. it is computationally efficient because the encoder only processes the visible patches, while the masked patches are reconstructed by a lightweight decoder. This makes MAE easier to scale to large datasets and large models.
    2. MAE uses a simple reconstruction objective, which is stable and general. It does not need complex teacher-student training, multi-crop augmentation, or collapse-prevention tricks like centering.

b) DINO is still preferred for some computer vision only tasks such as segmentation because it often learns stronger semantic and spatial representations. Its attention maps can naturally focus on object regions, so the learned features are useful for tasks that require object localization and dense visual understanding.

I would choose DINO for medical images when the task requires strong semantic and spatial representations, such as lesion localization, organ segmentation, abnormality detection, or attention-based interpretation. DINO learns by matching different views of the same image using a teacher-student framework, which encourages the model to focus on meaningful regions rather than only reconstructing pixels. This is useful in medical imaging because disease patterns are often local and subtle, and interpretability is important.







