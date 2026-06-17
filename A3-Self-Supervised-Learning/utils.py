import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms

CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
MAE_MEAN = [0.4914, 0.4822, 0.4465]
MAE_STD = [0.247, 0.243, 0.261]

EVAL_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

MAE_TEST_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MAE_MEAN, MAE_STD),
])


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def ensure_dirs():
    Path('saved').mkdir(exist_ok=True)
    Path('figures').mkdir(exist_ok=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def save_stats(stats, path):
    import json
    clean = {}
    for k, v in stats.items():
        if isinstance(v, (list, tuple)):
            clean[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
        elif isinstance(v, (np.floating, np.integer)):
            clean[k] = float(v)
        else:
            clean[k] = v
    with open(path, 'w') as f:
        json.dump(clean, f, indent=2)
