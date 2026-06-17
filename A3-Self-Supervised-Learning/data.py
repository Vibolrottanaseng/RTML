import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from utils import CIFAR_MEAN, CIFAR_STD, EVAL_TF, MAE_MEAN, MAE_STD, MAE_TEST_TF


class SimCLRAugmentation:
    """Returns two independently augmented views of the same image."""
    def __init__(self, image_size=32):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class CIFAR10SSL(Dataset):
    def __init__(self, root='./data', train=True):
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        self.augment = SimCLRAugmentation()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        x_i, x_j = self.augment(img)
        return x_i, x_j, label


class DINOAugmentation:
    """
    Creates:
      - 2 global crops, scale 0.4 to 1.0
      - n_local local crops, scale 0.05 to 0.4
    Teacher only sees global crops; student sees all crops.
    """
    def __init__(self, image_size=32, n_local=4):
        normalize = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        flip_jitter = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
            *flip_jitter,
            transforms.ToTensor(),
            normalize,
        ])
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.05, 0.4)),
            *flip_jitter,
            transforms.ToTensor(),
            normalize,
        ])
        self.n_local = n_local

    def __call__(self, img):
        global1 = self.global_transform(img)
        global2 = self.global_transform(img)
        locals_ = [self.local_transform(img) for _ in range(self.n_local)]
        return [global1, global2] + locals_


class CIFAR10DINO(Dataset):
    def __init__(self, root='./data', train=True, n_local=4):
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        self.augment = DINOAugmentation(n_local=n_local)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return self.augment(img), label


def dino_collate(batch):
    crops_list, labels = zip(*batch)
    n_views = len(crops_list[0])
    stacked = [torch.stack([crops_list[i][v] for i in range(len(crops_list))]) for v in range(n_views)]
    return stacked, torch.tensor(labels)


def get_eval_loaders(batch_size=256, num_workers=2, root='./data'):
    train_lbl = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=EVAL_TF)
    test_lbl = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=EVAL_TF)
    trl = DataLoader(train_lbl, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    tel = DataLoader(test_lbl, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trl, tel


def get_simclr_loader(batch_size=256, num_workers=2, root='./data'):
    return DataLoader(
        CIFAR10SSL(root=root),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )


def get_dino_loader(batch_size=64, n_local=4, num_workers=2, root='./data'):
    dataset = CIFAR10DINO(root=root, n_local=n_local)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=dino_collate,
    )


def get_mae_loader(batch_size=128, num_workers=2, root='./data'):
    mae_train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MAE_MEAN, MAE_STD),
    ])
    mae_train_ds = torchvision.datasets.CIFAR10(root, train=True, transform=mae_train_tf, download=True)
    return DataLoader(
        mae_train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def get_mae_eval_loaders(batch_size=256, num_workers=2, root='./data'):
    mae_clf_train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MAE_MEAN, MAE_STD),
    ])
    mae_clf_train_ds = torchvision.datasets.CIFAR10(root, train=True, transform=mae_clf_train_tf, download=True)
    mae_clf_test_ds = torchvision.datasets.CIFAR10(root, train=False, transform=MAE_TEST_TF, download=True)
    mae_trl = DataLoader(mae_clf_train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    mae_tel = DataLoader(mae_clf_test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return mae_trl, mae_tel
