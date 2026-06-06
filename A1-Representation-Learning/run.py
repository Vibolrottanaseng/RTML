import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models

from models.alexnet import AlexNetNoLRN
from models.googlenet import GoogLeNet
from models.resnet18 import ResNet18
from models.vit_small import ViTSmall


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def get_dataloaders(dataset="cifar10", batch_size=64, image_size=32):
    if dataset.lower() != "cifar10":
        raise ValueError("Only cifar10 is supported in this assignment.")

    transform_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        ),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        ),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return trainloader, testloader


def get_model(model_name, num_classes=10):
    if model_name == "alexnet":
        model = AlexNetNoLRN(num_classes=num_classes)

    elif model_name == "googlenet":
        model = GoogLeNet(num_classes=num_classes)

    elif model_name == "resnet18":
        model = ResNet18(num_classes=num_classes)

    elif model_name == "vit_small":
        model = ViTSmall(
            image_size=32,
            patch_size=4,
            num_classes=num_classes,
            embed_dim=256,
            depth=6,
            num_heads=8,
            mlp_dim=512,
            dropout=0.1
        )

    elif model_name == "resnet18_pretrained":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "vit_b16_pretrained":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads = nn.Linear(768, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model

def get_image_size(model_name):
    if model_name in ["alexnet","resnet18_pretrained", "vit_b16_pretrained"]:
        return 224

    # If you use original AlexNet, change this to 224.
    # If you use CIFAR AlexNet, keep 32.
    return 32


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, trainloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in trainloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        if isinstance(outputs, tuple):
            main_output, aux1, aux2 = outputs

            loss_main = criterion(main_output, labels)
            loss_aux1 = criterion(aux1, labels)
            loss_aux2 = criterion(aux2, labels)

            loss = loss_main + 0.3 * loss_aux1 + 0.3 * loss_aux2
            outputs_for_acc = main_output
        else:
            loss = criterion(outputs, labels)
            outputs_for_acc = outputs

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs_for_acc.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(trainloader)
    train_acc = 100.0 * correct / total

    return train_loss, train_acc


def test_model(model, testloader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = running_loss / len(testloader)
    test_acc = 100.0 * correct / total

    return test_loss, test_acc


def train_model(model, trainloader, testloader, criterion, optimizer, device, epochs, save_path):
    best_acc = 0.0

    for epoch in range(epochs):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, trainloader, criterion, optimizer, device
        )

        test_loss, test_acc = test_model(
            model, testloader, criterion, device
        )

        epoch_time = time.time() - start_time

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.2f}% "
            f"Test Loss: {test_loss:.4f} "
            f"Test Acc: {test_acc:.2f}% "
            f"Time: {epoch_time:.2f}s"
        )

        if test_acc > best_acc:
            best_acc = test_acc

            if save_path is not None:
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model to {save_path}")

    print("Best Test Accuracy:", best_acc)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--weights", type=str, default=None)

    args = parser.parse_args()

    if not args.train and not args.test:
        raise ValueError("Please use either --train or --test")

    device = get_device()

    image_size = get_image_size(args.model)

    trainloader, testloader = get_dataloaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
        image_size=image_size
    )

    model = get_model(args.model, num_classes=10)
    model = model.to(device)

    print("Model:", args.model)
    print("Trainable parameters:", count_parameters(model))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs("weights", exist_ok=True)

    if args.weights is None:
        save_path = f"weights/{args.model}_{args.dataset}.pth"
    else:
        save_path = args.weights

    if args.train:
        train_model(
            model=model,
            trainloader=trainloader,
            testloader=testloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=args.epochs,
            save_path=save_path
        )

    if args.test:
        if args.weights is None:
            raise ValueError("Please provide --weights when using --test")

        model.load_state_dict(torch.load(args.weights, map_location=device))

        test_loss, test_acc = test_model(
            model=model,
            testloader=testloader,
            criterion=criterion,
            device=device
        )

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()