import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(
        self,
        in_channels,
        ch1x1,
        ch3x3_reduce,
        ch3x3,
        ch5x5_reduce,
        ch5x5,
        pool_proj
    ):
        super(Inception, self).__init__()

        # 1x1 branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # 1x1 -> 3x3 branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 1x1 -> 5x5 branch
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )

        # maxpool -> 1x1 branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        super(AuxiliaryClassifier, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10, aux_logits=True):
        super(GoogLeNet, self).__init__()

        self.aux_logits = aux_logits

        # For CIFAR-10 image size: 3 x 32 x 32
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Inception 3
        self.inception3a = Inception(
            in_channels=192,
            ch1x1=64,
            ch3x3_reduce=96,
            ch3x3=128,
            ch5x5_reduce=16,
            ch5x5=32,
            pool_proj=32
        )

        self.inception3b = Inception(
            in_channels=256,
            ch1x1=128,
            ch3x3_reduce=128,
            ch3x3=192,
            ch5x5_reduce=32,
            ch5x5=96,
            pool_proj=64
        )

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception 4
        self.inception4a = Inception(
            in_channels=480,
            ch1x1=192,
            ch3x3_reduce=96,
            ch3x3=208,
            ch5x5_reduce=16,
            ch5x5=48,
            pool_proj=64
        )

        self.inception4b = Inception(
            in_channels=512,
            ch1x1=160,
            ch3x3_reduce=112,
            ch3x3=224,
            ch5x5_reduce=24,
            ch5x5=64,
            pool_proj=64
        )

        self.inception4c = Inception(
            in_channels=512,
            ch1x1=128,
            ch3x3_reduce=128,
            ch3x3=256,
            ch5x5_reduce=24,
            ch5x5=64,
            pool_proj=64
        )

        self.inception4d = Inception(
            in_channels=512,
            ch1x1=112,
            ch3x3_reduce=144,
            ch3x3=288,
            ch5x5_reduce=32,
            ch5x5=64,
            pool_proj=64
        )

        self.inception4e = Inception(
            in_channels=528,
            ch1x1=256,
            ch3x3_reduce=160,
            ch3x3=320,
            ch5x5_reduce=32,
            ch5x5=128,
            pool_proj=128
        )

        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception 5
        self.inception5a = Inception(
            in_channels=832,
            ch1x1=256,
            ch3x3_reduce=160,
            ch3x3=320,
            ch5x5_reduce=32,
            ch5x5=128,
            pool_proj=128
        )

        self.inception5b = Inception(
            in_channels=832,
            ch1x1=384,
            ch3x3_reduce=192,
            ch3x3=384,
            ch5x5_reduce=48,
            ch5x5=128,
            pool_proj=128
        )

        # Auxiliary classifiers
        self.aux1 = AuxiliaryClassifier(512, num_classes)
        self.aux2 = AuxiliaryClassifier(528, num_classes)

        # Final classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pre_layers(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool1(x)

        x = self.inception4a(x)

        aux1 = None
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        aux2 = None
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool2(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.training and self.aux_logits:
            return x, aux1, aux2

        return x