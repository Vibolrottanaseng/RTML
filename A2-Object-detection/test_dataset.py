from torch.utils.data import DataLoader
from utils.datasets import ListDataset, collate_fn


dataset = ListDataset("data/train.txt", img_size=608)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,
)

paths, imgs, targets = next(iter(loader))

print("Number of images:", len(paths))
print("Image batch shape:", imgs.shape)
print("Targets shape:", targets.shape)
print("First image path:", paths[0])
print("First targets:")
print(targets[:5])