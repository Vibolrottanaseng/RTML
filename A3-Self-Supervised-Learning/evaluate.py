import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm
import matplotlib.pyplot as plt

from data import get_eval_loaders, get_mae_eval_loaders
from models import MAE, SimCLR, build_dino_model
from utils import CLASSES, CIFAR_MEAN, CIFAR_STD, MAE_MEAN, MAE_STD, MAE_TEST_TF


def linear_eval_simclr(device, weights='saved/simclr.pt', epochs=10, batch_size=256):
    simclr = SimCLR().to(device)
    simclr.load_state_dict(torch.load(weights, map_location=device))
    for p in simclr.encoder.parameters():
        p.requires_grad = False

    clf = nn.Linear(512, 10).to(device)
    trl, tel = get_eval_loaders(batch_size=batch_size)
    opt_clf = torch.optim.Adam(clf.parameters(), lr=1e-3)

    for epoch in range(epochs):
        clf.train()
        correct = total = 0
        for imgs, labels in tqdm(trl, desc=f'Linear Eval {epoch + 1}/{epochs}'):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                h = torch.flatten(simclr.encoder(imgs), 1)
            loss = F.cross_entropy(clf(h), labels)
            opt_clf.zero_grad()
            loss.backward()
            opt_clf.step()
            correct += (clf(h).argmax(1) == labels).sum().item()
            total += labels.size(0)
        print(f'Train Acc: {correct / total * 100:.2f}%')

    clf.eval()
    correct = total = 0
    embeddings, labels_all = [], []
    with torch.no_grad():
        for imgs, labels in tel:
            imgs, labels = imgs.to(device), labels.to(device)
            h = torch.flatten(simclr.encoder(imgs), 1)
            correct += (clf(h).argmax(1) == labels).sum().item()
            total += labels.size(0)
            embeddings.append(h.cpu())
            labels_all.append(labels.cpu())
    embeddings = torch.cat(embeddings)
    labels_all = torch.cat(labels_all)
    acc = correct / total * 100
    print(f'\nSimCLR Linear Eval Test Accuracy: {acc:.2f}%')
    return acc, embeddings, labels_all


def linear_eval_dino(device, weights='saved/dino.pt', epochs=10, batch_size=256):
    student_vit, _ = build_dino_model()
    ckpt = torch.load(weights, map_location=device)
    student_vit.load_state_dict(ckpt['student_vit'])
    student_vit = student_vit.to(device)
    for p in student_vit.parameters():
        p.requires_grad = False

    embed_dim = student_vit.embed_dim
    clf_dino = nn.Linear(embed_dim, 10).to(device)
    trl, tel = get_eval_loaders(batch_size=batch_size)
    opt_dino_clf = torch.optim.Adam(clf_dino.parameters(), lr=1e-3)

    for epoch in range(epochs):
        clf_dino.train()
        correct = total = 0
        for imgs, labels in tqdm(trl, desc=f'DINO Linear Eval {epoch + 1}/{epochs}'):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                h = student_vit(imgs)
            loss = F.cross_entropy(clf_dino(h), labels)
            opt_dino_clf.zero_grad()
            loss.backward()
            opt_dino_clf.step()
            correct += (clf_dino(h).argmax(1) == labels).sum().item()
            total += labels.size(0)
        print(f'Train Acc: {correct / total * 100:.2f}%')

    clf_dino.eval()
    correct = total = 0
    embeddings, labels_all = [], []
    with torch.no_grad():
        for imgs, labels in tel:
            imgs, labels = imgs.to(device), labels.to(device)
            h = student_vit(imgs)
            correct += (clf_dino(h).argmax(1) == labels).sum().item()
            total += labels.size(0)
            embeddings.append(h.cpu())
            labels_all.append(labels.cpu())
    embeddings = torch.cat(embeddings)
    labels_all = torch.cat(labels_all)
    acc = correct / total * 100
    print(f'\nDINO Linear Eval Test Accuracy: {acc:.2f}%')
    return acc, embeddings, labels_all


def linear_eval_mae(device, weights='saved/mae_encoder.pt', epochs=10, batch_size=256, mask_ratio=0.75):
    mae_model = MAE(mask_ratio=mask_ratio).to(device)
    mae_model.encoder.load_state_dict(torch.load(weights, map_location=device))
    mae_model.encoder.eval()
    for p in mae_model.encoder.parameters():
        p.requires_grad = False
    mae_model.encoder.mask_ratio = 0.0

    clf_mae = nn.Linear(mae_model.encoder.embed_dim, 10).to(device)
    mae_trl, mae_tel = get_mae_eval_loaders(batch_size=batch_size)
    opt_mae_clf = torch.optim.Adam(clf_mae.parameters(), lr=1e-3)

    for ep in range(epochs):
        clf_mae.train()
        correct = total = 0
        for imgs, labels in tqdm(mae_trl, desc=f'MAE Linear Eval {ep + 1}/{epochs}'):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                x_vis, _, _ = mae_model.encoder(imgs)
                feats = x_vis.mean(dim=1)
            logits = clf_mae(feats)
            loss = F.cross_entropy(logits, labels)
            opt_mae_clf.zero_grad()
            loss.backward()
            opt_mae_clf.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        print(f'Train Acc: {correct / total * 100:.2f}%')

    clf_mae.eval()
    correct = total = 0
    embeddings, labels_all = [], []
    with torch.no_grad():
        for imgs, labels in mae_tel:
            imgs, labels = imgs.to(device), labels.to(device)
            x_vis, _, _ = mae_model.encoder(imgs)
            feats = x_vis.mean(dim=1)
            correct += (clf_mae(feats).argmax(1) == labels).sum().item()
            total += labels.size(0)
            embeddings.append(feats.cpu())
            labels_all.append(labels.cpu())
    embeddings = torch.cat(embeddings)
    labels_all = torch.cat(labels_all)
    acc = correct / total * 100
    print(f'\nMAE Linear Eval Test Accuracy: {acc:.2f}%')
    return acc, embeddings, labels_all


def unpatchify(patches, p, h, w, in_ch=3):
    N = patches.size(0)
    x = patches.reshape(N, h, w, p, p, in_ch)
    x = x.permute(0, 5, 1, 3, 2, 4)
    return x.reshape(N, in_ch, h * p, w * p)


def visualize_mae_reconstruction(device, weights='saved/mae_encoder.pt', mask_ratio=0.75,
                                  output_path='saved/mae_reconstruction.png'):
    import torchvision
    from torch.utils.data import DataLoader

    mae_model = MAE(mask_ratio=mask_ratio).to(device)
    mae_model.encoder.load_state_dict(torch.load(weights, map_location=device))
    mae_model.encoder.mask_ratio = mask_ratio
    mae_model.eval()

    imgs_viz, _ = next(iter(DataLoader(
        torchvision.datasets.CIFAR10('./data', train=False, transform=MAE_TEST_TF, download=True),
        batch_size=8,
        shuffle=True,
    )))
    imgs_viz = imgs_viz.to(device)

    with torch.no_grad():
        loss_viz, pred, mask = mae_model(imgs_viz)

    p = mae_model.patch_size
    h_g = w_g = 32 // p
    pred_imgs = unpatchify(pred.cpu(), p, h_g, w_g)

    mean_t = torch.tensor(MAE_MEAN).view(3, 1, 1)
    std_t = torch.tensor(MAE_STD).view(3, 1, 1)
    orig_np = (imgs_viz.cpu() * std_t + mean_t).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    pred_np = (pred_imgs * std_t + mean_t).clamp(0, 1).permute(0, 2, 3, 1).numpy()

    mask_exp = mask.cpu().view(-1, h_g, w_g).unsqueeze(1)
    mask_exp = mask_exp.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)
    mask_np = mask_exp.expand(-1, 3, -1, -1).permute(0, 2, 3, 1).numpy()
    masked_np = orig_np.copy()
    masked_np[mask_np.astype(bool)] = 0.5

    n_show = 4
    fig, axes = plt.subplots(3, n_show, figsize=(2 * n_show, 6))
    for row, (imgs_row, title) in enumerate(zip([orig_np, masked_np, pred_np],
                                                 ['Original', 'Masked', 'Reconstructed'])):
        axes[row, 0].set_ylabel(title, fontsize=10)
        for col in range(n_show):
            axes[row, col].imshow(imgs_row[col])
            axes[row, col].axis('off')
    plt.suptitle('MAE Reconstruction on CIFAR-10', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f'Reconstruction loss: {loss_viz.item():.4f}')


def plot_tsne_dino_mae(dino_embeddings, dino_labels, mae_embeddings, mae_labels, output_path='saved/tsne_dino_mae.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for ax, (name, emb, lbls) in zip(axes, [
        ('DINO ViT-Tiny', dino_embeddings, dino_labels),
        ('MAE ViT', mae_embeddings, mae_labels),
    ]):
        n = min(len(emb), 2000)
        idx = np.random.choice(len(emb), n, replace=False)
        proj = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(emb[idx].numpy())
        for c in range(10):
            mask_c = lbls[idx].numpy() == c
            ax.scatter(proj[mask_c, 0], proj[mask_c, 1], c=[colors[c]], label=CLASSES[c], alpha=0.6, s=10)
        ax.set_title(name, fontsize=12)
        ax.legend(fontsize=7, markerscale=2)
        ax.axis('off')
    plt.suptitle('t-SNE: DINO vs MAE Learned Representations on CIFAR-10', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.show()
