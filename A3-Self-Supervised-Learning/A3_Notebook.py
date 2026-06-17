# This file preserves all code cells from the original notebook in order.
# Notebook: A3-Self-Supervised-Learning (1).ipynb


# ==================== Notebook code cell 1 ====================
# Original notebook install command:
# !pip install torch torchvision timm scikit-learn tqdm matplotlib numpy pillow -q

# ==================== Notebook code cell 2 ====================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.manifold import TSNE
import random, os, math, time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)
os.makedirs('saved', exist_ok=True)

CLASSES = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

EVAL_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

# ==================== Notebook code cell 5 ====================
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
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
    def __call__(self, x):
        return self.transform(x), self.transform(x)


class CIFAR10SSL(Dataset):
    def __init__(self, root='./data', train=True):
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        self.augment = SimCLRAugmentation()
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        x_i, x_j = self.augment(img)
        return x_i, x_j, label


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    def forward(self, z_i, z_j):
        N = z_i.shape[0]
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = torch.mm(z, z.T) / self.temperature
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, float('-inf'))
        labels = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)]).to(z.device)
        return F.cross_entropy(sim, labels)


class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.projector = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 128)
        )
    def forward(self, x_i, x_j):
        h_i = torch.flatten(self.encoder(x_i), 1)
        h_j = torch.flatten(self.encoder(x_j), 1)
        return self.projector(h_i), self.projector(h_j), h_i, h_j

# ==================== Notebook code cell 6 ====================
import time

# --- Train SimCLR ---
BATCH_SIZE, EPOCHS = 256, 10
train_loader = DataLoader(CIFAR10SSL(), batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=2, drop_last=True)
simclr    = SimCLR().to(device)
criterion = NTXentLoss(temperature=0.5)
optimizer = torch.optim.Adam(simclr.parameters(), lr=3e-4, weight_decay=1e-4)

simclr_losses = []
epoch_times = []
total_start = time.time()

for epoch in range(EPOCHS):
    simclr.train()
    ep = []
    t0 = time.time()
    for x_i, x_j, _ in tqdm(train_loader, desc=f'SimCLR {epoch+1}/{EPOCHS}'):
        x_i, x_j = x_i.to(device), x_j.to(device)
        z_i, z_j, _, _ = simclr(x_i, x_j)
        loss = criterion(z_i, z_j)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        ep.append(loss.item())
    elapsed = time.time() - t0
    epoch_times.append(elapsed)
    simclr_losses.append(np.mean(ep))
    print(f'Epoch {epoch+1:02d} | Loss: {np.mean(ep):.4f} | Time: {elapsed:.1f}s')

total_time = time.time() - total_start
print(f'\nTotal: {total_time/60:.1f} min  |  Avg/epoch: {np.mean(epoch_times):.1f}s')
torch.save(simclr.state_dict(), 'saved/simclr.pt')

# ==================== Notebook code cell 7 ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
ax1.plot(simclr_losses, marker='o')
ax1.set_title('SimCLR Training Loss'); 
ax1.set_xlabel('Epoch'); 
ax1.set_ylabel('NT-Xent Loss'); 
ax1.grid(True)
ax2.bar(range(1, len(epoch_times)+1), epoch_times, color='steelblue')
ax2.set_title('SimCLR Time per Epoch'); 
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Seconds'); 
ax2.grid(True, axis='y')
plt.tight_layout(); 
plt.show()

# ==================== Notebook code cell 9 ====================
simclr.load_state_dict(torch.load('saved/simclr.pt', map_location=device))
for p in simclr.encoder.parameters(): p.requires_grad = False

clf = nn.Linear(512, 10).to(device)

train_lbl = torchvision.datasets.CIFAR10('./data', train=True,  download=True, transform=EVAL_TF)
test_lbl  = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=EVAL_TF)
trl = DataLoader(train_lbl, batch_size=256, shuffle=True,  num_workers=2)
tel = DataLoader(test_lbl,  batch_size=256, shuffle=False, num_workers=2)

opt_clf = torch.optim.Adam(clf.parameters(), lr=1e-3)
for epoch in range(10):
    clf.train(); correct = total = 0
    for imgs, labels in tqdm(trl, desc=f'Linear Eval {epoch+1}/10'):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad(): h = torch.flatten(simclr.encoder(imgs), 1)
        loss = F.cross_entropy(clf(h), labels)
        opt_clf.zero_grad(); loss.backward(); opt_clf.step()
        correct += (clf(h).argmax(1) == labels).sum().item(); total += labels.size(0)
    print(f'  Train Acc: {correct/total*100:.2f}%')

clf.eval(); correct = total = 0
simclr_embeddings, simclr_labels = [], []
with torch.no_grad():
    for imgs, labels in tel:
        imgs, labels = imgs.to(device), labels.to(device)
        h = torch.flatten(simclr.encoder(imgs), 1)
        correct += (clf(h).argmax(1) == labels).sum().item(); total += labels.size(0)
        simclr_embeddings.append(h.cpu()); simclr_labels.append(labels.cpu())
simclr_embeddings = torch.cat(simclr_embeddings)
simclr_labels     = torch.cat(simclr_labels)
print(f'\n✅ SimCLR Linear Eval Test Accuracy: {correct/total*100:.2f}%')

# ==================== Notebook code cell 13 ====================
# ─── DINO Multi-Crop Augmentation ────────────────────────────────────────────

class DINOAugmentation:
    """
    Creates:
      - 2 global crops (large, scale 0.4–1.0)
      - n_local local crops (small, scale 0.05–0.4)
    Teacher only sees global crops; student sees all.
    """
    def __init__(self, image_size=32, n_local=4):
        normalize = transforms.Normalize([0.4914,0.4822,0.4465],[0.2023,0.1994,0.2010])
        flip_jitter = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.2,0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
            *flip_jitter,
            transforms.ToTensor(), normalize
        ])
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.05, 0.4)),
            *flip_jitter,
            transforms.ToTensor(), normalize
        ])
        self.n_local = n_local

    def __call__(self, img):
        global1 = self.global_transform(img)
        global2 = self.global_transform(img)
        locals_ = [self.local_transform(img) for _ in range(self.n_local)]
        return [global1, global2] + locals_   # teacher uses [0,1]; student uses all


class CIFAR10DINO(Dataset):
    def __init__(self, root='./data', train=True, n_local=4):
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        self.augment = DINOAugmentation(n_local=n_local)
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return self.augment(img), label

# ==================== Notebook code cell 14 ====================
# ─── DINO: Student & Teacher Networks ────────────────────────────────────────
import timm

class DINOHead(nn.Module):
    def __init__(self, in_dim=192, hidden_dim=512, out_dim=256, n_layers=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, out_dim, bias=False))
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.utils.weight_norm(nn.Linear(out_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


def build_dino_model(out_dim=256):
    vit = timm.create_model('vit_tiny_patch16_224', pretrained=False,
                             img_size=32, patch_size=4, num_classes=0)
    embed_dim = vit.embed_dim
    head = DINOHead(in_dim=embed_dim, out_dim=out_dim)
    return vit, head


student_vit, student_head = build_dino_model()
teacher_vit, teacher_head = build_dino_model()

student_vit, student_head = student_vit.to(device), student_head.to(device)
teacher_vit, teacher_head = teacher_vit.to(device), teacher_head.to(device)

teacher_vit.load_state_dict(student_vit.state_dict())
teacher_head.load_state_dict(student_head.state_dict())

for p in teacher_vit.parameters():  p.requires_grad = False
for p in teacher_head.parameters(): p.requires_grad = False

total = sum(p.numel() for p in student_vit.parameters()) + sum(p.numel() for p in student_head.parameters())
print(f'Student parameters: {total:,}')

# ==================== Notebook code cell 15 ====================
# ─── DINO Loss ────────────────────────────────────────────────────────────────

class DINOLoss(nn.Module):
    def __init__(self, out_dim=256, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer('center', torch.zeros(1, out_dim))

    def forward(self, student_out, teacher_out):
        # student_out: list of (N, out_dim) — all crops (global + local)
        # teacher_out: list of (N, out_dim) — global crops only (index 0, 1)

        s_probs = [F.log_softmax(s / self.student_temp, dim=-1) for s in student_out]
        t_probs = [F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach()
                   for t in teacher_out]

        total_loss = 0
        n_loss_terms = 0
        for t_idx, t_prob in enumerate(t_probs):
            for s_idx, s_log_prob in enumerate(s_probs):
                # skip same view: student global crop i vs teacher global crop i
                if s_idx == t_idx:
                    continue
                loss = -(t_prob * s_log_prob).sum(dim=-1).mean()
                total_loss += loss
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(torch.stack(teacher_out).mean(dim=0))
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_mean):
        self.center = self.center * self.center_momentum + teacher_mean * (1 - self.center_momentum)

# ==================== Notebook code cell 16 ====================
# ─── DINO Training ───────────────────────────────────────────────────────────

N_LOCAL   = 4
OUT_DIM   = 256
EPOCHS_D  = 10
BATCH_D   = 64
EMA_M     = 0.996

dino_dataset = CIFAR10DINO(n_local=N_LOCAL)

def dino_collate(batch):
    crops_list, labels = zip(*batch)
    n_views = len(crops_list[0])
    stacked = [torch.stack([crops_list[i][v] for i in range(len(crops_list))]) for v in range(n_views)]
    return stacked, torch.tensor(labels)

dino_loader = DataLoader(dino_dataset, batch_size=BATCH_D, shuffle=True,
                          num_workers=2, drop_last=True, collate_fn=dino_collate)

dino_loss_fn = DINOLoss(out_dim=OUT_DIM).to(device)
optimizer_d  = torch.optim.AdamW(
    list(student_vit.parameters()) + list(student_head.parameters()),
    lr=5e-4, weight_decay=0.04
)

dino_losses = []
dino_epoch_times = []
total_start = time.time()

for epoch in range(EPOCHS_D):
    student_vit.train(); student_head.train()
    ep = []
    t0 = time.time()

    for crops, _ in tqdm(dino_loader, desc=f'DINO {epoch+1}/{EPOCHS_D}'):
        crops = [c.to(device) for c in crops]
        student_out = [student_head(student_vit(c)) for c in crops]
        with torch.no_grad():
            teacher_out = [teacher_head(teacher_vit(crops[0])),
                           teacher_head(teacher_vit(crops[1]))]
        loss = dino_loss_fn(student_out, teacher_out)
        optimizer_d.zero_grad(); loss.backward(); optimizer_d.step()
        with torch.no_grad():
            for s_p, t_p in zip(student_vit.parameters(), teacher_vit.parameters()):
                t_p.data = EMA_M * t_p.data + (1 - EMA_M) * s_p.data
            for s_p, t_p in zip(student_head.parameters(), teacher_head.parameters()):
                t_p.data = EMA_M * t_p.data + (1 - EMA_M) * s_p.data
        ep.append(loss.item())

    elapsed = time.time() - t0
    dino_epoch_times.append(elapsed)
    dino_losses.append(np.mean(ep))
    print(f'Epoch {epoch+1:02d} | Loss: {np.mean(ep):.4f} | Center norm: {dino_loss_fn.center.norm().item():.4f} | Time: {elapsed:.1f}s')

total_time = time.time() - total_start
print(f'\nTotal: {total_time/60:.1f} min  |  Avg/epoch: {np.mean(dino_epoch_times):.1f}s')
torch.save({'student_vit': student_vit.state_dict(),
            'student_head': student_head.state_dict()}, 'saved/dino.pt')

# ==================== Notebook code cell 17 ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
ax1.plot(dino_losses, marker='o', color='darkorange')
ax1.set_title('DINO Training Loss'); 
ax1.set_xlabel('Epoch'); 
ax1.set_ylabel('Cross-Entropy'); 
ax1.grid(True)
ax2.bar(range(1, len(dino_epoch_times)+1), dino_epoch_times, color='darkorange')
ax2.set_title('DINO Time per Epoch'); 
ax2.set_xlabel('Epoch'); 
ax2.set_ylabel('Seconds'); 
ax2.grid(True, axis='y')
plt.tight_layout(); 
plt.show()

# ==================== Notebook code cell 19 ====================
ckpt = torch.load('saved/dino.pt', map_location=device)
student_vit.load_state_dict(ckpt['student_vit'])
for p in student_vit.parameters(): p.requires_grad = False

embed_dim = student_vit.embed_dim
clf_dino  = nn.Linear(embed_dim, 10).to(device)
opt_dino_clf = torch.optim.Adam(clf_dino.parameters(), lr=1e-3)

for epoch in range(10):
    clf_dino.train(); correct = total = 0
    for imgs, labels in tqdm(trl, desc=f'DINO Linear Eval {epoch+1}/10'):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad(): h = student_vit(imgs)
        loss = F.cross_entropy(clf_dino(h), labels)
        opt_dino_clf.zero_grad(); loss.backward(); opt_dino_clf.step()
        correct += (clf_dino(h).argmax(1)==labels).sum().item(); total += labels.size(0)
    print(f'  Train Acc: {correct/total*100:.2f}%')

clf_dino.eval(); correct = total = 0
dino_embeddings, dino_labels = [], []
with torch.no_grad():
    for imgs, labels in tel:
        imgs, labels = imgs.to(device), labels.to(device)
        h = student_vit(imgs)
        correct += (clf_dino(h).argmax(1)==labels).sum().item(); total += labels.size(0)
        dino_embeddings.append(h.cpu()); dino_labels.append(labels.cpu())
dino_embeddings = torch.cat(dino_embeddings)
dino_labels     = torch.cat(dino_labels)
print(f'\n✅ DINO Linear Eval Test Accuracy: {correct/total*100:.2f}%')

# ==================== Notebook code cell 21 ====================
student_vit.eval()

img_mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
img_std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1)

attentions = {}
attn_module = student_vit.blocks[-1].attn
_original_forward = attn_module.forward

def _patched_attn_forward(x, **kwargs):
    B, N, C = x.shape
    qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    attn_w = (q @ k.transpose(-2, -1)) * attn_module.scale
    attn_w = attn_w.softmax(dim=-1)
    attentions['last'] = attn_w.detach()
    attn_w = attn_module.attn_drop(attn_w)
    x = (attn_w @ v).transpose(1, 2).reshape(B, N, C)
    x = attn_module.proj(x)
    x = attn_module.proj_drop(x)
    return x

attn_module.forward = _patched_attn_forward

raw_test = torchvision.datasets.CIFAR10('./data', train=False, transform=EVAL_TF)
img_loader = DataLoader(raw_test, batch_size=1, shuffle=True)

n_heads = student_vit.blocks[-1].attn.num_heads
patch_h = patch_w = 32 // 4   # patch_size=4 → 8×8 grid

fig, axes = plt.subplots(5, n_heads + 1, figsize=(2*(n_heads+1), 12))

sample_iter = iter(img_loader)
for row in range(5):
    img_tensor, label = next(sample_iter)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        _ = student_vit(img_tensor)

    attn = attentions['last']
    cls_attn = attn[0, :, 0, 1:]  # (n_heads, n_patches)

    img_disp = torch.clamp(img_tensor[0].cpu() * img_std + img_mean, 0, 1).permute(1,2,0).numpy()
    axes[row][0].imshow(img_disp)
    axes[row][0].set_title(f'{CLASSES[label.item()]}', fontsize=9)
    axes[row][0].axis('off')

    for h in range(n_heads):
        head_map = cls_attn[h].reshape(patch_h, patch_w).cpu().numpy()
        head_map = (head_map - head_map.min()) / (head_map.max() - head_map.min() + 1e-8)
        head_up = np.array(Image.fromarray((head_map * 255).astype(np.uint8)).resize((32, 32)))
        axes[row][h+1].imshow(img_disp, alpha=0.4)
        axes[row][h+1].imshow(head_up, cmap='hot', alpha=0.7, vmin=0, vmax=255)
        if row == 0: axes[row][h+1].set_title(f'Head {h+1}', fontsize=8)
        axes[row][h+1].axis('off')

plt.suptitle(
    'DINO Self-Attention Maps: [CLS] token → patches\n'
    'Emergent object segmentation — no segmentation labels used!',
    fontsize=11, 
    y=1.01
)
plt.tight_layout(); plt.show()

# ==================== Notebook code cell 24 ====================
# ─── Patch Embedding ──────────────────────────────────────────────────────────
# For CIFAR-10 (32×32) with patch_size=4: (32/4)² = 64 patches

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3, embed_dim=192):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)       # (N, embed_dim, H//p, W//p)
        x = x.flatten(2)       # (N, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (N, n_patches, embed_dim)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """Fixed 2D sinusoidal positional embeddings. Returns (grid_size**2, embed_dim)."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)

    def sincos_1d(pos, dim):
        omega = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
        out = pos.reshape(-1, 1) * omega.reshape(1, -1)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    half = embed_dim // 2
    emb = np.concatenate([sincos_1d(grid_h.flatten(), half),
                           sincos_1d(grid_w.flatten(), half)], axis=1)
    return torch.tensor(emb, dtype=torch.float32)

# ==================== Notebook code cell 25 ====================
# ─── MAE Encoder ─────────────────────────────────────────────────────────────
# Key innovation: only processes VISIBLE (unmasked) patches.

class MAEEncoder(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3,
                 embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.0,
                 mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, embed_dim)

        pos_embed = get_2d_sincos_pos_embed(embed_dim, img_size // patch_size)
        self.register_buffer('pos_embed', pos_embed.unsqueeze(0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.0, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def random_masking(self, x):
        N, L, D = x.shape
        n_keep = int(L * (1 - self.mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)

        ids_keep = ids_shuffle[:, :n_keep]
        x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(N, L, device=x.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)

        return x_visible, mask, ids_restore

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x_vis, mask, ids_restore = self.random_masking(x)
        x_vis = self.norm(self.transformer(x_vis))
        return x_vis, mask, ids_restore

# ==================== Notebook code cell 26 ====================
# ─── MAE Decoder ─────────────────────────────────────────────────────────────
# Intentionally shallow (4 layers, 128-dim) — forces semantic info into encoder.

class MAEDecoder(nn.Module):
    def __init__(self, n_patches, patch_size=4, in_ch=3,
                 encoder_dim=192, decoder_dim=128,
                 depth=4, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        patch_pixels = patch_size * patch_size * in_ch
        grid_size = int(math.sqrt(n_patches))

        self.embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        pos_embed = get_2d_sincos_pos_embed(decoder_dim, grid_size)
        self.register_buffer('pos_embed', pos_embed.unsqueeze(0))

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=num_heads,
            dim_feedforward=int(decoder_dim * mlp_ratio),
            dropout=0.0, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, patch_pixels)

        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x_vis, ids_restore):
        N = x_vis.size(0)
        x = self.embed(x_vis)

        n_masked = ids_restore.size(1) - x.size(1)
        mask_tokens = self.mask_token.expand(N, n_masked, -1)
        x_full = torch.cat([x, mask_tokens], dim=1)
        x_full = torch.gather(
            x_full, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, x_full.size(-1))
        )

        x_full = x_full + self.pos_embed
        x_full = self.norm(self.transformer(x_full))
        return self.pred(x_full)  # (N, n_patches, patch_pixels)

# ==================== Notebook code cell 27 ====================
# ─── MAE Full Model + Loss ────────────────────────────────────────────────────
# Loss: MSE on masked patches only (not visible ones).
# norm_pix_loss: normalize each patch before MSE — prevents low-variance patches from dominating.

class MAE(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3,
                 encoder_dim=192, encoder_depth=6, encoder_heads=3,
                 decoder_dim=128, decoder_depth=4, decoder_heads=4,
                 mask_ratio=0.75, norm_pix_loss=True):
        super().__init__()
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.norm_pix_loss = norm_pix_loss

        self.encoder = MAEEncoder(
            img_size, patch_size, in_ch,
            encoder_dim, encoder_depth, encoder_heads,
            mask_ratio=mask_ratio
        )
        n_patches = self.encoder.patch_embed.n_patches
        self.decoder = MAEDecoder(
            n_patches, patch_size, in_ch,
            encoder_dim, decoder_dim, decoder_depth, decoder_heads
        )

    def patchify(self, imgs):
        p = self.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], self.in_ch, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        return x.reshape(imgs.shape[0], h * w, p * p * self.in_ch)

    def forward(self, imgs):
        x_vis, mask, ids_restore = self.encoder(imgs)
        pred = self.decoder(x_vis, ids_restore)

        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var  = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss, pred, mask


mae_model = MAE(
    img_size=32, patch_size=4, in_ch=3,
    encoder_dim=192, encoder_depth=6, encoder_heads=3,
    decoder_dim=128, decoder_depth=4, decoder_heads=4,
    mask_ratio=0.75, norm_pix_loss=True,
).to(device)

enc_params = sum(p.numel() for p in mae_model.encoder.parameters())
dec_params = sum(p.numel() for p in mae_model.decoder.parameters())
print(f'MAE Encoder params: {enc_params:,}')
print(f'MAE Decoder params: {dec_params:,}  ({100*dec_params/enc_params:.1f}% of encoder)')

# ==================== Notebook code cell 28 ====================
# ─── MAE Training ────────────────────────────────────────────────────────────

EPOCHS_M  = 10
BATCH_M   = 128
LR_M      = 1.5e-4

mae_mean = [0.4914, 0.4822, 0.4465]
mae_std  = [0.247,  0.243,  0.261]

mae_train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mae_mean, mae_std),
])
mae_test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mae_mean, mae_std),
])

mae_train_ds = torchvision.datasets.CIFAR10('./data', train=True,  transform=mae_train_tf, download=True)
mae_loader   = DataLoader(mae_train_ds, batch_size=BATCH_M, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)

optimizer_m = torch.optim.AdamW(mae_model.parameters(), lr=LR_M, weight_decay=0.05,
                                 betas=(0.9, 0.95))
scheduler_m = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_m, T_max=EPOCHS_M)

mae_losses = []
mae_epoch_times = []
mae_model.train()
total_start = time.time()

for epoch in range(EPOCHS_M):
    ep = []
    t0 = time.time()
    for imgs, _ in tqdm(mae_loader, desc=f'MAE {epoch+1}/{EPOCHS_M}'):
        imgs = imgs.to(device)
        loss, _, _ = mae_model(imgs)
        optimizer_m.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(mae_model.parameters(), max_norm=1.0)
        optimizer_m.step()
        ep.append(loss.item())
    scheduler_m.step()
    elapsed = time.time() - t0
    mae_epoch_times.append(elapsed)
    mae_losses.append(np.mean(ep))
    print(f'Epoch {epoch+1:02d} | Recon Loss: {np.mean(ep):.4f} | Time: {elapsed:.1f}s')

total_time = time.time() - total_start
print(f'\nTotal: {total_time/60:.1f} min  |  Avg/epoch: {np.mean(mae_epoch_times):.1f}s')
torch.save(mae_model.encoder.state_dict(), 'saved/mae_encoder.pt')

# ==================== Notebook code cell 29 ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
ax1.plot(mae_losses, marker='o', color='steelblue')
ax1.set_title('MAE Training Loss'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('MSE (masked patches)'); ax1.grid(True)
ax2.bar(range(1, len(mae_epoch_times)+1), mae_epoch_times, color='steelblue')
ax2.set_title('MAE Time per Epoch'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Seconds'); ax2.grid(True, axis='y')
plt.tight_layout(); plt.show()

# ==================== Notebook code cell 31 ====================
mae_model.encoder.load_state_dict(torch.load('saved/mae_encoder.pt', map_location=device))
mae_model.encoder.eval()
for p in mae_model.encoder.parameters(): p.requires_grad = False

mae_model.encoder.mask_ratio = 0.0  # disable masking for evaluation

clf_mae = nn.Linear(mae_model.encoder.embed_dim, 10).to(device)
opt_mae_clf = torch.optim.Adam(clf_mae.parameters(), lr=1e-3)

mae_clf_train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize(mae_mean, mae_std)
])
mae_clf_train_ds = torchvision.datasets.CIFAR10('./data', train=True,  transform=mae_clf_train_tf)
mae_clf_test_ds  = torchvision.datasets.CIFAR10('./data', train=False, transform=mae_test_tf)
mae_trl = DataLoader(mae_clf_train_ds, batch_size=256, shuffle=True,  num_workers=2)
mae_tel = DataLoader(mae_clf_test_ds,  batch_size=256, shuffle=False, num_workers=2)

for ep in range(10):
    clf_mae.train(); correct = total = 0
    for imgs, labels in tqdm(mae_trl, desc=f'MAE Linear Eval {ep+1}/10'):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            x_vis, _, _ = mae_model.encoder(imgs)
            feats = x_vis.mean(dim=1)  # global average pooling over patch tokens
        logits = clf_mae(feats)
        loss = F.cross_entropy(logits, labels)
        opt_mae_clf.zero_grad(); loss.backward(); opt_mae_clf.step()
        correct += (logits.argmax(1) == labels).sum().item(); total += labels.size(0)
    print(f'  Train Acc: {correct/total*100:.2f}%')

clf_mae.eval(); correct = total = 0
mae_embeddings, mae_labels_list = [], []
with torch.no_grad():
    for imgs, labels in mae_tel:
        imgs, labels = imgs.to(device), labels.to(device)
        x_vis, _, _ = mae_model.encoder(imgs)
        feats = x_vis.mean(dim=1)
        correct += (clf_mae(feats).argmax(1) == labels).sum().item(); total += labels.size(0)
        mae_embeddings.append(feats.cpu()); mae_labels_list.append(labels.cpu())
mae_embeddings  = torch.cat(mae_embeddings)
mae_labels_list = torch.cat(mae_labels_list)
print(f'\n✅ MAE Linear Eval Test Accuracy: {correct/total*100:.2f}%')

# ==================== Notebook code cell 33 ====================
mae_model.encoder.mask_ratio = 0.75  # restore masking for visualization
mae_model.eval()

imgs_viz, _ = next(iter(DataLoader(
    torchvision.datasets.CIFAR10('./data', train=False, transform=mae_test_tf),
    batch_size=8, shuffle=True
)))
imgs_viz = imgs_viz.to(device)

with torch.no_grad():
    loss_viz, pred, mask = mae_model(imgs_viz)

p = mae_model.patch_size
h_g = w_g = 32 // p

def unpatchify(patches, p, h, w, in_ch=3):
    N = patches.size(0)
    x = patches.reshape(N, h, w, p, p, in_ch)
    x = x.permute(0, 5, 1, 3, 2, 4)
    return x.reshape(N, in_ch, h*p, w*p)

pred_imgs = unpatchify(pred.cpu(), p, h_g, w_g)

mean_t = torch.tensor(mae_mean).view(3,1,1)
std_t  = torch.tensor(mae_std).view(3,1,1)
orig_np = (imgs_viz.cpu() * std_t + mean_t).clamp(0,1).permute(0,2,3,1).numpy()
pred_np = (pred_imgs       * std_t + mean_t).clamp(0,1).permute(0,2,3,1).numpy()

mask_exp = mask.cpu().view(-1, h_g, w_g).unsqueeze(1)
mask_exp = mask_exp.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)
mask_np  = mask_exp.expand(-1,3,-1,-1).permute(0,2,3,1).numpy()
masked_np = orig_np.copy()
masked_np[mask_np.astype(bool)] = 0.5

N_show = 4
fig, axes = plt.subplots(3, N_show, figsize=(2*N_show, 6))
for row, (imgs_row, title) in enumerate(zip([orig_np, masked_np, pred_np],
                                             ['Original', 'Masked (75%)', 'Reconstructed'])):
    axes[row, 0].set_ylabel(title, fontsize=10)
    for col in range(N_show):
        axes[row, col].imshow(imgs_row[col])
        axes[row, col].axis('off')
plt.suptitle('MAE Reconstruction (CIFAR-10)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('saved/mae_reconstruction.png', dpi=120, bbox_inches='tight')
plt.show()
print(f'Reconstruction loss: {loss_viz.item():.4f}')

# ==================== Notebook code cell 35 ====================
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
colors = plt.cm.tab10(np.linspace(0, 1, 10))

for ax, (name, emb, lbls) in zip(axes, [
    ('SimCLR (ResNet-18)', simclr_embeddings, simclr_labels),
    ('DINO (ViT-Tiny)',    dino_embeddings,   dino_labels),
    ('MAE (ViT)',          mae_embeddings,    mae_labels_list),
]):
    idx = np.random.choice(len(emb), 2000, replace=False)
    proj = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(emb[idx].numpy())
    for c in range(10):
        mask_c = lbls[idx].numpy() == c
        ax.scatter(proj[mask_c,0], proj[mask_c,1], c=[colors[c]], label=CLASSES[c], alpha=0.6, s=10)
    ax.set_title(name, fontsize=12)
    ax.legend(fontsize=7, markerscale=2)
    ax.axis('off')

plt.suptitle('t-SNE: Learned Representations on CIFAR-10 (no labels used in training)', fontsize=13)
plt.tight_layout(); plt.show()
