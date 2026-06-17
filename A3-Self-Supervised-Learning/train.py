import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data import get_dino_loader, get_mae_loader, get_simclr_loader
from models import DINOLoss, MAE, NTXentLoss, SimCLR, build_dino_model
from utils import count_parameters


def train_simclr(device, epochs=10, batch_size=256, lr=3e-4, weight_decay=1e-4,
                 save_path='saved/simclr.pt'):
    train_loader = get_simclr_loader(batch_size=batch_size)
    simclr = SimCLR().to(device)
    criterion = NTXentLoss(temperature=0.5)
    optimizer = torch.optim.Adam(simclr.parameters(), lr=lr, weight_decay=weight_decay)

    losses = []
    epoch_times = []
    total_start = time.time()

    for epoch in range(epochs):
        simclr.train()
        ep = []
        t0 = time.time()
        for x_i, x_j, _ in tqdm(train_loader, desc=f'SimCLR {epoch + 1}/{epochs}'):
            x_i, x_j = x_i.to(device), x_j.to(device)
            z_i, z_j, _, _ = simclr(x_i, x_j)
            loss = criterion(z_i, z_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep.append(loss.item())
        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        losses.append(np.mean(ep))
        print(f'Epoch {epoch + 1:02d} | Loss: {np.mean(ep):.4f} | Time: {elapsed:.1f}s')

    total_time = time.time() - total_start
    print(f'\nTotal: {total_time / 60:.1f} min | Avg/epoch: {np.mean(epoch_times):.1f}s')
    torch.save(simclr.state_dict(), save_path)
    return {'losses': losses, 'epoch_times': epoch_times, 'avg_time': float(np.mean(epoch_times)), 'save_path': save_path}


def train_dino(device, epochs=10, batch_size=64, n_local=4, out_dim=256,
               ema_m=0.996, use_centering=True, save_path='saved/dino.pt'):
    student_vit, student_head = build_dino_model(out_dim=out_dim)
    teacher_vit, teacher_head = build_dino_model(out_dim=out_dim)

    student_vit, student_head = student_vit.to(device), student_head.to(device)
    teacher_vit, teacher_head = teacher_vit.to(device), teacher_head.to(device)

    teacher_vit.load_state_dict(student_vit.state_dict())
    teacher_head.load_state_dict(student_head.state_dict())
    for p in teacher_vit.parameters():
        p.requires_grad = False
    for p in teacher_head.parameters():
        p.requires_grad = False

    total = count_parameters(student_vit) + count_parameters(student_head)
    print(f'Student parameters: {total:,}')

    dino_loader = get_dino_loader(batch_size=batch_size, n_local=n_local)
    dino_loss_fn = DINOLoss(out_dim=out_dim, use_centering=use_centering).to(device)
    optimizer = torch.optim.AdamW(
        list(student_vit.parameters()) + list(student_head.parameters()),
        lr=5e-4,
        weight_decay=0.04,
    )

    losses = []
    center_norms = []
    epoch_times = []
    total_start = time.time()

    for epoch in range(epochs):
        student_vit.train()
        student_head.train()
        ep = []
        t0 = time.time()
        for crops, _ in tqdm(dino_loader, desc=f'DINO {epoch + 1}/{epochs}'):
            crops = [c.to(device) for c in crops]
            student_out = [student_head(student_vit(c)) for c in crops]
            with torch.no_grad():
                teacher_out = [
                    teacher_head(teacher_vit(crops[0])),
                    teacher_head(teacher_vit(crops[1])),
                ]
            loss = dino_loss_fn(student_out, teacher_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for s_p, t_p in zip(student_vit.parameters(), teacher_vit.parameters()):
                    t_p.data = ema_m * t_p.data + (1 - ema_m) * s_p.data
                for s_p, t_p in zip(student_head.parameters(), teacher_head.parameters()):
                    t_p.data = ema_m * t_p.data + (1 - ema_m) * s_p.data
            ep.append(loss.item())

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        losses.append(np.mean(ep))
        center_norms.append(float(dino_loss_fn.center.norm().item()))
        print(
            f'Epoch {epoch + 1:02d} | Loss: {np.mean(ep):.4f} | '
            f'Center norm: {center_norms[-1]:.4f} | Time: {elapsed:.1f}s'
        )

    total_time = time.time() - total_start
    print(f'\nTotal: {total_time / 60:.1f} min | Avg/epoch: {np.mean(epoch_times):.1f}s')
    torch.save({'student_vit': student_vit.state_dict(), 'student_head': student_head.state_dict()}, save_path)
    return {
        'losses': losses,
        'center_norms': center_norms,
        'epoch_times': epoch_times,
        'avg_time': float(np.mean(epoch_times)),
        'save_path': save_path,
    }


def train_mae(device, epochs=10, batch_size=128, lr=1.5e-4, mask_ratio=0.75,
              save_path='saved/mae_encoder.pt'):
    mae_model = MAE(
        img_size=32,
        patch_size=4,
        in_ch=3,
        encoder_dim=192,
        encoder_depth=6,
        encoder_heads=3,
        decoder_dim=128,
        decoder_depth=4,
        decoder_heads=4,
        mask_ratio=mask_ratio,
        norm_pix_loss=True,
    ).to(device)

    enc_params = count_parameters(mae_model.encoder)
    dec_params = count_parameters(mae_model.decoder)
    print(f'MAE Encoder params: {enc_params:,}')
    print(f'MAE Decoder params: {dec_params:,} ({100 * dec_params / enc_params:.1f}% of encoder)')

    mae_loader = get_mae_loader(batch_size=batch_size)
    optimizer = torch.optim.AdamW(mae_model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    losses = []
    epoch_times = []
    mae_model.train()
    total_start = time.time()

    for epoch in range(epochs):
        ep = []
        t0 = time.time()
        for imgs, _ in tqdm(mae_loader, desc=f'MAE {epoch + 1}/{epochs}'):
            imgs = imgs.to(device)
            loss, _, _ = mae_model(imgs)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(mae_model.parameters(), max_norm=1.0)
            optimizer.step()
            ep.append(loss.item())
        scheduler.step()
        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        losses.append(np.mean(ep))
        print(f'Epoch {epoch + 1:02d} | Recon Loss: {np.mean(ep):.4f} | Time: {elapsed:.1f}s')

    total_time = time.time() - total_start
    print(f'\nTotal: {total_time / 60:.1f} min | Avg/epoch: {np.mean(epoch_times):.1f}s')
    torch.save(mae_model.encoder.state_dict(), save_path)
    return {'losses': losses, 'epoch_times': epoch_times, 'avg_time': float(np.mean(epoch_times)), 'save_path': save_path}
