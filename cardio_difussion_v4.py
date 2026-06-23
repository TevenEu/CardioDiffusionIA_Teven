# -*- coding: utf-8 -*-
"""
CardioECG Diffusion PRO - Entrenamiento Optimizado v4
Mejoras v4 sobre v3:
  1. Gold Standard de validación — registros 100 y 103 excluidos
     del entrenamiento y usados exclusivamente para validación.
  2. DataLoader acelerado — num_workers=4 + persistent_workers=True
  3. Detección de picos QRS en GPU con PyTorch (sin .cpu().numpy()
     ni scipy en el loop de entrenamiento → elimina cuello de botella CPU)
"""

import os
import glob
import math
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb
from torch.utils.data import Dataset, DataLoader, random_split


# CONFIGURACION GLOBAL

BASE_DIR       = r"C:\Users\steev\OneDrive\Documentos\Proyecto_Cardio_Final"
RUTA_MITBIH    = os.path.join(BASE_DIR, "data", "MIT_BIH",
                              "physionet.org", "files", "mitdb", "1.0.0")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_v4")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Registros reservados exclusivamente para validación (Gold Standard)
# 100 → señal muy limpia, ritmo sinusal normal
# 103 → otro registro limpio de referencia
GOLD_STANDARD = {'100', '103', '105', '109', '117'}

# Hiperparámetros
EPOCHS     = 300
LR         = 1e-4
BATCH_SIZE = 16
SEQ_LEN    = 512
SEED       = 42
PATIENCE   = 50
# Seeds fijas
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Dispositivo: {device}")



# ARQUITECTURA

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device   = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class DiffusionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.conv     = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.norm     = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act      = nn.SiLU()

    def forward(self, x, emb):
        h = self.conv(x)
        h = self.norm(h)
        h = h + self.emb_proj(emb).unsqueeze(-1)
        return self.act(h)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm  = nn.GroupNorm(min(8, channels), channels)
        self.q     = nn.Conv1d(channels, channels, 1)
        self.k     = nn.Conv1d(channels, channels, 1)
        self.v     = nn.Conv1d(channels, channels, 1)
        self.proj  = nn.Conv1d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        B, C, L = x.shape
        h    = self.norm(x)
        q    = self.q(h).transpose(1, 2)
        k    = self.k(h).transpose(1, 2)
        v    = self.v(h).transpose(1, 2)
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * self.scale, dim=-1)
        out  = torch.bmm(attn, v).transpose(1, 2)
        return x + self.proj(out)


class CardioDiffusionV4(nn.Module):
    def __init__(self, num_leads=2, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )
        self.lead_emb = nn.Embedding(num_leads, time_dim)

        self.enc1     = DiffusionBlock(3,   128, time_dim)
        self.enc2     = DiffusionBlock(128, 256, time_dim)
        self.bot      = DiffusionBlock(256, 512, time_dim)
        self.attn_bot = AttentionBlock(512)

        self.up2  = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec2 = DiffusionBlock(512, 256, time_dim)

        self.up1  = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec1 = DiffusionBlock(256, 128, time_dim)

        self.final = nn.Conv1d(128, 2, kernel_size=1)

    def forward(self, x_noisy, mask, t, lead_idx):
        emb  = self.time_mlp(t) + self.lead_emb(lead_idx)
        x_in = torch.cat([x_noisy, mask], dim=1)

        h1 = self.enc1(x_in, emb)
        h2 = self.enc2(F.max_pool1d(h1, 2), emb)
        b  = self.bot(F.max_pool1d(h2, 2), emb)
        b  = self.attn_bot(b)

        u2 = self.up2(b)[..., :h2.shape[-1]]
        d2 = self.dec2(torch.cat([u2, h2], dim=1), emb)

        u1 = self.up1(d2)[..., :h1.shape[-1]]
        d1 = self.dec1(torch.cat([u1, h1], dim=1), emb)

        return self.final(d1)

class EMA:
    def __init__(self, model, decay=0.999):
        self.model  = model
        self.decay  = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.data

    def apply_shadow(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data = self.shadow[n]

    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data = self.backup[n]

# SCHEDULER DE DIFUSION

class CardioScheduler:
    def __init__(self, T=1000, device='cuda'):
        self.T         = T
        self.beta      = torch.linspace(1e-4, 0.02, T).to(device)
        self.alpha     = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x_0, t):
        noise             = torch.randn_like(x_0)
        sqrt_ab           = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1)
        sqrt_one_minus_ab = torch.sqrt(1. - self.alpha_bar[t]).view(-1, 1, 1)
        return sqrt_ab * x_0 + sqrt_one_minus_ab * noise, noise


# DETECCION DE PICOS QRS EN GPU (sin scipy, sin .cpu())

def find_qrs_peaks_gpu(signal, min_distance=40, threshold_factor=0.8):
    """
    Detecta picos R directamente en GPU usando max_pool1d.
    Reemplaza scipy.find_peaks — sin mover datos a CPU.

    signal: (B, C, L) tensor en GPU
    Retorna: lista de tensors con índices de picos por muestra
    """
    B, C, L = signal.shape
    canal   = signal[:, 0:1, :]  # Canal MLII (B, 1, L)

    # Máximos locales: punto es pico si es el máximo en ventana de min_distance
    pool_size = min_distance * 2 + 1
    padding   = min_distance
    maxpool   = F.max_pool1d(canal, kernel_size=pool_size, stride=1, padding=padding)
    es_maximo = (canal == maxpool)  # (B, 1, L)

    # Umbral dinámico por muestra
    std_por_muestra = canal.std(dim=-1, keepdim=True)  # (B, 1, 1)
    umbral          = std_por_muestra * threshold_factor
    sobre_umbral    = canal > umbral                   # (B, 1, L)

    # Excluir bordes
    borde = 60
    mascara_borde            = torch.zeros_like(es_maximo)
    mascara_borde[..., borde:L-borde] = 1

    picos = (es_maximo & sobre_umbral & mascara_borde).squeeze(1)  # (B, L)
    return picos  # Boolean tensor: True donde hay pico R


def get_smart_mask_gpu(batch, seq_len=512):
    """
    Genera máscaras centradas en picos QRS reales.
    Todo en GPU — sin transfers CPU/GPU en el loop de entrenamiento.

    batch: (B, 2, L) tensor en GPU
    Retorna: mask (B, 1, L) en GPU
    """
    B = batch.shape[0]
    device = batch.device
    masks  = torch.ones(B, 1, seq_len, device=device)

    # Detectar picos en todo el batch de una vez
    picos_batch = find_qrs_peaks_gpu(batch)  # (B, L) boolean

    for i in range(B):
        picos_idx = picos_batch[i].nonzero(as_tuple=True)[0]  # índices de picos

        if len(picos_idx) > 0:
            # Elegir pico aleatorio
            idx_aleatorio = torch.randint(len(picos_idx), (1,)).item()
            peak          = picos_idx[idx_aleatorio].item()
            length        = torch.randint(60, 150, (1,)).item()
            start         = max(0, peak - length // 2)
            end           = min(seq_len, start + length)
            masks[i, 0, start:end] = 0
        else:
            # Fallback: bloque central aleatorio
            length = torch.randint(int(seq_len * 0.10), int(seq_len * 0.35), (1,)).item()
            start  = torch.randint(int(seq_len * 0.1), int(seq_len * 0.6), (1,)).item()
            masks[i, 0, start:start + length] = 0

    return masks



# DATASET MIT-BIH CON GOLD STANDARD

class MITBIH_Dataset(Dataset):
    def __init__(self, ruta, seq_len=512, stride=None, excluir=None):
        """
        excluir: set de nombres de registro a excluir (ej. {'100', '103'})
        """
        self.seq_len = seq_len
        self.stride  = stride or seq_len // 2
        self.index   = []
        excluir      = excluir or set()

        paths = sorted(glob.glob(os.path.join(ruta, "*.dat")))
        print(f"Indexando registros (excluyendo: {excluir})...")

        for path in paths:
            nombre = os.path.basename(path).replace('.dat', '')
            if nombre in excluir:
                continue
            record = path.replace('.dat', '')
            try:
                header     = wfdb.rdheader(record)
                n_ventanas = (header.sig_len - seq_len) // self.stride
                for j in range(n_ventanas):
                    self.index.append((record, j * self.stride))
            except Exception:
                continue

        print(f"  Total ventanas: {len(self.index)}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        record, start = self.index[idx]
        try:
            sig, _ = wfdb.rdsamp(record, sampfrom=start, sampto=start + self.seq_len)
        except Exception:
            sig = np.zeros((self.seq_len, 2), dtype=np.float32)

        sig = (sig - np.mean(sig, axis=0)) / (np.std(sig, axis=0) + 1e-8)
        sig = np.clip(sig, -5, 5).astype(np.float32)

        x        = torch.tensor(sig.T)
        lead_idx = torch.tensor(0, dtype=torch.long)
        return x, lead_idx


class MITBIH_GoldStandard(Dataset):
    """Dataset exclusivo para validación con registros Gold Standard."""
    def __init__(self, ruta, registros, seq_len=512, stride=None):
        self.seq_len = seq_len
        self.stride  = stride or seq_len // 2
        self.index   = []

        print(f"Indexando Gold Standard: {registros}")
        for nombre in registros:
            record = os.path.join(ruta, nombre)
            try:
                header     = wfdb.rdheader(record)
                n_ventanas = (header.sig_len - seq_len) // self.stride
                for j in range(n_ventanas):
                    self.index.append((record, j * self.stride))
            except Exception as e:
                print(f"  Error en {nombre}: {e}")

        print(f"  Total ventanas Gold Standard: {len(self.index)}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        record, start = self.index[idx]
        try:
            sig, _ = wfdb.rdsamp(record, sampfrom=start, sampto=start + self.seq_len)
        except Exception:
            sig = np.zeros((self.seq_len, 2), dtype=np.float32)

        sig = (sig - np.mean(sig, axis=0)) / (np.std(sig, axis=0) + 1e-8)
        sig = np.clip(sig, -5, 5).astype(np.float32)

        x        = torch.tensor(sig.T)
        lead_idx = torch.tensor(0, dtype=torch.long)
        return x, lead_idx



# FUNCION DE PERDIDA HIBRIDA

def hybrid_loss(pred, target, mask):
    inv_mask = 1 - mask
    p = pred   * inv_mask
    t = target * inv_mask

    pesos     = (1 + torch.abs(t)) ** 2
    mse_loss  = (pesos * (p - t) ** 2).mean()
    fft_full = F.mse_loss(torch.abs(torch.fft.rfft(p, dim=-1)),
                      torch.abs(torch.fft.rfft(t, dim=-1)))
    fft_low  = F.mse_loss(torch.abs(torch.fft.rfft(p, dim=-1)[..., :50]),
                      torch.abs(torch.fft.rfft(t, dim=-1)[..., :50]))
    fft_loss = 0.5 * fft_full + 0.5 * fft_low
    grad_loss = F.mse_loss(torch.diff(p, dim=-1), torch.diff(t, dim=-1))

    global_loss = F.mse_loss(pred, target)
    return mse_loss + 0.3 * fft_loss + 1.5 * grad_loss + 0.1 * global_loss

# PASOS DE ENTRENAMIENTO Y VALIDACION

def train_step(model, batch, lead_idx, scheduler, optimizer, scaler):
    model.train()
    B = batch.shape[0]
    t = torch.randint(0, scheduler.T, (B,), device=device).long()

    x_noisy, noise_real = scheduler.add_noise(batch, t)

    # Máscaras en GPU — sin .cpu().numpy()
    mask = get_smart_mask_gpu(batch, seq_len=batch.shape[-1])

    optimizer.zero_grad()
    with torch.amp.autocast('cuda'):
        noise_pred = model(x_noisy, mask, t, lead_idx)
        loss       = hybrid_loss(noise_pred, noise_real, mask)

    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


@torch.no_grad()
def val_step(model, batch, lead_idx, scheduler):
    model.eval()
    B = batch.shape[0]
    t = torch.randint(0, scheduler.T, (B,), device=device).long()

    x_noisy, noise_real = scheduler.add_noise(batch, t)
    mask = get_smart_mask_gpu(batch, seq_len=batch.shape[-1])

    with torch.amp.autocast('cuda'):
        noise_pred = model(x_noisy, mask, t, lead_idx)
        
        loss       = hybrid_loss(noise_pred, noise_real, mask)

    a_bar   = scheduler.alpha_bar[t].view(-1, 1, 1)
    x0_pred = (x_noisy - torch.sqrt(1 - a_bar) * noise_pred) / torch.sqrt(a_bar)
    x0_pred = x0_pred.clamp(-5, 5)
    rmse    = torch.sqrt(torch.mean((x0_pred - batch) ** 2)).item()
    corr_val = torch.mean((x0_pred - x0_pred.mean()) * (batch - batch.mean()))
    corr     = (corr_val / (x0_pred.std() * batch.std() + 1e-8)).item()
    return loss.item(), rmse, corr



# BUCLE DE ENTRENAMIENTO

def train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    diff_scheduler = CardioScheduler(T=1000, device=device)
    optimizer      = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler         = torch.amp.GradScaler('cuda')

    best_val    = float('inf')
    start_epoch = 0
    patience_count = 0
    # Reanudar desde checkpoint
    checkpoints = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "epoch_*.pth")))
    if checkpoints:
        ultimo = checkpoints[-1]
        ckpt   = torch.load(ultimo, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        if 'scheduler_state' in ckpt:
            lr_scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch']
        best_val    = ckpt.get('best_val', float('inf'))
        print(f"Reanudando desde época {start_epoch} ({os.path.basename(ultimo)})")
        print(f"Mejor Val Loss: {best_val:.6f}")

        # Forzar LR actual
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        print(f"LR forzado a: {lr}")
    ema = EMA(model, decay=0.999)
    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_loss = sum(
            train_step(model, b.to(device), li.to(device), diff_scheduler, optimizer, scaler)
            for b, li in train_loader
        ) / len(train_loader)
        ema.update()
        val_results = [val_step(model, b.to(device), li.to(device), diff_scheduler) for b, li in val_loader]
        val_loss = sum(r[0] for r in val_results) / len(val_loader)
        val_rmse = sum(r[1] for r in val_results) / len(val_loader)
        val_corr = sum(r[2] for r in val_results) / len(val_loader)
        
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        elapsed    = time.time() - t0

        vram = torch.cuda.memory_allocated() / 1e9
        print(f"Época {epoch+1:3d}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | RMSE: {val_rmse:.4f} | Corr: {val_corr:.4f} | LR: {current_lr:.2e} | VRAM: {vram:.1f}GB | {elapsed:.0f}s")
        if val_rmse < best_val - 1e-4:
            patience_count = 0
        else:
            patience_count += 1

        if val_rmse < best_val:
            best_val  = val_rmse
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_path = os.path.join(CHECKPOINT_DIR, f"mejor_{timestamp}_val{val_loss:.4f}.pth")
            ema.apply_shadow()
            torch.save({
                'epoch':           epoch + 1,
                'model_state':     model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': lr_scheduler.state_dict(),
                'val_loss':        best_val,
            }, best_path)
            ema.restore()
            print(f"  ✅ Mejor modelo: {os.path.basename(best_path)}")

        # Checkpoint cada 10 épocas
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1:03d}.pth")
            torch.save({
                'epoch':           epoch + 1,
                'model_state':     model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': lr_scheduler.state_dict(),
                'val_loss':        val_loss,
                'best_val':        best_val,
            }, ckpt_path)
            print(f"  Checkpoint: epoch_{epoch+1:03d}.pth")

        if patience_count >= PATIENCE:
            print(f"Early stopping — sin mejoras en {PATIENCE} épocas.")
            break


# MAIN

if __name__ == "__main__":
    # Dataset de entrenamiento — excluye Gold Standard
    train_ds = MITBIH_Dataset(
        RUTA_MITBIH,
        seq_len=SEQ_LEN,
        excluir=GOLD_STANDARD
    )

    # Dataset de validación — solo Gold Standard (nunca visto en train)
    val_ds = MITBIH_GoldStandard(
        RUTA_MITBIH,
        registros=list(GOLD_STANDARD),
        seq_len=SEQ_LEN
    )

    print(f"\nTrain: {len(train_ds)} ventanas | Val (Gold): {len(val_ds)} ventanas")

    # DataLoaders acelerados (num_workers=4, persistent_workers=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    # Modelo
    model    = CardioDiffusionV4(num_leads=2, time_dim=128).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {n_params:,}")

    # Entrenar
    print(f"\nIniciando entrenamiento v4 (seed={SEED}, batch={BATCH_SIZE}, lr={LR})...")
    train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR)

    # Guardar modelo final
    final_path = os.path.join(BASE_DIR, "cardio_diffusion_v4_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nModelo final: {final_path}")
