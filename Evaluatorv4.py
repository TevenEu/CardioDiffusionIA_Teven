# -*- coding: utf-8 -*-
"""
Evaluación de CardioDiffusion V4 – Enfocada en QRS
====================================================
Usa inferencia directa con poco ruido (t_val=10),
detección de picos en GPU para centrar el hueco,
y corrección de amplitud.
"""

import os, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACION
# ============================================================
BASE_DIR       = r"C:\Users\steev\OneDrive\Documentos\Proyecto_Cardio_Final"
RUTA_MITBIH    = os.path.join(BASE_DIR, "data", "MIT_BIH",
                              "physionet.org", "files", "mitdb", "1.0.0")
CHECKPOINT     = os.path.join(BASE_DIR, "checkpoints_v4",
                              "mejor_20260503_050145_val2.1558.pth")
RESULTADOS_DIR = os.path.join(BASE_DIR, "resultados_qrs_v2")
os.makedirs(RESULTADOS_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Dispositivo: {device}")

SEQ_LEN = 512
FS      = 360
GAP_LEN = 80     # muestras de hueco (~220 ms)

# Registros a evaluar
REGISTROS_TEST = ['107', '118', '222', '234', '101']
MUESTRAS_POR_REG = 3   # cuántas ventanas distintas por registro

# ============================================================
# ARQUITECTURA (idéntica a CardioDiffusionV4)
# ============================================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class DiffusionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.SiLU()
    def forward(self, x, emb):
        h = self.conv(x)
        h = self.norm(h)
        return self.act(h + self.emb_proj(emb).unsqueeze(-1))

class AttentionBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.q = nn.Conv1d(ch, ch, 1)
        self.k = nn.Conv1d(ch, ch, 1)
        self.v = nn.Conv1d(ch, ch, 1)
        self.proj = nn.Conv1d(ch, ch, 1)
        self.scale = ch ** -0.5
    def forward(self, x):
        B, C, L = x.shape
        h = self.norm(x)
        q, k, v = self.q(h).transpose(1,2), self.k(h).transpose(1,2), self.v(h).transpose(1,2)
        attn = torch.softmax(torch.bmm(q, k.transpose(1,2)) * self.scale, dim=-1)
        out = torch.bmm(attn, v).transpose(1,2)
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
        self.enc1 = DiffusionBlock(3,   128, time_dim)
        self.enc2 = DiffusionBlock(128, 256, time_dim)
        self.bot  = DiffusionBlock(256, 512, time_dim)
        self.attn = AttentionBlock(512)
        self.up2 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec2 = DiffusionBlock(512, 256, time_dim)
        self.up1 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec1 = DiffusionBlock(256, 128, time_dim)
        self.final = nn.Conv1d(128, 2, kernel_size=1)

    def forward(self, x_noisy, mask, t, lead_idx):
        emb = self.time_mlp(t) + self.lead_emb(lead_idx)
        x_in = torch.cat([x_noisy, mask], dim=1)
        h1 = self.enc1(x_in, emb)
        h2 = self.enc2(F.max_pool1d(h1, 2), emb)
        b  = self.bot(F.max_pool1d(h2, 2), emb)
        b  = self.attn(b)
        u2 = self.up2(b)[:, :, :h2.shape[-1]]
        d2 = self.dec2(torch.cat([u2, h2], dim=1), emb)
        u1 = self.up1(d2)[:, :, :h1.shape[-1]]
        d1 = self.dec1(torch.cat([u1, h1], dim=1), emb)
        return self.final(d1)

# ============================================================
# SCHEDULER
# ============================================================
class CardioScheduler:
    def __init__(self, T=1000, device='cuda'):
        self.T = T
        self.beta = torch.linspace(1e-4, 0.02, T).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_ab = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1)
        sqrt_1m_ab = torch.sqrt(1. - self.alpha_bar[t]).view(-1, 1, 1)
        return sqrt_ab * x0 + sqrt_1m_ab * noise, noise

# ============================================================
# DETECCIÓN DE PICOS EN GPU (para centrar el hueco en el QRS)
# ============================================================
def find_qrs_peaks_gpu(signal, min_distance=40, threshold_factor=0.7):
    B, C, L = signal.shape
    canal = signal[:, 0:1, :]  # canal MLII

    pool_size = min_distance * 2 + 1
    padding   = min_distance
    maxpool   = F.max_pool1d(canal, kernel_size=pool_size, stride=1, padding=padding)
    es_maximo = (canal == maxpool)

    std_por_muestra = canal.std(dim=-1, keepdim=True)
    umbral          = std_por_muestra * threshold_factor
    sobre_umbral    = canal > umbral

    borde = 60
    mascara_borde = torch.zeros_like(es_maximo)
    mascara_borde[..., borde:L-borde] = 1

    picos = (es_maximo & sobre_umbral & mascara_borde).squeeze(1)
    return picos  # (B, L) booleano

# ============================================================
# RECONSTRUCCIÓN DIRECTA CON BAJO RUIDO
# ============================================================
@torch.no_grad()
def reconstruir_directa(model, x_original, mask, scheduler, t_val=10):
    """
    Predicción de un solo paso con nivel de ruido bajo.
    t_val=10 preserva los detalles de alta frecuencia (QRS).
    """
    model.eval()
    B = x_original.shape[0]
    t = torch.full((B,), t_val, device=device, dtype=torch.long)
    x_noisy, _ = scheduler.add_noise(x_original, t)
    x_noisy = x_noisy * (1 - mask) + x_original * mask   # mantener zona conocida
    lead_idx = torch.zeros(B, dtype=torch.long, device=device)

    with torch.amp.autocast('cuda'):
        noise_pred = model(x_noisy, mask, t, lead_idx)

    a_bar = scheduler.alpha_bar[t].view(-1, 1, 1)
    x0_pred = (x_noisy - torch.sqrt(1 - a_bar) * noise_pred) / torch.sqrt(a_bar)
    x0_pred = x0_pred.clamp(-3, 3)
    return x0_pred * (1 - mask) + x_original * mask

def corregir_amplitud(x_rec, x_original, mask):
    zona = (mask[0,0] == 0)
    zona_ok = (mask[0,0] == 1)
    if zona.sum() < 5:
        return x_rec
    for ch in range(x_rec.shape[1]):
        orig_ch = x_original[0, ch]
        recon_ch = x_rec[0, ch].clone()
        p_ok = orig_ch[zona_ok].abs().max()
        p_o  = orig_ch[zona].abs().max()
        p_r  = recon_ch[zona].abs().max()
        if p_r > 1e-6 and p_o > p_ok * 0.5:
            factor = (p_o / p_r).clamp(0.5, 5.0)
            media = recon_ch[zona].mean()
            x_rec[0,ch][zona] = (recon_ch[zona] - media) * factor + media
    return x_rec

# ============================================================
# EVALUACIÓN DE UN SEGMENTO
# ============================================================
def evaluar_segmento(record_path, start, model, scheduler):
    try:
        sig, _ = wfdb.rdsamp(record_path, sampfrom=start, sampto=start+SEQ_LEN)
    except:
        return None
    if sig.shape[1] < 2:
        return None

    # Normalizar
    sig = (sig - np.mean(sig, axis=0)) / (np.std(sig, axis=0) + 1e-8)
    x_orig = torch.tensor(sig.T, dtype=torch.float32).unsqueeze(0).to(device)

    # Detectar pico R para centrar el hueco
    picos = find_qrs_peaks_gpu(x_orig)
    idxs = picos[0].nonzero(as_tuple=True)[0]
    if len(idxs) > 0:
        peak = idxs[len(idxs)//2].item()  # primer pico central
    else:
        peak = SEQ_LEN // 2                # fallback al centro

    # Crear máscara alrededor del pico
    gap_start = max(0, peak - GAP_LEN//2)
    gap_end   = min(SEQ_LEN, gap_start + GAP_LEN)
    if gap_end - gap_start < 20:   # ventana mínima
        gap_start = max(0, SEQ_LEN//2 - GAP_LEN//2)
        gap_end   = min(SEQ_LEN, gap_start + GAP_LEN)

    mask = torch.ones(1, 1, SEQ_LEN, device=device)
    mask[0, 0, gap_start:gap_end] = 0

    # Reconstruir
    x_rec = reconstruir_directa(model, x_orig, mask, scheduler, t_val=16)
    x_rec = corregir_amplitud(x_rec, x_orig, mask)

    orig = x_orig[0].cpu().numpy()
    recon = x_rec[0].cpu().numpy()
    msk = mask[0, 0].cpu().numpy()
    zona_hueco = (msk == 0)

    # Métricas dentro del hueco
    diff = orig[:, zona_hueco] - recon[:, zona_hueco]
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))

    if np.std(recon[0, zona_hueco]) < 1e-8:
        corr = 0.0
    else:
        corr = float(np.corrcoef(orig[0, zona_hueco], recon[0, zona_hueco])[0,1])

    den = np.sqrt(np.sum(orig[0, zona_hueco]**2))
    prd = float((np.sqrt(np.sum(diff[0]**2)) / den) * 100) if den > 1e-8 else 999.0

    return {
        'orig': orig,
        'recon': recon,
        'mask': msk,
        'gap': (gap_start, gap_end),
        'MSE': mse, 'MAE': mae, 'Corr': corr, 'PRD': prd,
        'record': os.path.basename(record_path),
    }

# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================
if __name__ == "__main__":
    # Cargar modelo con corrección de nombres
    model = CardioDiffusionV4(num_leads=2, time_dim=128).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)

    # --- Corrección attn_bot → attn ---
    state_dict = ckpt['model_state']
    new_sd = {}
    for k, v in state_dict.items():
        new_sd[k.replace('attn_bot.', 'attn.')] = v
    model.load_state_dict(new_sd)
    print("Modelo cargado (época 276, corrección attn_bot → attn).")

    model.eval()
    scheduler = CardioScheduler(T=1000, device=device)

    metricas = []

    for reg in REGISTROS_TEST:
        record = os.path.join(RUTA_MITBIH, reg)
        try:
            header = wfdb.rdheader(record)
        except:
            print(f"Saltando {reg}: no se pudo leer.")
            continue

        # Elegir varias ventanas aleatorias
        for _ in range(MUESTRAS_POR_REG):
            start = random.randint(0, max(0, header.sig_len - SEQ_LEN - 1))
            res = evaluar_segmento(record, start, model, scheduler)
            if res is None:
                continue

            metricas.append({
                'archivo': f"{res['record']}_{start}",
                'MSE': res['MSE'],
                'MAE': res['MAE'],
                'Corr': res['Corr'],
                'PRD': res['PRD'],
            })

            # Graficar
            fig, axes = plt.subplots(2, 1, figsize=(14,6), sharex=True)
            t = np.arange(SEQ_LEN)
            for ch, ax in enumerate(axes):
                ax.plot(t, res['orig'][ch], color='steelblue', linewidth=1.0, label='Original', alpha=0.9)
                ax.plot(t, res['recon'][ch], color='tomato', linewidth=1.0, label='Reconstruido', alpha=0.9, linestyle='--')
                gs, ge = res['gap']
                ax.axvspan(gs, ge, color='yellow', alpha=0.2, label='Zona reconstruida')
                ax.set_ylabel(f'Derivación {ch+1}')
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)

            nombre_fig = f"eval_{res['record']}_{start}"
            fig.suptitle(f"{nombre_fig} | MSE={res['MSE']:.4f} MAE={res['MAE']:.4f} Corr={res['Corr']:.4f} PRD={res['PRD']:.2f}%",
                         fontsize=11)
            plt.xlabel('Muestras')
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTADOS_DIR, f"{nombre_fig}.png"), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Guardada: {nombre_fig}.png")

    # Resumen final
    print("\n" + "="*70)
    print(f"{'Archivo':<25} {'MSE':>8} {'MAE':>8} {'Corr':>8} {'PRD(%)':>8}")
    print("-"*70)
    for m in metricas:
        print(f"{m['archivo']:<25} {m['MSE']:8.4f} {m['MAE']:8.4f} {m['Corr']:8.4f} {m['PRD']:8.2f}")
    print("-"*70)
    mse_avg  = np.mean([m['MSE'] for m in metricas])
    mae_avg  = np.mean([m['MAE'] for m in metricas])
    corr_avg = np.mean([m['Corr'] for m in metricas])
    prd_avg  = np.mean([m['PRD'] for m in metricas])
    print(f"{'PROMEDIO':<25} {mse_avg:8.4f} {mae_avg:8.4f} {corr_avg:8.4f} {prd_avg:8.2f}")
    print(f"\nGráficas guardadas en: {RESULTADOS_DIR}")