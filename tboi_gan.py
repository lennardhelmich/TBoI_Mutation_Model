#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tboi_gan_film_r1.py

BicycleGAN-Variante für 13×7 px 1-Kanal-BMPs
mit:
- FiLM-Z-Injektion in jedem Decoder-Block
- Patch-D mit echtem Patch-Output (2×1)
- R1-Reg (alle N Schritte)
- λ_L1-Annealing
- Adaptives D/G-Update-Ratio
"""

from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch import LightningDataModule
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

import optuna
import time
import os
import numpy as np


# -------------------------------------------------------------------
# Padding 13×7 ↔ 16×8 für voll-conv
# -------------------------------------------------------------------
def pad_to_pow2(img: torch.Tensor, th: int = 16, tw: int = 8) -> torch.Tensor:
    _, _, h, w = img.shape
    # konstantes Padding (Tile 0) statt replicate  # <<<
    return F.pad(img, (0, tw - w, 0, th - h), mode="constant", value=-1.0)

def unpad(img: torch.Tensor, h: int = 13, w: int = 7) -> torch.Tensor:
    return img[..., :h, :w]


# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------
class NestedBmpDataset(Dataset):
    def __init__(self, root: str | Path, split: str="train", split_ratio: float=0.9, seed: int=42):
        self.src_dir = Path(root) / "Mutations"
        self.dst_dir = Path(root) / "Inverted_Mutations"

        pairs = []
        for bmp_k in sorted(self.src_dir.iterdir()):
            if not bmp_k.is_dir(): 
                continue
            for src in bmp_k.glob("mutation_*.bmp"):
                stem = src.stem
                tgt_dir = self.dst_dir / bmp_k.name / stem
                if tgt_dir.is_dir():
                    bmps = sorted(tgt_dir.glob("*.bmp"))
                    if bmps:
                        pairs.append((src, bmps[0]))

        if not pairs:
            raise RuntimeError("Keine Paare in Mutations/… gefunden")

        torch.manual_seed(seed)
        idx = torch.randperm(len(pairs))
        cut = int(len(pairs) * split_ratio)
        sel = idx[:cut] if split=="train" else idx[cut:]
        self.pairs = [pairs[i] for i in sel]

        self.to_tensor = T.Compose([
            lambda p: Image.open(p).convert("L"),
            T.PILToTensor(),
            lambda t: (t.float()/127.5) - 1.0,
        ])

    def __len__(self): 
        return len(self.pairs)
    def __getitem__(self, i):
        x, y = self.pairs[i]
        return self.to_tensor(x), self.to_tensor(y)


class BmpDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int=64, num_workers: int=0, split_ratio: float=0.9):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.train_set = NestedBmpDataset(self.hparams.data_dir, "train", self.hparams.split_ratio)

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.hparams.batch_size,
                          shuffle=True,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True)


# -------------------------------------------------------------------
# NN-Bausteine
# -------------------------------------------------------------------
def conv(in_c, out_c, k=3, s=1, p=1, norm=True, act=True):
    layers = [nn.ReflectionPad2d(p), nn.Conv2d(in_c, out_c, k, s, 0, bias=not norm)]
    if norm: 
        layers.append(nn.InstanceNorm2d(out_c))
    if act:  
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation mit z."""
    def __init__(self, c: int, z_dim: int):
        super().__init__()
        self.gamma = nn.Linear(z_dim, c)
        self.beta  = nn.Linear(z_dim, c)

    def forward(self, x, z):
        g = self.gamma(z).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(z).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + g) + b

class UNetGenerator(nn.Module):
    def __init__(self, in_ch, out_ch, z_dim=128, nf=64):
        super().__init__()
        self.d1 = conv(in_ch, nf, 4, 2, 1)                # 16x8 -> 8x4
        self.d2 = conv(nf, nf*2, 4, 2, 1)                 # 8x4 -> 4x2
        self.d3 = conv(nf*2, nf*4, 4, 2, 1, norm=False, act=False)  # 4x2 -> 2x1 (no norm/act)

        # Decoder
        self.u1 = nn.ConvTranspose2d(nf*4, nf*2, 4, 2, 1)  # 2x1 -> 4x2
        self.in1 = nn.InstanceNorm2d(nf*2)
        self.film1 = FiLMBlock(nf*2, z_dim)

        self.u2 = nn.ConvTranspose2d(nf*4, nf, 4, 2, 1)    # 4x2 -> 8x4
        self.in2 = nn.InstanceNorm2d(nf)
        self.film2 = FiLMBlock(nf, z_dim)

        self.fin = nn.ConvTranspose2d(nf*2, out_ch, 4, 2, 1)  # 8x4 -> 16x8
        self.z_dim = z_dim

    def forward(self, x, z):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)

        u1 = self.u1(d3)
        u1 = self.in1(u1)
        u1 = F.relu(self.film1(u1, z), inplace=True)
        u1 = torch.cat([u1, d2], 1)

        u2 = self.u2(u1)
        u2 = self.in2(u2)
        u2 = F.relu(self.film2(u2, z), inplace=True)
        u2 = torch.cat([u2, d1], 1)

        out = torch.tanh(self.fin(u2))
        return out

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch, nf=64):
        super().__init__()
        self.body = nn.Sequential(
            conv(in_ch, nf, 4, 2, 1, norm=False),  # 16x8 -> 8x4
            conv(nf, nf*2, 4, 2, 1),               # 8x4 -> 4x2
            conv(nf*2, nf*4, 3, 1, 1),             # 4x2 -> 4x2
        )
        # Final Layer: 4x2 -> 2x1, lässt Patch-Struktur übrig
        self.last = nn.Conv2d(nf*4, 1, (3,2), stride=2, padding=0)

    def forward(self, x):
        h = self.body(x)
        return self.last(h)  # [B,1,2,1]


class Encoder(nn.Module):
    def __init__(self, in_ch, z_dim=128, nf=64):
        super().__init__()
        self.body = nn.Sequential(
            conv(in_ch, nf, 4, 2, 1),
            conv(nf, nf*2, 4, 2, 1),
            conv(nf*2, nf*4, 4, 2, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mu     = nn.Linear(nf*4, z_dim)
        self.logvar = nn.Linear(nf*4, z_dim)

    def forward(self, y):
        h = self.body(y).view(y.size(0), -1)
        return self.mu(h), self.logvar(h)


# -------------------------------------------------------------------
# Lightning-Modul
# -------------------------------------------------------------------
class LitBicycleGAN(LightningModule):
    def __init__(self, in_ch=1, z_dim=128, lr=2e-4,
                 lambda_l1=10.0, lambda_kl=0.01,
                 decay_start=200, total_epochs=400,
                 r1_every=16, r1_gamma=5e-3,         # <<< R1 settings
                 d_suppress_thresh=0.3,              # <<< adaptive D
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.G = UNetGenerator(in_ch, in_ch, z_dim)
        self.D = PatchDiscriminator(in_ch*2)
        self.E = Encoder(in_ch, z_dim)
        self.automatic_optimization = False

    @staticmethod
    def hinge(logits, real=True):
        return F.relu(1.-logits).mean() if real else F.relu(1.+logits).mean()

    @staticmethod
    def kl(mu, logvar):
        return 0.5 * torch.mean(torch.sum(mu**2 + torch.exp(logvar) - 1. - logvar, dim=1))

    def forward(self, x, z):
        return self.G(x, z)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_pad, y_pad = pad_to_pow2(x), pad_to_pow2(y)

        opt_g, opt_d, opt_e = self.optimizers()

        # ---- Encode z
        mu, logvar = self.E(y_pad)
        std        = torch.exp(0.5*logvar)
        z_enc      = mu + torch.randn_like(std)*std
        z_rand     = torch.randn_like(z_enc)

        fake_enc  = self.G(x_pad, z_enc)
        fake_rand = self.G(x_pad, z_rand)

        # ---- D step (evtl. aussetzen)
        run_d = True
        if hasattr(self, "_last_d_loss"):
            if self._last_d_loss < self.hparams.d_suppress_thresh:
                run_d = False

        if run_d:
            opt_d.zero_grad(set_to_none=True)
            d_real = self.D(torch.cat([x_pad, y_pad],1))
            d_fake = self.D(torch.cat([x_pad, fake_rand.detach()],1))
            d_loss = self.hinge(d_real, True) + self.hinge(d_fake, False)

            # R1 every N steps
            if (self.global_step % self.hparams.r1_every) == 0:
                real_in = torch.cat([x_pad, y_pad],1).requires_grad_(True)
                out = self.D(real_in)
                grad = torch.autograd.grad(out.sum(), real_in, create_graph=True)[0]
                r1 = grad.pow(2).reshape(grad.size(0), -1).sum(1).mean()
                d_loss = d_loss + self.hparams.r1_gamma * r1
                self.log("d_r1", r1, prog_bar=False)

            self.manual_backward(d_loss)
            opt_d.step()
            self._last_d_loss = d_loss.detach().item()
            self.log("d_loss", d_loss, prog_bar=True)

        # ---- G+E step
        opt_g.zero_grad(set_to_none=True)
        opt_e.zero_grad(set_to_none=True)

        # Adversarial nur auf fake_rand (Diversität) – könnte man splitten
        g_adv = self.hinge(self.D(torch.cat([x_pad, fake_rand],1)), True)
        g_l1  = F.l1_loss(fake_enc, y_pad)
        g_kl  = self.kl(mu, logvar)

        # Anneal λ_L1
        epoch = self.current_epoch
        lam_l1 = self.hparams.lambda_l1
        if self.hparams.decay_start > 0:
            # Beispiel: in den ersten 100 Epochen linear auf 0.1*λ runter
            decay_ep = min(self.hparams.decay_start, 100)
            if epoch < decay_ep:
                lam_l1 = lam_l1 * (1.0 - 0.9 * (epoch / decay_ep))

        g_loss = g_adv + lam_l1*g_l1 + self.hparams.lambda_kl*g_kl

        self.manual_backward(g_loss)
        opt_g.step()
        opt_e.step()

        # Logs
        self.log_dict({"g_adv": g_adv, "g_l1": g_l1, "g_kl": g_kl, "lam_l1": lam_l1},
                      prog_bar=True)
        self.log("lr", opt_g.param_groups[0]['lr'], prog_bar=True, logger=False)

        # Optional: Grad-Normen
        # for name, p in self.G.named_parameters():
        #     if p.grad is not None:
        #         self.log(f"gnorm_G/{name}", p.grad.data.norm(2), prog_bar=False, logger=False)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.G.parameters(), lr=self.hparams.lr,  betas=(0.5,0.999))
        opt_d = torch.optim.Adam(self.D.parameters(), lr=self.hparams.lr*0.5, betas=(0.5,0.999))
        opt_e = torch.optim.Adam(self.E.parameters(), lr=self.hparams.lr,  betas=(0.5,0.999))

        def lr_lambda(epoch):
            if epoch < self.hparams.decay_start: 
                return 1.0
            frac = (epoch - self.hparams.decay_start) / max(1, self.hparams.total_epochs - self.hparams.decay_start)
            return 1.0 - frac

        scheds = [torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
                  for opt in (opt_g, opt_d, opt_e)]
        return [opt_g, opt_d, opt_e], [{"scheduler": s, "interval":"epoch"} for s in scheds]

    @torch.no_grad()
    def sample(self, x_np, z_seed: int|None=None):
        if z_seed is not None: torch.manual_seed(z_seed)
        x = torch.tensor(x_np).float().unsqueeze(0).to(self.device)
        x_pad = pad_to_pow2(x)
        z = torch.randn(1, self.hparams.z_dim, device=self.device)
        return unpad(self.G(x_pad, z)).squeeze(0).cpu()


# ---------------------------------------------------------------------------
# Optuna Hyperparameter-Optimierung
# ---------------------------------------------------------------------------
def objective(trial):
    """Optuna Objective Function für GAN Hyperparameter-Optimierung"""
    import time
    
    trial_start_time = time.time()
    
    # GPU-Status und Trial-Info
    print(f"\n{'='*60}")
    print(f"🚀 GAN TRIAL {trial.number} GESTARTET")
    print(f"{'='*60}")
    print(f"🔥 DEVICE: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"💻 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Hyperparameter-Vorschläge
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    z_dim = trial.suggest_categorical('z_dim', [64, 128, 256, 512])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    lambda_l1 = trial.suggest_uniform('lambda_l1', 1.0, 20.0)
    lambda_kl = trial.suggest_loguniform('lambda_kl', 1e-3, 1e-1)
    d_suppress_thresh = trial.suggest_uniform('d_suppress_thresh', 0.1, 0.5)
    r1_gamma = trial.suggest_loguniform('r1_gamma', 1e-4, 1e-2)
    
    print(f"📋 HYPERPARAMETER:")
    print(f"   Learning Rate: {lr:.2e}")
    print(f"   Z Dimension:   {z_dim}")
    print(f"   Batch Size:    {batch_size}")
    print(f"   Lambda L1:     {lambda_l1:.3f}")
    print(f"   Lambda KL:     {lambda_kl:.4f}")
    print(f"   D Suppress:    {d_suppress_thresh:.3f}")
    print(f"   R1 Gamma:      {r1_gamma:.2e}")
    print(f"{'='*60}")
    
    # Dataset
    print("📁 Lade Dataset...")
    dm = BmpDataModule("Bitmaps", batch_size, num_workers=0, split_ratio=0.9)
    dm.setup()
    train_dl = dm.train_dataloader()
    print(f"✅ Dataset geladen: {len(dm.train_set)} Samples")
    
    # Modell mit Trial-Parametern
    print("🏗️  Erstelle GAN...")
    model = LitBicycleGAN(
        in_ch=1,
        z_dim=z_dim,
        lr=lr,
        lambda_l1=lambda_l1,
        lambda_kl=lambda_kl,
        decay_start=15,  # Kurz für Optuna
        total_epochs=25,
        d_suppress_thresh=d_suppress_thresh,
        r1_gamma=r1_gamma
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Modell-Info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 GAN Parameter: {total_params:,}")
    
    # Optimizers
    opts = model.configure_optimizers()
    opt_g, opt_d, opt_e = opts[0]
    
    best_g_l1 = float('inf')
    epochs = 25  # Kurze Epochen für Optuna
    
    print(f"🎯 Starte GAN Training für {epochs} Epochen...")
    epoch_times = []
    
    for ep in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        
        # Training Loop
        g_losses = []
        d_losses = []
        g_l1_losses = []
        
        for batch_idx, batch in enumerate(train_dl):
            if torch.cuda.is_available():
                batch = [b.cuda() for b in batch]
            
            x, y = batch
            x_pad, y_pad = pad_to_pow2(x), pad_to_pow2(y)
            
            # ---- Encode z
            mu, logvar = model.E(y_pad)
            std = torch.exp(0.5*logvar)
            z_enc = mu + torch.randn_like(std)*std
            z_rand = torch.randn_like(z_enc)
            
            fake_enc = model.G(x_pad, z_enc)
            fake_rand = model.G(x_pad, z_rand)
            
            # ---- D step
            opt_d.zero_grad()
            d_real = model.D(torch.cat([x_pad, y_pad], 1))
            d_fake = model.D(torch.cat([x_pad, fake_rand.detach()], 1))
            d_loss = model.hinge(d_real, True) + model.hinge(d_fake, False)
            d_loss.backward()
            opt_d.step()
            
            # ---- G+E step
            opt_g.zero_grad()
            opt_e.zero_grad()
            
            g_adv = model.hinge(model.D(torch.cat([x_pad, fake_rand], 1)), True)
            g_l1 = F.l1_loss(fake_enc, y_pad)
            g_kl = model.kl(mu, logvar)
            g_loss = g_adv + lambda_l1*g_l1 + lambda_kl*g_kl
            
            g_loss.backward()
            opt_g.step()
            opt_e.step()
            
            # Sammle Losses
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            g_l1_losses.append(g_l1.item())
            
            # Früh stoppen bei explodierenden Losses
            if g_loss.item() > 100 or d_loss.item() > 100:
                print(f"💥 Training gestoppt - Loss explodiert!")
                raise optuna.TrialPruned()
            
            # Limitiere Batches für Optuna
            if batch_idx > 50:  # Nur 50 Batches pro Epoche
                break
        
        # Epoch Stats
        avg_g_loss = np.mean(g_losses)
        avg_d_loss = np.mean(d_losses)
        avg_g_l1 = np.mean(g_l1_losses)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # Progress Logging
        if ep % 5 == 0 or ep == 1:
            avg_epoch_time = np.mean(epoch_times)
            remaining_time = avg_epoch_time * (epochs - ep)
            print(f"📊 Epoche {ep:2d}/{epochs} | "
                  f"G: {avg_g_loss:.4f} | D: {avg_d_loss:.4f} | "
                  f"L1: {avg_g_l1:.4f} | Zeit: {epoch_time:.1f}s | "
                  f"ETA: {remaining_time/60:.1f}min")
        
        # Optuna Reporting (verwende G_L1 als Metrik)
        trial.report(avg_g_l1, ep)
        
        # Pruning
        if trial.should_prune():
            print(f"✂️  Trial {trial.number} PRUNED nach Epoche {ep}")
            raise optuna.TrialPruned()
        
        if avg_g_l1 < best_g_l1:
            best_g_l1 = avg_g_l1
            print(f"🎉 Neue beste G_L1: {best_g_l1:.6f}")
        
        # Früh stoppen wenn Modell nicht konvergiert
        if ep > 10 and avg_g_l1 > 5.0:
            print(f"⚠️  Trial gestoppt - keine Konvergenz")
            break
    
    trial_time = time.time() - trial_start_time
    
    # Trial-Zusammenfassung
    print(f"\n{'='*60}")
    print(f"📈 GAN TRIAL {trial.number} ABGESCHLOSSEN")
    print(f"{'='*60}")
    print(f"⏱️  Gesamtzeit: {trial_time/60:.1f} Minuten")
    print(f"🎯 Beste G_L1: {best_g_l1:.6f}")
    print(f"📊 Finale G_L1: {avg_g_l1:.6f}")
    
    # Speichere Trial-Ergebnisse
    os.makedirs("Optuna/GAN", exist_ok=True)
    trial_results = {
        'trial_number': trial.number,
        'params': trial.params,
        'best_g_l1': best_g_l1,
        'final_g_l1': avg_g_l1,
        'trial_time_minutes': trial_time/60,
        'epochs_completed': ep
    }
    
    np.save(f"Optuna/GAN/trial_{trial.number}_results.npy", trial_results)
    
    # GPU Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🧹 GPU Memory bereinigt")
    
    print(f"{'='*60}\n")
    
    return best_g_l1

def run_gan_optuna_study(n_trials=50, n_jobs=1):
    """Starte Optuna Hyperparameter-Suche für GAN"""
    import time
    
    study_start_time = time.time()
    
    print(f"\n{'🔬'*20}")
    print(f"🚀 GAN OPTUNA HYPERPARAMETER SEARCH")
    print(f"{'🔬'*20}")
    print(f"🎯 Trials geplant: {n_trials}")
    print(f"⚡ Parallele Jobs: {n_jobs}")
    print(f"🔥 Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"📅 Startzeit: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Erstelle oder lade Study
    study_name = "tboi_gan_optimization"
    storage = f"sqlite:///optuna_{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='minimize',  # Minimiere G_L1 Loss
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=5,
            interval_steps=1
        )
    )
    
    # Zeige bisherige Trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) > 0:
        print(f"📋 Bestehende Study gefunden mit {len(study.trials)} Trials")
        print(f"✅ Erfolgreich abgeschlossen: {len(completed_trials)}")
        try:
            print(f"🏆 Beste bisherige G_L1 Loss: {study.best_value:.6f}")
            print(f"📊 Beste Parameter: {study.best_params}")
        except ValueError:
            print(f"⚠️  Noch keine gültigen Ergebnisse in der Datenbank")
        print(f"{'='*60}")
    else:
        print(f"📋 Neue Study wird erstellt")
        print(f"{'='*60}")
    
    # Fortschritts-Callback
    def progress_callback(study, trial):
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        
        print(f"\n📊 GAN FORTSCHRITT UPDATE:")
        print(f"   ✅ Abgeschlossen: {completed}")
        print(f"   ✂️  Pruned: {pruned}")
        print(f"   ❌ Fehler: {failed}")
        
        if completed > 0:
            try:
                print(f"   🏆 Aktuelle beste G_L1: {study.best_value:.6f}")
            except ValueError:
                print(f"   🏆 Beste G_L1: Noch keine gültigen Trials")
        
        if completed > 0:
            elapsed = time.time() - study_start_time
            avg_time_per_trial = elapsed / (completed + pruned + failed)
            remaining_trials = n_trials - (completed + pruned + failed)
            eta = remaining_trials * avg_time_per_trial
            print(f"   ⏱️  Durchschn. Zeit/Trial: {avg_time_per_trial/60:.1f}min")
            print(f"   🎯 ETA: {eta/3600:.1f} Stunden")
        print(f"{'='*60}")
    
    # Optimierung mit Callback
    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, callbacks=[progress_callback])
    except KeyboardInterrupt:
        print(f"\n⚠️  Unterbrochen durch Benutzer!")
    
    study_time = time.time() - study_start_time
    completed_final = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    # Finale Ergebnisse
    print(f"\n{'🎉'*20}")
    print(f"🏁 GAN OPTUNA STUDIE ABGESCHLOSSEN")
    print(f"{'🎉'*20}")
    print(f"⏱️  Gesamtzeit: {study_time/3600:.1f} Stunden")
    print(f"🎯 Trials durchgeführt: {len(study.trials)}")
    print(f"✅ Erfolgreich: {len(completed_final)}")
    print(f"✂️  Gepruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"❌ Fehler: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    print(f"{'='*60}")
    
    if len(completed_final) > 0:
        try:
            print(f"🏆 BESTE G_L1 LOSS: {study.best_value:.6f}")
            print(f"📊 BESTE HYPERPARAMETER:")
            for key, value in study.best_params.items():
                print(f"   {key}: {value}")
        except ValueError:
            print(f"⚠️  Keine gültigen Trials verfügbar")
    else:
        print(f"⚠️  Keine erfolgreichen Trials abgeschlossen!")
    
    print(f"{'='*60}")
    
    # Speichere Ergebnisse
    os.makedirs("Optuna/GAN", exist_ok=True)
    with open("Optuna/GAN/study_results.txt", "w") as f:
        f.write(f"GAN Optuna Study Results\n")
        f.write(f"========================\n")
        f.write(f"Study Time: {study_time/3600:.1f} hours\n")
        f.write(f"Total Trials: {len(study.trials)}\n")
        f.write(f"Completed Trials: {len(completed_final)}\n")
        if len(completed_final) > 0:
            try:
                f.write(f"Best G_L1 Loss: {study.best_value:.6f}\n")
                f.write(f"Best Params: {study.best_params}\n")
            except ValueError:
                f.write(f"Best Loss: No valid trials\n")
    
    return study

def show_gan_study_status():
    """Zeige aktuellen Status der GAN Optuna Study"""
    try:
        study = optuna.load_study(
            study_name="tboi_gan_optimization",
            storage="sqlite:///optuna_tboi_gan_optimization.db"
        )
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        
        print(f"\n📊 GAN OPTUNA STATUS:")
        print(f"{'='*40}")
        print(f"✅ Abgeschlossene Trials: {completed}")
        print(f"✂️  Geprune Trials: {pruned}")
        print(f"❌ Fehlgeschlagene Trials: {failed}")
        print(f"📊 Total Trials: {len(study.trials)}")
        
        if completed > 0:
            try:
                print(f"🏆 Beste G_L1 Loss: {study.best_value:.6f}")
                print(f"📋 Beste Parameter:")
                for key, value in study.best_params.items():
                    print(f"   {key}: {value}")
            except ValueError:
                print(f"⚠️  Keine gültigen Ergebnisse verfügbar")
        print(f"{'='*40}")
    except:
        print("❌ Keine GAN Study gefunden")

# Ändere cli_main() zu main():
def main():  # Statt cli_main()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir")
    parser.add_argument("--max_epochs",   type=int,   default=400)
    parser.add_argument("--decay_start_epoch", type=int, default=200)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--split_ratio",    type=float, default=0.9)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--accelerator",    default="auto", choices=["cpu","gpu","auto"])
    parser.add_argument("--devices",        default="auto")
    parser.add_argument("--r1_every", type=int, default=16)
    parser.add_argument("--r1_gamma", type=float, default=5e-3)
    parser.add_argument("--d_suppress_thresh", type=float, default=0.3)
    
    # Neue Optuna Argumente
    parser.add_argument("--mode", default="train", choices=["train", "optuna", "status"],
                       help="Modus: train, optuna, oder status")
    parser.add_argument("--optuna_trials", type=int, default=50, 
                       help="Anzahl Optuna Trials")
    parser.add_argument("--optuna_jobs", type=int, default=1, 
                       help="Parallele Optuna Jobs")
    
    args = parser.parse_args()

    if args.mode == "train":
        if not args.data_dir:
            print("❌ --data_dir ist erforderlich für Training")
            return
            
        seed_everything(args.seed, workers=True)

        dm = BmpDataModule(args.data_dir, args.batch_size,
                           num_workers=0, split_ratio=args.split_ratio)

        model = LitBicycleGAN(
            in_ch=1,
            decay_start=args.decay_start_epoch,
            total_epochs=args.max_epochs,
            r1_every=args.r1_every,
            r1_gamma=args.r1_gamma,
            d_suppress_thresh=args.d_suppress_thresh
        )

        ckpt_cb = ModelCheckpoint(save_last=True, save_top_k=3, monitor="g_l1", mode="min")
        lr_cb   = LearningRateMonitor(logging_interval="epoch")
        pbar_cb = TQDMProgressBar(refresh_rate=10)
        tb_logger = TensorBoardLogger("tb_logs", name="tboi_gan", default_hp_metric=False)

        trainer = Trainer(
            accelerator=args.accelerator,
            devices=args.devices,
            max_epochs=args.max_epochs,
            precision="16-mixed",
            callbacks=[ckpt_cb, lr_cb, pbar_cb],
            logger=tb_logger,
        )
        trainer.fit(model, datamodule=dm)
        
    elif args.mode == "optuna":
        print(f"🚀 Starte GAN Optuna mit {args.optuna_trials} Trials")
        study = run_gan_optuna_study(n_trials=args.optuna_trials, n_jobs=args.optuna_jobs)
        
        # Optional: Trainiere bestes Modell
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed_trials) > 0:
            try:
                print(f"\n🎯 Trainiere bestes GAN-Modell...")
                best_params = study.best_params
                
                seed_everything(42, workers=True)
                dm = BmpDataModule("Bitmaps", best_params['batch_size'], num_workers=0)
                
                model = LitBicycleGAN(
                    in_ch=1,
                    z_dim=best_params['z_dim'],
                    lr=best_params['lr'],
                    lambda_l1=best_params['lambda_l1'],
                    lambda_kl=best_params['lambda_kl'],
                    decay_start=200,
                    total_epochs=400,
                    d_suppress_thresh=best_params['d_suppress_thresh'],
                    r1_gamma=best_params['r1_gamma']
                )
                
                ckpt_cb = ModelCheckpoint(
                    dirpath="./best_optuna_gan_model",
                    save_last=True, save_top_k=3, monitor="g_l1", mode="min"
                )
                lr_cb = LearningRateMonitor(logging_interval="epoch")
                pbar_cb = TQDMProgressBar(refresh_rate=10)
                tb_logger = TensorBoardLogger("tb_logs", name="tboi_gan_optuna_best")
                
                trainer = Trainer(
                    accelerator="auto",
                    devices="auto",
                    max_epochs=400,
                    precision="16-mixed",
                    callbacks=[ckpt_cb, lr_cb, pbar_cb],
                    logger=tb_logger,
                )
                trainer.fit(model, datamodule=dm)
                
            except ValueError:
                print(f"⚠️  Kann bestes Modell nicht trainieren")
        else:
            print(f"⚠️  Keine erfolgreichen Trials!")
            
    elif args.mode == "status":
        show_gan_study_status()

if __name__ == "__main__":
    main()
