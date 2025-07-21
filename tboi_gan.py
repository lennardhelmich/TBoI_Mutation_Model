#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tboi_gan.py

BicycleGAN für 13×7 px 1‑Kanal‑BMPs in verschachtelter Ordnerstruktur.
Validation ist komplett aus—kein CombinedLoader mehr.
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


# -------------------------------------------------------------------
# Padding 13×7 ↔ 16×8 für voll-conv
# -------------------------------------------------------------------
def pad_to_pow2(img: torch.Tensor, th: int = 16, tw: int = 8) -> torch.Tensor:
    _, _, h, w = img.shape
    return F.pad(img, (0, tw - w, 0, th - h), mode="replicate")

def unpad(img: torch.Tensor, h: int = 13, w: int = 7) -> torch.Tensor:
    return img[..., :h, :w]


# -------------------------------------------------------------------
# 1) Dataset für deine Ordnerstruktur
# -------------------------------------------------------------------
class NestedBmpDataset(Dataset):
    def __init__(self, root: str | Path, split: str="train", split_ratio: float=0.9, seed: int=42):
        self.src_dir = Path(root) / "Mutations"
        self.dst_dir = Path(root) / "Inverted_Mutations"

        pairs = []
        for bmp_k in sorted(self.src_dir.iterdir()):
            if not bmp_k.is_dir(): continue
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

    def __len__(self): return len(self.pairs)
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
                          num_workers=self.hparams.num_workers,  # Auf 0 gesetzt
                          pin_memory=True)

    # ENTFERNE val_dataloader() komplett oder gib einen Dummy zurück
    # def val_dataloader(self):
    #     return None


# -------------------------------------------------------------------
# 2) Netz-Bausteine
# -------------------------------------------------------------------
def conv(in_c, out_c, k=3, s=1, p=1, norm=True, act=True):
    layers = [nn.ReflectionPad2d(p), nn.Conv2d(in_c, out_c, k, s, 0, bias=not norm)]
    if norm: layers.append(nn.InstanceNorm2d(out_c))
    if act:  layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class UNetGenerator(nn.Module):
    def __init__(self, in_ch, out_ch, z_dim=128, nf=64):
        super().__init__()
        self.d1 = conv(in_ch, nf, 4, 2, 1)
        self.d2 = conv(nf+z_dim, nf*2, 4, 2, 1)
        self.d3 = conv(nf*2, nf*4, 4, 2, 1, norm=False, act=False)
        self.u1 = nn.Sequential(nn.ConvTranspose2d(nf*4, nf*2, 4, 2, 1),
                                nn.InstanceNorm2d(nf*2), nn.ReLU(True))
        self.u2 = nn.Sequential(nn.ConvTranspose2d(nf*4, nf,   4, 2, 1),
                                nn.InstanceNorm2d(nf),   nn.ReLU(True))
        self.fin= nn.Sequential(nn.ConvTranspose2d(nf*2, out_ch, 4, 2, 1), nn.Tanh())
        self.z_dim = z_dim

    def forward(self, x, z):
        d1 = self.d1(x)
        B, _, H, W = d1.shape
        z_img = z.view(B, self.z_dim, 1, 1).expand(-1, -1, H, W)
        d2 = self.d2(torch.cat([d1, z_img], 1))
        d3 = self.d3(d2)
        u1 = self.u1(d3); u1 = torch.cat([u1, d2], 1)
        u2 = self.u2(u1); u2 = torch.cat([u2, d1], 1)
        return self.fin(u2)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch, nf=64):
        super().__init__()
        self.net = nn.Sequential(
            # Erste Layer: 16×8 → 8×4
            conv(in_ch, nf, 4, 2, 1, norm=False),
            # Zweite Layer: 8×4 → 4×2  
            conv(nf, nf*2, 4, 2, 1),
            # Dritte Layer: 4×2 → 2×1 (kleinere Kernel!)
            conv(nf*2, nf*4, 3, 1, 1),  # 3×3 statt 4×4
            # Final Layer: 2×1 → 1×1
            nn.Conv2d(nf*4, 1, 2, 1, 0),  # 2×2 statt 4×4
        )

    def forward(self, x): return self.net(x)


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
# 3) Lightning-Modul mit Scheduler
# -------------------------------------------------------------------
class LitBicycleGAN(LightningModule):
    def __init__(self, in_ch=1, z_dim=128, lr=2e-4,
                 lambda_l1=10.0, lambda_kl=0.01,
                 decay_start=200, total_epochs=400):
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

    def training_step(self, batch, _):
        x, y = batch
        x_pad, y_pad = pad_to_pow2(x), pad_to_pow2(y)
        opt_g, opt_d, opt_e = self.optimizers()

        mu, logvar = self.E(y_pad)
        std        = torch.exp(0.5*logvar)
        z_enc      = mu + torch.randn_like(std)*std
        z_rand     = torch.randn_like(z_enc)

        fake_enc  = self.G(x_pad, z_enc)
        fake_rand = self.G(x_pad, z_rand)

        # D step
        opt_d.zero_grad(set_to_none=True)
        d_real = self.D(torch.cat([x_pad, y_pad],1))
        d_fake = self.D(torch.cat([x_pad, fake_rand.detach()],1))
        d_loss = self.hinge(d_real, True) + self.hinge(d_fake, False)
        self.manual_backward(d_loss); opt_d.step()

        # G+E step
        opt_g.zero_grad(set_to_none=True); opt_e.zero_grad(set_to_none=True)
        g_adv = self.hinge(self.D(torch.cat([x_pad, fake_rand],1)), True)
        g_l1  = F.l1_loss(fake_enc, y_pad)
        g_kl  = self.kl(mu, logvar)
        g_loss= g_adv + self.hparams.lambda_l1*g_l1 + self.hparams.lambda_kl*g_kl
        self.manual_backward(g_loss); opt_g.step(); opt_e.step()

        self.log_dict({"d_loss": d_loss, "g_adv": g_adv, "g_l1": g_l1, "g_kl": g_kl},
                      prog_bar=True)
        self.log("lr", opt_g.param_groups[0]['lr'], prog_bar=True, logger=False)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.G.parameters(), lr=self.hparams.lr,  betas=(0.5,0.999))
        opt_d = torch.optim.Adam(self.D.parameters(), lr=self.hparams.lr*0.5, betas=(0.5,0.999))
        opt_e = torch.optim.Adam(self.E.parameters(), lr=self.hparams.lr,  betas=(0.5,0.999))

        def lr_lambda(epoch):
            if epoch < self.hparams.decay_start: return 1.0
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


# -------------------------------------------------------------------
# 4) CLI: Trainer.fit mit datamodule=
# -------------------------------------------------------------------
def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",          required=True)
    parser.add_argument("--max_epochs",   type=int,   default=400)
    parser.add_argument("--decay_start_epoch", type=int, default=200)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--split_ratio",    type=float, default=0.9)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--accelerator",    default="auto", choices=["cpu","gpu","auto"])
    parser.add_argument("--devices",        default="auto")
    args = parser.parse_args()

    seed_everything(args.seed, workers=True)

    dm = BmpDataModule(args.data_dir, args.batch_size,
                       num_workers=0, split_ratio=args.split_ratio)  # num_workers=0
    model = LitBicycleGAN(in_ch=1,
                          decay_start=args.decay_start_epoch,
                          total_epochs=args.max_epochs)

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
        # ENTFERNE alle validation-bezogenen Parameter
        # num_sanity_val_steps=0,    
        # limit_val_batches=0,       
    )
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    cli_main()
