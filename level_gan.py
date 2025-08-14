#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
level_gan_interpolator.py
=========================

Conditional GAN (Pix2Pix-Variante) für The Binding of Isaac Level-Bitmaps
(12 Klassen, Raster 13×7 → intern H×W = 7×13).

Vereinfachung: KEINE Klassen-Gewichtung, KEIN Focal-Loss.
Training lernt: Inverted_Mutations  →  Mutations.

Neu:
- d_lr_factor: Diskriminator-LR wird relativ zur Generator-LR skaliert (z. B. 0.5)
- d_every: D wird nur jedes n-te Batch geupdatet (z. B. 2 ⇒ halb so oft)
- Optuna-Studie mit Pruning über val_px, inkl. Batch-Size als HParam und finalem 100-Epochen-Training mit Best-Params.
"""

from __future__ import annotations
import random
from pathlib import Path
import argparse
import os

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

# Optional: Optuna
try:
    import optuna
    from optuna.integration import PyTorchLightningPruningCallback
    _HAS_OPTUNA = True
except Exception:
    _HAS_OPTUNA = False


# -----------------------------------------------------------------------------#
# 1) Klassen & Mapping
# -----------------------------------------------------------------------------#

NUM_CLASSES = 12
CLASS_NAMES = [
    "WALL", "DOOR", "FREE_SPACE", "STONE", "PIT", "BLOCK",
    "ENTITY", "PICKUP", "MACHINE", "FIRE", "POOP", "SPIKE"
]

def _build_pixel_maps():
    """Versuche Mapping aus tboi_bitmap, sonst gleichmäßige 12 Graustufen."""
    try:
        from tboi_bitmap import TBoI_Bitmap, EntityType
        tboi = TBoI_Bitmap()
        entity_to_pixel = {e.value: tboi.get_pixel_value_with_entity_id(e) for e in EntityType}
        pixel_to_entity = np.zeros(256, dtype=np.uint8)
        for cid, px in entity_to_pixel.items():
            pixel_to_entity[px] = cid
        return entity_to_pixel, pixel_to_entity
    except Exception:
        entity_to_pixel = {cid: round(cid * 255 / (NUM_CLASSES - 1)) for cid in range(NUM_CLASSES)}
        pixel_to_entity = np.zeros(256, dtype=np.uint8)
        for cid, px in entity_to_pixel.items():
            pixel_to_entity[px] = cid
        return entity_to_pixel, pixel_to_entity

ENTITY_TO_PIXEL, PIXEL_TO_ENTITY = _build_pixel_maps()


# -----------------------------------------------------------------------------#
# 2) Dataset & DataModule
# -----------------------------------------------------------------------------#

def _remove_outer_border(arr: np.ndarray) -> np.ndarray:
    """Entfernt den 1-px Rand: (9,15) → (7,13). Lässt (7,13) unverändert."""
    if arr.shape == (9, 15):
        return arr[1:-1, 1:-1]
    return arr

def _add_outer_border(arr_hw: np.ndarray) -> np.ndarray:
    """Fügt 1-px Border hinzu: (7,13) → (9,15), Rand=0 (WALL)."""
    out = np.zeros((arr_hw.shape[0] + 2, arr_hw.shape[1] + 2), dtype=np.uint8)
    out[1:-1, 1:-1] = arr_hw
    return out

class LevelDataset(Dataset):
    """
    Struktur:
      root/
        ├─ Mutations/                     (TARGET Y)
        │     <gruppe>/ mutation_*.bmp
        └─ Inverted_Mutations/            (INPUT X)
              <gruppe>/ mutation_*/ *.bmp

    Lädt X (inverted) als One-Hot (C,H,W) float und Y (mutation) als Klassen-IDs (H,W) long.
    Training lernt: Inverted_Mutations  →  Mutations.
    """
    def __init__(self, root: str | Path, pick_random_input=False, seed=42):
        super().__init__()
        self.root_in = Path(root) / "Inverted_Mutations"  # INPUT
        self.root_out = Path(root) / "Mutations"          # TARGET
        self.pick_random_input = pick_random_input
        self.rng = random.Random(seed)
        self.pairs = self._collect_pairs()

    def _collect_pairs(self):
        pairs: list[tuple[Path, Path]] = []
        for grp_dir in sorted(self.root_in.iterdir()):
            if not grp_dir.is_dir():
                continue
            for inv_dir in sorted(grp_dir.glob("mutation_*")):
                if not inv_dir.is_dir():
                    continue
                inv_bmps = sorted(inv_dir.glob("*.bmp"))
                if not inv_bmps:
                    continue
                target = self.root_out / grp_dir.name / f"{inv_dir.name}.bmp"
                if not target.is_file():
                    continue
                p_in = self.rng.choice(inv_bmps) if self.pick_random_input else inv_bmps[0]
                pairs.append((p_in, target))
        if not pairs:
            raise RuntimeError(f"Keine Paare gefunden (Inverted_Mutations → Mutations) unter {self.root_in}")
        return pairs

    @staticmethod
    def _bmp_to_ids(path: Path) -> torch.Tensor:
        """BMP → (H,W) Klassen-IDs, nach Border-Crop H×W = 7×13."""
        arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
        arr = _remove_outer_border(arr)
        cls = torch.from_numpy(PIXEL_TO_ENTITY[arr]).long()
        return cls

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        p_in, p_out = self.pairs[idx]
        x_ids = self._bmp_to_ids(p_in)                          # (7,13)
        y_ids = self._bmp_to_ids(p_out)                         # (7,13)
        x_onehot = F.one_hot(x_ids.long(), NUM_CLASSES).permute(2,0,1).float()  # (C,7,13)
        return x_onehot, y_ids


class LevelDataModule(pl.LightningDataModule):
    def __init__(self, root="Bitmaps", batch_size=256, num_workers=4, val_pct=0.1, seed=42):
        super().__init__()
        self.root = root
        self.bs = batch_size
        self.nw = num_workers
        self.val_pct = val_pct
        self.seed = seed

    def setup(self, stage=None):
        full = LevelDataset(self.root, pick_random_input=False, seed=self.seed)
        n = len(full)
        n_val = max(1, int(self.val_pct * n))
        n_train = n - n_val
        self.train, self.val = torch.utils.data.random_split(
            full, [n_train, n_val],
            generator=torch.Generator().manual_seed(self.seed)
        )

    def _dl(self, ds, shuffle=False):
        return DataLoader(ds, self.bs, shuffle, num_workers=self.nw, pin_memory=True)

    def train_dataloader(self): return self._dl(self.train, True)
    def val_dataloader(self):   return self._dl(self.val, False)


# -----------------------------------------------------------------------------#
# 3) Architektur (U-Net Generator + PatchGAN Discriminator)
# -----------------------------------------------------------------------------#

def _conv_block(in_ch, out_ch, bn=True):
    layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)]
    if bn:
        layers += [nn.BatchNorm2d(out_ch)]
    layers += [nn.LeakyReLU(0.2, inplace=True)]
    return nn.Sequential(*layers)

class _Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(_conv_block(in_ch, out_ch), _conv_block(out_ch, out_ch))
        self.pool = nn.MaxPool2d(2, ceil_mode=True)   # robust für 7×13

    def forward(self, x):
        h = self.conv(x)
        return self.pool(h), h

class _Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(_conv_block(in_ch, out_ch), _conv_block(out_ch, out_ch))

    def forward(self, x, skip):
        x = self.up(x)
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], 1)
        return self.conv(x)

class GeneratorUNet(nn.Module):
    """Kompakter U-Net Generator – ausgelegt auf (C,7,13)."""
    def __init__(self, in_ch=NUM_CLASSES, out_ch=NUM_CLASSES, base=32):
        super().__init__()
        self.d1 = _Down(in_ch,   base)
        self.d2 = _Down(base,    base*2)
        self.d3 = _Down(base*2,  base*4)
        self.mid = _conv_block(base*4, base*8)
        self.u3 = _Up(base*8,  base*4)
        self.u2 = _Up(base*4,  base*2)
        self.u1 = _Up(base*2,  base)
        self.out = nn.Conv2d(base, out_ch, 1)  # Logits (ohne Softmax)

    def forward(self, x):
        x, s1 = self.d1(x)
        x, s2 = self.d2(x)
        x, s3 = self.d3(x)
        x = self.mid(x)
        x = self.u3(x, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)
        return self.out(x)

class PatchGAN(nn.Module):
    """Spektral-normalisiertes Patch-GAN über (x|y), angepasst für 7×13."""
    def __init__(self, in_ch=NUM_CLASSES*2, base=32, n_layers=2):
        super().__init__()
        layers = [
            spectral_norm(nn.Conv2d(in_ch, base, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        nf = base
        # nur 2 Downsamplings -> stabil für H=7
        for n in range(1, n_layers):
            nf_prev, nf = nf, min(base * 2**n, 256)
            layers += [
                spectral_norm(nn.Conv2d(nf_prev, nf, 4, 2, 1)),
                nn.BatchNorm2d(nf),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        # Letzte 3×3-Conv ohne Stride liefert Patch-Map (B,1,1,3)
        layers.append(spectral_norm(nn.Conv2d(nf, 1, 3, padding=1)))
        self.model = nn.Sequential(*layers)

    def forward(self, x):   # x: (B, 24, 7, 13)
        return self.model(x)  # (B, 1, 1, 3)



# -----------------------------------------------------------------------------#
# 4) Lightning-Modul (manuelle Optimierung; d_lr_factor & d_every)
# -----------------------------------------------------------------------------#

class LitPix2Pix(pl.LightningModule):
    def __init__(self, lr=2e-4, lambda_px=10.0, d_lr_factor=0.5, d_every=2):
        super().__init__()
        self.save_hyperparameters()
        self.G = GeneratorUNet()
        self.D = PatchGAN(n_layers=2)
        self.ce  = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.automatic_optimization = False

    def forward(self, x):  # für Inferenz
        return self.G(x).softmax(1)

    @staticmethod
    def _soft_one_hot(y_ids: torch.Tensor) -> torch.Tensor:
        return F.one_hot(y_ids, NUM_CLASSES).permute(0,3,1,2).float()

    def _adv(self, pred, real=True):
        tgt = torch.ones_like(pred) if real else torch.zeros_like(pred)
        return self.bce(pred, tgt)

    def configure_optimizers(self):
        # D-LR wird über d_lr_factor relativ zur G-LR gesetzt
        opt_g = torch.optim.Adam(self.G.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.D.parameters(), lr=self.hparams.lr * self.hparams.d_lr_factor, betas=(0.5, 0.999))
        return [opt_g, opt_d]

    def training_step(self, batch, batch_idx):
        """
        Manuelle Optimierung:
          - D-Step nur jedes d_every-te Batch
          - G-Step in jedem Batch
        """
        x, y_ids = batch  # x=(B,C,7,13) [Inverted], y_ids=(B,7,13) [Mutation]
        y_soft = self._soft_one_hot(y_ids)

        opt_g, opt_d = self.optimizers()

        # ---- 1) Discriminator step (selten) ----
        do_d_step = (batch_idx % self.hparams.d_every == 0)
        if do_d_step:
            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake_soft = self.G(x).softmax(1)
            d_real = self.D(torch.cat([x, y_soft], 1))
            d_fake = self.D(torch.cat([x, fake_soft], 1))
            loss_d = 0.5 * (self._adv(d_real, True) + self._adv(d_fake, False))
            self.manual_backward(loss_d)
            opt_d.step()
            self.log("d_loss", loss_d, prog_bar=True, on_step=True, on_epoch=True)

        # ---- 2) Generator step (immer) ----
        opt_g.zero_grad(set_to_none=True)
        fake_logits = self.G(x)
        fake_soft = fake_logits.softmax(1)
        adv_g = self._adv(self.D(torch.cat([x, fake_soft], 1)), True)
        px = self.ce(fake_logits, y_ids)
        loss_g = adv_g + self.hparams.lambda_px * px
        self.manual_backward(loss_g)
        opt_g.step()

        # Logging
        self.log_dict(
            {"g_adv": adv_g, "px": px, "g_loss": loss_g},
            prog_bar=True, on_step=True, on_epoch=True
        )

    def validation_step(self, batch, _):
        x, y_ids = batch
        logits = self.G(x)
        px = self.ce(logits, y_ids)
        acc = (logits.argmax(1) == y_ids).float().mean()
        self.log_dict({"val_px": px, "val_acc": acc}, prog_bar=True)


# -----------------------------------------------------------------------------#
# 5) Interpolation (Input-Space der INVERTEDs)
# -----------------------------------------------------------------------------#

def _bmp_to_one_hot_tensor(path: str | Path) -> torch.Tensor:
    """BMP → (1,C,7,13) float (nimmt **Inverted_Mutations**-Bitmap an)."""
    arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    arr = _remove_outer_border(arr)
    y = torch.from_numpy(PIXEL_TO_ENTITY[arr]).long()         # (7,13)
    x = F.one_hot(y, NUM_CLASSES).permute(2,0,1).unsqueeze(0).float()
    return x  # (1,C,7,13)

def _ids_to_bmp_with_border(y_ids: np.ndarray) -> Image.Image:
    """(7,13) → BMP (9,15) mit Rand=0 (WALL)."""
    inner = np.vectorize(lambda k: ENTITY_TO_PIXEL[int(k)])(y_ids).astype(np.uint8)
    with_border = _add_outer_border(inner)
    return Image.fromarray(with_border, mode="L")

@torch.no_grad()
def interpolate_inputs_and_decode(ckpt: str, bmp_a: str, bmp_b: str,
                                  steps=5, out_dir="interp_out", device="auto"):
    """
    Lineare Interpolation im **INPUT-Space der Inverted_Mutations**:
    x(a..b) → G → BMPs speichern (Output-Stil: Mutations).
    """
    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else device)
    model = LitPix2Pix.load_from_checkpoint(ckpt, map_location=dev).to(dev).eval()

    xa = _bmp_to_one_hot_tensor(bmp_a).to(dev)
    xb = _bmp_to_one_hot_tensor(bmp_b).to(dev)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    alphas = torch.linspace(0, 1, steps, device=dev)
    for i, a in enumerate(alphas):
        xmix = (1.0 - a) * xa + a * xb
        logits = model.G(xmix)
        y_ids = logits.argmax(1)[0].cpu().numpy()
        img = _ids_to_bmp_with_border(y_ids)
        img.save(out_path / f"interp_{i:02d}_a{float(a):.2f}.bmp")


# -----------------------------------------------------------------------------#
# 6) Optuna (Studie + Final-Training)
# -----------------------------------------------------------------------------#

def _make_trainer(max_epochs, monitor="val_px", mode="min", gpus=1, log_every=20, enable_checkpointing=True):
    ckpt_cb = pl.callbacks.ModelCheckpoint(monitor=monitor, mode=mode, save_top_k=1, save_last=False)
    lr_cb = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    pbar = pl.callbacks.TQDMProgressBar(refresh_rate=10)
    accelerator = "gpu" if (gpus and torch.cuda.is_available()) else "cpu"
    devices = gpus if accelerator == "gpu" else None
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator, devices=devices,
        log_every_n_steps=log_every,
        callbacks=[ckpt_cb, lr_cb, pbar],
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        enable_checkpointing=enable_checkpointing,
        default_root_dir=Path("lightning_logs")
    )
    return trainer, ckpt_cb

def _objective_optuna(trial, root, num_workers, val_pct, gpus, seed=42, opt_epochs=20):
    # --- Suchraum ---
    lr          = trial.suggest_loguniform("lr", 1e-5, 5e-4)
    lambda_px   = trial.suggest_float("lambda_px", 5.0, 20.0)
    d_lr_factor = trial.suggest_categorical("d_lr_factor", [0.25, 0.5, 0.75, 1.0])
    d_every     = trial.suggest_int("d_every", 1, 3)
    batch_size  = trial.suggest_categorical("batch_size", [128, 256, 512])

    # Daten
    dm = LevelDataModule(root=root, batch_size=batch_size, num_workers=num_workers, val_pct=val_pct, seed=seed)
    dm.setup()

    # Modell
    model = LitPix2Pix(lr=lr, lambda_px=lambda_px, d_lr_factor=d_lr_factor, d_every=d_every)

    # Trainer mit Pruning-Callback
    trainer, ckpt_cb = _make_trainer(max_epochs=opt_epochs, monitor="val_px", mode="min", gpus=gpus)
    pruning_cb = PyTorchLightningPruningCallback(trial, monitor="val_px") if _HAS_OPTUNA else None
    if pruning_cb:
        trainer.callbacks.append(pruning_cb)

    trainer.fit(model, datamodule=dm)

    # Bestes val_px aus CheckpointCallback
    if ckpt_cb.best_model_score is None:
        return float("inf")
    return ckpt_cb.best_model_score.item()

def run_optuna(root, trials=30, n_jobs=1, num_workers=0, gpus=1, val_pct=0.1, study_name="level_gan_study",
               storage="sqlite:///optuna_level_gan.db", seed=42, opt_epochs=20, final_epochs=100, out_dir="optuna_final"):
    if not _HAS_OPTUNA:
        raise RuntimeError("Optuna ist nicht installiert. Bitte 'pip install optuna' ausführen.")

    pl.seed_everything(seed, workers=True)

    # Studie laden/erstellen (fortsetzbar)
    study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage, load_if_exists=True)

    # Optimize
    study.optimize(
        lambda t: _objective_optuna(t, root=root, num_workers=num_workers, val_pct=val_pct, gpus=gpus,
                                    seed=seed, opt_epochs=opt_epochs),
        n_trials=trials,
        n_jobs=n_jobs
    )

    print("\n🏁 Optuna abgeschlossen.")
    print(f"Beste val_px: {study.best_value:.6f}")
    print(f"Beste Parameter: {study.best_params}")

    # --- Finales Training 100 Epochen mit besten Parametern ---
    best = study.best_params
    bs   = int(best["batch_size"])
    dm   = LevelDataModule(root=root, batch_size=bs, num_workers=num_workers, val_pct=val_pct, seed=seed)
    dm.setup()

    model = LitPix2Pix(
        lr=float(best["lr"]),
        lambda_px=float(best["lambda_px"]),
        d_lr_factor=float(best["d_lr_factor"]),
        d_every=int(best["d_every"])
    )

    trainer, ckpt_cb = _make_trainer(max_epochs=final_epochs, monitor="val_px", mode="min", gpus=gpus)
    trainer.fit(model, datamodule=dm)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Speichere bestes Modell (Lightning Checkpoint)
    if ckpt_cb.best_model_path:
        final_ckpt = Path(out_dir) / "best_final.ckpt"
        # Kopiere die Datei
        from shutil import copyfile
        copyfile(ckpt_cb.best_model_path, final_ckpt)
        print(f"✅ Finales bestes Modell gespeichert unter: {final_ckpt}")
    else:
        print("⚠️ Kein bestes Modell im Final-Training gefunden (keine ckpt_cb.best_model_path).")

    return study


# -----------------------------------------------------------------------------#
# 7) CLI
# -----------------------------------------------------------------------------#

def main():
    p = argparse.ArgumentParser(description="Level-GAN Training & Interpolation (Inverted → Mutations)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- Train ---
    p_train = sub.add_parser("train", help="Trainiere das cGAN (Inverted_Mutations → Mutations)")
    p_train.add_argument("--root", type=str, default="Bitmaps")
    p_train.add_argument("--epochs", type=int, default=200)
    p_train.add_argument("--batch_size", type=int, default=512)
    p_train.add_argument("--num_workers", type=int, default=0)
    p_train.add_argument("--lr", type=float, default=2e-4)
    p_train.add_argument("--lambda_px", type=float, default=10.0)
    # NEU:
    p_train.add_argument("--d_lr_factor", type=float, default=0.5, help="D-LR = d_lr_factor * lr")
    p_train.add_argument("--d_every", type=int, default=2, help="Update D nur jedes n-te Batch")
    p_train.add_argument("--val_pct", type=float, default=0.1, help="Validierungsanteil für Split")
    p_train.add_argument("--gpus", type=int, default=1)

    # --- Interpolation ---
    p_interp = sub.add_parser("interp", help="Interpoliert zwischen zwei **Inverted_Mutations**-Bitmaps und decodiert via G")
    p_interp.add_argument("ckpt", type=str, help="Pfad zum Lightning .ckpt")
    p_interp.add_argument("bmp_a", type=str, help="Pfad zu Inverted-Bitmap A")
    p_interp.add_argument("bmp_b", type=str, help="Pfad zu Inverted-Bitmap B")
    p_interp.add_argument("--steps", type=int, default=5)
    p_interp.add_argument("--out_dir", type=str, default="interp_out")
    p_interp.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])

    # --- Optuna ---
    p_opt = sub.add_parser("optuna", help="Optuna-Hyperparameter-Suche mit Pruning (monitor=val_px) + Finaltraining")
    p_opt.add_argument("--root", type=str, default="Bitmaps")
    p_opt.add_argument("--trials", type=int, default=40)
    p_opt.add_argument("--n_jobs", type=int, default=1, help="Parallele Trials (bei 1 GPU meist 1)")
    p_opt.add_argument("--num_workers", type=int, default=0)
    p_opt.add_argument("--gpus", type=int, default=1)
    p_opt.add_argument("--val_pct", type=float, default=0.2, help="Validierungsanteil für Split in der Studie")
    p_opt.add_argument("--study_name", type=str, default="level_gan_study")
    p_opt.add_argument("--storage", type=str, default="sqlite:///optuna_level_gan.db")
    p_opt.add_argument("--opt_epochs", type=int, default=20, help="Epochen pro Trial")
    p_opt.add_argument("--final_epochs", type=int, default=100, help="Epochen für das abschließende Training")
    p_opt.add_argument("--out_dir", type=str, default="optuna_gan_final", help="Ordner für finales bestes Modell")

    args = p.parse_args()

    if args.cmd == "train":
        pl.seed_everything(42, workers=True)
        dm = LevelDataModule(root=args.root,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             val_pct=args.val_pct)
        model = LitPix2Pix(lr=args.lr,
                           lambda_px=args.lambda_px,
                           d_lr_factor=args.d_lr_factor,
                           d_every=args.d_every)
        trainer, ckpt_cb = _make_trainer(max_epochs=args.epochs, monitor="val_px", mode="min", gpus=args.gpus)
        trainer.fit(model, datamodule=dm)
        print("\n✅ Training abgeschlossen.")
        if ckpt_cb.best_model_path:
            print("Bestes Checkpoint:", ckpt_cb.best_model_path)

    elif args.cmd == "interp":
        interpolate_inputs_and_decode(args.ckpt, args.bmp_a, args.bmp_b,
                                      steps=args.steps, out_dir=args.out_dir, device=args.device)
        print(f"\n✅ Interpolation fertig. BMPs unter: {args.out_dir}")

    elif args.cmd == "optuna":
        run_optuna(root=args.root,
                   trials=args.trials,
                   n_jobs=args.n_jobs,
                   num_workers=args.num_workers,
                   gpus=args.gpus,
                   val_pct=args.val_pct,
                   study_name=args.study_name,
                   storage=args.storage,
                   opt_epochs=args.opt_epochs,
                   final_epochs=args.final_epochs,
                   out_dir=args.out_dir)

if __name__ == "__main__":
    main()
