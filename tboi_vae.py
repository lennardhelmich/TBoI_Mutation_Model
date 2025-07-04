#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
conv_vae_tboi.py  –  β-VAE für Binding-of-Isaac-Mutationen
( Center-Crop-Fix, Median-Freq-Klassen-Gewichtung, LR-Scheduler )
"""

# ---------------------------------------------------------------------------
# 0  Imports
# ---------------------------------------------------------------------------
import argparse, math, random, time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# 1  Konstanten
# ---------------------------------------------------------------------------
NUM_CLASSES = 12
IMG_H, IMG_W = 7, 13          # Ziel-Auflösung nach Crop
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# 2  Datensatz  (unverändert)
# ---------------------------------------------------------------------------
class MutationDataset(Dataset):
    def __init__(self, root_dir: str | Path):
        self.root = Path(root_dir)
        self.tboi = TBoI_Bitmap()
        self._pmap = np.vectorize(
            lambda px: self.tboi.get_entity_id_with_pixel_value(px).value,
            otypes=[np.uint8],
        )
        self.pairs: list[tuple[np.ndarray, np.ndarray]] = []
        self._gather_pairs()

    def _load_bitmap(self, path: Path) -> np.ndarray:
        arr = np.asarray(Image.open(path).convert("L"))[1:-1, 1:-1]
        return self._pmap(arr)

    def _gather_pairs(self):
        for in_bmp in sorted((self.root / "InputRooms").glob("bitmap_*.bmp")):
            in_arr = self._load_bitmap(in_bmp)
            mut_dir = self.root / "Mutations" / in_bmp.stem
            if not mut_dir.is_dir():  continue
            for mut_bmp in mut_dir.glob("*.bmp"):
                mut_arr = self._load_bitmap(mut_bmp)
                self.pairs.append((in_arr, mut_arr))

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx: int):
        inp, tgt = self.pairs[idx]
        return (
            torch.from_numpy(inp).unsqueeze(0).float(),
            torch.from_numpy(tgt).long(),
        )

# ---------------------------------------------------------------------------
# 3  Modell (Center-Crop-Fix unverändert)
# ---------------------------------------------------------------------------
class ConvVAE(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, latent_dim: int = 64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),   # 4×7
            nn.Conv2d(64,128, 3, stride=2, padding=1), nn.ReLU(),   # 2×4
        )
        self.enc_out = 128 * 2 * 4
        self.fc_mu     = nn.Linear(self.enc_out, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, self.enc_out)
        self.dec = nn.Sequential(
            nn.Unflatten(1, (128, 2, 4)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1,
                               output_padding=1), nn.ReLU(),        # 4×8
            nn.ConvTranspose2d(64,  32, 3, stride=2, padding=1,
                               output_padding=1), nn.ReLU(),        # 8×16
            nn.Conv2d(32, num_classes, 1)
        )

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5*logvar) * torch.randn_like(logvar)

    @staticmethod
    def _center_crop(t, hw=(IMG_H, IMG_W)):
        h, w = t.shape[-2:]
        th, tw = hw
        i0 = (h - th) // 2
        j0 = (w - tw) // 2
        return t[..., i0:i0+th, j0:j0+tw]

    def decode(self, z):
        return self._center_crop(self.dec(self.fc_dec(z)))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# ---------------------------------------------------------------------------
# 4  Loss
# ---------------------------------------------------------------------------
def beta_vae_loss(logits, targets, mu, logvar, *,
                  beta=1.0, class_weights=None):
    recon = F.cross_entropy(logits, targets,
                            weight=class_weights, reduction="mean")
    kld   = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kld, recon, kld

# ---------------------------------------------------------------------------
# 5  Median-Frequency-Gewichte           ### NEW/CHANGED
# ---------------------------------------------------------------------------
def compute_class_weights(dataset, cap=5.0) -> torch.Tensor:
    """
    Median-Frequency-Balancing nach Eigen & Fergus (2015):
        w_c = median(freq) / freq_c
    Zusätzlich Clamp, damit extreme Seltenheit nicht explodiert.
    """
    cnt = Counter()
    for _, tgt in dataset:
        cnt.update(tgt.flatten().tolist())
    counts = torch.tensor([cnt[i] for i in range(NUM_CLASSES)],
                          dtype=torch.float32)
    freq = counts / counts.sum()
    median = torch.median(freq[freq > 0])
    w = median / (freq + 1e-6)
    w = torch.clamp(w, max=cap)           # Obergrenze
    w = w / w.sum() * NUM_CLASSES         # normiert (optional)
    return w.to(DEVICE)

# ---------------------------------------------------------------------------
# 6  Training
# ---------------------------------------------------------------------------
def train(data_root: str | Path,
          epochs: int = 80,
          batch_size: int = 64,
          lr: float = 3e-4,
          latent_dim: int = 128,
          beta_max: float = .5,     ### NEW/CHANGED
          beta_warmup: int = 30,
          val_split: float = .10,
          out_dir: str | Path = "checkpoints",
          weighted: bool = True):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    full_ds   = MutationDataset(data_root)
    val_len   = int(len(full_ds)*val_split)
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(
        full_ds, [train_len, val_len],
        generator=torch.Generator().manual_seed(SEED)
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size)

    class_weights = compute_class_weights(train_ds) if weighted else None
    if weighted:
        print("➜ Median-Freq-Gewichte aktiviert:", class_weights.cpu().tolist())

    model = ConvVAE(latent_dim=latent_dim).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3     # kein 'verbose'
    )

    best_val = math.inf
    for ep in range(1, epochs+1):
        t0 = time.time(); model.train()
        beta = beta_max * min(1.0, ep / beta_warmup)   # sanfter & kleiner

        tr_loss = tr_rec = tr_kld = 0.0
        prog = tqdm(train_dl, desc=f"Epoch {ep}/{epochs}", unit="batch")
        for x, y in prog:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits, mu, logvar = model(x)
            loss, rec, kld = beta_vae_loss(
                logits, y, mu, logvar,
                beta=beta, class_weights=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            bsz = x.size(0)
            tr_loss += loss.item()*bsz
            tr_rec  += rec.item()*bsz
            tr_kld  += kld.item()*bsz
            prog.set_postfix(loss=f"{tr_loss/((prog.n+1)*batch_size):.4f}",
                             rec =f"{tr_rec /((prog.n+1)*batch_size):.4f}",
                             kld =f"{tr_kld /((prog.n+1)*batch_size):.4f}",
                             beta=f"{beta:.2f}")

        # ---------- Validation ----------
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits, mu, logvar = model(x)
                vloss, *_ = beta_vae_loss(
                    logits, y, mu, logvar,
                    beta=1.0, class_weights=class_weights)
                val_loss += vloss.item()*x.size(0)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        print(f"\nEp {ep:03d} train {tr_loss/len(train_ds):.4f} "
              f"(rec {tr_rec/len(train_ds):.4f}  kld {tr_kld/len(train_ds):.4f}) | "
              f"val {val_loss:.4f} | {time.time()-t0:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = out_dir/"best.pt"
            torch.save(dict(model_state=model.state_dict(),
                            epoch=ep, val_loss=val_loss,
                            args=dict(latent_dim=latent_dim)), ckpt)
            print(f"   ✅  Neues bestes Modell → {ckpt}")

# ---------------------------------------------------------------------------
# 7  Sampling  (unverändert)
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_random_rooms(ckpt_path: str|Path, n_samples=5, out_folder="samples"):
    out = Path(out_folder); out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    latent_dim = ckpt["args"]["latent_dim"]
    model = ConvVAE(latent_dim=latent_dim).to(DEVICE)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    tboi = TBoI_Bitmap(width=IMG_W, height=IMG_H)
    z = torch.randn(n_samples, latent_dim, device=DEVICE)
    preds = torch.argmax(model.decode(z), dim=1).cpu().numpy()
    for idx, arr in enumerate(preds):
        img = TBoI_Bitmap(width=IMG_W, height=IMG_H)
        for x in range(IMG_W):
            for y in range(IMG_H):
                ent = EntityType(int(arr[y, x]))
                px  = tboi.get_pixel_value_with_entity_id(ent)
                img.bitmap.putpixel((x, y), px)
        img.save_bitmap_in_folder(f"sample_{idx}", out)
        print(f"Sample {idx} gespeichert.")

# ---------------------------------------------------------------------------
# 8  CLI (kurz)
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser("β-VAE für TBoI-Bitmaps")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--data", default="Bitmaps")
    t.add_argument("--epochs", type=int, default=80)
    t.add_argument("--bs", type=int, default=64)
    t.add_argument("--out", default="checkpoints")
    s = sub.add_parser("sample"); s.add_argument("ckpt")
    s.add_argument("--n", type=int, default=5); s.add_argument("--out", default="samples")
    a = p.parse_args()
    if a.cmd=="train":
        train(a.data, epochs=a.epochs, batch_size=a.bs, out_dir=a.out)
    else:
        sample_random_rooms(a.ckpt, n_samples=a.n, out_folder=a.out)

# ---------------------------------------------------------------------------
# 9  Stub, falls tboi_bitmap fehlt (unverändert)
# ---------------------------------------------------------------------------
try:
    from tboi_bitmap import TBoI_Bitmap, EntityType
except ImportError:                                     # pragma: no cover
    class EntityType(int): pass
    class TBoI_Bitmap:
        def __init__(self, width=13, height=7):
            self.width, self.height = width, height
            self.bitmap = Image.new("L", (width, height), 0)
        def get_entity_id_with_pixel_value(self, px): return EntityType(px % NUM_CLASSES)
        def get_pixel_value_with_entity_id(self, ent): return int(ent)
        def save_bitmap_in_folder(self, name, folder):
            folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
            self.bitmap.save(folder / f"{name}.bmp")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
