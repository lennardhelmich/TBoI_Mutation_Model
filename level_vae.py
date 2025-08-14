#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
conv_vae_onehot.py – β-VAE (One-Hot Input) für Binding-of-Isaac Mutationen
Training: Inverted_Mutations (Input, One-Hot)  →  Mutations (Target, Klassen-IDs)
Speichert Outputs als 15×9-BMP (Innen 13×7 + 1-px Rand)

Features
- One-Hot Eingabe (C=12, H=7, W=13), Ziel = Klassen-IDs (H×W)
- β-Annealing mit fester Warmup-Ep.-Zahl (beta_warmup) ODER fraktionsbasiert (beta_warmup_frac)
- AdamW, Grad-Clipping, optional Median-Freq-Gewichtung
- Stabiler Decoder (Transposed-Conv) + Center-Crop auf 7×13
- Sampling (random z), Transform (einzelne BMP), Interpolation (latent space)
- Optuna Hyperparameter-Suche (resumierbar), Pruning über val_loss,
  finales Training mit besten Parametern (TensorBoard)
"""

import argparse, math, random, time, os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# Optuna
try:
    import optuna
except Exception:
    optuna = None

# ------------------------------- Konstanten ----------------------------------
NUM_CLASSES = 12
IMG_H, IMG_W = 7, 13              # Innenraster (nach Border-Crop)
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------- TBoI-Helper / Stub ------------------------------
try:
    from tboi_bitmap import TBoI_Bitmap, EntityType
except ImportError:  # Fallback-Stubs (Pixelwert = Entity-ID % 12)
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


# ---------------------------- Utility: Border --------------------------------
def remove_border(arr: np.ndarray) -> np.ndarray:
    """(9×15) → (7×13); (7×13) bleibt unverändert."""
    if arr.shape == (9, 15):
        return arr[1:-1, 1:-1]
    return arr

def add_border(arr_hw: np.ndarray) -> np.ndarray:
    """(7×13) → (9×15), Rand=0 (WALL)."""
    out = np.zeros((arr_hw.shape[0] + 2, arr_hw.shape[1] + 2), dtype=np.uint8)
    out[1:-1, 1:-1] = arr_hw
    return out


# -------------------------------- Dataset ------------------------------------
class MutationDataset(Dataset):
    """
    Lädt Paare: Inverted_Mutations/<bitmap>/<mutation>/*.bmp  →  Mutations/<bitmap>/mutation_*.bmp

    __getitem__:
      x_onehot : (12, 7, 13) float  – One-Hot der invertierten Mutation
      y_ids    : (7, 13) long       – Klassen-IDs der originalen Mutation
    """
    def __init__(self, root_dir: str | Path, seed: int = 42):
        self.root = Path(root_dir)
        self.tboi = TBoI_Bitmap()
        self.rng = random.Random(seed)
        self._p2eid = np.vectorize(
            lambda px: self.tboi.get_entity_id_with_pixel_value(px).value,
            otypes=[np.uint8],
        )
        self.pairs: list[tuple[Path, Path]] = []
        self._gather_pairs()

    def _load_ids(self, path: Path) -> np.ndarray:
        arr = np.asarray(Image.open(path).convert("L"))
        arr = remove_border(arr)  # (7,13)
        return self._p2eid(arr)   # (7,13) uint8 [0..11]

    def _gather_pairs(self):
        mutations_dir = self.root / "Mutations2"
        inverted_dir  = self.root / "Inverted_Mutations2"

        # Index der invertierten (schnell & robust)
        inv_index: dict[str, Path] = {}
        for bitmap_folder in inverted_dir.glob("bitmap_*"):
            if not bitmap_folder.is_dir(): continue
            for mut_dir in bitmap_folder.glob("mutation_*"):
                if not mut_dir.is_dir(): continue
                bmp_files = list(mut_dir.glob("*.bmp"))
                if bmp_files:
                    inv_index[f"{bitmap_folder.name}/{mut_dir.name}"] = sorted(bmp_files)[0]

        # Paare bilden
        for bitmap_folder in sorted(mutations_dir.glob("bitmap_*")):
            if not bitmap_folder.is_dir(): continue
            for mut_bmp in sorted(bitmap_folder.glob("mutation_*.bmp")):
                key = f"{bitmap_folder.name}/{mut_bmp.stem}"
                if key in inv_index:
                    self.pairs.append((inv_index[key], mut_bmp))  # (inverted, mutation)

        if not self.pairs:
            raise RuntimeError("Keine Paare gefunden (Inverted_Mutations → Mutations).")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx: int):
        p_inv, p_mut = self.pairs[idx]
        x_ids = self._load_ids(p_inv)       # (7,13)
        y_ids = self._load_ids(p_mut)       # (7,13)
        # one_hot benötigt LongTensor
        x_oh  = F.one_hot(torch.from_numpy(x_ids).long(), NUM_CLASSES) \
                  .permute(2,0,1).float()   # (12,7,13)
        return x_oh, torch.from_numpy(y_ids).long()


# --------------------------------- Modell ------------------------------------
class ConvVAE(nn.Module):
    """
    Encoder:  in  (B, 12, 7, 13)
              -> (B, 32, 7, 13) -> (B, 64, 4, 7) -> (B,128, 2, 4)
    Latent:   μ, logσ² ∈ R^latent_dim
    Decoder:  z → (B,128,2,4) → (B, 64,4,8) → (B,32,8,16) → (B,12,?,?)
              Center-Crop → (B,12,7,13)
    """
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(NUM_CLASSES, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),   # 7×13 → 4×7
            nn.Conv2d(64,128, 3, stride=2, padding=1), nn.ReLU(inplace=True),   # 4×7 → 2×4
        )
        self.enc_out = 128 * 2 * 4
        self.fc_mu     = nn.Linear(self.enc_out, latent_dim)
        # logvar = log(σ^2)
        self.fc_logvar = nn.Linear(self.enc_out, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, self.enc_out)
        self.dec = nn.Sequential(
            nn.Unflatten(1, (128, 2, 4)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(inplace=True),  # 4×8
            nn.ConvTranspose2d( 64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(inplace=True),  # 8×16
            nn.Conv2d(32, NUM_CLASSES, 1)  # Logits
        )

    @staticmethod
    def _center_crop(t, hw=(IMG_H, IMG_W)):
        h, w = t.shape[-2:]
        th, tw = hw
        i0 = (h - th) // 2
        j0 = (w - tw) // 2
        return t[..., i0:i0+th, j0:j0+tw]

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.exp(0.5*logvar) * torch.randn_like(logvar)

    def decode(self, z):
        return self._center_crop(self.dec(self.fc_dec(z)))  # (B,12,7,13)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# --------------------------------- Verluste ----------------------------------
def beta_vae_loss(logits, targets, mu, logvar, *, beta=1.0, class_weights=None):
    """
    logits:  (B, C, H, W)
    targets: (B, H, W)  Klassen-IDs
    """
    recon = F.cross_entropy(logits, targets, weight=class_weights, reduction="mean")
    kld   = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kld, recon, kld


# -------- Median-Frequency-Klassengewichte (optional, robust bei Unbalance) ---
def compute_class_weights(dataset, cap=5.0) -> torch.Tensor:
    cnt = Counter()
    for _, tgt in dataset:
        cnt.update(tgt.flatten().tolist())
    counts = torch.tensor([cnt[i] for i in range(NUM_CLASSES)], dtype=torch.float32)
    freq = counts / counts.sum().clamp_min(1.0)
    median = torch.median(freq[freq > 0])
    w = median / (freq + 1e-6)
    w = torch.clamp(w, max=cap)
    w = w / w.sum() * NUM_CLASSES
    return w.to(DEVICE)


# --------------------------------- Training ----------------------------------
def train(data_root: str | Path,
          epochs: int = 80,
          batch_size: int = 64,
          lr: float = 3e-4,
          latent_dim: int = 128,
          beta_max: float = 0.05,
          beta_warmup: int = 60,
          beta_warmup_frac: float | None = None,
          weighted: bool = False,
          out_dir: str = "checkpoints",
          val_split: float = 0.2):

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "tensorboard"; log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    print("📥 Lade Dataset…")
    full_ds = MutationDataset(data_root)
    val_len = int(len(full_ds) * val_split)
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len],
                                    generator=torch.Generator().manual_seed(SEED))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size)
    print(f"✅ Samples: train={len(train_ds)}, val={len(val_ds)}")

    class_weights = compute_class_weights(train_ds) if weighted else None
    if weighted:
        print("⚖️  Klassen-Gewichte aktiviert:", class_weights.detach().cpu().tolist())

    model = ConvVAE(latent_dim=latent_dim).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    # Warmup-Epochen aus Fraktion ableiten (falls gesetzt)
    warmup_epochs = int(math.ceil(beta_warmup_frac * epochs)) if beta_warmup_frac is not None else beta_warmup
    warmup_epochs = max(1, warmup_epochs)

    best_val = math.inf
    for ep in range(1, epochs+1):
        t0 = time.time(); model.train()
        beta = beta_max * min(1.0, ep / float(warmup_epochs))

        tr_loss = tr_rec = tr_kld = 0.0
        seen = 0
        prog = tqdm(train_dl, desc=f"Epoch {ep}/{epochs}", unit="batch")
        for x, y in prog:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits, mu, logvar = model(x)
            loss, rec, kld = beta_vae_loss(logits, y, mu, logvar, beta=beta, class_weights=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            bsz = x.size(0); seen += bsz
            tr_loss += loss.item() * bsz
            tr_rec  += rec.item()  * bsz
            tr_kld  += kld.item()  * bsz
            prog.set_postfix(loss=f"{tr_loss/seen:.4f}", rec=f"{tr_rec/seen:.4f}",
                             kld=f"{tr_kld/seen:.4f}", beta=f"{beta:.3f}")

        # ---------- Validation ----------
        model.eval(); val_loss = 0.0; vseen = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits, mu, logvar = model(x)
                vloss, _, _ = beta_vae_loss(logits, y, mu, logvar,
                                            beta=1.0, class_weights=class_weights)
                val_loss += vloss.item() * x.size(0); vseen += x.size(0)
        val_loss /= max(1, vseen)
        scheduler.step(val_loss)

        # TensorBoard
        writer.add_scalar("Loss/train_total", tr_loss/seen, ep)
        writer.add_scalar("Loss/train_recon", tr_rec/seen,  ep)
        writer.add_scalar("Loss/train_kld",   tr_kld/seen,  ep)
        writer.add_scalar("Loss/val_total",   val_loss,     ep)
        writer.add_scalar("Hyper/beta",       beta,         ep)
        writer.add_scalar("Hyper/beta_warmup_epochs", warmup_epochs, ep)
        writer.add_scalar("Hyper/lr",         opt.param_groups[0]['lr'], ep)

        print(f"\nEp {ep:03d}  train {tr_loss/seen:.4f} (rec {tr_rec/seen:.4f}  kld {tr_kld/seen:.4f})"
              f" | val {val_loss:.4f} | β {beta:.3f} (warmup {warmup_epochs} ep)"
              f" | {time.time()-t0:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = out_dir / "best.pt"
            torch.save(dict(model_state=model.state_dict(),
                            epoch=ep, val_loss=val_loss,
                            args=dict(latent_dim=latent_dim)), ckpt)
            print(f"   ✅ Neues bestes Modell gespeichert → {ckpt}")
    writer.close()


# ------------------------------- I/O Utilities -------------------------------
def ids_to_bmp_with_border(y_ids: np.ndarray) -> Image.Image:
    """(7,13) Klassen-IDs → BMP (9,15) mit Rand=0."""
    tboi = TBoI_Bitmap(width=15, height=9)
    bmp = Image.new("L", (15, 9), 0)
    for x in range(13):
        for y in range(7):
            ent = EntityType(int(y_ids[y, x]))
            px  = tboi.get_pixel_value_with_entity_id(ent)
            bmp.putpixel((x+1, y+1), px)  # innen
    return bmp


# ---------------------------- Inferenz / Tools -------------------------------
@torch.no_grad()
def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = ConvVAE(latent_dim=ckpt["args"]["latent_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def bitmap_to_onehot(path: str | Path) -> torch.Tensor:
    """BMP → (1,12,7,13) float (One-Hot)."""
    tboi = TBoI_Bitmap()
    arr = np.asarray(Image.open(path).convert("L"))
    arr = remove_border(arr)
    p2eid = np.vectorize(lambda px: tboi.get_entity_id_with_pixel_value(px).value, otypes=[np.uint8])
    ids = p2eid(arr)
    x = F.one_hot(torch.from_numpy(ids).long(), NUM_CLASSES).permute(2,0,1).unsqueeze(0).float()
    return x  # (1,12,7,13)

@torch.no_grad()
def sample_random_rooms(ckpt_path: str | Path, n_samples=5, out_folder="samples"):
    out = Path(out_folder); out.mkdir(parents=True, exist_ok=True)
    model = load_model(ckpt_path)
    z = torch.randn(n_samples, model.fc_mu.out_features, device=DEVICE)
    preds = torch.argmax(model.decode(z), dim=1).cpu().numpy()  # (N,7,13)
    for i, arr in enumerate(preds):
        img = ids_to_bmp_with_border(arr)
        img.save(out / f"sample_{i}.bmp")
        print(f"💾 sample_{i}.bmp gespeichert.")

@torch.no_grad()
def transform_bitmap(ckpt_path: str | Path, input_bitmap_path: str | Path,
                     output_name: str = "transformed", out_folder="samples"):
    out = Path(out_folder); out.mkdir(parents=True, exist_ok=True)
    model = load_model(ckpt_path)
    x = bitmap_to_onehot(input_bitmap_path).to(DEVICE)
    mu, logvar = model.encode(x)
    z = model.reparameterize(mu, logvar)
    pred = torch.argmax(model.decode(z), dim=1)[0].cpu().numpy()
    img = ids_to_bmp_with_border(pred)
    img.save(out / f"{output_name}.bmp")
    print(f"💾 {out / f'{output_name}.bmp'} gespeichert.")

@torch.no_grad()
def transform_folder(ckpt_path: str | Path, input_folder: str | Path,
                     out_folder="samples", max_files: int | None = None):
    out = Path(out_folder); out.mkdir(parents=True, exist_ok=True)
    model = load_model(ckpt_path)
    paths = sorted([p for p in Path(input_folder).glob("*.bmp")])
    if max_files is not None:
        paths = paths[:max_files]
    for p in paths:
        x = bitmap_to_onehot(p).to(DEVICE)
        mu, logvar = model.encode(x)
        z = model.reparameterize(mu, logvar)
        pred = torch.argmax(model.decode(z), dim=1)[0].cpu().numpy()
        img = ids_to_bmp_with_border(pred)
        img.save(out / f"{p.stem}_vae.bmp")
    print(f"✅ {len(paths)} Dateien transformiert → {out}")

@torch.no_grad()
def encode_bitmap_to_latent(model: ConvVAE, bitmap_path: str | Path) -> torch.Tensor:
    """BMP → μ (1, latent_dim) – deterministische Repräsentation."""
    x = bitmap_to_onehot(bitmap_path).to(DEVICE)
    mu, _ = model.encode(x)
    return mu

@torch.no_grad()
def interpolate_between_bitmaps(ckpt_path: str | Path, bitmap1_path: str | Path,
                                bitmap2_path: str | Path, steps=5, out_folder="interpolations"):
    model = load_model(ckpt_path)
    z1 = encode_bitmap_to_latent(model, bitmap1_path)
    z2 = encode_bitmap_to_latent(model, bitmap2_path)
    out = Path(out_folder); out.mkdir(parents=True, exist_ok=True)

    for i, a in enumerate(torch.linspace(0, 1, steps)):
        z = (1.0 - a) * z1 + a * z2
        pred = torch.argmax(model.decode(z), dim=1)[0].cpu().numpy()
        img = ids_to_bmp_with_border(pred)
        img.save(out / f"interp_{i:02d}_a{float(a):.2f}.bmp")
        print(f"💾 interp_{i:02d}_a{float(a):.2f}.bmp")

    print(f"✅ Interpolation abgeschlossen → {out}")


# ------------------------------ Optuna Helpers --------------------------------
def _objective_vae(trial,
                   data_root: str,
                   val_pct: float,
                   trial_epochs: int,
                   batch_sizes: list[int],
                   lr_min: float, lr_max: float,
                   latent_choices: list[int],
                   beta_max_min: float, beta_max_max: float,
                   warmup_frac_min: float, warmup_frac_max: float,
                   weighted: bool):

    # Vorschlag der Hyperparameter
    lr          = trial.suggest_float("lr", lr_min, lr_max, log=True)
    latent_dim  = trial.suggest_categorical("latent_dim", latent_choices)
    batch_size  = trial.suggest_categorical("batch_size", batch_sizes)
    beta_max    = trial.suggest_float("beta_max", beta_max_min, beta_max_max)
    beta_warmup_frac = trial.suggest_float("beta_warmup_frac", warmup_frac_min, warmup_frac_max)
    warmup_epochs = max(1, int(math.ceil(beta_warmup_frac * trial_epochs)))

    # Seed
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    # Dataset & Split
    full_ds = MutationDataset(data_root)
    val_len = int(len(full_ds) * val_pct)
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(
        full_ds, [train_len, val_len],
        generator=torch.Generator().manual_seed(SEED)
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size)

    class_weights = compute_class_weights(train_ds) if weighted else None

    # Modell
    model = ConvVAE(latent_dim=latent_dim).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val = float("inf")
    for ep in range(1, trial_epochs+1):
        model.train()
        beta = beta_max * min(1.0, ep / float(warmup_epochs))

        # --- Training ---
        tr_loss = tr_rec = tr_kld = 0.0; seen = 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits, mu, logvar = model(x)
            loss, rec, kld = beta_vae_loss(logits, y, mu, logvar, beta=beta, class_weights=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            b = x.size(0); seen += b
            tr_loss += loss.item()*b; tr_rec += rec.item()*b; tr_kld += kld.item()*b

        # --- Validation ---
        model.eval(); val_loss = 0.0; vseen = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits, mu, logvar = model(x)
                vloss, _, _ = beta_vae_loss(logits, y, mu, logvar, beta=1.0, class_weights=class_weights)
                val_loss += vloss.item()*x.size(0); vseen += x.size(0)
        val_loss /= max(1, vseen)

        # Reporting & Pruning
        trial.report(val_loss, ep)
        if val_loss < best_val:
            best_val = val_loss
        if trial.should_prune():
            raise optuna.TrialPruned()

        if not math.isfinite(val_loss) or val_loss > 10.0:
            break

        if ep % 5 == 0 or ep == 1:
            print(f"[Trial {trial.number}] Ep {ep}/{trial_epochs} | "
                  f"train {tr_loss/seen:.4f} | val {val_loss:.4f} | "
                  f"β {beta:.3f} | warmup {warmup_epochs} ep (~{beta_warmup_frac:.2f})"
                  f" | bs {batch_size} | lr {lr:.2e}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val


def run_optuna_study(data_root: str,
                     trials: int = 40,
                     n_jobs: int = 1,
                     val_pct: float = 0.2,
                     trial_epochs: int = 30,
                     batch_sizes: list[int] = [16, 32, 64, 128],
                     lr_min: float = 1e-5, lr_max: float = 1e-2,
                     latent_choices: list[int] = [32, 64, 128, 256],
                     beta_max_min: float = 0.01, beta_max_max: float = 0.2,
                     warmup_frac_min: float = 0.2, warmup_frac_max: float = 1.0,
                     storage: str = "sqlite:///optuna_vae_2.db",
                     study_name: str = "vae_onehot_optimization",
                     weighted: bool = False):
    if optuna is None:
        raise RuntimeError("Optuna ist nicht installiert. Bitte mit `pip install optuna` nachinstallieren.")

    print("\n================ Optuna Study ================")
    print(f"Trials: {trials} | Jobs: {n_jobs} | Device: {DEVICE}")
    print(f"Validation Anteil (val_pct): {val_pct}")
    print("==============================================\n")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
    )

    def _objective(trial):
        return _objective_vae(
            trial=trial,
            data_root=data_root,
            val_pct=val_pct,
            trial_epochs=trial_epochs,
            batch_sizes=batch_sizes,
            lr_min=lr_min, lr_max=lr_max,
            latent_choices=latent_choices,
            beta_max_min=beta_max_min, beta_max_max=beta_max_max,
            warmup_frac_min=warmup_frac_min, warmup_frac_max=warmup_frac_max,
            weighted=weighted
        )

    try:
        study.optimize(_objective, n_trials=trials, n_jobs=n_jobs, gc_after_trial=True)
    except KeyboardInterrupt:
        print("⏹️  Studie vom Benutzer abgebrochen – aktueller Stand wird gespeichert.")

    print("\n================ Ergebnisse ==================")
    if len(study.trials) > 0 and study.best_trial is not None:
        print(f"Beste Val-Loss: {study.best_value:.6f}")
        print(f"Beste Parameter: {study.best_trial.params}")
    else:
        print("Keine abgeschlossenen Trials vorhanden.")
    print("==============================================\n")

    return study


# ------------------------------------ CLI ------------------------------------
def main():
    p = argparse.ArgumentParser("β-VAE (One-Hot Input) für TBoI-Bitmaps")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data",      default="Bitmaps")
    t.add_argument("--epochs",    type=int, default=80)
    t.add_argument("--bs",        type=int, default=64)
    t.add_argument("--lr",        type=float, default=3e-4)
    t.add_argument("--latent_dim",type=int, default=128)
    t.add_argument("--beta_max",  type=float, default=0.05)
    t.add_argument("--beta_warmup", type=int, default=60, help="Warmup-Epochen bis β_max (ignoriert, wenn --beta_warmup_frac gesetzt ist)")
    t.add_argument("--beta_warmup_frac", type=float, help="Fraktion der Gesamt-Epochen bis β_max, z. B. 0.5 ⇒ Hälfte der Epochen")
    t.add_argument("--weighted",  action="store_true")
    t.add_argument("--out",       default="./checkpoints")
    t.add_argument("--val_split", type=float, default=0.2)

    s = sub.add_parser("sample")
    s.add_argument("ckpt")
    s.add_argument("--n", type=int, default=5)
    s.add_argument("--out", default="samples")

    tr = sub.add_parser("transform")
    tr.add_argument("ckpt")
    tr.add_argument("--input", required=True)
    tr.add_argument("--output", default="transformed")
    tr.add_argument("--out_folder", default="samples")

    trf = sub.add_parser("transform_folder")
    trf.add_argument("ckpt")
    trf.add_argument("--input", required=True)
    trf.add_argument("--out_folder", default="samples")
    trf.add_argument("--max_files", type=int)

    interp = sub.add_parser("interpolate")
    interp.add_argument("ckpt")
    interp.add_argument("--input1", required=True)
    interp.add_argument("--input2", required=True)
    interp.add_argument("--steps", type=int, default=5)
    interp.add_argument("--out", default="interpolations")

    # --------- Optuna ---------
    o = sub.add_parser("optuna", help="Starte/resumiere Optuna-Studie und trainiere am Ende bestes Modell")
    o.add_argument("--data", default="Bitmaps")
    o.add_argument("--trials", type=int, default=100)
    o.add_argument("--jobs", type=int, default=1, help="Parallelität (GPU → 1)")
    o.add_argument("--val_pct", type=float, default=0.2, help="Validierungsanteil für die Trials")
    o.add_argument("--trial_epochs", type=int, default=20, help="Epochen je Trial")
    o.add_argument("--batch_sizes", type=int, nargs="+", default=[32, 64, 128])
    o.add_argument("--lr_min", type=float, default=1e-5)
    o.add_argument("--lr_max", type=float, default=1e-2)
    o.add_argument("--latent_choices", type=int, nargs="+", default=[32, 64, 128, 256])
    o.add_argument("--beta_max_min", type=float, default=0.01)
    o.add_argument("--beta_max_max", type=float, default=0.2)
    o.add_argument("--warmup_frac_min", type=float, default=0.2, help="Untergrenze für beta_warmup_frac (0..1)")
    o.add_argument("--warmup_frac_max", type=float, default=1.0, help="Obergrenze für beta_warmup_frac (0..1)")
    o.add_argument("--study_name", type=str, default="vae_onehot_optimization_2")
    o.add_argument("--storage", type=str, default="sqlite:///optuna_vae_2.db")
    o.add_argument("--weighted", action="store_true", help="Klassengewichte in Trials nutzen")
    o.add_argument("--final_epochs", type=int, default=100, help="Finales Training nach der Studie")
    o.add_argument("--out", type=str, default="./best_optuna_vae_model_2", help="Output-Ordner für finales Modell")

    a = p.parse_args()

    if a.cmd == "train":
        train(a.data, epochs=a.epochs, batch_size=a.bs, lr=a.lr,
              latent_dim=a.latent_dim, beta_max=a.beta_max,
              beta_warmup=a.beta_warmup, beta_warmup_frac=a.beta_warmup_frac,
              weighted=a.weighted, out_dir=a.out, val_split=a.val_split)

    elif a.cmd == "sample":
        sample_random_rooms(a.ckpt, n_samples=a.n, out_folder=a.out)

    elif a.cmd == "transform":
        transform_bitmap(a.ckpt, a.input, a.output, a.out_folder)

    elif a.cmd == "transform_folder":
        transform_folder(a.ckpt, a.input, a.out_folder, a.max_files)

    elif a.cmd == "interpolate":
        interpolate_between_bitmaps(a.ckpt, a.input1, a.input2, a.steps, a.out)

    elif a.cmd == "optuna":
        study = run_optuna_study(
            data_root=a.data,
            trials=a.trials,
            n_jobs=a.jobs,
            val_pct=a.val_pct,
            trial_epochs=a.trial_epochs,
            batch_sizes=a.batch_sizes,
            lr_min=a.lr_min, lr_max=a.lr_max,
            latent_choices=a.latent_choices,
            beta_max_min=a.beta_max_min, beta_max_max=a.beta_max_max,
            warmup_frac_min=a.warmup_frac_min, warmup_frac_max=a.warmup_frac_max,
            storage=a.storage,
            study_name=a.study_name,
            weighted=a.weighted
        )

        # Finales Training mit den besten Parametern
        if len(study.trials) > 0 and study.best_trial is not None:
            params = study.best_trial.params
            print("\n🚀 Starte finales Training mit besten Parametern:")
            print(params)

            # Backwards-compat: Falls alte Studie beta_warmup (int) statt beta_warmup_frac enthält
            beta_warmup_frac = params.get("beta_warmup_frac", None)
            beta_warmup_val = params.get("beta_warmup", None)  # nur falls vorhanden

            train(
                data_root=a.data,
                epochs=a.final_epochs,
                batch_size=params["batch_size"],
                lr=params["lr"],
                latent_dim=params["latent_dim"],
                beta_max=params["beta_max"],
                beta_warmup=(beta_warmup_val if (beta_warmup_frac is None and beta_warmup_val is not None) else 1),
                beta_warmup_frac=beta_warmup_frac,  # falls gesetzt, hat Vorrang
                weighted=a.weighted,
                out_dir=a.out,
                val_split=a.val_pct  # konsistent zur Studie
            )
        else:
            print("⚠️  Kein bestes Trial verfügbar – finales Training übersprungen.")


if __name__ == "__main__":
    main()
