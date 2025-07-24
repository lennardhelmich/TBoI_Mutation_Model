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

import optuna

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
        mutations_dir = self.root / "Mutations"
        inverted_dir = self.root / "Inverted_Mutations"
        
        # Erstelle einen Index aller verfügbaren Inverted_Mutations einmal
        print("Erstelle Index der Inverted_Mutations...")
        inverted_index = {}
        for bitmap_folder in inverted_dir.glob("bitmap_*"):
            if not bitmap_folder.is_dir():
                continue
            for mutation_folder in bitmap_folder.glob("mutation_*"):
                if not mutation_folder.is_dir():
                    continue
                # Finde die erste .bmp Datei in diesem Ordner
                bmp_files = list(mutation_folder.glob("*.bmp"))
                if bmp_files:
                    key = f"{bitmap_folder.name}/{mutation_folder.name}"
                    inverted_index[key] = sorted(bmp_files)[0]
        
        print(f"Index erstellt mit {len(inverted_index)} Inverted_Mutations")
        
        # Jetzt gehe durch alle Mutations und nutze den Index
        processed = 0
        for bitmap_folder in sorted(mutations_dir.glob("bitmap_*")):
            if not bitmap_folder.is_dir():
                continue
                
            print(f"Verarbeite Ordner: {bitmap_folder.name}")
            mutation_files = list(bitmap_folder.glob("mutation_*.bmp"))
            
            for mutation_file in sorted(mutation_files):
                # Nutze den Index statt glob
                key = f"{bitmap_folder.name}/{mutation_file.stem}"
                if key not in inverted_index:
                    continue
                    
                inverted_file = inverted_index[key]
                
                try:
                    # Lade beide Bitmaps
                    mutation_arr = self._load_bitmap(mutation_file)
                    inverted_arr = self._load_bitmap(inverted_file)
                    
                    # Füge das Paar hinzu
                    self.pairs.append((mutation_arr, inverted_arr))
                    processed += 1
                    
                    # Progress feedback
                    if processed % 1000 == 0:
                        print(f"  {processed} Paare verarbeitet...")
                        
                except Exception as e:
                    print(f"Fehler beim Laden von {mutation_file} oder {inverted_file}: {e}")
                    continue
        
        print(f"Fertig! {len(self.pairs)} Paare geladen.")

    def __len__(self): 
        return len(self.pairs)

    def __getitem__(self, idx: int):
        mutation_arr, inverted_arr = self.pairs[idx]
        return (
            torch.from_numpy(inverted_arr).unsqueeze(0).float(),  # Input: inverted_mutation
            torch.from_numpy(mutation_arr).long(),                # Target: mutation
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
          epochs: int = 120,
          batch_size: int = 32,
          lr: float = 1e-3,
          latent_dim: int = 64,
          beta_max: float = 0.05,
          beta_warmup: int = 60,
          weighted: bool = False,      # Fehlender Parameter
          out_dir: str = "checkpoints", # Fehlender Parameter
          val_split: float = 0.2):     # Falls nicht definiert
    
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    
    print("Start loading Dataset...")
    full_ds   = MutationDataset(data_root)
    print("Finished loading Dataset.")
    
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
# 6.5  Optuna Hyperparameter-Optimierung (Mit Logging)
# ---------------------------------------------------------------------------
def objective(trial):
    """Optuna Objective Function für Hyperparameter-Optimierung"""
    import os
    import time
    
    trial_start_time = time.time()
    
    # GPU-Status und Trial-Info
    print(f"\n{'='*60}")
    print(f"🚀 TRIAL {trial.number} GESTARTET")
    print(f"{'='*60}")
    print(f"🔥 DEVICE: {DEVICE}")
    if torch.cuda.is_available():
        print(f"💻 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"💾 VRAM Frei: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f} GB")
    
    # Hyperparameter-Vorschläge
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    latent_dim = trial.suggest_categorical('latent_dim', [32, 64, 128, 256])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    beta_max = trial.suggest_uniform('beta_max', 0.01, 0.2)
    beta_warmup = trial.suggest_int('beta_warmup', 20, 80)
    
    print(f"📋 HYPERPARAMETER:")
    print(f"   Learning Rate: {lr:.2e}")
    print(f"   Latent Dim:    {latent_dim}")
    print(f"   Batch Size:    {batch_size}")
    print(f"   Beta Max:      {beta_max:.3f}")
    print(f"   Beta Warmup:   {beta_warmup}")
    print(f"{'='*60}")
    
    # Dataset laden
    print("📁 Lade Dataset...")
    full_ds = MutationDataset("Bitmaps")
    val_len = int(len(full_ds) * 0.2)
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(
        full_ds, [train_len, val_len],
        generator=torch.Generator().manual_seed(SEED)
    )
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    print(f"✅ Dataset geladen: {len(train_ds)} Train, {len(val_ds)} Val Samples")
    
    # Modell mit Trial-Parametern
    print("🏗️  Erstelle Modell...")
    model = ConvVAE(latent_dim=latent_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Modell-Info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Modell Parameter: {total_params:,}")
    
    best_val_loss = float('inf')
    epochs = 30  # Weniger Epochen für Optuna
    
    print(f"🎯 Starte Training für {epochs} Epochen...")
    epoch_times = []
    
    for ep in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        beta = beta_max * min(1.0, ep / beta_warmup)
        
        # Training
        train_loss = 0.0
        train_batches = 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits, mu, logvar = model(x)
            loss, rec, kld = beta_vae_loss(logits, y, mu, logvar, beta=beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_loss += loss.item()
            train_batches += 1
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits, mu, logvar = model(x)
                vloss, _, _ = beta_vae_loss(logits, y, mu, logvar, beta=1.0)
                val_loss += vloss.item()
                val_batches += 1
        
        val_loss /= val_batches
        train_loss /= train_batches
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # Progress Logging
        if ep % 5 == 0 or ep == 1:
            avg_epoch_time = np.mean(epoch_times)
            remaining_time = avg_epoch_time * (epochs - ep)
            print(f"📊 Epoche {ep:2d}/{epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Beta: {beta:.3f} | Zeit: {epoch_time:.1f}s | "
                  f"ETA: {remaining_time/60:.1f}min")
        
        # Optuna Reporting
        trial.report(val_loss, ep)
        
        # Pruning (frühzeitiges Stoppen schlechter Trials)
        if trial.should_prune():
            print(f"✂️  Trial {trial.number} PRUNED nach Epoche {ep}")
            raise optuna.TrialPruned()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"🎉 Neue beste Val-Loss: {best_val_loss:.6f}")
        
        # Früh stoppen wenn Loss explodiert
        if val_loss > 10.0:
            print(f"💥 Trial gestoppt - Loss explodiert: {val_loss:.2f}")
            break
    
    trial_time = time.time() - trial_start_time
    
    # Trial-Zusammenfassung
    print(f"\n{'='*60}")
    print(f"📈 TRIAL {trial.number} ABGESCHLOSSEN")
    print(f"{'='*60}")
    print(f"⏱️  Gesamtzeit: {trial_time/60:.1f} Minuten")
    print(f"🎯 Beste Val-Loss: {best_val_loss:.6f}")
    print(f"📊 Finale Val-Loss: {val_loss:.6f}")
    print(f"⚡ Epochen pro Minute: {epochs/(trial_time/60):.1f}")
    
    # Speichere Trial-Ergebnisse
    os.makedirs("Optuna/VAE", exist_ok=True)
    trial_results = {
        'trial_number': trial.number,
        'params': trial.params,
        'best_val_loss': best_val_loss,
        'final_val_loss': val_loss,
        'trial_time_minutes': trial_time/60,
        'epochs_completed': epochs
    }
    
    np.save(f"Optuna/VAE/trial_{trial.number}_results.npy", trial_results)
    
    # GPU Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🧹 GPU Memory bereinigt")
    
    print(f"{'='*60}\n")
    
    return best_val_loss

def run_optuna_study(n_trials=50, n_jobs=1):
    """Starte Optuna Hyperparameter-Suche mit verbessertem Logging"""
    import os
    import time
    
    study_start_time = time.time()
    
    print(f"\n{'🔬'*20}")
    print(f"🚀 OPTUNA HYPERPARAMETER SEARCH")
    print(f"{'🔬'*20}")
    print(f"🎯 Trials geplant: {n_trials}")
    print(f"⚡ Parallele Jobs: {n_jobs}")
    print(f"🔥 Device: {DEVICE}")
    print(f"📅 Startzeit: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Erstelle oder lade Study
    study_name = "tboi_vae_optimization"
    storage = f"sqlite:///optuna_{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='minimize',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
    )
    
    # Zeige bisherige Trials (SAFE VERSION)
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) > 0:
        print(f"📋 Bestehende Study gefunden mit {len(study.trials)} Trials")
        print(f"✅ Erfolgreich abgeschlossen: {len(completed_trials)}")
        try:
            print(f"🏆 Beste bisherige Loss: {study.best_value:.6f}")
            print(f"📊 Beste Parameter: {study.best_params}")
        except ValueError:
            print(f"⚠️  Noch keine gültigen Ergebnisse in der Datenbank")
        print(f"{'='*60}")
    else:
        print(f"📋 Neue Study wird erstellt - keine vorherigen Trials gefunden")
        print(f"{'='*60}")
    
    # Fortschritts-Callback (SAFE VERSION)
    def progress_callback(study, trial):
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        
        print(f"\n📊 FORTSCHRITT UPDATE:")
        print(f"   ✅ Abgeschlossen: {completed}")
        print(f"   ✂️  Pruned: {pruned}")
        print(f"   ❌ Fehler: {failed}")
        
        # Safe best value access
        if completed > 0:
            try:
                print(f"   🏆 Aktuelle beste Loss: {study.best_value:.6f}")
            except ValueError:
                print(f"   🏆 Beste Loss: Noch keine gültigen Trials")
        else:
            print(f"   🏆 Beste Loss: Noch keine abgeschlossenen Trials")
        
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
    
    # Finale Ergebnisse (SAFE VERSION)
    completed_final = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    print(f"\n{'🎉'*20}")
    print(f"🏁 OPTUNA STUDIE ABGESCHLOSSEN")
    print(f"{'🎉'*20}")
    print(f"⏱️  Gesamtzeit: {study_time/3600:.1f} Stunden")
    print(f"🎯 Trials durchgeführt: {len(study.trials)}")
    print(f"✅ Erfolgreich: {len(completed_final)}")
    print(f"✂️  Gepruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"❌ Fehler: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    print(f"{'='*60}")
    
    # Safe best value access
    if len(completed_final) > 0:
        try:
            print(f"🏆 BESTE VALIDIERUNGSLOSS: {study.best_value:.6f}")
            print(f"📊 BESTE HYPERPARAMETER:")
            for key, value in study.best_params.items():
                print(f"   {key}: {value}")
        except ValueError:
            print(f"⚠️  Keine gültigen Trials für beste Parameter verfügbar")
            # Zeige stattdessen den besten verfügbaren Trial
            if completed_final:
                best_trial = min(completed_final, key=lambda t: t.value)
                print(f"🏆 BESTE VERFÜGBARE LOSS: {best_trial.value:.6f}")
                print(f"📊 PARAMETER:")
                for key, value in best_trial.params.items():
                    print(f"   {key}: {value}")
    else:
        print(f"⚠️  Keine erfolgreichen Trials abgeschlossen!")
    
    print(f"{'='*60}")
    
    # Speichere finale Ergebnisse
    os.makedirs("Optuna/VAE", exist_ok=True)
    with open("Optuna/VAE/study_results.txt", "w") as f:
        f.write(f"Optuna Study Results\n")
        f.write(f"==================\n")
        f.write(f"Study Time: {study_time/3600:.1f} hours\n")
        f.write(f"Total Trials: {len(study.trials)}\n")
        f.write(f"Completed Trials: {len(completed_final)}\n")
        if len(completed_final) > 0:
            try:
                f.write(f"Best Loss: {study.best_value:.6f}\n")
                f.write(f"Best Params: {study.best_params}\n")
            except ValueError:
                f.write(f"Best Loss: No valid trials\n")
    
    # Top 5 Trials anzeigen
    if len(completed_final) >= 5:
        print(f"🔝 TOP 5 TRIALS:")
        sorted_trials = sorted(completed_final, key=lambda t: t.value)
        for i, trial in enumerate(sorted_trials[:5]):
            print(f"   {i+1}. Trial {trial.number}: {trial.value:.6f}")
    
    return study

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

@torch.no_grad()
def transform_bitmap(ckpt_path: str|Path, input_bitmap_path: str|Path, 
                    output_name: str = "transformed", out_folder="samples"):
    """Transformiere eine einzelne Input-Bitmap mit dem trainierten VAE"""
    out = Path(out_folder)
    out.mkdir(parents=True, exist_ok=True)
    
    # Lade trainiertes Modell
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    latent_dim = ckpt["args"]["latent_dim"]
    model = ConvVAE(latent_dim=latent_dim).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    print(f"🔥 Modell geladen (Latent Dim: {latent_dim})")
    print(f"📁 Input: {input_bitmap_path}")
    
    # Lade Input-Bitmap (15×9)
    img = Image.open(input_bitmap_path).convert("L")
    print(f"📏 Original size: {img.size}")
    
    # Crop borders: 15×9 → 13×7 (entferne 1px Rand)
    arr = np.array(img)[1:-1, 1:-1]  # Remove 1px border on all sides
    print(f"📏 After crop: {arr.shape} (sollte 7×13 sein)")
    
    # TBoI Bitmap Helper für Pixel → Entity Conversion
    tboi = TBoI_Bitmap()
    
    # Convert Pixelwerte (0-255) zu Entity IDs (0-11)
    pmap = np.vectorize(
        lambda px: tboi.get_entity_id_with_pixel_value(px).value,
        otypes=[np.uint8]
    )
    entity_arr = pmap(arr)
    
    print(f"🔢 Entity range: {entity_arr.min()}-{entity_arr.max()}")
    print(f"🔢 Entity shape: {entity_arr.shape}")
    
    # Zu Tensor konvertieren (1, 1, 7, 13)
    x = torch.from_numpy(entity_arr).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    print(f"🔢 Input tensor shape: {x.shape}")
    
    # VAE Forward Pass: Encode → Sample → Decode
    mu, logvar = model.encode(x)
    z = model.reparameterize(mu, logvar)  # Sample aus gelernter Verteilung
    output_logits = model.decode(z)       # (1, 12, 7, 13)
    pred = torch.argmax(output_logits, dim=1)[0].cpu().numpy()  # (7, 13)
    
    print(f"📊 Latent stats: μ={mu.mean():.3f}±{mu.std():.3f}, logvar={logvar.mean():.3f}")
    print(f"🎯 Output shape: {pred.shape}")
    print(f"🎯 Output entity range: {pred.min()}-{pred.max()}")
    
    # Speichere transformierte Bitmap
    save_transformed_bitmap(pred, output_name, out)
    
    print(f"✅ Transformierte Bitmap gespeichert: {out / f'{output_name}.bmp'}")
    
    return pred

def save_transformed_bitmap(arr, name, out_folder):
    """Speichere 7×13 Entity Array als 15×9 TBoI-Bitmap (mit Rand)"""
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    
    # Erstelle 15×9 Bitmap
    tboi = TBoI_Bitmap(width=15, height=9)
    
    # Fülle den Rand mit Entity 0 (EMPTY/WALL)
    for x in range(15):
        for y in range(9):
            if x == 0 or x == 14 or y == 0 or y == 8:
                # Rand: Setze Entity 0
                ent = EntityType(0)
            else:
                # Innenbereich: Nutze predicted entities (7×13 in 15×9)
                inner_y = y - 1  # 0-6
                inner_x = x - 1  # 0-12
                entity_id = int(arr[inner_y, inner_x])
                ent = EntityType(entity_id)
            
            # Convert Entity zu Pixelwert
            px = tboi.get_pixel_value_with_entity_id(ent)
            tboi.bitmap.putpixel((x, y), px)
    
    # Speichere als BMP
    output_path = out_folder / f"{name}.bmp"
    tboi.bitmap.save(output_path)
    print(f"💾 Bitmap gespeichert: {output_path}")

@torch.no_grad()
def transform_with_variations(ckpt_path, input_bitmap_path, n_variations=5, out_folder="variations"):
    """Erzeuge mehrere Variationen derselben Input-Bitmap"""
    # Lade Modell
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = ConvVAE(latent_dim=ckpt["args"]["latent_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    # Lade Input
    img = Image.open(input_bitmap_path).convert("L")
    arr = np.array(img)[1:-1, 1:-1]  # Crop
    
    # Convert zu Entity IDs
    tboi = TBoI_Bitmap()
    pmap = np.vectorize(lambda px: tboi.get_entity_id_with_pixel_value(px).value)
    entity_arr = pmap(arr)
    x = torch.from_numpy(entity_arr).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    
    # Encode zu μ und σ
    mu, logvar = model.encode(x)
    
    for i in range(n_variations):
        # Sample verschiedene z-Vektoren aus der gelernten Verteilung
        z = model.reparameterize(mu, logvar)  # Jedes Mal anders!
        
        # Decode
        output_logits = model.decode(z)
        pred = torch.argmax(output_logits, dim=1)[0].cpu().numpy()
        
        # Speichere
        save_transformed_bitmap(pred, f"variation_{i}", out_folder)
        print(f"✅ Variation {i} gespeichert")

@torch.no_grad()
def load_model(ckpt_path):
    """Lade ein trainiertes VAE-Modell"""
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    latent_dim = ckpt["args"]["latent_dim"]
    model = ConvVAE(latent_dim=latent_dim).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

@torch.no_grad()
def interpolate_between_bitmaps(ckpt_path, bitmap1_path, bitmap2_path, steps=5, out_folder="interpolations"):
    """Interpoliere zwischen zwei Bitmaps im latent space"""
    # Lade Modell (korrigiert)
    model = load_model(ckpt_path)
    
    print(f"🔀 Interpoliere zwischen {bitmap1_path} und {bitmap2_path}")
    
    # Encode beide Bitmaps zu latent vectors
    z1 = encode_bitmap_to_latent(model, bitmap1_path)  
    z2 = encode_bitmap_to_latent(model, bitmap2_path)  
    
    out = Path(out_folder)
    out.mkdir(parents=True, exist_ok=True)
    
    for i in range(steps):
        # Linear interpolation
        alpha = i / (steps - 1)
        z_interp = (1 - alpha) * z1 + alpha * z2
        
        print(f"🎯 Schritt {i}: α={alpha:.2f}")
        
        # Decode interpolierten Vektor
        output_logits = model.decode(z_interp)
        pred = torch.argmax(output_logits, dim=1)[0].cpu().numpy()
        
        save_transformed_bitmap(pred, f"interp_{i:02d}_alpha_{alpha:.2f}", out_folder)
    
    print(f"✅ Interpolation abgeschlossen! {steps} Schritte in {out_folder}")

def encode_bitmap_to_latent(model, bitmap_path):
    """Hilfsfunktion: Bitmap → Latent Vector (korrigiert)"""
    img = Image.open(bitmap_path).convert("L")
    arr = np.array(img)[1:-1, 1:-1]
    
    tboi = TBoI_Bitmap()
    pmap = np.vectorize(lambda px: tboi.get_entity_id_with_pixel_value(px).value)
    entity_arr = pmap(arr)
    x = torch.from_numpy(entity_arr).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    
    mu, logvar = model.encode(x)
    return mu  # Nutze μ für deterministische Interpolation
# ---------------------------------------------------------------------------
# 8  CLI (kurz)
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser("β-VAE für TBoI-Bitmaps")
    sub = p.add_subparsers(dest="cmd", required=True)
    
    # Train subparser mit allen benötigten Argumenten
    t = sub.add_parser("train")
    t.add_argument("--data", default="Bitmaps")
    t.add_argument("--epochs", type=int, default=80)
    t.add_argument("--bs", type=int, default=64)
    t.add_argument("--lr", type=float, default=3e-4)           # Neu hinzugefügt
    t.add_argument("--latent_dim", type=int, default=128)      # Neu hinzugefügt
    t.add_argument("--weighted", action="store_true")          # Neu hinzugefügt
    t.add_argument("--out", default="./checkpoints")
    
    # Sample subparser (unverändert)
    s = sub.add_parser("sample")
    s.add_argument("ckpt")
    s.add_argument("--n", type=int, default=5)
    s.add_argument("--out", default="samples")
    
    # Neuer Optuna subparser
    o = sub.add_parser("optuna")
    o.add_argument("--trials", type=int, default=50, help="Anzahl Optuna Trials")
    o.add_argument("--jobs", type=int, default=1, help="Parallele Jobs (1 für GPU)")
    
    # Neuer Transform subparser
    tr = sub.add_parser("transform")
    tr.add_argument("ckpt", help="Pfad zum trainierten Modell")
    tr.add_argument("--input", required=True, help="Input Bitmap oder Ordner")
    tr.add_argument("--output", default="transformed", help="Output Name/Ordner")
    tr.add_argument("--out_folder", default="samples", help="Output Verzeichnis")
    tr.add_argument("--max_files", type=int, help="Max. Anzahl Dateien (für Ordner)")
    
    # Neuer Variationen subparser
    var = sub.add_parser("variations")
    var.add_argument("ckpt")
    var.add_argument("--input", required=True)
    var.add_argument("--n", type=int, default=5)
    var.add_argument("--out", default="variations")

    # Neuer Interpolation subparser
    interp = sub.add_parser("interpolate") 
    interp.add_argument("ckpt")
    interp.add_argument("--input1", required=True)
    interp.add_argument("--input2", required=True)
    interp.add_argument("--steps", type=int, default=5)

    a = p.parse_args()
    
    if a.cmd == "train":
        train(a.data, epochs=a.epochs, batch_size=a.bs, lr=a.lr, 
              latent_dim=a.latent_dim, weighted=a.weighted, out_dir=a.out)
    elif a.cmd == "sample":
        sample_random_rooms(a.ckpt, n_samples=a.n, out_folder=a.out)
    elif a.cmd == "optuna":
        print(f"🚀 Starte Optuna mit {a.trials} Trials und {a.jobs} Jobs")
        study = run_optuna_study(n_trials=a.trials, n_jobs=a.jobs)
        
        # Optional: Trainiere bestes Modell (SAFE VERSION)
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed_trials) > 0:
            try:
                print(f"\n🎯 Trainiere bestes Modell mit optimalen Hyperparametern...")
                best_params = study.best_params
                train("Bitmaps", 
                      epochs=75, 
                      batch_size=best_params['batch_size'],
                      lr=best_params['lr'],
                      latent_dim=best_params['latent_dim'],
                      beta_max=best_params['beta_max'],
                      beta_warmup=best_params['beta_warmup'],
                      out_dir="./best_optuna_model")
            except ValueError:
                print(f"⚠️  Kann bestes Modell nicht trainieren - keine gültigen Parameter")
        else:
            print(f"⚠️  Keine erfolgreichen Trials gefunden!")
    elif a.cmd == "transform":
        input_path = Path(a.input)
        if input_path.is_file():
            # Einzelne Datei transformieren
            transform_bitmap(a.ckpt, a.input, a.output, a.out_folder)
        elif input_path.is_dir():
            # Ganzen Ordner transformieren  
            print(f"❌ Ordner-Transformation noch nicht implementiert: {a.input}")
            print(f"💡 Nutze stattdessen einzelne Dateien mit --input datei.bmp")
        else:
            print(f"❌ Input nicht gefunden: {a.input}")
    elif a.cmd == "variations":
        transform_with_variations(a.ckpt, a.input, a.n, a.out)
    elif a.cmd == "interpolate":
        interpolate_between_bitmaps(a.ckpt, a.input1, a.input2, a.steps)

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

