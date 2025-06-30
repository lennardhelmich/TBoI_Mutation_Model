import optuna
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tboi_bitmap import TBoI_Bitmap, EntityType

NUM_CLASSES = 12

class MutationDataset(Dataset):
    def __init__(self, data_folder):
        self.inputs = []
        self.mutated = []

        input_rooms_folder = os.path.join(os.path.dirname(data_folder), "InputRooms")
        input_files = sorted([f for f in os.listdir(input_rooms_folder) if f.endswith(".bmp")])

        tboi_bitmap = TBoI_Bitmap()

        for input_file in input_files:
            input_path = os.path.join(input_rooms_folder, input_file)
            input_img = Image.open(input_path).convert("L")
            input_arr = np.array(input_img)
            input_arr = input_arr[1:-1, 1:-1]

            for i in range(input_arr.shape[0]):
                for j in range(input_arr.shape[1]):
                    entity = tboi_bitmap.get_entity_id_with_pixel_value(input_arr[i, j])
                    input_arr[i, j] = entity.value

            input_arr = input_arr[np.newaxis, :, :]

            mutation_folder = os.path.join(data_folder, "Mutations", input_file.replace("bitmap_", "bitmap_").replace(".bmp", ""))
            if not os.path.isdir(mutation_folder):
                continue
            mutation_files = [f for f in os.listdir(mutation_folder) if f.endswith(".bmp")]
            for mfile in mutation_files:
                mpath = os.path.join(mutation_folder, mfile)
                mimg = Image.open(mpath).convert("L")
                marr = np.array(mimg)
                marr = marr[1:-1, 1:-1]
                for i in range(marr.shape[0]):
                    for j in range(marr.shape[1]):
                        entity = tboi_bitmap.get_entity_id_with_pixel_value(marr[i, j])
                        marr[i, j] = entity.value
                self.inputs.append(input_arr)
                self.mutated.append(marr)

        self.inputs = torch.tensor(np.stack(self.inputs), dtype=torch.float32)
        self.mutated = torch.tensor(np.stack(self.mutated), dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.mutated[idx]

class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(13*7, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 128)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 13*7*NUM_CLASSES),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        out = self.decoder(h)
        out = out.view(-1, NUM_CLASSES, 13, 7)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar

def vae_loss(recon_logits, target, mu, logvar):
    recon_loss = nn.CrossEntropyLoss()(recon_logits, target)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

def train_vae():
    dataset = MutationDataset("Bitmaps/")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    vae = VAE()
    optimizer = optim.Adam(vae.parameters(), lr=0.0002)

    for epoch in range(100):
        for input_bmp, mutated_bmp in dataloader:
            optimizer.zero_grad()
            mutated_bmp = mutated_bmp.permute(0, 2, 1)
            recon_logits, mu, logvar = vae(input_bmp.float())
            loss = vae_loss(recon_logits, mutated_bmp.long(), mu, logvar)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}: Loss={loss.item():.4f}")

    with torch.no_grad():
        recon_logits, _, _ = vae(input_bmp.float())
        pred = torch.argmax(recon_logits, dim=1)
        os.makedirs("Bitmaps/VAE", exist_ok=True)
        tboi_bitmap = TBoI_Bitmap(width=13, height=7)
        for idx in range(pred.shape[0]):
            arr = pred[idx].cpu().numpy()
            img = TBoI_Bitmap(width=13, height=7)
            for x in range(13):
                for y in range(7):
                    entity_id = EntityType(arr[x, y])
                    pixel_value = tboi_bitmap.get_pixel_value_with_entity_id(entity_id)
                    img.bitmap.putpixel((x, y), pixel_value)
            img.save_bitmap_in_folder(idx, "Bitmaps/VAE")

    input_img = Image.open(os.path.join("Bitmaps", "InputRooms", "bitmap_0.bmp")).convert("L")
    input_arr = np.array(input_img)[1:-1, 1:-1]
    tboi_bitmap = TBoI_Bitmap()
    for i in range(input_arr.shape[0]):
        for j in range(input_arr.shape[1]):
            entity = tboi_bitmap.get_entity_id_with_pixel_value(input_arr[i, j])
            input_arr[i, j] = entity.value
    input_tensor = torch.tensor(input_arr[np.newaxis, np.newaxis, :, :], dtype=torch.float32)

    vae.eval()
    for new_idx in range(5):
        with torch.no_grad():
            mu, logvar = vae.encode(input_tensor)
            z = vae.reparameterize(mu, logvar)
            recon_logits = vae.decode(z)
            pred = torch.argmax(recon_logits, dim=1)[0].cpu().numpy()

        img = TBoI_Bitmap(width=13, height=7)
        for x in range(13):
            for y in range(7):
                entity_id = EntityType(pred[x, y])
                pixel_value = tboi_bitmap.get_pixel_value_with_entity_id(entity_id)
                img.bitmap.putpixel((x, y), pixel_value)
        img.save_bitmap_in_folder(f"new_{new_idx}", "Bitmaps/VAE")

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    latent_dim = trial.suggest_categorical('latent_dim', [32, 64, 128])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    dataset = MutationDataset("Bitmaps/")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vae = VAE(latent_dim=latent_dim)
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    best_loss = float('inf')
    epoch_losses = []

    for epoch in range(50):
        for input_bmp, mutated_bmp in dataloader:
            optimizer.zero_grad()
            mutated_bmp = mutated_bmp.permute(0, 2, 1)
            recon_logits, mu, logvar = vae(input_bmp.float())
            loss = vae_loss(recon_logits, mutated_bmp.long(), mu, logvar)
            loss.backward()
            optimizer.step()
        epoch_losses.append(loss.item())
        trial.report(loss.item(), epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        if loss.item() < best_loss:
            best_loss = loss.item()

    os.makedirs("Optuna/VAE", exist_ok=True)
    epoch_losses.append(0)
    epoch_losses.append(lr)
    epoch_losses.append(latent_dim)
    epoch_losses.append(batch_size)
    np.save(f"Optuna/VAE/optuna_trial_{trial.number}_losses.npy", np.array(epoch_losses))

    return best_loss

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=200, n_jobs=8)
    print("Beste Hyperparameter:", study.best_params)