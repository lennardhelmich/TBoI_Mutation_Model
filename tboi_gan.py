import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from tboi_bitmap import TBoI_Bitmap, EntityType
import numpy as np
import optuna
import matplotlib.pyplot as plt
import re

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

            mutation_folder = os.path.join(data_folder,"Mutations", input_file.replace("bitmap_", "bitmap_").replace(".bmp", ""))
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

                fitness_str = mfile.split("_")[-1].replace(".bmp", "").replace(",", ".")
                try:
                    fitness_val = float(fitness_str)
                except ValueError:
                    continue
                
                self.inputs.append(input_arr)
                self.mutated.append(marr)

        self.inputs = torch.tensor(np.stack(self.inputs), dtype=torch.float32)
        self.mutated = torch.tensor(np.stack(self.mutated), dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.mutated[idx]

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(13*7*2, 128),
            nn.ReLU(),
            nn.Linear(128, 13*7*NUM_CLASSES),
        )

    def forward(self, x, cond):
        x = torch.cat([x, cond], dim=1)
        x = x.view(x.size(0), -1)
        x = self.model(x)
        x = x.view(-1, NUM_CLASSES, 13, 7)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2*13*7, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, cond, y):
        x = torch.cat([cond, y], dim=1)
        x = x.view(x.size(0), -1)
        return self.model(x)

def train_cgan():
    dataset = MutationDataset("Bitmaps/")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    G = Generator()
    D = Discriminator()
    l1_loss = nn.L1Loss()
    bce_loss = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

    for epoch in range(100):
        use_adv = epoch >= 10
        train_D = use_adv and (epoch % 5 == 0)
        for input_bmp, mutated_bmp in dataloader:
            input_bmp = input_bmp.float()
            mutated_bmp = mutated_bmp.unsqueeze(1).float()

            if train_D:
                optimizer_D.zero_grad()
                real_labels = torch.ones(input_bmp.size(0), 1)
                fake_labels = torch.zeros(input_bmp.size(0), 1)

                output_real = D(input_bmp, mutated_bmp)
                loss_real = bce_loss(output_real, real_labels)

                logits_fake = G(torch.zeros_like(input_bmp), input_bmp)
                pred_fake = torch.argmax(logits_fake, dim=1).unsqueeze(1).float().permute(0, 1, 3, 2)
                output_fake = D(input_bmp, pred_fake.detach())
                loss_fake = bce_loss(output_fake, fake_labels)

                loss_D = (loss_real + loss_fake) / 2
                loss_D.backward()
                optimizer_D.step()

            optimizer_G.zero_grad()
            logits_fake = G(torch.zeros_like(input_bmp), input_bmp)
            pred_fake = torch.argmax(logits_fake, dim=1).unsqueeze(1).float().permute(0, 1, 3, 2)
            probs = torch.softmax(logits_fake, dim=1)
            expected = torch.sum(probs * torch.arange(NUM_CLASSES, device=probs.device).view(1, -1, 1, 1), dim=1, keepdim=True)
            if expected.shape != mutated_bmp.shape:
                if expected.shape[2] == mutated_bmp.shape[3] and expected.shape[3] == mutated_bmp.shape[2]:
                    expected = expected.permute(0, 1, 3, 2)
            l1 = l1_loss(expected, mutated_bmp)
            if train_D:
                output_fake = D(input_bmp, pred_fake)
                adv_loss = bce_loss(output_fake, real_labels)
                loss_G = adv_loss + 10 * l1
            else:
                loss_G = 10 * l1
            loss_G.backward()
            optimizer_G.step()

        if use_adv:
            print(f"Epoch {epoch}: Loss_D={loss_D.item():.4f}, Loss_G={loss_G.item():.4f}")
        else:
            print(f"Epoch {epoch}: Loss_G={loss_G.item():.4f} (nur L1)")

    with torch.no_grad():
        logits = G(torch.zeros_like(input_bmp), input_bmp)
        pred = torch.argmax(logits, dim=1)
        os.makedirs("Bitmaps/GAN", exist_ok=True)
        tboi_bitmap = TBoI_Bitmap(width=13, height=7)
        for idx in range(pred.shape[0]):
            arr = pred[idx].cpu().numpy()
            img = TBoI_Bitmap(width=13, height=7)
            for x in range(13):
                for y in range(7):
                    entity_id = EntityType(arr[x, y])
                    pixel_value = tboi_bitmap.get_pixel_value_with_entity_id(entity_id)
                    img.bitmap.putpixel((x, y), pixel_value)
            img.save_bitmap_in_folder(idx, "Bitmaps/GAN")

def objective(trial):
    lr_g = trial.suggest_loguniform('lr_g', 1e-5, 2e-3)
    lr_d = trial.suggest_loguniform('lr_d', 1e-5, 1e-2)
    l1_weight = trial.suggest_float('l1_weight', 1, 20)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    dataset = MutationDataset("Bitmaps/")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    G = Generator()
    D = Discriminator()
    l1_loss = nn.L1Loss()
    bce_loss = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=lr_g)
    optimizer_D = optim.Adam(D.parameters(), lr=lr_d)

    best_val_loss = float('inf')
    epoch_losses = []

    for epoch in range(50):
        for input_bmp, mutated_bmp in dataloader:
            input_bmp = input_bmp.float()
            mutated_bmp = mutated_bmp.unsqueeze(1).float()

            optimizer_D.zero_grad()
            real_labels = torch.ones(input_bmp.size(0), 1)
            fake_labels = torch.zeros(input_bmp.size(0), 1)

            output_real = D(input_bmp, mutated_bmp)
            loss_real = bce_loss(output_real, real_labels)

            logits_fake = G(torch.zeros_like(input_bmp), input_bmp)
            pred_fake = torch.argmax(logits_fake, dim=1).unsqueeze(1).float().permute(0, 1, 3, 2)
            output_fake = D(input_bmp, pred_fake.detach())
            loss_fake = bce_loss(output_fake, fake_labels)

            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            logits_fake = G(torch.zeros_like(input_bmp), input_bmp)
            pred_fake = torch.argmax(logits_fake, dim=1).unsqueeze(1).float().permute(0, 1, 3, 2)
            probs = torch.softmax(logits_fake, dim=1)
            expected = torch.sum(probs * torch.arange(NUM_CLASSES, device=probs.device).view(1, -1, 1, 1), dim=1, keepdim=True)
            if expected.shape != mutated_bmp.shape:
                if expected.shape[2] == mutated_bmp.shape[3] and expected.shape[3] == mutated_bmp.shape[2]:
                    expected = expected.permute(0, 1, 3, 2)
            l1 = l1_loss(expected, mutated_bmp)
            adv_loss = bce_loss(D(input_bmp, pred_fake), real_labels)
            loss_G = adv_loss + l1_weight * l1
            loss_G.backward()
            optimizer_G.step()

        epoch_losses.append(loss_G.item())
        trial.report(loss_G.item(), epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if loss_G.item() < best_val_loss:
            best_val_loss = loss_G.item()

    epoch_losses.append(0)
    epoch_losses.append(lr_d)
    epoch_losses.append(lr_g)
    epoch_losses.append(l1_weight)
    epoch_losses.append(batch_size)
    np.save(f"Optuna/GAN/optuna_trial_{trial.number}_losses.npy", np.array(epoch_losses))

    return best_val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=200, n_jobs=8)
    print("Beste Hyperparameter:", study.best_params)