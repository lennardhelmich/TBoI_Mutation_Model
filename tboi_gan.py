import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from tboi_bitmap import TBoI_Bitmap
import numpy as np
from tboi_bitmap import EntityType

NUM_CLASSES = 12

class MutationDataset(Dataset):
    def __init__(self, data_folder):
        self.inputs = []
        self.mutated = []
        self.fitness = []

        input_rooms_folder = os.path.join(os.path.dirname(data_folder), "InputRooms")
        input_files = sorted([f for f in os.listdir(input_rooms_folder) if f.endswith(".bmp")])

        tboi_bitmap = TBoI_Bitmap()

        for input_file in input_files:
            input_path = os.path.join(input_rooms_folder, input_file)
            input_img = Image.open(input_path).convert("L")
            input_arr = np.array(input_img)
            input_arr = input_arr[1:-1, 1:-1]  # von 15x9 auf 13x7

            for i in range(input_arr.shape[0]):
                for j in range(input_arr.shape[1]):
                    entity = tboi_bitmap.get_entity_id_with_pixel_value(input_arr[i, j])
                    input_arr[i, j] = entity.value

            input_arr = input_arr[np.newaxis, :, :]  # [1, 13, 7]

            mutation_folder = os.path.join(data_folder,"Mutations", input_file.replace("bitmap_", "bitmap_").replace(".bmp", ""))
            if not os.path.isdir(mutation_folder):
                continue
            mutation_files = [f for f in os.listdir(mutation_folder) if f.endswith(".bmp")]
            for mfile in mutation_files:
                mpath = os.path.join(mutation_folder, mfile)
                mimg = Image.open(mpath).convert("L")
                marr = np.array(mimg)
                marr = marr[1:-1, 1:-1]  # von 15x9 auf 13x7

                for i in range(marr.shape[0]):
                    for j in range(marr.shape[1]):
                        entity = tboi_bitmap.get_entity_id_with_pixel_value(marr[i, j])
                        marr[i, j] = entity.value
                # Extract fitness from filename: e.g. mutation_0_0,908.bmp

                fitness_str = mfile.split("_")[-1].replace(".bmp", "").replace(",", ".")
                try:
                    fitness_val = float(fitness_str)
                except ValueError:
                    continue
                
                self.inputs.append(input_arr)
                self.mutated.append(marr)
                self.fitness.append(fitness_val)

        self.inputs = torch.tensor(np.stack(self.inputs), dtype=torch.float32)
        self.mutated = torch.tensor(np.stack(self.mutated), dtype=torch.long)
        self.fitness = torch.tensor(self.fitness, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.mutated[idx], self.fitness[idx]

# Generator: Eingabe-Bitmap -> Mutierte Bitmap (als Klassen)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(13*7, 128),
            nn.ReLU(),
            nn.Linear(128, 13*7*NUM_CLASSES),
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, NUM_CLASSES, 13, 7)  # [batch, classes, 13, 7]
        return x  # logits, noch kein Softmax!

# Training-Loop (stark vereinfacht)
def train_gan():
    dataset = MutationDataset("Bitmaps/")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    G = Generator()
    criterion = nn.CrossEntropyLoss()  # Für Klassifikation pro Pixel
    optimizer_G = optim.Adam(G.parameters(), lr=0.0002)

    for epoch in range(100):
        for input_bmp, mutated_bmp, fitness in dataloader:
            optimizer_G.zero_grad()
            logits = G(input_bmp.float())  # [batch, classes, 13, 7]
            mutated_bmp = mutated_bmp.permute(0, 2, 1)  # [batch, 13, 7]
            loss = criterion(logits, mutated_bmp.long())  # mutated_bmp: [batch, 13, 7]
            loss.backward()
            optimizer_G.step()
        print(f"Epoch {epoch}: Loss_G={loss.item():.4f}")

    # Beispiel: Mutierte Bitmap generieren (als Integer-Matrix)
    with torch.no_grad():
        logits = G(input_bmp.float())
        pred = torch.argmax(logits, dim=1)  # [batch, 13, 7], Werte 0-7
    
    os.makedirs("Bitmaps/GAN", exist_ok=True)
    tboi_bitmap = TBoI_Bitmap(width=13, height=7)

    for idx in range(pred.shape[0]):
        arr = pred[idx].cpu().numpy()  # [13, 7]
        img = TBoI_Bitmap(width=13, height=7)
        for x in range(13):
            for y in range(7):
                # arr[x, y] ist der EntityType-Index
                entity_id = EntityType(arr[x, y])
                pixel_value = tboi_bitmap.get_pixel_value_with_entity_id(entity_id)
                img.bitmap.putpixel((x, y), pixel_value)
        img.save_bitmap_in_folder(idx, "Bitmaps/GAN")
    
    index = 0

if __name__ == "__main__":
    train_gan()