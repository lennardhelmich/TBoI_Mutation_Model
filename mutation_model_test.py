from level_gan import interpolate_inputs_and_decode
from level_vae import interpolate_between_bitmaps
import random
import os
from pathlib import Path
from fitness_function import Fitness_Function
from PIL import Image

def test_gan_interpolation(ckpt, bmp_a, bmp_b, steps=5, out_dir="gan_interp_test", device="cpu"):
    """Testet die GAN-Interpolation zwischen zwei Bitmaps."""
    interpolate_inputs_and_decode(ckpt, bmp_a, bmp_b, steps=steps, out_dir=out_dir, device=device)
    print(f"GAN-Interpolation gespeichert in: {out_dir}")

def test_vae_interpolation(ckpt, bmp_a, bmp_b, steps=5, out_dir="vae_interp_test"):
    """Testet die VAE-Interpolation zwischen zwei Bitmaps."""
    interpolate_between_bitmaps(ckpt, bmp_a, bmp_b, steps=steps, out_folder=out_dir)
    print(f"VAE-Interpolation gespeichert in: {out_dir}")

def get_test_bitmaps():
    """Gibt zwei Bitmaps zurück:
    1. Zufällige Bitmap aus Bitmaps/InputRooms
    2. Zufällige Bitmap aus Bitmaps/Inverted_Mutations/<Bitmap1NameOhne.bmp>/<randomOrdner>/<randomBitmap>
    """
    # 1. Zufällige Bitmap aus InputRooms
    input_dir = Path("Bitmaps/InputRooms")
    bmp1 = random.choice(list(input_dir.glob("*.bmp")))

    inv_dir = Path("Bitmaps/Inverted_Mutations") / bmp1.stem
    subfolders = [f for f in inv_dir.iterdir() if f.is_dir()]
    if not subfolders:
        raise RuntimeError(f"Keine Unterordner in {inv_dir} gefunden.")
    random_folder = random.choice(subfolders)
    bmp2_candidates = list(random_folder.glob("*.bmp"))
    if not bmp2_candidates:
        raise RuntimeError(f"Keine Bitmaps in {random_folder} gefunden.")
    bmp2 = random.choice(bmp2_candidates)

    return str(bmp1), str(bmp2)

def get_two_inverted_mutations():
    """
    Wählt einen zufälligen bitmap_X-Ordner aus Inverted_Mutations und gibt daraus zwei unterschiedliche Bitmaps aus allen mutation_* Unterordnern zurück.
    Gibt zusätzlich den Namen der gewählten bitmap_X zurück.
    """
    base_dir = Path("Bitmaps/Inverted_Mutations")
    bitmap_folders = [f for f in base_dir.iterdir() if f.is_dir()]
    if not bitmap_folders:
        raise RuntimeError("Keine bitmap_X Ordner in Inverted_Mutations gefunden.")
    chosen_bitmap_folder = random.choice(bitmap_folders)
    bitmap_name = chosen_bitmap_folder.name  # z.B. "bitmap_0"

    bmp_paths = []
    for mutation_folder in chosen_bitmap_folder.glob("mutation_*"):
        if mutation_folder.is_dir():
            bmp_files = list(mutation_folder.glob("*.bmp"))
            bmp_paths.extend(bmp_files)

    if len(bmp_paths) < 2:
        raise RuntimeError(f"Nicht genug Bitmaps in {chosen_bitmap_folder} gefunden.")

    bmp1, bmp2 = random.sample(bmp_paths, 2)
    return str(bmp1), str(bmp2), bitmap_name

def test_gan_interpolation_alpha(ckpt, bmp_a, bmp_b, alpha=0.5, out_path="gan_interp_alpha.bmp", device="cpu"):
    """Interpoliert GAN mit gegebenem alpha und speichert genau eine Mutation."""
    # Lade Modell
    from level_gan import LitPix2Pix
    import torch
    model = LitPix2Pix.load_from_checkpoint(ckpt, map_location=device).to(device).eval()

    # Lade Bitmaps und preprocess
    def load_one(path):
        import numpy as np
        from PIL import Image
        from level_gan import PIXEL_TO_ENTITY, NUM_CLASSES
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.uint8)
        if arr.shape == (9, 15):
            arr = arr[1:-1, 1:-1]
        cls = torch.from_numpy(PIXEL_TO_ENTITY[arr]).long()
        x_oh = torch.nn.functional.one_hot(cls, NUM_CLASSES).permute(2,0,1).unsqueeze(0).float()
        return x_oh.to(device)

    x1 = load_one(bmp_a)
    x2 = load_one(bmp_b)
    x_interp = (1.0 - alpha) * x1 + alpha * x2

    with torch.no_grad():
        logits = model.G(x_interp)
        pred = logits.argmax(1)[0].cpu().numpy()

    # Rücktransformation zu Bitmap
    from level_gan import ENTITY_TO_PIXEL
    import numpy as np
    from PIL import Image

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    output_with_border = np.zeros((9, 15), dtype=np.uint8)
    output_with_border[1:-1, 1:-1] = np.vectorize(lambda k: ENTITY_TO_PIXEL[int(k)])(pred).astype(np.uint8)
    img = Image.fromarray(output_with_border, mode='L')
    img.save(out_path)
    print(f"GAN-Interpolation mit alpha={alpha} gespeichert: {out_path}")

def test_vae_interpolation_alpha(ckpt, bmp_a, bmp_b, alpha=0.5, out_path="vae_interp_alpha.bmp"):
    """Interpoliert VAE mit gegebenem alpha und speichert genau eine Mutation."""
    from level_vae import load_model, encode_bitmap_to_latent, ids_to_bmp_with_border
    import torch

    model = load_model(ckpt)
    z1 = encode_bitmap_to_latent(model, bmp_a)
    z2 = encode_bitmap_to_latent(model, bmp_b)
    z = (1.0 - alpha) * z1 + alpha * z2

    with torch.no_grad():
        pred = torch.argmax(model.decode(z), dim=1)[0].cpu().numpy()
        img = ids_to_bmp_with_border(pred)
        img.save(out_path)
    print(f"VAE-Interpolation mit alpha={alpha} gespeichert: {out_path}")

def evaluate_generated_bitmaps_normal_inverted(ckpt_gan, ckpt_vae, a_min = 0.2, a_max = 0.8, n=100):
    """Generiert n Bitmaps mit GAN und VAE und bewertet sie mit Fitness_Function."""
    gan_scores = []
    vae_scores = []

    for i in range(n):
        # Zufällige Bitmaps für Input
        bmp1, bmp2 = get_test_bitmaps()
        bmp1_img = Image.open(bmp1)
        alpha = random.uniform(a_min, a_max)

        # GAN generieren
        out_gan = f"interpolation_test_bmp_inv/temp_gan_{i}.bmp"
        test_gan_interpolation_alpha(ckpt_gan, bmp1, bmp2, alpha=alpha, out_path=out_gan)
        gan_bitmap = Image.open(out_gan)

        # VAE generieren
        out_vae = f"interpolation_test_bmp_inv/temp_vae_{i}.bmp"
        test_vae_interpolation_alpha(ckpt_vae, bmp1, bmp2, alpha=alpha, out_path=out_vae)
        vae_bitmap = Image.open(out_vae)

        gan_bitmap = add_doors_to_bitmap(gan_bitmap)
        vae_bitmap = add_doors_to_bitmap(vae_bitmap)

        print(f"Bitmap {i+1}/{n} generiert: GAN={gan_bitmap}, VAE={vae_bitmap}")

        # Fitness bewerten (nutze bmp1 als Startbitmap)
        gan_fitness = Fitness_Function(bmp1_img, gan_bitmap)
        gan_fitness.calc_fitness_function()
        gan_scores.append(gan_fitness.functionValue)

        vae_fitness = Fitness_Function(bmp1_img, vae_bitmap)
        vae_fitness.calc_fitness_function()
        vae_scores.append(vae_fitness.functionValue)

        print(f"[{i+1}/{n}] GAN: {gan_fitness.functionValue} | VAE: {vae_fitness.functionValue}")

        # Optional: temporäre Dateien löschen

    return gan_scores, vae_scores

def evaluate_generated_bitmaps_inverted_inverted(ckpt_gan, ckpt_vae, n=100):
    """Generiert n Bitmaps mit GAN und VAE und bewertet sie mit Fitness_Function.
    Nutzt get_two_inverted_mutations und gibt die passende InputRooms-Bitmap als Startbitmap in die Fitness-Funktion."""
    gan_scores = []
    vae_scores = []

    for i in range(n):
        # Hole zwei Bitmaps und den bitmap_X Namen
        bmp1, bmp2, bitmap_name = get_two_inverted_mutations()
        # Hole die zugehörige InputRooms-Bitmap
        input_bitmap_path = Path("Bitmaps/InputRooms") / f"{bitmap_name}.bmp"
        bmp1_img = Image.open(input_bitmap_path)
        alpha = random.uniform(0.2, 0.8)

        # GAN generieren
        out_gan = f"interpolation_test_inv_inv/temp_gan_{i}.bmp"
        test_gan_interpolation_alpha(ckpt_gan, bmp1, bmp2, alpha=alpha, out_path=out_gan)
        gan_bitmap = Image.open(out_gan)

        # VAE generieren
        out_vae = f"interpolation_test_inv_inv/temp_vae_{i}.bmp"
        test_vae_interpolation_alpha(ckpt_vae, bmp1, bmp2, alpha=alpha, out_path=out_vae)
        vae_bitmap = Image.open(out_vae)

        gan_bitmap = add_doors_to_bitmap(gan_bitmap)
        vae_bitmap = add_doors_to_bitmap(vae_bitmap)

        print(f"Bitmap {i+1}/{n} generiert: GAN={gan_bitmap}, VAE={vae_bitmap}")

        # Fitness bewerten (nutze InputRooms-Bitmap als Startbitmap)
        gan_fitness = Fitness_Function(bmp1_img, gan_bitmap)
        gan_fitness.calc_fitness_function()
        gan_scores.append(gan_fitness.functionValue)

        vae_fitness = Fitness_Function(bmp1_img, vae_bitmap)
        vae_fitness.calc_fitness_function()
        vae_scores.append(vae_fitness.functionValue)

        print(f"[{i+1}/{n}] GAN: {gan_fitness.functionValue} | VAE: {vae_fitness.functionValue}")

    return gan_scores, vae_scores

def save_scores_json(gan_scores, vae_scores, out_name):
    import os, json
    os.makedirs("evaluation_result", exist_ok=True)
    with open(f"evaluation_result/{out_name}_gan_scores.json", "w") as f:
        json.dump(gan_scores, f)
    with open(f"evaluation_result/{out_name}_vae_scores.json", "w") as f:
        json.dump(vae_scores, f)
    print(f"Saved: evaluation_result/{out_name}_gan_scores.json, evaluation_result/{out_name}_vae_scores.json")

def print_score_summary(gan_scores, vae_scores):
    """Gibt Mittelwert, Median, Min, Max und Std für alle Fitness-Komponenten von GAN und VAE aus.
    Zählt zusätzlich, wie oft der Gesamtwert 0 ist."""
    import numpy as np

    gan_arr = np.array(gan_scores)  # shape: (n, 6)
    vae_arr = np.array(vae_scores)  # shape: (n, 6)

    fitness_names = [
        "Gesamtwert",
        "Balance",
        "Enemies",
        "Symmetry",
        "Variation"
    ]

    gan_mask = gan_arr[:, 0] != 0
    vae_mask = vae_arr[:, 0] != 0
    gan_arr = gan_arr[gan_mask]
    vae_arr = vae_arr[vae_mask]

    print("\n--- GAN Scores ---")
    print(f"Verbleibende Bitmaps: {len(gan_arr)} (ignoriert: {np.sum(~gan_mask)})")
    for i, name in enumerate(fitness_names):
        vals = gan_arr[:, i]
        print(f"{name}:")
        print(f"  Mittelwert: {vals.mean():.4f}")
        print(f"  Median:    {np.median(vals):.4f}")
        print(f"  Min:       {vals.min():.4f}")
        print(f"  Max:       {vals.max():.4f}")
        print(f"  Std:       {vals.std():.4f}")

    gan_zero_count = np.sum(gan_arr[:, 0] == 0)
    print(f"\nGAN: Anzahl functionValue[0] == 0: {gan_zero_count} von {len(gan_arr)}")

    print("\n--- VAE Scores ---")
    print(f"Verbleibende Bitmaps: {len(vae_arr)} (ignoriert: {np.sum(~vae_mask)})")
    for i, name in enumerate(fitness_names):
        vals = vae_arr[:, i]
        print(f"{name}:")
        print(f"  Mittelwert: {vals.mean():.4f}")
        print(f"  Median:    {np.median(vals):.4f}")
        print(f"  Min:       {vals.min():.4f}")
        print(f"  Max:       {vals.max():.4f}")
        print(f"  Std:       {vals.std():.4f}")

    vae_zero_count = np.sum(vae_arr[:, 0] == 0)
    print(f"\nVAE: Anzahl functionValue[0] == 0: {vae_zero_count} von {len(vae_arr)}")

    print("\n--- Vergleich Gesamtwert ---")
    diff = gan_arr[:, 0].mean() - vae_arr[:, 0].mean()
    print(f"Durchschnittlicher Unterschied (GAN - VAE, Gesamtwert): {diff:.4f}")

def add_doors_to_bitmap(img, door_value=23):
    """
    Fügt an jedem Rand der Bitmap in der Mitte eine Tür hinzu.
    img: PIL.Image oder np.ndarray mit shape (9, 15) oder (7, 13)
    door_value: Pixelwert für die Tür (z.B. 46)
    """
    import numpy as np
    from PIL import Image

    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape

    # Nur für Bitmaps mit Rand (9x15)
    if h == 9 and w == 15:
        # Mitte berechnen
        mid_row = h // 2
        mid_col = w // 2
        # Türen setzen
        arr[0, mid_col] = door_value      # oben
        arr[-1, mid_col] = door_value     # unten
        arr[mid_row, 0] = door_value      # links
        arr[mid_row, -1] = door_value     # rechts
    else:
        # Falls kein Rand vorhanden, nichts tun oder anpassen
        pass
    return Image.fromarray(arr, mode="L")

# test_gan_interpolation(
#     "optuna_gan_final/best_final.ckpt",
#     "Bitmaps/InputRooms/bitmap_0.bmp",
#     "C:\\Users\\Lennard Arbeit\\Bachelor Arbeit\\TBoI_Mutation_Model\\Bitmaps\\Inverted_Mutations\\bitmap_0\\mutation_0_0,8643206000348856\\mutation_0_0,7546815429468491.bmp",
#     steps=10
# )

# test_vae_interpolation(
#     "best_optuna_vae_model_2/best.pt",
#     "Bitmaps/InputRooms/bitmap_0.bmp",
#     "C:\\Users\\Lennard Arbeit\\Bachelor Arbeit\\TBoI_Mutation_Model\\Bitmaps\\Inverted_Mutations\\bitmap_0\\mutation_0_0,8643206000348856\\mutation_0_0,7546815429468491.bmp",
#     steps=10
# )

# bmp1, bmp2 = get_test_bitmaps()
# test_gan_interpolation_alpha("optuna_gan_final/best_final.ckpt", bmp1, bmp2, alpha=0.7, out_path="interp/gan_alpha_07.bmp")
# test_vae_interpolation_alpha("best_optuna_vae_model_2/best.pt", bmp1, bmp2, alpha=0.7, out_path="interp/vae_alpha_07.bmp")

# gan_scores, vae_scores = evaluate_generated_bitmaps_normal_inverted("optuna_gan_final/best_final.ckpt", "best_optuna_vae_model/best.pt", n=1000)
# gan_scores_2, vae_scores_2 = evaluate_generated_bitmaps_inverted_inverted("optuna_gan_final/best_final.ckpt", "best_optuna_vae_model/best.pt", n=1000)

# print_score_summary(gan_scores, vae_scores)
# print_score_summary(gan_scores_2, vae_scores_2)

def run_evaluation_series():
    gan_scores, vae_scores = evaluate_generated_bitmaps_normal_inverted("optuna_gan_final/best_final.ckpt", "best_optuna_vae_model/best.pt", a_min=0.0, a_max=1.0, n=1000)
    save_scores_json(gan_scores, vae_scores, "alpha_00_10")
    gan_scores2, vae_scores2 = evaluate_generated_bitmaps_normal_inverted("optuna_gan_final/best_final.ckpt", "best_optuna_vae_model/best.pt", a_min=0.1, a_max=0.9, n=1000)
    save_scores_json(gan_scores2, vae_scores2, "alpha_01_09")
    gan_scores3, vae_scores3 = evaluate_generated_bitmaps_normal_inverted("optuna_gan_final/best_final.ckpt", "best_optuna_vae_model/best.pt", a_min=0.2, a_max=0.8, n=1000)
    save_scores_json(gan_scores3, vae_scores3, "alpha_02_08")
    gan_scores4, vae_scores4 = evaluate_generated_bitmaps_normal_inverted("optuna_gan_final/best_final.ckpt", "best_optuna_vae_model/best.pt", a_min=0.3, a_max=0.7, n=1000)
    save_scores_json(gan_scores4, vae_scores4, "alpha_03_07")


run_evaluation_series()


