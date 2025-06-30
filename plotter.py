import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re

def extract_index(fname):
    match = re.search(r'(\d+)', fname)
    return int(match.group(1)) if match else -1

def plot_optuna_gan_hyperparams(optuna_folder="Optuna/GAN"):
    files = [f for f in os.listdir(optuna_folder) if f.endswith(".npy")]
    if not files:
        print("Keine .npy-Dateien gefunden.")
        return

    files_sorted = sorted(files, key=extract_index)
    x_indices = [extract_index(f) for f in files_sorted]

    lr_d_list, lr_g_list, l1_weight_list, batch_size_list = [], [], [], []
    for fname in files_sorted:
        print(f"Verarbeite Datei: {fname}")
        arr = np.load(os.path.join(optuna_folder, fname))
        lr_d, lr_g, l1_weight, batch_size = arr[-4:]
        lr_d_list.append(lr_d)
        lr_g_list.append(lr_g)
        l1_weight_list.append(l1_weight)
        batch_size_list.append(batch_size)

    print(lr_d_list, lr_g_list, l1_weight_list, batch_size_list)

    os.makedirs("Optuna/Plots", exist_ok=True)
    print("1")
    plt.figure(figsize=(10, 6))
    print("1")
    plt.subplot(2, 2, 1)
    print("1")
    plt.plot(x_indices, lr_d_list, 'o-')
    plt.title("lr_d")
    plt.xlabel("Trial Index")
    plt.subplot(2, 2, 2)
    plt.plot(x_indices, lr_g_list, 'o-')
    plt.title("lr_g")
    plt.xlabel("Trial Index")
    plt.subplot(2, 2, 3)
    plt.plot(x_indices, l1_weight_list, 'o-')
    plt.title("l1_weight")
    plt.xlabel("Trial Index")
    plt.subplot(2, 2, 4)
    plt.plot(x_indices, batch_size_list, 'o-')
    plt.title("batch_size")
    plt.xlabel("Trial Index")
    plt.tight_layout()
    plt.savefig(os.path.join("Optuna/Plots", "optuna_gan_hyperparams.png"))
    plt.close()

def plot_optuna_vae_hyperparams(optuna_folder="Optuna/VAE"):
    files = [f for f in os.listdir(optuna_folder) if f.endswith(".npy")]
    if not files:
        print("Keine .npy-Dateien gefunden.")
        return

    files_sorted = sorted(files, key=extract_index)
    x_indices = [extract_index(f) for f in files_sorted]

    lr_list, latent_dim_list, batch_size_list = [], [], []
    for fname in files_sorted:
        print(f"Verarbeite Datei: {fname}")
        arr = np.load(os.path.join(optuna_folder, fname))
        lr, latent_dim, batch_size = arr[-3:]
        lr_list.append(lr)
        latent_dim_list.append(latent_dim)
        batch_size_list.append(batch_size)

    print(lr_list, latent_dim_list, batch_size_list)

    os.makedirs("Optuna/Plots", exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.plot(x_indices, lr_list, 'o-')
    plt.title("lr")
    plt.xlabel("Trial Index")
    plt.subplot(1, 3, 2)
    plt.plot(x_indices, latent_dim_list, 'o-')
    plt.title("latent_dim")
    plt.xlabel("Trial Index")
    plt.subplot(1, 3, 3)
    plt.plot(x_indices, batch_size_list, 'o-')
    plt.title("batch_size")
    plt.xlabel("Trial Index")
    plt.tight_layout()
    plt.savefig(os.path.join("Optuna/Plots", "optuna_vae_hyperparams.png"))
    plt.close()

if __name__ == "__main__":
    plot_optuna_vae_hyperparams()
    print("Plots der Hyperparameter für GAN- und VAE-Optuna-Studien wurden erstellt.")