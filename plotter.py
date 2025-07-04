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

def plot_min_losses(optuna_folder="Optuna/GAN", model_name="GAN"):
    files = [f for f in os.listdir(optuna_folder) if f.endswith(".npy")]
    if not files:
        print("Keine .npy-Dateien gefunden.")
        return

    files_sorted = sorted(files, key=extract_index)
    x_indices = [extract_index(f) for f in files_sorted]
    
    min_losses = []
    
    for fname in files_sorted:
        arr = np.load(os.path.join(optuna_folder, fname))
        
        i = 0
        while i < len(arr) and arr[i] != 0:
            i += 1

        losses = arr[:i] if i > 0 else arr
        
        if len(losses) > 0:
            min_loss = np.min(losses)
            min_losses.append(min_loss)
            print(f"Niedrigster Loss in {fname}: {min_loss:.4f}")
        else:
            print(f"Keine Loss-Werte in {fname} gefunden.")
            min_losses.append(float('inf'))
    
    # Plotten
    os.makedirs("Optuna/Plots", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(x_indices, min_losses, 'o-')
    plt.title(f"Niedrigste Loss-Werte pro Trial ({model_name})")
    plt.xlabel("Trial Index")
    plt.ylabel("Niedrigster Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("Optuna/Plots", f"min_losses_{model_name.lower()}.png"))
    plt.close()
    
    print(f"Niedrigste Loss-Werte: {min_losses}")
    print(f"Bester Trial: {x_indices[np.argmin(min_losses)]} mit Loss: {np.min(min_losses):.4f}")

def plot_trial_losses(optuna_folder, trial_number, model_name="Model"):
    """
    Plottet die Loss-Kurve für einen bestimmten Trial und zeigt die Hyperparameter an.
    
    Args:
        optuna_folder (str): Pfad zum Optuna-Ordner
        trial_number (int): Nummer des Trials
        model_name (str): Name des Modells (für den Titel)
    """
    # Suche nach der entsprechenden Datei
    filename = f"optuna_trial_{trial_number}_losses.npy"
    filepath = os.path.join(optuna_folder, filename)
    
    if not os.path.exists(filepath):
        print(f"Trial {trial_number} nicht gefunden in {optuna_folder}")
        return
    
    # Lade das Array
    arr = np.load(filepath)
    
    # Finde den Index der ersten 0
    zero_index = None
    for i, val in enumerate(arr):
        if val == 0:
            zero_index = i
            break
    
    if zero_index is None:
        print(f"Keine 0 im Array gefunden für Trial {trial_number}")
        return
    
    # Teile das Array auf
    losses = arr[:zero_index]
    hyperparams = arr[zero_index+1:]  # Überspringe die 0
    
    # Erstelle den Plot
    os.makedirs("Optuna/Plots", exist_ok=True)
    plt.figure(figsize=(12, 6))
    
    # Loss-Kurve plotten
    plt.subplot(1, 2, 1)
    plt.plot(range(len(losses)), losses, 'b-', linewidth=2)
    plt.title(f"Loss-Kurve für Trial {trial_number} ({model_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    
    # Hyperparameter als Text anzeigen
    plt.subplot(1, 2, 2)
    plt.axis('off')
    
    # Bestimme die Hyperparameter basierend auf dem Modell
    if model_name.upper() == "GAN":
        if len(hyperparams) >= 4:
            param_text = f"Trial {trial_number} Hyperparameter:\n\n"
            param_text += f"lr_d: {hyperparams[0]:.6f}\n"
            param_text += f"lr_g: {hyperparams[1]:.6f}\n"
            param_text += f"l1_weight: {hyperparams[2]:.2f}\n"
            param_text += f"batch_size: {int(hyperparams[3])}\n\n"
            param_text += f"Bester Loss: {np.min(losses):.6f}\n"
            param_text += f"Finaler Loss: {losses[-1]:.6f}\n"
            param_text += f"Anzahl Epochen: {len(losses)}"
        else:
            param_text = f"Nicht genügend Hyperparameter gefunden\nGefunden: {len(hyperparams)}"
    
    elif model_name.upper() == "VAE":
        if len(hyperparams) >= 3:
            param_text = f"Trial {trial_number} Hyperparameter:\n\n"
            param_text += f"lr: {hyperparams[0]:.6f}\n"
            param_text += f"latent_dim: {int(hyperparams[1])}\n"
            param_text += f"batch_size: {int(hyperparams[2])}\n\n"
            param_text += f"Bester Loss: {np.min(losses):.6f}\n"
            param_text += f"Finaler Loss: {losses[-1]:.6f}\n"
            param_text += f"Anzahl Epochen: {len(losses)}"
        else:
            param_text = f"Nicht genügend Hyperparameter gefunden\nGefunden: {len(hyperparams)}"
    
    else:
        param_text = f"Trial {trial_number} Hyperparameter:\n\n"
        for i, param in enumerate(hyperparams):
            param_text += f"Param {i+1}: {param}\n"
        param_text += f"\nBester Loss: {np.min(losses):.6f}\n"
        param_text += f"Finaler Loss: {losses[-1]:.6f}\n"
        param_text += f"Anzahl Epochen: {len(losses)}"
    
    plt.text(0.1, 0.9, param_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.savefig(os.path.join("Optuna/Plots", f"trial_{trial_number}_{model_name.lower()}_losses.png"))
    plt.close()
    
    print(f"Plot für Trial {trial_number} erstellt.")
    print(f"Bester Loss: {np.min(losses):.6f}")
    print(f"Hyperparameter: {hyperparams}")

def plot_progressive_best_hyperparams(optuna_folder="Optuna/GAN", model_name="GAN"):
    """
    Plottet nur die Hyperparameter von Trials, die einen neuen besten (niedrigsten) finalen Loss erreichen.
    Startet bei 0 und nimmt nur Trials, deren finaler Loss besser als alle vorherigen ist.
    
    Args:
        optuna_folder (str): Pfad zum Optuna-Ordner
        model_name (str): Name des Modells ("GAN" oder "VAE")
    """
    files = [f for f in os.listdir(optuna_folder) if f.endswith(".npy")]
    if not files:
        print("Keine .npy-Dateien gefunden.")
        return

    files_sorted = sorted(files, key=extract_index)
    
    # Listen für progressive beste Trials
    x_indices = []
    best_loss_so_far = float('inf')
    
    if model_name.upper() == "GAN":
        lr_d_list, lr_g_list, l1_weight_list, batch_size_list = [], [], [], []
        
        for fname in files_sorted:
            trial_idx = extract_index(fname)
            arr = np.load(os.path.join(optuna_folder, fname))
            
            # Finde erste 0 (Trenner zwischen Losses und Hyperparametern)
            zero_index = None
            for i, val in enumerate(arr):
                if val == 0:
                    zero_index = i
                    break
            
            if zero_index is None or zero_index == 0:
                continue
                
            losses = arr[:zero_index]
            hyperparams = arr[zero_index+1:]
            
            # Prüfe ob genügend Hyperparameter vorhanden
            if len(hyperparams) < 4:
                continue
                
            # Prüfe ob finaler Loss besser als bisheriger bester ist
            final_loss = losses[-1]
            if final_loss < best_loss_so_far:
                best_loss_so_far = final_loss
                x_indices.append(trial_idx)
                lr_d_list.append(hyperparams[0])
                lr_g_list.append(hyperparams[1])
                l1_weight_list.append(hyperparams[2])
                batch_size_list.append(hyperparams[3])
                print(f"Neuer bester Trial {trial_idx}: Final Loss = {final_loss:.4f} (vorher: {best_loss_so_far:.4f})")
        
        if not x_indices:
            print("Keine progressiv verbessernden Trials gefunden.")
            return
        
        # Plotten
        os.makedirs("Optuna/Plots", exist_ok=True)
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(x_indices, lr_d_list, 'ro-', linewidth=2, markersize=8)
        plt.title("lr_d (Progressive Best Trials)")
        plt.xlabel("Trial Index")
        plt.ylabel("lr_d")
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(x_indices, lr_g_list, 'bo-', linewidth=2, markersize=8)
        plt.title("lr_g (Progressive Best Trials)")
        plt.xlabel("Trial Index")
        plt.ylabel("lr_g")
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(x_indices, l1_weight_list, 'go-', linewidth=2, markersize=8)
        plt.title("l1_weight (Progressive Best Trials)")
        plt.xlabel("Trial Index")
        plt.ylabel("l1_weight")
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(x_indices, batch_size_list, 'mo-', linewidth=2, markersize=8)
        plt.title("batch_size (Progressive Best Trials)")
        plt.xlabel("Trial Index")
        plt.ylabel("batch_size")
        plt.grid(True)
        
        plt.suptitle(f"Progressive Best Hyperparameter für {model_name}", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join("Optuna/Plots", f"progressive_best_{model_name.lower()}_hyperparams.png"))
        plt.close()
        
        print(f"Progressive beste Hyperparameter geplottet für {len(x_indices)} Trials")
        print(f"Beste Trials: {x_indices}")
        
    elif model_name.upper() == "VAE":
        lr_list, latent_dim_list, batch_size_list = [], [], []
        
        for fname in files_sorted:
            trial_idx = extract_index(fname)
            arr = np.load(os.path.join(optuna_folder, fname))
            
            # Finde erste 0
            zero_index = None
            for i, val in enumerate(arr):
                if val == 0:
                    zero_index = i
                    break
            
            if zero_index is None or zero_index == 0:
                continue
                
            losses = arr[:zero_index]
            hyperparams = arr[zero_index+1:]
            
            # Prüfe ob genügend Hyperparameter vorhanden
            if len(hyperparams) < 3:
                continue
                
            # Prüfe ob finaler Loss besser als bisheriger bester ist
            final_loss = losses[-1]
            if final_loss < best_loss_so_far:
                best_loss_so_far = final_loss
                x_indices.append(trial_idx)
                lr_list.append(hyperparams[0])
                latent_dim_list.append(hyperparams[1])
                batch_size_list.append(hyperparams[2])
                print(f"Neuer bester Trial {trial_idx}: Final Loss = {final_loss:.4f}")
        
        if not x_indices:
            print("Keine progressiv verbessernden Trials gefunden.")
            return
        
        # Plotten
        os.makedirs("Optuna/Plots", exist_ok=True)
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(x_indices, lr_list, 'ro-', linewidth=2, markersize=8)
        plt.title("lr (Progressive Best Trials)")
        plt.xlabel("Trial Index")
        plt.ylabel("lr")
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(x_indices, latent_dim_list, 'bo-', linewidth=2, markersize=8)
        plt.title("latent_dim (Progressive Best Trials)")
        plt.xlabel("Trial Index")
        plt.ylabel("latent_dim")
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(x_indices, batch_size_list, 'go-', linewidth=2, markersize=8)
        plt.title("batch_size (Progressive Best Trials)")
        plt.xlabel("Trial Index")
        plt.ylabel("batch_size")
        plt.grid(True)
        
        plt.suptitle(f"Progressive Best Hyperparameter für {model_name}", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join("Optuna/Plots", f"progressive_best_{model_name.lower()}_hyperparams.png"))
        plt.close()
        
        print(f"Progressive beste Hyperparameter geplottet für {len(x_indices)} Trials")
        print(f"Beste Trials: {x_indices}")

if __name__ == "__main__":
    path = r"C:\Users\Lennard Arbeit\Bachelor Arbeit\Datensatz 1\VAE_Datensatz1_OptunaHPS3"
    #plot_min_losses(path, "VAE")
    plot_trial_losses(path, 78, "VAE")
    #plot_progressive_best_hyperparams(path, "VAE")
    #plot_optuna_vae_hyperparams(path)