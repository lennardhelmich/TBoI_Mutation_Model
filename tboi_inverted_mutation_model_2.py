import multiprocessing
from xml_to_bitmap_converter import convert_xml_to_bitmap, convert_generated_bitmaps_to_xml
from PIL import Image
import os
from deap import base, creator, tools
from tboi_bitmap import TBoI_Bitmap
import random
import copy
from constants_2 import Constants
from tboi_inverted_room_mutation_ea_2 import Inverted_TBoI_Room_Mutation
import time
import numpy as np
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_bitmap(image_path):
    image = Image.open(image_path)
    width,height = image.size
    pixel_data = []
    for i in range(height):
        row = []
        for j in range(width):
            pixel = image.getpixel((j,i))
            row.append(pixel)
        pixel_data.append(row)
    return pixel_data

def calculate_mutations_for_room(room_mutation_ea):
    room_mutations, fitness_history = room_mutation_ea.calculate_mutations(
        Constants.NUMBER_GENERATIONS,
        Constants.CROSSOVER_PROBABILITY,
        Constants.MUTATION_PROBABILITY,
        Constants.POPULATION_SIZE,
        Constants.NUMBER_ELITES)
    return room_mutations, fitness_history

def delete_mutations(folder):
    try:
        dirList = os.listdir(folder)
        if(len(dirList) != 0):
            for filename in dirList:
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path) 
    except FileNotFoundError:
        print("Mutations did not exist")

def save_room_mutations_for_room(room_path, room_mutations, offset):
    folder = room_path.split(".")[0] + "/"
    index = 0 + offset*Constants.NUMBER_ELITES
    bitmap = room_mutations[0]
    tboi_bitmap = TBoI_Bitmap() 
    tboi_bitmap.bitmap = Image.fromarray(np.array(bitmap, dtype=np.uint8))
    fitness_str = str(bitmap.fitness.values[0]).replace('.', ',')
    tboi_bitmap.save_mutation_in_folder_with_fitness(index, fitness_str, folder)

def plot_progress(fitness_history, filename="fitness_progress.png"):
    """
    fitness_history: List of lists, each inner list is functionValue of best individual per generation
    """
    print(fitness_history)
    fitness_history = np.array(fitness_history)
    labels = [
        "Gesamtfitness",
        "Inverted_Symmetry",
        "Inverted_Variation",
        "Topology",
        "Detour",
        "Enemy_Proximity",
        "Enemy_Difference",
        "Bitmap_Changes"
    ]

    for i in range(fitness_history.shape[1]):
        plt.plot(fitness_history[:, i], label=labels[i])
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Progress (alle Komponenten)")
    plt.legend()
    os.makedirs("Plots", exist_ok=True)
    plt.savefig(os.path.join("Plots", filename))
    plt.close()

if __name__ == "__main__":
    mutations_root = "Bitmaps/Mutations2"
    inputRoomNumber = 48000
    for foldername in os.listdir(mutations_root):
        folder_path = os.path.join(mutations_root, foldername)
        if os.path.isdir(folder_path) and foldername.startswith("bitmap_"):
            for filename in os.listdir(folder_path):
                path = os.path.join(folder_path, filename)
                bitmap = load_bitmap(path)
                inputRoomNumber += 1
                logging.info(f"Generating mutations for room {inputRoomNumber} ({foldername}/{filename})")
                start_time = time.time()
                room_mutation_ea = Inverted_TBoI_Room_Mutation(bitmap)
                mutations, fitness_history = calculate_mutations_for_room(room_mutation_ea)
                save_room_mutations_for_room(
                    os.path.join("Bitmaps/Inverted_Mutations2", foldername, filename),
                    mutations,
                    0
                )
                if inputRoomNumber % 1000 == 1:
                    plot_progress(fitness_history, f"fitness_progress_{inputRoomNumber}.png")
                elapsed_time = time.time() - start_time
                logging.info(f"Time used for it : {elapsed_time:.2f} sec")








