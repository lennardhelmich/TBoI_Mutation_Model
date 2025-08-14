import multiprocessing
from xml_to_bitmap_converter import convert_xml_to_bitmap, convert_generated_bitmaps_to_xml
from PIL import Image
import os
from fitness_function_2 import Fitness_Function
from deap import base, creator, tools
from tboi_bitmap import TBoI_Bitmap
import random
import copy
from constants_2 import Constants
from tboi_room_mutation_ea_2 import TBoI_Room_Mutation
import time
import numpy as np
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def path_finding_test():
    xml_file_path = 'OutputXmls/Raum0.xml'
    convert_xml_to_bitmap(xml_file_path, "Bitmaps/PathfindingTests")
    expectedResult = [False, False, False, False, True, True, True, True, True]
    returnedResult = []
    for filename in os.listdir("Bitmaps/PathfindingTests"):
        path = "Bitmaps/PathfindingTests/" + filename
        bitmap = Image.open(path)
        fitness = Fitness_Function(bitmap, bitmap)
        if(fitness.check_every_traversability()):
            returnedResult.append(True)
        else:
            returnedResult.append(False)
        
    if(expectedResult == returnedResult):
        print("Pathfinding Test : Success")
    else:
        print("Pathfinding Test : Failure")

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

    for bitmap in room_mutations:
        tboi_bitmap = TBoI_Bitmap() 
        tboi_bitmap.bitmap = Image.fromarray(np.array(bitmap, dtype=np.uint8))
        fitness_str = str(bitmap.fitness.values[0]).replace('.', ',')
        tboi_bitmap.save_mutation_in_folder_with_fitness(index, fitness_str, folder)
        index+=1

def plot_progress(fitness_history, filename="fitness_progress.png"):
    """
    fitness_history: List of lists, each inner list is functionValue of best individual per generation
    """
    fitness_history = np.array(fitness_history)
    labels = [
        "Gesamtfitness",
        "Balance",
        "Bitmap Changes",
        "Enemies",
        "Symmetry",
        "Variation",
        "PoopFire"
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


# ✅ NEUE FUNKTIONEN für dynamische Fitness-Parameter hinzufügen

def get_dynamic_fitness_parameters(iteration: int) -> tuple[int, int]:
    """
    Berechnet dynamische Fitness-Parameter basierend auf der Iteration.
    
    Schema:
    - Alle 60 Iterationen: TARGETED_BITMAP_DIFFERENCE +5 (35→55)
    - Alle 20 Iterationen: DESIRED_FREESPACE_ENTITY_RATIO_PERCENT +10 (30→60, Reset bei Phase-Wechsel)
    """
    
    # Berechne welche 60er-Phase wir sind (0, 1, 2, 3, 4)
    phase = iteration // 60
    targeted_bitmap_diff = 30 + (phase * 5)
    
    # Position innerhalb der 60er-Phase (0-59)
    phase_position = iteration % 60
    
    # Berechne DESIRED_FREESPACE_ENTITY_RATIO_PERCENT
    # Alle 20 Iterationen um 10 erhöhen, startend bei 30
    ratio_increases = phase_position // 20
    desired_freespace_ratio = 30 + (ratio_increases * 10)
    
    return targeted_bitmap_diff, desired_freespace_ratio

def update_fitness_constants(iteration: int) -> None:
    """
    Aktualisiert nur die Fitness-Constants basierend auf der Iteration.
    """
    targeted_diff, freespace_ratio = get_dynamic_fitness_parameters(iteration)
    Constants.TARGETED_BITMAP_DIFFERENCE = targeted_diff
    Constants.DESIRED_FREESPACE_ENTITY_RATIO_PERCENT = freespace_ratio


# ✅ AKTUALISIERTE MAIN-FUNKTION
if __name__ == "__main__":
    # initXml = "Rooms/First_20_Rooms_For_Dataset.xml"
    saveFolder = "Bitmaps/InputRooms"
    # convert_xml_to_bitmap(initXml, saveFolder)
    
    inputRoomNumber = 0
    
    logging.info("🚀 Starte Mutation-Generierung mit dynamischen Fitness-Parametern")
    logging.info("=" * 60)
    
    for filename in os.listdir(saveFolder):
        path = saveFolder + "/" + filename
        bitmap = load_bitmap(path)
        inputRoomNumber += 1
        
        logging.info(f"📁 Verarbeite Raum {inputRoomNumber}: {filename}")
        
        for i in range(300):
            
            # ✅ Aktualisiere nur Fitness-Parameter vor jeder Iteration
            update_fitness_constants(i)
            
            logging.info(f"Generating mutations {i+1} for room {inputRoomNumber}")
            start_time = time.time()
            
            # ✅ EA verwendet jetzt die aktualisierten Constants
            room_mutation_ea = TBoI_Room_Mutation(bitmap)
            mutations, fitness_history = calculate_mutations_for_room(room_mutation_ea)
            save_room_mutations_for_room("Bitmaps/Mutations2/" + filename, mutations, i)
            # Log Parameter alle 20 Iterationen
            if i % 20 == 0:
                plot_progress(fitness_history, f"fitness_progress_{inputRoomNumber}_{i+1}.png")
            
            elapsed_time = time.time() - start_time
            logging.info(f"Time used for it : {elapsed_time:.2f} sec")
    
    # convert_generated_bitmaps_to_xml(initXml)
    logging.info("🎉 Alle Mutationen generiert!")








