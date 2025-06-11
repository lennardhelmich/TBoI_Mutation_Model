import multiprocessing
from xml_to_bitmap_converter import convert_xml_to_bitmap, convert_generated_bitmaps_to_xml
from PIL import Image
import os
from fitness_function import Fitness_Function
from deap import base, creator, tools
from tboi_bitmap import TBoI_Bitmap
import random
import copy
from constants import Constants
from tboi_room_mutation_ea import TBoI_Room_Mutation
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
    fitness_history = np.array(fitness_history)  # shape: [num_generations, num_metrics]
    labels = [
        "Gesamtfitness",
        "Balance",
        "Bitmap Changes",
        "Enemies",
        "Symmetry",
        "Variation"
    ]
    for i in range(fitness_history.shape[1]):
        plt.plot(fitness_history[:, i], label=labels[i])
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Progress (alle Komponenten)")
    plt.legend()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    initXml = "Rooms/FirstInputRoom.xml"
    saveFolder = "Bitmaps/InputRooms"
    convert_xml_to_bitmap(initXml, saveFolder)
    inputRoomNumber = 0
    for filename in os.listdir(saveFolder):
        path = saveFolder + "/" + filename
        bitmap = load_bitmap(path)
        inputRoomNumber += 1
        for i in range(10):
            logging.info(f"Generating mutations {i+1} for room {inputRoomNumber}")
            start_time = time.time()
            room_mutation_ea = TBoI_Room_Mutation(bitmap)
            mutations, fitness_history = calculate_mutations_for_room(room_mutation_ea)
            save_room_mutations_for_room("Bitmaps/Mutations/" + filename, mutations, i)
            plot_progress(fitness_history, f"fitness_progress_{inputRoomNumber}_{i+1}.png")
            elapsed_time = time.time() - start_time
            logging.info(f"Time used for it : {elapsed_time:.2f} sec")
    
    convert_generated_bitmaps_to_xml(initXml)








