from xml_to_bitmap_converter import convert_xml_to_bitmap
from PIL import Image
import os
from fitness_function import Fitness_Function
from deap import base, creator, tools
from tboi_bitmap import TBoI_Bitmap
import random
import copy
from constants import Constants
from tboi_room_mutation_ea import TBoI_Room_Mutation 
import numpy as np

# PathFinding Test with pre-defined Rooms
# To-Do : Split traversable and non-traversable rooms in 2 folders to simplify verification
def path_finding_test():
    xml_file_path = 'Rooms/PathfindingTestRooms.xml'
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

def calculate_mutations_for_room(room_path):
    bitmap = load_bitmap(room_path)
    room_mutation_ea = TBoI_Room_Mutation(bitmap)
    room_mutations = room_mutation_ea.calculate_mutations(
        Constants.NUMBER_GENERATIONS,
        Constants.CROSSOVER_PROBABILITY,
        Constants.MUTATION_PROBABILITY,
        Constants.POPULATION_SIZE,
        Constants.NUMBER_ELITES)
    return room_mutations

def save_room_mutations_for_room(room_path, room_mutations):
    folder = room_path.split(".")[0] + "/"
    index = 0
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path) 
    for bitmap in room_mutations:
        tboi_bitmap = TBoI_Bitmap() 
        tboi_bitmap.bitmap = Image.fromarray(np.array(bitmap, dtype=np.uint8))
        fitness_str = str(bitmap.fitness.values[0]).replace('.', ',')
        tboi_bitmap.save_mutation_in_folder_with_fitness(index, fitness_str, folder)
        index+=1


if __name__ == "__main__":
    first_room_path = "Bitmaps/InitRooms/bitmap_0.bmp"
    mutations = calculate_mutations_for_room(first_room_path)
    save_room_mutations_for_room(first_room_path, mutations)
    


    
        
    
