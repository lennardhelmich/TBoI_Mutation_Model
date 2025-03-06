from xml_to_bitmap_converter import convert_xml_to_bitmap
from PIL import Image
import os
from fitness_function import Fitness_Function
from deap import base, creator, tools
from tboi_bitmap import TBoI_Bitmap
import random
import copy
from constants import Constants


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

bitmap = TBoI_Bitmap()

ALLOWED_VALUES = bitmap.allowed_room_entities()

def random_bit():
    return random.choice(ALLOWED_VALUES)

def create_individual(data):
    return data

toolbox.register("individual", creator.Individual)
toolbox.register("population", list)

def fitness_function(individual):
    fitness = Fitness_Function(individual[0], individual[1])
    fitness.calc_fitness_function()
    return fitness.functionValue

def custom_mutation(individual):
    current_bitmap = individual[1]

    for i in range(1,len(current_bitmap)-1):
        for j in range(1,len(current_bitmap[i])-1):
            if random.random() < Constants.MUTATION_CHANCE_FOR_EACH_SPACE:
                current_bitmap[i][j] = random.choice(ALLOWED_VALUES)
    
    return individual

toolbox.register("mutate", custom_mutation)

toolbox.register("select", tools.selTournament, tournsize=3)

toolbox.register("evaluate", fitness_function)

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

if __name__ == "__main__":

    init_room_path = "Bitmaps/InitRooms/"

    #Turn InitRooms.xml into InitRooms/*.bmp
    xml_file_path = 'Rooms/InitRooms.xml'
    convert_xml_to_bitmap(xml_file_path, init_room_path)

    #Get Bitmaps of InitRooms
    bitmap = TBoI_Bitmap()
    initial_individuals = []
    for rooms in os.listdir(init_room_path):
        bitmap = Image.open(init_room_path+rooms)
        width,height = bitmap.size
        pixel_data = []
        for i in range(height):
            row = []
            for j in range(width):
                pixel = bitmap.getpixel((j,i))
                row.append(pixel)
            pixel_data.append(row)
        initial_individual = []
        initial_individual.append(copy.deepcopy(pixel_data))
        initial_individual.append(copy.deepcopy(pixel_data))
        initial_individuals.append(initial_individual)
    
    population = toolbox.population()

    #Extend Population with InitRooms
    population.extend(toolbox.individual(data)
                      for data
                      in initial_individuals)
    
    index = 0
    for pop in population:
        # print("Start Room : \n")
        # for line in pop[0]:
        #     print(line)
        # print("End Room : \n")
        # for line in pop[1]:
        #     print(line)
        print(fitness_function(pop))
    pop0 = population[0]
    for line in pop0[0]:
        print(line)
    print("End Room : \n")
    for line in pop0[1]:
        print(line)
    custom_mutation(pop0)
    print("Start Room : \n")
    for line in pop0[0]:
        print(line)
    print("End Room : \n")
    for line in pop0[1]:
        print(line)
    index = 0
        
    
