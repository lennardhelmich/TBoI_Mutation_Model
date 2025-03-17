from deap import base, creator, tools
import random
import copy
from fitness_function import Fitness_Function
from tboi_bitmap import TBoI_Bitmap

class TBoI_Room_Mutation:

    def make_fitness_function(self):
        def fitness_function(individual):
            fitness = Fitness_Function(startBitmap=self.startBitmap, resultBitmap=individual)
            fitness.calc_fitness_function()
            return fitness.functionValue,
        
        return fitness_function
    
    
    def make_mutate_function(self):
        def mutate_pixel(individual):
            height = len(individual)
            width = len(individual[0])
            
            i = random.randint(1, height - 2)
            j = random.randint(1, width - 2)

            individual[i][j] = random.choice(self.ALLOWED_VALUES)

        return mutate_pixel
    
    def make_mate_function(self):
        def mate_bitmaps(ind1, ind2):
            crossover_point = random.randint(1, min(len(ind1), len(ind2)) - 1) 
            ind1[:crossover_point], ind2[:crossover_point] = ind2[:crossover_point], ind1[:crossover_point]
        
        return mate_bitmaps
    
    def __init__(self, startBitmap):
        self.startBitmap = startBitmap

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        self.toolbox.register("individual", tools.initIterate, creator.Individual, lambda: [copy.deepcopy(row) for row in self.startBitmap])
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.make_fitness_function())
        self.toolbox.register("mutate", self.make_mutate_function())
        self.toolbox.register("mate", self.make_mate_function())
        self.toolbox.register("select", tools.selBest)

        bitmap = TBoI_Bitmap()
        self.ALLOWED_VALUES = bitmap.allowed_room_entities()
    
    def calculate_mutations(self, NGEN, CXPB, MUTPB, popSize, numElites):
        population = self.toolbox.population(n=popSize)

        for gen in range(NGEN):
            fits = list(map(self.toolbox.evaluate, population))

            for ind, fit in zip(population, fits):
                ind.fitness.values = fit

            elites = self.toolbox.select(population, numElites)
            
            offspring = [copy.deepcopy(ind) for ind in elites]  # Start offspring with elites
            
            while len(offspring) < len(population):  # Fill up offspring till full population size
                parent1, parent2 = random.sample(elites, 2)  # Randomly select two parents from elites
                
                if random.random() < CXPB:
                    self.toolbox.mate(parent1, parent2)
                    del parent1.fitness.values  
                    del parent2.fitness.values  

                offspring.append(parent1)
                offspring.append(parent2)

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

        return self.toolbox.select(population, numElites)  # Optionally return final population or best individual

   


    


    