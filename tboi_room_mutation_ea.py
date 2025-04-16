from deap import base, creator, tools
import random
import copy
from fitness_function import Fitness_Function
from tboi_bitmap import TBoI_Bitmap
from constants import Constants

def symmetric_horizontal(height, i, j):
    return height - 1 - i, j
    
def symmetric_vertical(width, i, j):
    return i, width - 1 - j

def symmetric_center(width, height, i, j):
    return height - 1 - i, width - 1 - j

def rectangle_i_j(i,j):
    new_i = i + 1 if i < 4 else i - 1
    new_j = j + 1 if j < 7 else j - 1
    return new_i, new_j
        

class TBoI_Room_Mutation:

    def make_select_function(self):
        def select_best_unique(individuals, k):
            seen_fitness = set()
            selected = []

            # Sort individuals by fitness descending (best first)
            sorted_inds = sorted(individuals, key=lambda ind: ind.fitness.values[0], reverse=True)

            for ind in sorted_inds:
                fitness_val = ind.fitness.values[0]
                if fitness_val not in seen_fitness:
                    seen_fitness.add(fitness_val)
                    selected.append(ind)
                if len(selected) == k:
                    break
        
            while len(selected) < k:
                selected.append(sorted_inds[0])
            
            return selected

        return select_best_unique

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
            i, j = random.randint(1, height - 2), random.randint(1, width - 2)

            while (i, j) in [(1, 7), (4, 1), (4, 13), (7, 7)]:
                i, j = random.randint(1, height - 2), random.randint(1, width - 2)

            selected_mutation = random.choices(self.AVAILABLE_MUTATIONS, weights=self.MUT_PROB, k=1)[0]
            new_ind = random.choices(self.ALLOWED_VALUES, weights=self.TILE_PROB, k=1)[0]

            if selected_mutation == 1:
                individual[i][j] = new_ind
            elif selected_mutation == 2:
                i_h, j_h = symmetric_horizontal(height, i, j)
                individual[i][j] = individual[i_h][j_h] = new_ind
            elif selected_mutation == 3:
                i_v, j_v = symmetric_vertical(width, i, j)
                individual[i][j] = individual[i_v][j_v] = new_ind
            elif selected_mutation == 4:
                i_c, j_c = symmetric_center(width, height, i, j)
                individual[i][j] = individual[i_c][j_c] = new_ind
            elif selected_mutation == 5:
                new_i, new_j = rectangle_i_j(i, j)
                individual[i][j] = individual[i][new_j] = individual[new_i][j] = individual[new_i][new_j] = new_ind

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
        self.toolbox.register("select", self.make_select_function())

        bitmap = TBoI_Bitmap()
        self.ALLOWED_VALUES = bitmap.allowed_room_entities()

        #calculate probabilities out of constants
        sum = Constants.PROB_BLOCK + Constants.PROB_ENTITY + Constants.PROB_FIRE + Constants.PROB_FREE_SPACE + Constants.PROB_PIT + Constants.PROB_POOP + Constants.PROB_SPIKE
        self.TILE_PROB = []
        self.TILE_PROB.append((Constants.PROB_FREE_SPACE/sum))
        self.TILE_PROB.append((Constants.PROB_STONE/sum))
        self.TILE_PROB.append((Constants.PROB_PIT/sum))
        self.TILE_PROB.append((Constants.PROB_BLOCK/sum))
        self.TILE_PROB.append((Constants.PROB_ENTITY/sum))
        self.TILE_PROB.append((Constants.PROB_FIRE/sum))
        self.TILE_PROB.append((Constants.PROB_POOP/sum))
        self.TILE_PROB.append((Constants.PROB_SPIKE/sum))

        self.AVAILABLE_MUTATIONS = [1,2,3,4,5]
        self.MUT_PROB = [0.2,0.2,0.2,0.2,0.2]
    
    def calculate_mutations(self, NGEN, CXPB, MUTPB, popSize, numElites):
        population = self.toolbox.population(n=popSize)
        fits = list(map(self.toolbox.evaluate, population))

        for ind, fit in zip(population, fits):
            ind.fitness.values = fit

        for gen in range(NGEN):
            elites = self.toolbox.select(population, numElites)
            
            offspring = [copy.deepcopy(ind) for ind in elites]  # Start offspring with elites
            
            while len(offspring) < len(population):  # Fill up offspring till full population size
                parent1, parent2 = random.sample(elites, 2)  # Randomly select two parents from elites
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values  
                    del child2.fitness.values  

                offspring.append(child1)
                offspring.append(child2)

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fits = list(self.toolbox.map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fits):
                ind.fitness.values = fit

            population[:] = offspring

        return population  # Optionally return final population or best individual







