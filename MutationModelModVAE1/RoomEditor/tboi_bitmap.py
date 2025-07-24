from PIL import Image
from enum import Enum
from collections import deque
from PIL import Image
import math
import os

class EntityType(Enum):
    WALL = 0
    DOOR = 1
    FREE_SPACE = 2
    STONE = 3
    PIT = 4
    BLOCK = 5
    ENTITY = 6
    PICKUP = 7
    MACHINE = 8
    FIRE = 9
    POOP = 10
    SPIKE = 11


# Function to parse the correct simplified EntityType out of TBoI entity_types
def handle_entity_values(value):
    if value == 0:
        return EntityType.FREE_SPACE
    elif value == 5:
        return EntityType.PICKUP
    elif value == 3000:
        return EntityType.PIT
    elif value == 33 or value == 1400:
        return EntityType.FIRE
    elif value == 6:
        return EntityType.MACHINE
    elif value == 1900:
        return EntityType.BLOCK
    elif 10 <= value <= 900:
        return EntityType.ENTITY
    elif 1000 <= value <= 1100:
        return EntityType.STONE
    elif 1490 <= value <= 1510:
        return EntityType.POOP
    elif 1930 <= value <= 2000:
        return EntityType.SPIKE
    else:
        valueStr = str(value)
        print("The following entity_type is not defined in BitmapValues : " + valueStr)

class TBoI_Bitmap:

# TBoI_Bitmap Constructor with width and height pre-defined and initial bitmap and pathFindingGraph with 0s
    def __init__(self, width=15, height=9):
        self.width = width
        self.height = height
        self.bitmap =  Image.new('L', (width, height), 0)
        self.pathFindingGraph = [[0 for _ in range(width)] for _ in range(height)]


# Function to evenly assign pixel values with count of different entities
    def get_pixel_value_with_entity_id(self, entity_id):
        entity_count = len(EntityType) - 1
        steps = 255 / entity_count
        return math.floor(entity_id.value*steps)

# Function to get EntityType out of given pixel_value
    def get_entity_id_with_pixel_value(self, pixel_value):
        entity_count = len(EntityType) - 1
        steps = 255 / entity_count
        return EntityType(round(pixel_value/steps))

# Function to set a pixel on a given position with the correct Pixel Value of given entity_id
    def set_pixel_with_entity_id(self, x, y, entity_id):
        self.bitmap.putpixel((x,y),self.get_pixel_value_with_entity_id(entity_id))

# Function to save bitmap in folder for future uses
    def save_bitmap_in_folder(self, index, directory):
        file_path = os.path.join(directory, f"bitmap_{index}.bmp")
        self.bitmap.save(file_path)
    
    def save_mutation_in_folder_with_fitness(self, index, fitness, directory):
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, f"mutation_{index}_{fitness}.bmp")
        self.bitmap.save(file_path)

# Function to assign correct values for pathFindingGraph computed out of current bitmap
    def create_graph_out_of_bitmap(self):
        for x in range(self.bitmap.width):
            for y in range(self.bitmap.height):
                bitmap_value = self.bitmap.getpixel((x,y))
                entity_id = self.get_entity_id_with_pixel_value(bitmap_value)
                hallo = "" + str(bitmap_value) + ""
                if(entity_id == EntityType.WALL or entity_id == EntityType.STONE or entity_id == EntityType.PIT or entity_id == EntityType.BLOCK or entity_id == EntityType.SPIKE):
                    self.pathFindingGraph[y][x] = 1

# Path Finding Algorithm (Breadth-First Search (BFS)) for fitness function
    def is_path_existent(self, start, targets):
        visited = set()
        queue = deque([start])
        remaining_targets = set(targets)

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            if current in remaining_targets:
                remaining_targets.remove(current)
                if not remaining_targets:
                    return True

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited and self.pathFindingGraph[neighbor[1]][neighbor[0]] == 0:
                    queue.append(neighbor)

        return False

    def get_all_neighbors(self, position):
        x, y = position
        directions = [(0, -1), (0, 1),(-1, 0), (1, 0),(-1, -1), (-1, 1),(1, -1), (1, 1)]
        return [(x + dx, y + dy) for dx, dy in directions if self.is_within_bounds((x + dx, y + dy))]

    def get_neighbors(self, position):
        x, y = position
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        return [(x + dx, y + dy) for dx, dy in directions if self.is_within_bounds((x + dx, y + dy))]

# Get all current neighbors for path finding
    def get_neighbors(self, position):
        x, y = position
        neighbors = []

        directions = [(0,-1), (0, 1), (-1, 0), (1, 0)]

        for dx, dy in directions:
            new_x = x + dx
            new_y = y + dy
            if self.is_within_bounds((new_x, new_y)):
                neighbors.append((new_x,new_y))
        
        return neighbors

# Check if given position is in bounds of graph
    def is_within_bounds(self, position):
        x,y = position
        return 0 <= x < self.width and 0 <= y < self.height
    
# 
    def allowed_room_entities(self):
        returnList = []
        for entities in EntityType:
            if(entities != EntityType.DOOR and entities != EntityType.WALL and entities != EntityType.MACHINE and entities != EntityType.PICKUP):
                returnList.append(self.get_pixel_value_with_entity_id(entities))
        return returnList
    
if __name__ == "__main__":
    tboi_bitmap = TBoI_Bitmap()
    th = tboi_bitmap.allowed_room_entities()
    tboi_bitmap.create_graph_out_of_bitmap()
    graph = tboi_bitmap.pathFindingGraph
    for x in range(tboi_bitmap.height):
        wer = ""
        for y in range(tboi_bitmap.width):
            wer += str(graph[x][y])
            wer += " "
        print(wer)