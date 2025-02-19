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

class TBoI_Bitmap:

    def __init__(self, width=15, height=9):
        self.width = width
        self.height = height
        self.bitmap =  Image.new('L', (width, height), 0)
        self.pathFindingGraph = [[0 for _ in range(width)] for _ in range(height)]

# Function to evenly assign pixel values with count of different entities
    def hallo():
        return 0
    
    def get_pixel_value_with_entity_id(self, entity_id):
        entity_count = len(EntityType) - 1
        steps = 255 / entity_count
        return math.floor(entity_id.value*steps)
    
    def get_entity_id_with_pixel_value(self, pixel_value):
        entity_count = len(EntityType) - 1
        steps = 255 / entity_count
        return EntityType(round(pixel_value/steps))
    
    def set_pixel_with_entity_id(self, x, y, entity_id):
        self.bitmap.putpixel((x,y),self.get_pixel_value_with_entity_id(entity_id))

    def save_bitmap_in_folder(self, index):
        directory = "Bitmaps"
        file_path = os.path.join(directory, f"bitmap_{index}.bmp")
        self.bitmap.save(file_path)

    def create_graph_out_of_bitmap(self):
        for x in range(self.bitmap.width):
            for y in range(self.bitmap.height):
                bitmap_value = self.bitmap.getpixel((x,y))
                entity_id = self.get_entity_id_with_pixel_value(bitmap_value)
                hallo = "" + str(bitmap_value) + ""
                if(entity_id == EntityType.WALL or entity_id == EntityType.STONE or entity_id == EntityType.PIT or entity_id == EntityType.BLOCK or entity_id == EntityType.SPIKE):
                    self.pathFindingGraph[y][x] = 1

    def is_path_existent(self, start, end):
        if not self.is_within_bounds(start) or not self.is_within_bounds(end):
            print("No path exists because start/end is not within bounds.")
            return False
        
        if self.pathFindingGraph[start[1]][start[0]] == 1 or self.pathFindingGraph[end[1]][end[0]] == 1:
            print("No path exists because start/end is not traversable")
            return False
        
        queue = deque([start])
        visited = set()

        while queue:
            current = queue.popleft()
            if(current == end):
                return True
            visited.add(current)
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited and self.pathFindingGraph[neighbor[1]][neighbor[0]] == 0:
                    queue.append(neighbor)
        
        return False

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

    def is_within_bounds(self, position):
        x,y = position
        return 0 <= x < self.width and 0 <= y < self.height
    
if __name__ == "__main__":
    tboi_bitmap = TBoI_Bitmap()
    tboi_bitmap.bitmap = Image.open("Bitmaps/bitmap_5.bmp")
    tboi_bitmap.create_graph_out_of_bitmap()
    graph = tboi_bitmap.pathFindingGraph
    for x in range(tboi_bitmap.height):
        wer = ""
        for y in range(tboi_bitmap.width):
            wer += str(graph[x][y])
            wer += " "
        print(wer)