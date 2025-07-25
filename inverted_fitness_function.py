from tboi_bitmap import TBoI_Bitmap
import math
from tboi_bitmap import EntityType
from PIL import Image
from constants import Constants
import numpy as np

class Inverted_Fitness_Function:
    def __init__(self, startBitmap, resultBitmap):

        start_tboi_bitmap = TBoI_Bitmap()
        start_tboi_bitmap.bitmap = Image.fromarray(np.array(startBitmap,dtype=np.uint8), mode="L")
        start_tboi_bitmap.create_graph_out_of_bitmap()

        result_tboi_bitmap = TBoI_Bitmap()
        result_tboi_bitmap.bitmap = Image.fromarray(np.array(resultBitmap,dtype=np.uint8), mode="L")
        result_tboi_bitmap.create_graph_out_of_bitmap()

        self.startBitmap = start_tboi_bitmap
        self.resultBitmap = result_tboi_bitmap
        self.functionValue = []

    def check_every_traversability(self):
        doors = []
        enemies = []
        for x in range(self.resultBitmap.width):
            for y in range(self.resultBitmap.height):
                pixelValue = self.resultBitmap.bitmap.getpixel((x, y))
                entity_id = self.resultBitmap.get_entity_id_with_pixel_value(pixelValue)
                if entity_id == EntityType.DOOR:
                    doors.append((x, y))
                elif entity_id == EntityType.ENTITY:
                    enemies.append((x, y))

        if not doors:
            return False

        firstDoor = doors[0]
        targets = set(doors + enemies)

        # Use the optimized is_path_existent method
        return self.resultBitmap.is_path_existent(firstDoor, targets)
    
    def check_spawn(self):
        pixelValue1 = self.resultBitmap.bitmap.getpixel((7, 1))
        entity_id1 = self.resultBitmap.get_entity_id_with_pixel_value(pixelValue1)
        if(entity_id1 != EntityType.FREE_SPACE):
            return False
        pixelValue2 = self.resultBitmap.bitmap.getpixel((7, 7))
        entity_id2 = self.resultBitmap.get_entity_id_with_pixel_value(pixelValue2)
        if(entity_id2 != EntityType.FREE_SPACE):
            return False
        pixelValue3 = self.resultBitmap.bitmap.getpixel((1, 4))
        entity_id3 = self.resultBitmap.get_entity_id_with_pixel_value(pixelValue3)
        if(entity_id3 != EntityType.FREE_SPACE):
            return False
        pixelValue4 = self.resultBitmap.bitmap.getpixel((13, 4))
        entity_id4 = self.resultBitmap.get_entity_id_with_pixel_value(pixelValue4)
        if(entity_id4 != EntityType.FREE_SPACE):
            return False
        return True
    
    def inverted_pixel_variation_score(self):
        bitmap = self.resultBitmap
        width, height = bitmap.bitmap.size
        total_matches = 0
        total_comparisons = 0

        # Iterate over all pixels (excluding border pixels)
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                current_pixel = bitmap.bitmap.getpixel((x, y))

                neighbors = bitmap.get_all_neighbors((x, y))

                for neighbor in neighbors:
                    if current_pixel == bitmap.bitmap.getpixel(neighbor):
                        total_matches += 1
                    total_comparisons += 1

        normalized_score = total_matches / total_comparisons

        return 1 - normalized_score
    
    def inverted_vertical_symmetric_score(self, width, height, bitmap):
        total_compared_pixels = 0
        vertical_matches = 0

        #Vertical symmetry
        for i in range(1, height-1):
            for j in range(1, math.ceil(width/2)):
                total_compared_pixels += 1
                mirror_j_value = width-1-j
                if(bitmap.getpixel((j,i)) == bitmap.getpixel((mirror_j_value,i))):
                    vertical_matches += 1

        return 1 - (vertical_matches/total_compared_pixels)
    
    def inverted_horizontal_symmetric_score(self,width,height,bitmap):
        total_compared_pixels = 0
        horizontal_matches = 0

        #Horizontal symmetry
        for i in range(1, math.ceil(height/2)):
            for j in range(1, width-1):
                total_compared_pixels+=1
                mirror_i_value = height-1-i
                if(bitmap.getpixel((j,i)) == bitmap.getpixel((j,mirror_i_value))):
                    horizontal_matches += 1

        return 1 - (horizontal_matches/total_compared_pixels)

    def inverted_central_symmetric_score(self,width,height,bitmap):
        total_compared_pixels = 0
        central_matches = 0

        #Central symmetry
        for i in range(1,height-1):
            for j in range(1,width-1):
                total_compared_pixels+=1
                i2 = (height-1)-i
                j2 = (width-1)-j
                if(bitmap.getpixel((j,i)) == bitmap.getpixel((j2,i2))):
                    central_matches += 1
        
        return 1 - (central_matches/total_compared_pixels)
                

    def inverted_symmetry_score(self):
        bitmap = self.resultBitmap.bitmap
        width, height = bitmap.size
        vertical_score = self.inverted_vertical_symmetric_score(width, height, bitmap)
        horizontal_score = self.inverted_horizontal_symmetric_score(width, height, bitmap)
        central_score = self.inverted_central_symmetric_score(width, height, bitmap)
        return (vertical_score + horizontal_score + central_score)/3


    # def balance_freespace_and_entities(self):
    #     totalCount = 0
    #     entityCount = 0
    #     bitmap = self.resultBitmap.bitmap
    #     for i in range(1,bitmap.height-1):
    #         for j in range(1, bitmap.width-1):
    #             pixel = bitmap.getpixel((j,i))
    #             entity = self.resultBitmap.get_entity_id_with_pixel_value(pixel)
    #             totalCount+=1
    #             if(entity == EntityType.BLOCK or entity == EntityType.STONE or entity == EntityType.PIT or entity == EntityType.FIRE or entity == EntityType.POOP or entity == EntityType.SPIKE):
    #                 entityCount+=1

    #     percent = (entityCount/totalCount) * 100
    #     deviation_of_desired_value = abs(Constants.DESIRED_FREESPACE_ENTITY_RATIO_PERCENT - percent)
    #     if(deviation_of_desired_value > Constants.DESIRED_FREESPACE_ENTITY_RATIO_PERCENT):
    #         return 0
    #     else :
    #         return (1-(deviation_of_desired_value/Constants.DESIRED_FREESPACE_ENTITY_RATIO_PERCENT))

    def enemy_difference_value(self):
        enemy_value = self.resultBitmap.get_pixel_value_with_entity_id(EntityType.ENTITY)
        enemies_now = list(self.startBitmap.bitmap.getdata()).count(enemy_value)
        enemies_prev = list(self.resultBitmap.bitmap.getdata()).count(enemy_value)
        difference = abs(enemies_now-enemies_prev)
        if(difference <= Constants.MAX_NUMBER_FREE_ENEMY_CHANGES):
            return 1
        else:
            return max(0, 1-(Constants.VALUE_REDUCTION_PER_ENEMY*(difference-Constants.MAX_NUMBER_FREE_ENEMY_CHANGES)))

    def bitmap_changes(self):
        list_prev = list(self.startBitmap.bitmap.getdata())
        list_now = list(self.resultBitmap.bitmap.getdata())
        total_count = (self.resultBitmap.bitmap.width-2) * (self.resultBitmap.bitmap.height-2)
        difference_in_pixels = sum(1 for p1, p2 in zip(list_prev, list_now) if p1 != p2)
        difference_percent = (difference_in_pixels/total_count)*100
        if(difference_percent > (Constants.TARGETED_BITMAP_DIFFERENCE*2)):
            return 0
        else:
            return (1-(abs(difference_percent-Constants.TARGETED_BITMAP_DIFFERENCE)*(1/Constants.TARGETED_BITMAP_DIFFERENCE)))

    def calc_fitness_function(self):
        value = 0
        total_weight = Constants.FITNESS_WEIGHT_CHANGES + Constants.FITNESS_WEIGHT_ENEMIES + Constants.FITNESS_WEIGHT_SYMMETRY + Constants.FITNESS_WEIGHT_VARIATION
        if(not self.check_every_traversability() ):
            self.functionValue = [0, 0, 0, 0, 0]
        elif(not self.check_spawn()):
            self.functionValue = [0, 0, 0, 0, 0]
        else:
            bitmap_changes = self.bitmap_changes()
            enemies = self.enemy_difference_value()
            symmetry = self.inverted_symmetry_score()
            variation = self.inverted_pixel_variation_score()
            value = (
                + Constants.FITNESS_WEIGHT_CHANGES * bitmap_changes
                + Constants.FITNESS_WEIGHT_ENEMIES * enemies
                + Constants.FITNESS_WEIGHT_SYMMETRY * symmetry
                + Constants.FITNESS_WEIGHT_VARIATION * variation
            )
            standardized_value = value / total_weight
            self.functionValue = [
                standardized_value,
                bitmap_changes,
                enemies,
                symmetry,
                variation]


if __name__ == "__main__":
    path = "Bitmaps/InitRooms/bitmap_32.bmp"
    bitmap = Image.open(path)
    fitness = Fitness_Function(bitmap, bitmap)
    fitness.calc_fitness_function()
    print(fitness.functionValue)


