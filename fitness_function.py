from tboi_bitmap import TBoI_Bitmap
import math
from tboi_bitmap import EntityType
from PIL import Image

class Fitness_Function:
    def __init__(self, startBitmap, resultBitmap):

        start_tboi_bitmap = TBoI_Bitmap()
        start_tboi_bitmap.bitmap = startBitmap
        start_tboi_bitmap.create_graph_out_of_bitmap()

        result_tboi_bitmap = TBoI_Bitmap()
        result_tboi_bitmap.bitmap = resultBitmap
        result_tboi_bitmap.create_graph_out_of_bitmap()

        self.startBitmap = start_tboi_bitmap
        self.resultBitmap = result_tboi_bitmap
        self.functionValue = 0

    def check_every_traversability(self):
        doors = set()
        enemies = set()
        for x in range(self.resultBitmap.width):
            for y in range(self.resultBitmap.height):
                pixelValue = self.resultBitmap.bitmap.getpixel((x,y))
                entity_id = self.resultBitmap.get_entity_id_with_pixel_value(pixelValue)
                if(entity_id == EntityType.DOOR):
                    doors.add((x,y))
                if(entity_id == EntityType.ENTITY):
                    enemies.add((x,y))

        firstDoor = doors.pop()
        for remainingDoor in doors:
            if not self.resultBitmap.is_path_existent(firstDoor, remainingDoor):
                return False
        
        for enemy in enemies:
            if not self.resultBitmap.is_path_existent(firstDoor, enemy):
                return False
            
        return True
    
    def vertical_symmetric_score(self, width, height, bitmap):
        total_compared_pixels = 0
        vertical_matches = 0

        #Vertical symmetry
        for i in range(1, height-1):
            for j in range(1, math.ceil(width/2)):
                total_compared_pixels += 1
                mirror_j_value = width-1-j
                if(bitmap.getpixel((j,i)) == bitmap.getpixel((mirror_j_value,i))):
                    vertical_matches += 1

        return vertical_matches/total_compared_pixels
    
    def horizontal_symmetric_score(self,width,height,bitmap):
        total_compared_pixels = 0
        horizontal_matches = 0

        #Horizontal symmetry
        for i in range(1, math.ceil(height/2)):
            for j in range(1, width-1):
                total_compared_pixels+=1
                mirror_i_value = height-1-i
                if(bitmap.getpixel((j,i)) == bitmap.getpixel((j,mirror_i_value))):
                    horizontal_matches += 1

        return horizontal_matches/total_compared_pixels

    def central_symmetric_score(self,width,height,bitmap):
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
        
        return central_matches/total_compared_pixels
                

    def symmetry_score(self):
        bitmap = self.resultBitmap.bitmap
        width, height = bitmap.size
        vertical_score = self.vertical_symmetric_score(width, height, bitmap)
        horizontal_score = self.horizontal_symmetric_score(width, height, bitmap)
        central_score = self.central_symmetric_score(width, height, bitmap)
        print("Vertical score is : " + str(vertical_score) + "\n" + "Horizontal score is : " + str(horizontal_score) + "\n" + "Central score is : " + str(central_score))


    

if __name__ == "__main__":
    path = "Bitmaps/bitmap_5.bmp"
    bitmap = Image.open(path)
    fitness = Fitness_Function(bitmap, bitmap)
    if(fitness.check_every_traversability()):
        print("In Bitmap : " + path + " everything is reachable")
    else:
        print("In Bitmap : " + path + " a non-reachable property was found.")
    fitness.symmetry_score()


