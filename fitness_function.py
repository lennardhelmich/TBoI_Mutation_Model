from tboi_bitmap import TBoI_Bitmap
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
    

if __name__ == "__main__":
    path = "Bitmaps/bitmap_5.bmp"
    bitmap = Image.open(path)
    fitness = Fitness_Function(bitmap, bitmap)
    if(fitness.check_every_traversability()):
        print("In Bitmap : " + path + " everything is reachable")
    else:
        print("In Bitmap : " + path + " a non-reachable property was found.")


