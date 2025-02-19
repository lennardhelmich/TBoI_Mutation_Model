from xml_to_bitmap_converter import convert_xml_to_bitmap
from PIL import Image
import os
from fitness_function import Fitness_Function

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
    path_finding_test()