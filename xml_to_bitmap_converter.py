import xml.etree.ElementTree as ET
import math
import os
from tboi_bitmap import EntityType 
from tboi_bitmap import TBoI_Bitmap
from PIL import Image
from enum import Enum
from tboi_xml_model import XMLRoomParser
import copy

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

#Function to convert a xml with Room Layout Data from basement_renovator into a 8bit bitmap 
def convert_xml_to_bitmap(file_path, save_directory):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        index = 0
        #Iterate through all defined Rooms from XML
        for room in root.findall('room'):

            #create Bitmap
            tboi_bitmap = TBoI_Bitmap()

            #Initialize the bitmap of the room with free Spaces
            room_width, room_height = 13,7
            for width_index in range(room_width):
                for height_index in range(room_height):
                    tboi_bitmap.set_pixel_with_entity_id(width_index+1, height_index+1, EntityType.FREE_SPACE)
            
            # Handle existing doors
            for door in room.findall('door'):
                exists = door.get('exists')
                x = int(door.get('x')) + 1
                y = int(door.get('y')) + 1
                if(exists):
                    tboi_bitmap.set_pixel_with_entity_id(x, y, EntityType.DOOR)

            # Handle every single entity
            for spawn in room.findall('spawn'):
                x = int(spawn.get('x')) + 1
                y = int(spawn.get('y')) + 1
                entity = spawn.find('entity')
                if entity is not None:
                    entity_type = int(entity.get('type'))
                    tboi_bitmap.set_pixel_with_entity_id(x,y,handle_entity_values(entity_type))

            # Save bitmap of current room 
            tboi_bitmap.save_bitmap_in_folder(index, save_directory)

            # Increase index for bitmap count
            index += 1

    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
    except FileNotFoundError:
        print("The specified file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def convert_generated_bitmaps_to_xml(inputXml_path):
    try:
        parser = XMLRoomParser(inputXml_path)
        tboi_bitmap = TBoI_Bitmap()
        index = 0

        for room in parser.rooms:
            bitmapPath = "Bitmaps/InputRooms/bitmap_" + str(index) + ".bmp"
            startBitmap = Image.open(bitmapPath)
            mutationParser = XMLRoomParser()
            mutationParser.rooms.append(copy.deepcopy(room))
            mutationFolder = "Bitmaps/Mutations/bitmap_" + str(index) + "/"
            width,height = startBitmap.size

            for file in os.listdir(mutationFolder):
                mutationBitmap = Image.open(file)
                newRoom = copy.deepcopy(room)
                changesList = []
                for x in range(width):
                    for y in range(height):
                        newPixelValue = mutationBitmap.getpixel(x,y)
                        if(newPixelValue != startBitmap.getpixel(x,y)):
                            changesList.append(((x,y),tboi_bitmap.get_entity_id_with_pixel_value(newPixelValue)))
                
                for change in changesList:
                    entity = room.get_spawns_by_coordinates(change[0][0],change[0][1])
                    newValue = change[1]
                    newSpawn = room.get_spawn_of_first_occurence_of_entity(newValue)
                    newSpawn.x = change[0][0]
                    newSpawn.y = change[0][1]
                    if entity.count == 0:
                        newRoom.add_spawn(newSpawn)
                    else:
                        newRoom.delete_spawn(entity[0])
                        newRoom.add_spawn(newSpawn)
                
                mutationParser.rooms.append(newRoom)
            
            outputPath = "OutputXmls/Raum" + str(index) + ".xml"
            mutationParser.save_as_xaml(outputPath)

    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
    except FileNotFoundError:
        print("The specified file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    

# Example usage
if __name__ == "__main__":
    xml_file_path = 'Rooms/SecondInputRooms.xml'
    convert_xml_to_bitmap(xml_file_path, "Bitmaps")