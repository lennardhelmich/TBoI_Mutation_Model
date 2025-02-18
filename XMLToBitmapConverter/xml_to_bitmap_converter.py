import xml.etree.ElementTree as ET
import math
import os
from PIL import Image
from enum import Enum

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

# Function to evenly assign pixel values with count of different entities
def get_pixel_value_with_entity_id(entity_id):
    entity_count = len(EntityType) - 1
    steps = 255 / entity_count
    return math.floor(entity_id.value*steps)

# Function to parse the correct simplified EntityType out of TBoI entity_types
def handle_entity_values(value):
    if value == 0:
        return EntityType.FREE_SPACE
    elif value == 5:
        return EntityType.PICKUP
    elif value == 3000:
        return EntityType.PIT
    elif value == 33:
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
        print("The following entity_type is not defined in BitmapValues : {value}")

#Function to convert a xml with Room Layout Data from basement_renovator into a 8bit bitmap 
def convert_xml_to_bitmap(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        index = 0
        #Iterate through all defined Rooms from XML
        for room in root.findall('room'):

            #create Bitmap
            width, height = 15,9
            bitmap_image = Image.new('L', (width, height), 0)

            #Initialize the bitmap of the room with free Spaces
            room_width, room_height = 13,7
            for width_index in range(room_width):
                for height_index in range(room_height):
                    bitmap_image.putpixel((width_index+1,height_index+1),get_pixel_value_with_entity_id(EntityType.FREE_SPACE))

            # Handle existing doors
            for door in room.findall('door'):
                exists = door.get('exists')
                x = int(door.get('x')) + 1
                y = int(door.get('y')) + 1
                if(exists):
                    bitmap_image.putpixel((x,y),get_pixel_value_with_entity_id(EntityType.DOOR))

            # Handle every single entity
            for spawn in room.findall('spawn'):
                x = int(spawn.get('x')) + 1
                y = int(spawn.get('y')) + 1
                entity = spawn.find('entity')
                if entity is not None:
                    entity_type = int(entity.get('type'))
                    bitmap_image.putpixel((x,y),get_pixel_value_with_entity_id(handle_entity_values(entity_type)))

            # Save bitmap of current room 
            directory = "Bitmaps"
            file_path = os.path.join(directory, f"bitmap_{index}.bmp")
            bitmap_image.save(file_path)

            # Increase index for bitmap count
            index += 1

    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
    except FileNotFoundError:
        print("The specified file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    xml_file_path = 'Rooms/SecondInputRooms.xml'
    convert_xml_to_bitmap(xml_file_path)