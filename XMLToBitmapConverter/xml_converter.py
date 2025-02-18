import xml.etree.ElementTree as ET
import math
import os
from PIL import Image
from enum import Enum

class BitmapValues(Enum):
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

def get_pixel_value_with_entity_id(entity_id):
    entity_count = len(BitmapValues) - 1
    print(f"Entity Count : {entity_count} \n")
    steps = 255 / entity_count
    print(f"Bit Value Steps : {steps} \n")
    return math.floor(entity_id.value*steps)

def handle_entity_values(value):
    if value == 0:
        return BitmapValues.FREE_SPACE
    elif value == 5:
        return BitmapValues.PICKUP
    elif value == 3000:
        return BitmapValues.PIT
    elif value == 33:
        return BitmapValues.FIRE
    elif value == 6:
        return BitmapValues.MACHINE
    elif value == 1900:
        return BitmapValues.BLOCK
    elif 10 <= value <= 900:
        return BitmapValues.ENTITY
    elif 1000 <= value <= 1100:
        return BitmapValues.STONE
    elif 1490 <= value <= 1510:
        return BitmapValues.POOP
    elif 1930 <= value <= 2000:
        return BitmapValues.SPIKE
    else:
        print("The following entity_type is not defined in BitmapValues : {value}")

def read_xml_file(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Print the root element
        print(f"Root element: {root.tag}")
        index = 0
        for room in root.findall('room'):
            print(f"\nRoom Attributes:")
            for attr in room.attrib:
                print(f"  {attr}: {room.attrib[attr]}")

            #create Bitmap
            width, height = 15,9
            bitmap_image = Image.new('L', (width, height), 0)

            room_width, room_height = 13,7
            for width_index in range(room_width):
                for height_index in range(room_height):
                    bitmap_image.putpixel((width_index+1,height_index+1),get_pixel_value_with_entity_id(BitmapValues.FREE_SPACE))

            # Handle doors
            print("\nDoors:")
            for door in room.findall('door'):
                exists = door.get('exists')
                x = int(door.get('x')) + 1
                y = int(door.get('y')) + 1
                if(exists):
                    bitmap_image.putpixel((x,y),get_pixel_value_with_entity_id(BitmapValues.DOOR))
                print(f"  Door exists: {exists}, Position: ({x}, {y})")

            # Handle spawns
            print("\nSpawns:")
            for spawn in room.findall('spawn'):
                x = int(spawn.get('x')) + 1
                y = int(spawn.get('y')) + 1
                entity = spawn.find('entity')
                if entity is not None:
                    entity_type = int(entity.get('type'))
                    variant = entity.get('variant')
                    subtype = entity.get('subtype')
                    weight = entity.get('weight')
                    bitmap_image.putpixel((x,y),get_pixel_value_with_entity_id(handle_entity_values(entity_type)))
                    print(f"  Spawn Position: ({x}, {y}), Entity Type: {entity_type}, Variant: {variant}, Subtype: {subtype}, Weight: {weight}")

            directory = "Bitmaps"
            file_path = os.path.join(directory, f"bitmap_{index}.bmp")
            bitmap_image.save(file_path)
            index += 1

    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
    except FileNotFoundError:
        print("The specified file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    xml_file_path = 'Rooms/SecondInputRooms.xml'  # Replace with your XML file path
    read_xml_file(xml_file_path)
    # print(get_pixel_value_with_entity_id(BitmapValues.WALL))
    # print(get_pixel_value_with_entity_id(BitmapValues.SPIKE))