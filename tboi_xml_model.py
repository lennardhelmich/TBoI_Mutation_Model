import xml.etree.ElementTree as ET
from xml_to_bitmap_converter import handle_entity_values
from xml_to_bitmap_converter import EntityType
import os

class Room:
    def __init__(self, variant, name, room_type, subtype, shape, width, height, difficulty, weight):
        self.variant = variant
        self.name = name
        self.type = room_type
        self.subtype = subtype
        self.shape = shape
        self.width = width
        self.height = height
        self.difficulty = difficulty
        self.weight = weight
        self.doors = []
        self.spawns = []

    def add_door(self, door):
        self.doors.append(door)

    def add_spawn(self, spawn):
        self.spawns.append(spawn)
    
    def delete_spawn(self, spawn):
        self.spawns.remove(spawn)
        
    def get_spawns_by_coordinates(self, x, y):
        return [spawn for spawn in self.spawns if spawn.x == x and spawn.y == y]
    
    def get_spawn_of_first_occurence_of_entity(self, entity):
        for spawn in self.spawns():
            entityType = spawn.entities[0].type
            if(handle_entity_values(entityType) == entity):
                return spawn
        
        newSpawn = Spawn(0,0)

        if(entity == EntityType.PICKUP):
            newEntity = Entity(5,0,0,1.0)
            newSpawn.entities.append(newEntity)
        elif(entity == EntityType.PIT):
            newEntity = Entity(3000,0,0,1.0)
            newSpawn.entities.append(newEntity)
        elif(entity == EntityType.FIRE):
            newEntity = Entity(33,0,0,1.0)
            newSpawn.entities.append(newEntity)
        elif(entity == EntityType.MACHINE):
            newEntity = Entity(6,0,0,1.0)
            newSpawn.entities.append(newEntity)
        elif(entity == EntityType.BLOCK):
            newEntity = Entity(1900,0,0,1.0)
            newSpawn.entities.append(newEntity)
        elif(entity == EntityType.ENTITY):
            newEntity = Entity(10,0,0,1.0)
            newSpawn.entities.append(newEntity)
        elif(entity == EntityType.STONE):
            newEntity = Entity(1000,0,0,1.0)
            newSpawn.entities.append(newEntity)
        elif(entity == EntityType.POOP):
            newEntity = Entity(1490,0,0,1.0)
            newSpawn.entities.append(newEntity)
        elif(entity == EntityType.SPIKE):
            newEntity = Entity(1930,0,0,1.0)
            newSpawn.entities.append(newEntity)
        elif(entity == EntityType.FREE_SPACE):
            return None
        
        return newSpawn




class Door:
    def __init__(self, exists, x, y):
        self.exists = exists == 'True'
        self.x = int(x)
        self.y = int(y)

class Spawn:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)
        self.entities = []

    def add_entity(self, entity):
        self.entities.append(entity)

class Entity:
    def __init__(self, entity_type, variant, subtype, weight):
        self.type = int(entity_type)
        self.variant = int(variant)
        self.subtype = int(subtype)
        self.weight = float(weight)

class XMLRoomParser:
    def __init__(self, xml_file=None):
        if(xml_file is not None):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            self.rooms = []
            
        # Parse all rooms in the XML file.
            for room_elem in root.findall('room'):
                room_variant    	= room_elem.get('variant')
                room_name        	= room_elem.get('name')
                room_type        	= room_elem.get('type')
                room_subtype     	= room_elem.get('subtype')
                room_shape        	= room_elem.get('shape')
                room_width        	= room_elem.get('width')
                room_height      	= room_elem.get('height')
                room_difficulty  	= room_elem.get('difficulty')
                room_weight      	= room_elem.get('weight')

                # Create a Room object.
                current_room = Room(room_variant,
                                        room_name,
                                        room_type,
                                        room_subtype,
                                        room_shape,
                                        room_width,
                                        room_height,
                                        room_difficulty,
                                        room_weight)

            # Parse doors.
                for door_elem in room_elem.findall('door'):
                    door_exists = door_elem.get('exists')
                    door_x = door_elem.get('x')
                    door_y = door_elem.get('y')

                    # Create a Door object and add it to the current_room.
                    current_door = Door(door_exists,
                                    door_x,
                                    door_y)
                    current_room.add_door(current_door)

            # Parse spawns.
                for spawn_elem in room_elem.findall('spawn'):
                    spawn_x = spawn_elem.get('x')
                    spawn_y = spawn_elem.get('y')

                # Create a Spawn object and add it to the current_room.
                    current_spawn = Spawn(spawn_x, spawn_y)

                # Parse entities within the spawn element.
                    for entity_elem in spawn_elem.findall('entity'):
                        entity_type = entity_elem.get('type')
                        entity_variant = entity_elem.get('variant')
                        entity_subtype = entity_elem.get('subtype') 
                        entity_weight = entity_elem.get("weight")
                        
                        # Create an Entity object and add it to the current_spawn.
                        current_entity 	= Entity(entity_type,
                                                entity_variant,
                                                entity_subtype,
                                                entity_weight) 
                        current_spawn.add_entity(current_entity) 

                # Add the spawn to the current_room after processing its entities.
                    current_room.add_spawn(current_spawn)
                
                self.rooms.append(current_room)
        else:
            self.rooms=[]

        
    def save_as_xaml(self, xaml_file_path):
        os.makedirs(os.path.dirname(xaml_file_path), exist_ok=True)
        with open(xaml_file_path , 'w') as f: 
                f.write('<?xml version="1.0" encoding="utf-8"?>\n')
                f.write('<rooms>\n')
                
                for room in self.rooms:
                    f.write(f'  <room variant="{room.variant}" name="{room.name}" type="{room.type}" '
                            f'subtype="{room.subtype}" shape="{room.shape}" '
                            f'width="{room.width}" height="{room.height}" '
                            f'difficulty="{room.difficulty}" weight="{room.weight}">\n')

                    for door in room.doors:
                        f.write(f'    <door exists="{door.exists}" x="{door.x}" y="{door.y}"/>\n')

                    for spawn in room.spawns:
                        f.write(f'    <spawn x="{spawn.x}" y="{spawn.y}">\n')
                        for entity in spawn.entities:
                            f.write(f'      <entity type="{entity.type}" variant="{entity.variant}" '
                                    f'subtype="{entity.subtype}" weight="{entity.weight}"/>\n')
                        f.write('    </spawn>\n')

                    f.write('  </room>\n')
                    
                f.write('</rooms>')

# Beispiel für die Verwendung der Klasse:    
if __name__ == "__main__":
    parser = XMLRoomParser("Rooms/FirstInputRoom.xml")
    parser.rooms[0].spawns.remove(parser.rooms[0].spawns[0])
    parser.save_as_xaml("XAMLTEST/1/Test.xml")