MutationModel = RegisterMod("MutationModel",1)
local game = Game()
local SaveState = {}

local BITMAP_VALUES = {
    WALL = 0,
    DOOR = 1,
    FREE_SPACE = 2,
    STONE = 3,
    PIT = 4,
    BLOCK = 5,
    ENTITY = 6,
    PICKUP = 7,
    MACHINE = 8,
    FIRE = 9,
    POOP = 10,
    SPIKE = 11
}

local function GetBitmapValueFromGridEntity(gridEntity)
    if not gridEntity then
        return BITMAP_VALUES.FREE_SPACE
    end
    
    local gridType = gridEntity:GetType()
    
    if gridType == GridEntityType.GRID_WALL then
        return BITMAP_VALUES.WALL
    elseif gridType == GridEntityType.GRID_DOOR then
        return BITMAP_VALUES.DOOR
    elseif gridType == GridEntityType.GRID_ROCK or gridType == GridEntityType.GRID_ROCKT or 
           gridType == GridEntityType.GRID_ROCK_BOMB or gridType == GridEntityType.GRID_ROCK_ALT then
            print("GridType = " .. gridEntity:GetType())
        return BITMAP_VALUES.STONE
    elseif gridType == GridEntityType.GRID_PIT then
        return BITMAP_VALUES.PIT
    elseif gridType == GridEntityType.GRID_BLOCK then
        return BITMAP_VALUES.BLOCK
    elseif gridType == GridEntityType.GRID_FIREPLACE then
        return BITMAP_VALUES.FIRE
    elseif gridType == GridEntityType.GRID_POOP then
        return BITMAP_VALUES.POOP
    elseif gridType == GridEntityType.GRID_SPIKES or gridType == GridEntityType.GRID_SPIKES_ONOFF then
        return BITMAP_VALUES.SPIKE
    else
        return BITMAP_VALUES.FREE_SPACE
    end
end

local function CreateRoomBitmapString(room)
    local roomWidth = room:GetGridWidth()
    local roomHeight = room:GetGridHeight()
    local bitmapString = ""
    local roomEntities = room:GetEntities()

    for y = 0, roomHeight - 1 do
        for x = 0, roomWidth - 1 do
            local gridIndex = y * roomWidth + x
            local gridEntity = room:GetGridEntity(gridIndex)
            local bitmapValue = GetBitmapValueFromGridEntity(gridEntity)

            if roomEntities then
                for i = 0, roomEntities.Size - 1 do
                    local entity = roomEntities:Get(i)
                    if entity then
                        local entityGridPos = room:GetGridIndex(entity.Position)
                        if entityGridPos == gridIndex then
                            value = entity.Type
                            if value >= 10 and value <= 900 then
                                bitmapValue = BITMAP_VALUES.ENTITY
                            end
                        end
                    end
                end
            end
            
            bitmapString = bitmapString .. tostring(bitmapValue)
            
            if x < roomWidth - 1 then
                bitmapString = bitmapString .. ","
            end
        end
        
        if y < roomHeight - 1 then
            bitmapString = bitmapString .. "\n"
        end
    end
    
    return bitmapString
end

MutationModel:AddCallback(ModCallbacks.MC_POST_NEW_ROOM, function()
    local room = Game():GetRoom()

    local bitmapString = CreateRoomBitmapString(room)

    local count = 0;
    MutationModel:SaveData(bitmapString)
end)