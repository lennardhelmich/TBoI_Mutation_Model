MutationModel = RegisterMod("MutationModel",1)
local game = Game()
local level = game:GetLevel()
local SaveState = {}
-- local visitedRooms = {}
local roomModifications = {} -- Store modifications for each room

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
                            if value == 245 then
                                bitmapValue = BITMAP_VALUES.POOP
                            end
                            if value == 33 then
                                bitmapValue = BITMAP_VALUES.FIREPLACE
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

local function ProcessBitmapChange(room, grid_index, old_value, new_value, first_entity)
    local gridPosition = room:GetGridPosition(grid_index)
    local entities = room:GetEntities()
    local needNewSet = false
    if (old_value == 9 or old_value == 6) then
        if not (entities.Size == 0) then
            -- Collect entities to remove first, then remove them
            local entitiesToRemove = {}
            for i = 0, entities.Size - 1 do  -- Fixed bounds
                local entity = entities:Get(i)
                if entity and room:GetGridIndex(entity.Position) == grid_index then
                    table.insert(entitiesToRemove, entity)
                end
            end
            
            -- Now safely remove all collected entities
            for _, entity in ipairs(entitiesToRemove) do
                entity:Remove()
            end
        end
    elseif old_value == 10 then
        print("Removing Poop")
        local gridEntity = room:GetGridEntity(grid_index)
        if not gridEntity then
            return
        else
            gridEntity:Destroy(true)
            print(gridEntity:GetType())
        end
    else
        local gridEntity = room:GetGridEntity(grid_index)
        if gridEntity ~= nil then
            room:RemoveGridEntity(grid_index,0, false)
        end
    end
    
    if new_value == 6 then
        if first_entity == nil then
            Isaac.Spawn(EntityType.ENTITY_ATTACKFLY, 0, 0, gridPosition, Vector.Zero, nil)
        else
            Isaac.Spawn(first_entity.Type, first_entity.Variant, first_entity.SubType, gridPosition, Vector.Zero, nil)
        end
        
    else
        --print("Spawning new grid entity for value: " .. new_value)
        if new_value == BITMAP_VALUES.STONE then
            --print("Spawning ROCK at index: " .. grid_index)
            room:SpawnGridEntity(grid_index, GridEntityType.GRID_ROCK, 0, 0, 0)
        elseif new_value == BITMAP_VALUES.PIT then
            --print("Spawning PIT at index: " .. grid_index)
            room:SpawnGridEntity(grid_index, GridEntityType.GRID_PIT, 0, 0, 0)
        elseif new_value == BITMAP_VALUES.BLOCK then
            --print("Spawning BLOCK at index: " .. grid_index)
            room:SpawnGridEntity(grid_index, GridEntityType.GRID_PIT, 0, 0, 0)
        elseif new_value == BITMAP_VALUES.FIRE then
            --print("Spawning FIREPLACE entity at position: " .. tostring(gridPosition))
            Isaac.Spawn(EntityType.ENTITY_FIREPLACE, 0, 0, gridPosition, Vector.Zero, nil)
        elseif new_value == BITMAP_VALUES.POOP then
            --print("Spawning POOP entity at position: " .. tostring(gridPosition))
            Isaac.Spawn(EntityType.ENTITY_POOP, 0, 0, gridPosition, Vector.Zero, nil)
        elseif new_value == BITMAP_VALUES.SPIKE then
            --print("Spawning SPIKES at index: " .. grid_index)
            room:SpawnGridEntity(grid_index, GridEntityType.GRID_SPIKES_ONOFF, 0, 0, 0)
        end
    end
end

local function GetFirstOccurringEntity(entities)
    if(entities.Size == 0) then
        return nil
    end
    for i = 0, entities.Size do
        local entity = entities:Get(i)
        if (entity) and (entity.Type >= 10 and entity.Type <= 900) then
            return entity
        end
    end
    return nil
end

MutationModel:AddCallback(ModCallbacks.MC_POST_NEW_ROOM, function()
    local room = Game():GetRoom()
    local roomType = room:GetType()
    if(roomType == RoomType.ROOM_DEFAULT) then
        local roomIndex = level:GetCurrentRoomIndex()
        
        -- Check if room has been visited before
        -- if visitedRooms[roomIndex] then
        --     print("Room " .. roomIndex .. " already visited - applying stored modifications")
        --     -- Apply stored modifications if any exist
        --     if roomModifications[roomIndex] then
        --         print("Applying " .. #roomModifications[roomIndex] .. " stored modifications to room " .. roomIndex)
        --         local entity = GetFirstOccurringEntity(room:GetEntities())
        --         for _, modification in ipairs(roomModifications[roomIndex]) do
        --             ProcessBitmapChange(room, modification.grid_index, modification.old_value, modification.new_value, entity)
        --         end
        --     end
        --     return
        -- else
            print("Processing room " .. roomIndex .. " - applying modifications")
            -- visitedRooms[roomIndex] = true
            
            local bitmapString = CreateRoomBitmapString(room)
            local count = 0;
            MutationModel:SaveData(bitmapString)
            
            -- Wait until the first row of SaveData is 1
            while true do
                local savedData = MutationModel:LoadData()
                if savedData and string.len(savedData) > 0 then
                    local firstChar = string.sub(savedData, 1, 1)
                    if firstChar == "1" then
                        print("First row is 1 - proceeding with modifications")
                        break
                    end
                end
            end
            
            -- Read and process all changes from the save file
            local savedData = MutationModel:LoadData()
            if savedData then
                local lines = {}
                for line in savedData:gmatch("[^\r\n]+") do
                    table.insert(lines, line)
                end
                
                -- Check if lines is empty
                if #lines == 0 then
                    print("No lines found in save data")
                    return
                end
                
                local entity = GetFirstOccurringEntity(room:GetEntities())
                local changesStarted = false
                
                for i, line in ipairs(lines) do
                    -- Skip the "1" and bitmap data, look for change lines
                    if changesStarted or (line and string.find(line, ",") and not string.find(line, "^[0-9],[0-9]") and string.len(line) < 50) then
                        changesStarted = true
                        if line and string.find(line, ",") then
                            local parts = {}
                            for part in line:gmatch("[^,]+") do
                                table.insert(parts, tonumber(string.match(part, "%d+")))
                            end
                            
                            if #parts == 3 then
                                local old_value = parts[1]
                                local new_value = parts[2]
                                local grid_index = parts[3]
                                ProcessBitmapChange(room, grid_index, old_value, new_value, entity)
                                
                                -- Store modification unless new_value is 6 (entities)
                                if new_value ~= 6 then
                                    if not roomModifications[roomIndex] then
                                        roomModifications[roomIndex] = {}
                                    end
                                    table.insert(roomModifications[roomIndex], {
                                        old_value = old_value,
                                        new_value = new_value,
                                        grid_index = grid_index
                                    })
                                end
                            end
                        end
                    elseif line and not string.find(line, ",") and string.len(line) > 10 then
                        -- This is likely the end of bitmap data, changes start next
                        changesStarted = true
                    end
                end
                MutationModel:SaveData("")
            end
        -- end
    end
end)

-- Reset visited rooms when starting a new level
MutationModel:AddCallback(ModCallbacks.MC_POST_NEW_LEVEL, function()
    MutationModel:SaveData("")
    -- visitedRooms = {}
    roomModifications = {}
    local roomIndex = level:GetCurrentRoomIndex()
    -- visitedRooms[roomIndex] = true
    print("New level started - cleared all room modifications")
end)