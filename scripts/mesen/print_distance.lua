function calculateDistance(x1, y1, x2, y2)
    local xDistance = x2 - x1
    local yDistance = y2 - y1
    return math.sqrt(xDistance^2 + yDistance^2)
end

-- Function to read memory and find the closest enemy
function findClosestEnemy()
    -- Read player's position
    local playerX = emu.read(0x70, emu.memType.nesMemory, false)
    local playerY = emu.read(0x84, emu.memType.nesMemory, false)

    local enemies = {}
    local closestEnemyIndex = -1
    local minDistance = math.huge

    -- Iterate over enemy indexes
    for i = 1, 0xa do
        local enemyX = emu.read(0x70 + i, emu.memType.nesMemory, false)
        local enemyY = emu.read(0x84 + i, emu.memType.nesMemory, false)
        local isEnemy = emu.read(0x34f + i, emu.memType.nesMemory, false)
        if isEnemy ~= 0 and isEnemy <= 0x48 then

	        local distance = calculateDistance(playerX, playerY, enemyX, enemyY)
	        table.insert(enemies, {index = i, distance = distance})

            --emu.log("Enemy " .. i .. " at " .. enemyX .. ", " .. enemyY .. " is " .. isEnemy)
	
	        -- Check if this enemy is the closest so far
	        if distance < minDistance then
	            minDistance = distance
	            closestEnemyIndex = i
	        end
        end
    end

    return closestEnemyIndex, enemies
end

local previous_enemies = {}
local prev_location = 0
function onFrame2()

    location = emu.read(0xeb, emu.memType.nesMemory, false)
    if location ~= prev_location then
        emu.log("Location: " .. location)
        prev_location = location
        previous_enemies = {}
    end

	local closestEnemyIndex, enemies = findClosestEnemy()

    -- get the closest enemy
    local enemy = nil
    for _, e in ipairs(enemies) do
        if e.index == closestEnemyIndex then
            enemy = e
            break
        end
    end

	if enemy ~= nil then
        bgColor = 0x302060FF
        fgColor = 0x30FF4040
        emu.drawRectangle(8, 8, 128, 24, bgColor, true, 1)
        emu.drawRectangle(8, 8, 128, 24, fgColor, false, 1)
        emu.drawString(12, 12, "Enemy: " .. enemy.index, 0xFFFFFF, 0xFF000000)
        emu.drawString(12, 21, "Dist:  " .. enemy.distance, 0xFFFFFF, 0xFF000000)
    end

    -- Detect killed enemies
    for idx, prevEnemy in pairs(previous_enemies) do
        if not enemies[prevEnemy.index] then
            emu.log("killed enemy " .. prevEnemy.index .. " at distance " .. prevEnemy.distance)
            previous_enemies[idx] = nil
            previous_enemies[prevEnemy.index] = nil
        end
    end

    -- Update previous enemies for the next frame
    previous_enemies = {}
    for _, enemy in ipairs(enemies) do
        previous_enemies[enemy.index] = enemy
    end
end

last_effects = {}
last_effects[0xb3] = 0
last_effects[0xb4] = 0
last_effects[0xb5] = 0
last_effects[0xb6] = 0
last_effects[0xb7] = 0

for k, v in pairs(last_effects) do
    last_effects[k] = emu.read(k, emu.memType.cpuMemory, false)
end

function readAndDiff(address, last)
    local current = emu.read(address, emu.memType.cpuMemory, false)
    if current ~= last then
        emu.log("Change at " .. address .. " from " .. last .. " to " .. current)
    end
    return current
end


function onFrame()
    val = emu.read(0xb7, emu.memType.nesMemory, false)
    if val ~= 0 then
        -- val in hex
        emu.log("0xb5: " .. string.format("%x", val))
    end
end

emu.addEventCallback(onFrame, emu.eventType.endFrame);