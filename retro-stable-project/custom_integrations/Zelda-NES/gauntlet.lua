last_location = nil
room_kill_count = 0
x = 0
y = 0
function get_reward()
    -- Pentalty for dying, no further processing
    if data.mode == 17 then
        return -1.0
    end

    -- No reward if we are not a controllable state.
    if data.mode ~= 5 then
        return 0.0
    end

    reward = 0.0

    -- Reward for moving right, penalty otherwise
    if last_location == nil then
        last_location = data.location
        last_x = data.x
        last_x = data.y
    end

    if data.location ~= last_location then
        if data.location < 120 or data.location > 127 then
            reward = -1.0
        else
            -- big reward for moving right, penalty for moving left
            if last_location < data.location then
                reward = 1.0
            else
                reward = -0.5
            end
        end

        last_location = data.location
        room_kill_count = 0
    else
        -- otherwise score link's x/y position
        if data.x < 6 then
            if data.x <= last_x then
                reward = reward - 0.05
            end
        else
            if data.x < last_x then
                reward = reward - 0.05
            else
                if data.x > last_x then
                    reward = reward + 0.05
                end
            end
        end

        last_x = data.x
        last_y = data.y
    end

    if room_kill_count < data.room_kill_count then
        reward = reward + 0.1
        room_kill_count = data.room_kill_count
    end

    return reward
end

function is_done()
    -- End if we are in the game over screen.
    if data.mode == 17 then
        return true
    end
    
    -- only check game conditions when playing
    if data.mode == 5 then
        if data.location < 120 or data.location >= 127 then
            return true
        end
    end

    return false
end