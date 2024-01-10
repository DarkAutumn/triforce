function writeValues(address, value)
  --emu.log(value)
end
items = {
    [0xb9] = "sword",
    [0xba] = "beams",
    [0xbb] = "boomerang",
    [0xbc] = "bomb1",
    [0xbd] = "bomb2",
    [0xbe] = "wand"
}
function writeValues(address, value)
  emu.log(items[address] .. " - " .. value)
  if address == 0xba and value == 16 then
    --emu.addMemoryCallback(readBeams, emu.memCallbackType.cpuRead, 0xba)
  else
    --emu.removeMemoryCallback(readBeams, emu.memCallbackType.cpuRead, 0xba)
  end
end


function writeDamageTable(address, value)
  prev = emu.read(address, emu.memType.cpu)
  address = address - 0x485
  emu.log("damage: enemy:" .. to_hex(address) .. " prev:" .. to_hex(prev) .. " curr:" .. to_hex(value))
end
  
function writeAnimationValues(address, value)
  emu.log(items[address] .. " - " .. value)
  if address == 0xba and value == 16 then
    --emu.addMemoryCallback(readBeams, emu.memCallbackType.cpuRead, 0xba)
  else
    --emu.removeMemoryCallback(readBeams, emu.memCallbackType.cpuRead, 0xba)
  end
end

function to_hex(num)
  return string.format("%x", num)
end

function readBeams(address, value)
  print("read: " .. address .. " " .. value)
  --emu.breakExecution()
end

function writeKillCount(address, value)
  print("kill: " .. to_hex(address) .. " " .. value)
end

emu.addMemoryCallback(writeAnimationValues, emu.memCallbackType.cpuWrite, 0xb9, 0xbe)
emu.addMemoryCallback(writeKillCount, emu.memCallbackType.cpuWrite, 0x627)
emu.addMemoryCallback(writeDamageTable, emu.memCallbackType.cpuWrite, 0x485, 0x485+12)

-- bomb==11 means one is placed
-- bomb===12-15 is explosion