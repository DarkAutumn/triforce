﻿function checkHearts(address, value)
  --emu.log(value)
end

function writeValues(address, value)
  sword = emu.read(0xb9, emu.memType.cpuDebug, false)
  beams = emu.read(0xba, emu.memType.cpuDebug, false)
  rang = emu.read(0xbb, emu.memType.cpuDebug, false)
  bomb1 = emu.read(0xbc, emu.memType.cpuDebug, false)
  bomb2 = emu.read(0xbd, emu.memType.cpuDebug, false)
  wand = emu.read(0xbe, emu.memType.cpuDebug, false)
  
  emu.log(string.format("sword=%x beams=%x, rang=%x, b1=%x, b2=%x, aw=%x", sword, beams, rang, bomb1, bomb2, wand))
end

emu.addMemoryCallback(checkHearts, emu.memCallbackType.writeValues, 0xb9)
emu.addMemoryCallback(checkHearts, emu.memCallbackType.writeValues, 0xba)
emu.addMemoryCallback(checkHearts, emu.memCallbackType.writeValues, 0xbb)
emu.addMemoryCallback(checkHearts, emu.memCallbackType.writeValues, 0xbc)
emu.addMemoryCallback(checkHearts, emu.memCallbackType.writeValues, 0xbd)
emu.addMemoryCallback(checkHearts, emu.memCallbackType.writeValues, 0xbe)
