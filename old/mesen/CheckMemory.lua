function writeValues(address, value)
  --emu.log(value)
end

function writeValues(address, value)
  sword = emu.read(0xb9, emu.memType.nesMemory, false)
  beams = emu.read(0xba, emu.memType.nesMemory, false)
  rang = emu.read(0xbb, emu.memType.nesMemory, false)
  bomb1 = emu.read(0xbc, emu.memType.nesMemory, false)
  bomb2 = emu.read(0xbd, emu.memType.nesMemory, false)
  wand = emu.read(0xbe, emu.memType.nesMemory, false)
  
  emu.log(string.format("wrote:%x=%x - sword=%x beams=%x, rang=%x, b1=%x, b2=%x, aw=%x", address, value, sword, beams, rang, bomb1, bomb2, wand))
end

function writeKillCount(address, value)
	emu.log(string.format("kill count %d", value))
end

emu.addMemoryCallback(writeKillCount, emu.callbackType.write, 0x627)
emu.addMemoryCallback(writeValues, emu.callbackType.write, 0xb9, 0xbe)
