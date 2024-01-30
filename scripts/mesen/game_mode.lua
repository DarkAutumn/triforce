function changeMode(addr, value)
emu.log(value)
end

emu.addMemoryCallback(changeMode, emu.callbackType.write, 0x12)
