# find the directory of this file
import os
import json

script_dir = os.path.dirname(os.path.realpath(__file__))
zelda_memory_file = os.path.join(script_dir, 'zelda_memory.txt')

def get_signed(value):
    if value == 'signed':
        return True
    elif value == 'unsigned':
        return False
    
    raise Exception(f'Invalid signed value: {value}')

def main():
    info = {}
    result = { 'info': info }

    #load the memory file
    with open(zelda_memory_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('0x'):
                parts = line.split(' ')
                name = parts[1]

                address = int(parts[0], 16)
                signed = get_signed(parts[2]) if len(parts) > 2 else False

                info[name] = {
                    'address': address,
                    'type': "=i1" if  signed else "=u1"
                    }
    
    # write the json out to a file
    output = os.path.join(script_dir, 'custom_integrations/Zelda-NES/data.json')
    with open(output, 'w') as f:
        json.dump(result, f, indent=4)

if __name__ == '__main__':
    main()

__all__ = []