import os

def remove_trailing_whitespace(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r+', encoding='utf-8') as f:
                    lines = f.readlines()
                    f.seek(0)
                    f.writelines([line.rstrip() + '\n' for line in lines])
                    f.truncate()

directory_to_scan = os.path.join(os.path.dirname(__file__), '..')
remove_trailing_whitespace(directory_to_scan)
