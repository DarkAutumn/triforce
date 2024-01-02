link_movements = ["none", "up", "down", "left", "right"]
link_actions = ["none,", "attack", "item", "swap_item"]

def get_movement(action):
    return link_movements[int(action / len(link_actions))]

def get_action(action):
    return link_actions[int(action % len(link_actions))]

total_actions = len(link_movements) * len(link_actions)