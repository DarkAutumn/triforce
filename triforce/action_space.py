# Design goals:
# - Actions are object oriented so we can easily modify/understand what the agent is doing.
# - The action space is reduced to only the actions that are possible in the game.
# - We keep an accurate

from numbers import Integral
from typing import List, Sequence
import gymnasium as gym
import numpy as np

from .link import Link
from .zelda_game import ZeldaGame
from .zelda_enums import Direction, SelectedEquipmentKind, ActionKind

class ActionTaken:
    """The action taken by a model."""
    def __init__(self, space, action : int):
        self.id = action
        self.multi_binary : List[bool] = space.index_to_buttons[action]
        self.buttons = space.index_to_button_names[action]
        action, direction = space.index_to_action_direction[action]
        self.kind : ActionKind = action
        self.direction : Direction = direction

    def select_equipment(self, link : Link):
        """Selects the equipment for the link."""

        match self.kind:
            case ActionKind.BOMBS:
                link.selected_equipment = SelectedEquipmentKind.BOMBS

            case ActionKind.ARROW:
                link.selected_equipment = SelectedEquipmentKind.ARROWS

            case ActionKind.WAND:
                link.selected_equipment = SelectedEquipmentKind.WAND

            case ActionKind.BOOMERANG:
                link.selected_equipment = SelectedEquipmentKind.BOOMERANG

            case ActionKind.WHISTLE:
                link.selected_equipment = SelectedEquipmentKind.WHISTLE

            case ActionKind.FOOD:
                link.selected_equipment = SelectedEquipmentKind.FOOD

            case ActionKind.POTION:
                link.selected_equipment = SelectedEquipmentKind.POTION

            case ActionKind.CANDLE:
                link.selected_equipment = SelectedEquipmentKind.CANDLE

    def __getitem__(self, index):
        return self.multi_binary[index]

    def __len__(self):
        return len(self.multi_binary)

class ZeldaActionSpace(gym.Wrapper):
    """A wrapper that shrinks the action space down to what's actually used in the game."""
    def __init__(self, env, actions_allowed : Sequence[ActionKind | str], prevent_wall_bumping : bool = True):
        super().__init__(env)

        if actions_allowed == 'all':
            actions_allowed = list(ActionKind)
        else:
            for i, action in enumerate(actions_allowed):
                if isinstance(action, str):
                    actions_allowed[i] = ActionKind(action)

        self.prevent_wall_bumping = prevent_wall_bumping
        self.button_count = env.action_space.n

        self.actions_allowed = set(actions_allowed)
        self.action_to_index = {}
        self.index_to_action_direction = []
        self.index_to_button_names = []
        self.index_to_buttons = {}

        self.up = env.unwrapped.buttons.index('UP')
        self.down = env.unwrapped.buttons.index('DOWN')
        self.right = env.unwrapped.buttons.index('RIGHT')
        self.left = env.unwrapped.buttons.index('LEFT')

        self.a = env.unwrapped.buttons.index('A')
        self.b = env.unwrapped.buttons.index('B')

        self.move_mask = np.ones(4, dtype=bool)

        self._setup_actions(actions_allowed)

        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        self.total_actions = len(self.index_to_button_names)
        self.action_space = gym.spaces.Discrete(self.total_actions)

        if not self.action_to_index:
            raise ValueError("Must select at least one kind of action.")

    def _setup_actions(self, actions_allowed):
        for action in actions_allowed:
            if action in self.action_to_index:
                continue

            match action:
                case ActionKind.MOVE:
                    self.action_to_index[action] = len(self.index_to_button_names)
                    self._add_directions(action)

                case ActionKind.SWORD:
                    self.action_to_index[action] = len(self.index_to_button_names)
                    self._add_directions(action, self.a)

                case ActionKind.BEAMS:
                    self.action_to_index[action] = len(self.index_to_button_names)
                    self._add_directions(action, self.a)

                case ActionKind.BOMBS:
                    self.action_to_index[action] = len(self.index_to_button_names)
                    self._add_directions(action, self.b)

                case ActionKind.ARROW:
                    self.action_to_index[action] = len(self.index_to_button_names)
                    self._add_directions(action, self.b)

                case ActionKind.WAND:
                    self.action_to_index[action] = len(self.index_to_button_names)
                    self._add_directions(action, self.b)

                case ActionKind.BOOMERANG:
                    self.action_to_index[action] = len(self.index_to_button_names)
                    self._add_directions(action, self.b)
                    self._add_diagonals(action, self.b)

                case ActionKind.WHISTLE:
                    self.action_to_index[action] = len(self.index_to_button_names)
                    self._add_button(action, self.b)

                case ActionKind.FOOD:
                    self.action_to_index[action] = len(self.index_to_button_names)
                    self._add_button(action, self.b)

                case ActionKind.POTION:
                    self.action_to_index[action] = len(self.index_to_button_names)
                    self._add_button(action, self.b)

                case ActionKind.CANDLE:
                    self.action_to_index[action] = len(self.index_to_button_names)
                    self._add_button(action, self.b)

        self.index_to_buttons[None] = np.array([False] * self.button_count)
        for i, buttons in enumerate(self.index_to_button_names):
            arr = np.array([False] * self.button_count)

            for button in buttons:
                arr[button] = True

            self.index_to_buttons[i] = arr

    def _add_button(self, action, *buttons):
        buttons = list(buttons)
        self.index_to_action_direction.append((action, Direction.NONE))
        self.index_to_button_names.append(buttons)

    def _add_directions(self, action, *buttons):
        buttons = list(buttons)
        self.index_to_action_direction.append((action, Direction.N))
        self.index_to_button_names.append([self.up] + buttons)

        self.index_to_action_direction.append((action, Direction.S))
        self.index_to_button_names.append([self.down] + buttons)

        self.index_to_action_direction.append((action, Direction.W))
        self.index_to_button_names.append([self.left] + buttons)

        self.index_to_action_direction.append((action, Direction.E))
        self.index_to_button_names.append([self.right] + buttons)

    def _add_diagonals(self, action, *buttons):
        buttons = list(buttons)
        self.index_to_action_direction.append((action, Direction.NW))
        self.index_to_button_names.append([self.up, self.left] + buttons)

        self.index_to_action_direction.append((action, Direction.NE))
        self.index_to_button_names.append([self.up, self.right] + buttons)

        self.index_to_action_direction.append((action, Direction.SW))
        self.index_to_button_names.append([self.down, self.left] + buttons)

        self.index_to_action_direction.append((action, Direction.SE))
        self.index_to_button_names.append([self.down, self.right] + buttons)

    def reset(self, **kwargs):
        observation, state = self.env.reset(**kwargs)
        self.move_mask = np.ones(4, dtype=bool)
        return observation, state

    def step(self, action):
        if isinstance(action, tuple):
            action = self._translate_action(*action)

        if isinstance(action, Integral):
            action = ActionTaken(self, action)
        elif not isinstance(action, ActionTaken):
            raise ValueError(f"Invalid action type {type(action)}.")

        observation, reward, terminated, truncated, state_change = self.env.step(action)
        self._handle_wall_bump(state_change)

        return observation, reward, terminated, truncated, state_change

    def _translate_action(self, action, direction):
        index = self.action_to_index[action]
        match direction:
            case Direction.N:
                index += 0
            case Direction.S:
                index += 1
            case Direction.W:
                index += 2
            case Direction.E:
                index += 3
            case Direction.NW:
                assert action == ActionKind.BOOMERANG
                index += 4
            case Direction.NE:
                assert action == ActionKind.BOOMERANG
                index += 5
            case Direction.SW:
                assert action == ActionKind.BOOMERANG
                index += 6
            case Direction.SE:
                assert action == ActionKind.BOOMERANG
                index += 7

            case _:
                raise ValueError(f"Invalid direction {direction}.")

        return index

    def _handle_wall_bump(self, state_change):
        if self.prevent_wall_bumping and state_change.action == ActionKind.MOVE:
            if state_change.previous.link.position == state_change.state.link.position:
                match state_change.action.direction:
                    case Direction.N:
                        self.move_mask[0] = False

                    case Direction.S:
                        self.move_mask[1] = False

                    case Direction.W:
                        self.move_mask[2] = False

                    case Direction.E:
                        self.move_mask[3] = False

                if np.any(self.move_mask):
                    return

        self.move_mask = np.ones(4, dtype=bool)

    def get_action_mask(self, state : ZeldaGame):
        """Returns the actions that are available to the agent."""

        actions_possible = self.actions_allowed & set(state.link.get_available_actions())
        assert actions_possible, "No actions available, we should have at least MOVE."

        mask = np.zeros(self.total_actions, dtype=bool)
        for action in actions_possible:
            index = self.action_to_index[action]
            match action:
                case ActionKind.MOVE:
                    mask[index:index + 4] = self.move_mask

                case ActionKind.SWORD:
                    mask[index:index + 4] = True

                case ActionKind.BEAMS:
                    mask[index:index + 4] = True

                case ActionKind.BOMBS:
                    mask[index:index + 4] = True

                case ActionKind.ARROW:
                    mask[index:index + 4] = True

                case ActionKind.WAND:
                    mask[index:index + 4] = True

                case ActionKind.CANDLE:
                    mask[index:index + 4] = True

                case ActionKind.BOOMERANG:
                    mask[index:index + 8] = True

                case ActionKind.WHISTLE:
                    mask[index] = True

                case ActionKind.POTION:
                    mask[index] = True

                case ActionKind.FOOD:
                    mask[index] = True

        return mask
