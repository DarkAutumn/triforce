# Design goals:
# - Actions are object oriented so we can easily modify/understand what the agent is doing.
# - The action space is reduced to only the actions that are possible in the game.
# - We keep an accurate

from numbers import Integral
from typing import List, Sequence
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import torch

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
    """A wrapper that shrinks the action space down to what's actually used in the game.

    When multihead=True, exposes a MultiDiscrete([K, 4]) action space for the two-head
    action decomposition (action type + direction). Masks become [K+4] concatenated format.
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, env, actions_allowed : Sequence[ActionKind | str],
                 multihead : bool = False):
        super().__init__(env)

        self.actions_allowed = ActionKind.get_from_list(actions_allowed)
        self.multihead = multihead

        self.button_count = env.action_space.n

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

        self._setup_actions(self.actions_allowed)

        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        self.total_actions = len(self.index_to_button_names)
        self.num_action_types = len(self.actions_allowed)

        if multihead:
            self.action_space = MultiDiscrete([self.num_action_types, 4])
        else:
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
        state.info['action_mask'] = self.get_action_mask(state)
        return observation, state

    def step(self, action):
        action = self.get_action_taken(action)

        observation, reward, terminated, truncated, state_change = self.env.step(action)
        state_change.action_mask = self.get_action_mask(state_change.state)
        state_change.state.info['action_mask'] = state_change.action_mask

        return observation, reward, terminated, truncated, state_change

    def get_action_taken(self, action) -> ActionTaken:
        """Returns the action taken by the agent based on the index.

        Accepts: flat int, (ActionKind, Direction) tuple, numpy/torch scalar,
        or 2-element numpy/torch array [type_idx, dir_idx] for multihead.
        """
        if isinstance(action, tuple):
            action = self._action_direction_to_index(*action)
        elif isinstance(action, np.ndarray):
            if action.shape == (2,):
                # Multihead: [action_type_idx, direction_idx]
                action = self.multihead_to_flat(int(action[0]), int(action[1]))
            else:
                action = action.item()
        elif isinstance(action, torch.Tensor):
            if action.shape == (2,):
                # Multihead: [action_type_idx, direction_idx]
                action = self.multihead_to_flat(int(action[0].item()), int(action[1].item()))
            else:
                action = action.item()

        if isinstance(action, Integral):
            action = ActionTaken(self, action)
        elif not isinstance(action, ActionTaken):
            raise ValueError(f"Invalid action type {type(action)}.")

        return action

    def multihead_to_flat(self, type_idx, dir_idx):
        """Convert multihead (action_type_index, direction_index) to flat action index."""
        action_kind = self.actions_allowed[type_idx]
        return self.action_to_index[action_kind] + dir_idx

    def flat_mask_to_multihead(self, flat_mask):
        """Decompose a flat action mask [N] into multihead format [K*4].

        Returns a tensor of shape [K*4] where mask[i*4 + j] is True if action
        type i with cardinal direction j (N=0, S=1, W=2, E=3) is valid.  This
        preserves the per-type direction constraints that are lost in a marginal
        [K+4] representation.

        The model derives the type mask as mask.view(K,4).any(dim=-1) and
        conditions the direction mask on the sampled type.
        """
        k = self.num_action_types
        multihead_mask = torch.zeros(k * 4, dtype=torch.bool)

        for i, action_kind in enumerate(self.actions_allowed):
            base = self.action_to_index[action_kind]

            # Determine how many flat entries this action type uses
            if i + 1 < k:
                next_base = self.action_to_index[self.actions_allowed[i + 1]]
            else:
                next_base = len(flat_mask)
            span = next_base - base

            chunk = flat_mask[base:next_base]
            if span >= 4:
                multihead_mask[i * 4:(i + 1) * 4] = chunk[:4]
            elif chunk.any():
                # Non-directional actions (whistle, potion, etc.): mark all 4
                # direction slots so the type is always selectable.
                multihead_mask[i * 4:(i + 1) * 4] = True

        return multihead_mask

    def _action_direction_to_index(self, action, direction):
        index = self.action_to_index[action]
        index += self._direction_to_index(direction)
        if direction in (Direction.NW, Direction.NE, Direction.SW, Direction.SE):
            assert action == ActionKind.BOOMERANG

        return index

    def get_action_mask(self, state : ZeldaGame):
        """Returns the actions that are available to the agent.

        When multihead=True, returns [K+4] concatenated mask (action type + direction).
        When multihead=False, returns flat [N] mask (one entry per flat action index).
        """
        flat_mask = self._get_flat_action_mask(state)
        if self.multihead:
            return self.flat_mask_to_multihead(flat_mask)
        return flat_mask

    def _get_flat_action_mask(self, state : ZeldaGame):
        """Computes the flat action mask [N] for the current state."""

        link = state.link
        actions_possible = set(self.actions_allowed)
        actions_possible &= link.get_available_actions(ActionKind.BEAMS in self.actions_allowed)
        assert actions_possible, "No actions available, we should have at least MOVE."

        invalid = {}
        for action, direction in state.info.get('invalid_actions', []):
            invalid.setdefault(action, []).append(direction)

        self._update_mask(state, invalid)

        mask = torch.zeros(self.total_actions, dtype=bool)
        for action in actions_possible:
            index = self.action_to_index[action]
            match action:
                case ActionKind.MOVE:
                    for direction in (Direction.N, Direction.S, Direction.W, Direction.E):
                        if state.can_link_move(direction):
                            mask[index + self._direction_to_index(direction)] = True

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

                case _:
                    if action in (ActionKind.SWORD, ActionKind.BEAMS):
                        # If there are no enemies or items, we can't use the sword.
                        # We allow items so we can pick up with a stab.
                        if not state.active_enemies and not state.items:
                            mask[index:index + 4] = False
                        else:
                            for direction in link.get_sword_directions_allowed():
                                mask[index + self._direction_to_index(direction)] = True

            if action in invalid:
                index = self.action_to_index[action]
                for direction in invalid[action]:
                    mask[index + self._direction_to_index(direction)] = False

        if not mask.any():
            self._handle_empty_mask(state, link, actions_possible, mask)

        return mask

    def _handle_empty_mask(self, _state, _link, _actions_possible, mask):
        """Fallback when all actions are masked — unmask all MOVE directions.

        Link can get into 'impossible' positions via sword knockback on certain
        tiles where can_link_move is conservatively False in every direction.
        Instead of crashing training, allow all moves and let the NES handle it.
        """
        move_index = self.action_to_index[ActionKind.MOVE]
        for i in range(4):
            mask[move_index + i] = True

    def _update_mask(self, state : ZeldaGame, invalid):
        """Removes certain actions if we are at the edge of the screen which link cannot perform."""
        link = state.link
        if state.level != 0:
            if link.tile.x <= 0x03 or link.tile.x >= 0x1c:
                for action in (ActionKind.SWORD, ActionKind.BEAMS):
                    invalid.setdefault(action, []).append(Direction.N)
                    invalid.setdefault(action, []).append(Direction.S)

            if link.tile.y <= 0x03 or link.tile.y >= 0x12:
                for action in (ActionKind.SWORD, ActionKind.BEAMS):
                    invalid.setdefault(action, []).append(Direction.W)
                    invalid.setdefault(action, []).append(Direction.E)

    def is_valid_action(self, action, action_mask):
        """Returns True if the action is valid.

        Handles both flat [N] and multihead [K*4] mask formats.
        """
        if action is None:
            return False

        action = self.get_action_taken(action)
        if len(action_mask) != self.total_actions:
            # Multihead [K*4] mask: direct (type, direction) lookup
            type_idx = self.actions_allowed.index(action.kind)
            dir_idx = self._direction_to_index(action.direction)
            return bool(action_mask[type_idx * 4 + dir_idx])
        return action_mask[action.id]

    def get_allowed_actions(self, state, action_mask):
        """Returns the allowed actions from the action mask.

        Handles both flat [N] and multihead [K*4] mask formats.
        """
        # For multihead mask, convert to flat for consistent processing
        if len(action_mask) != self.total_actions:
            action_mask = self._get_flat_action_mask(state)

        result = []

        link : Link = state.link
        actions_possible = set(self.actions_allowed)
        actions_possible &= link.get_available_actions(ActionKind.BEAMS in self.actions_allowed)

        for action in actions_possible:
            index = self.action_to_index[action]
            allowed_directions = []
            for direction in [Direction.N, Direction.S, Direction.W, Direction.E]:
                if action_mask[index + self._direction_to_index(direction)]:
                    allowed_directions.append(direction)

            if action == ActionKind.BOOMERANG:
                for direction in [Direction.NW, Direction.NE, Direction.SW, Direction.SE]:
                    if action_mask[index + self._direction_to_index(direction)]:
                        allowed_directions.append(direction)

            if allowed_directions:
                result.append((action, allowed_directions))

        return result

    def _direction_to_index(self, direction):
        value = None
        match direction:
            case Direction.N:
                value = 0
            case Direction.S:
                value = 1
            case Direction.W:
                value = 2
            case Direction.E:
                value = 3
            case Direction.NW:
                value = 4
            case Direction.NE:
                value = 5
            case Direction.SW:
                value = 6
            case Direction.SE:
                value = 7

        return value
