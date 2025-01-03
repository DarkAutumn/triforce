"""Makes a particular item usable and selected."""

from enum import Enum

class ItemOrder(Enum):
    """The order in which items are selected."""
    BOOMERANG = 0
    BOMBS = 1
    ARROWS = 2
    CANDLE = 4
    WHISTLE = 5
    FOOD = 6
    POTION = 7
    WAND = 8

class ItemSelector:
    """Helper class to select an item to use."""
    def __init__(self, env):
        self.env = env

    def select_boomerang(self, magic):
        data = self.env.unwrapped.data
        data.set_value('regular_boomerang', 1)
        if magic:
            data.set_value('magic_boomerang', 2)
        else:
            data.set_value('magic_boomerang', 0)

        self._select_item(data, ItemOrder.BOOMERANG)

    def select_arrows(self, silver):
        data = self.env.unwrapped.data
        data.set_value('arrows', 2 if silver else 1)
        data.set_value('bow', 1)
        data.set_value('rupees', 100)
        self._select_item(data, ItemOrder.ARROWS)

    def select_bombs(self):
        data = self.env.unwrapped.data
        data.set_value('bombs', 8)
        self._select_item(data, ItemOrder.BOMBS)

    def select_wand(self, book):
        data = self.env.unwrapped.data
        data.set_value('magic_rod', 1)
        data.set_value('book', 1 if book else 0)
        self._select_item(data, ItemOrder.WAND)

    def select_whistle(self):
        data = self.env.unwrapped.data
        data.set_value("whistle", 1)
        self._select_item(data, ItemOrder.WHISTLE)

    def select_candle(self, red):
        data = self.env.unwrapped.data
        data.set_value("candle", 2 if red else 1)
        self._select_item(data, ItemOrder.CANDLE)

    def select_potion(self, red):
        data = self.env.unwrapped.data
        data.set_value("potion", 2 if red else 1)
        self._select_item(data, ItemOrder.POTION)

    def select_food(self):
        data = self.env.unwrapped.data
        data.set_value("food", 1)
        self._select_item(data, ItemOrder.FOOD)

    def select_sword(self, level=1):
        data = self.env.unwrapped.data
        data.set_value("sword", level)
        data.set_value("hearts_and_containers", 0xfe)

    def select_beams(self, level=1):
        data = self.env.unwrapped.data
        data.set_value("sword", level)
        data.set_value("hearts_and_containers", 0xff)

    def _select_item(self, data, item : ItemOrder):
        data.set_value("selected_item", item.value)

__all__ = [ItemSelector.__name__]
