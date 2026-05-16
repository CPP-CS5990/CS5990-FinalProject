from snaikenet_server.game.list_dict import ListDict
from snaikenet_server.game.types import Position


class FoodPositions:
    def __init__(self):
        """For every food tile, the available food positions should be removed."""
        self._available_food_positions: ListDict[Position] = ListDict()
        self._num_food_tiles = 0

    def add_available_food_position(self, position: Position):
        self._available_food_positions.add_item(position)

    def remove_available_food_position(self, position: Position):
        self._available_food_positions.remove_item(position)

    def place_food(self, position: Position):
        self._num_food_tiles += 1
        self._available_food_positions.remove_item(position)

    def remove_food(self, position: Position):
        self._num_food_tiles -= 1
        self._available_food_positions.add_item(position)

    def choose_random_food_position(self) -> Position | None:
        if len(self._available_food_positions) == 0:
            return None
        return self._available_food_positions.choose_random_item()

    def get_num_food_tiles(self) -> int:
        return self._num_food_tiles
