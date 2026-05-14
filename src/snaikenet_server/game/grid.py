from enum import IntEnum

from loguru import logger
import numpy as np
from numpy.typing import NDArray
from snaikenet_server.game.food_positions import FoodPositions

from snaikenet_server.game.types import GridSize, Position, PlayerID

type GridStructure = list[list[TileData]]


class Grid:
    def __init__(self, grid_size: GridSize):
        self._grid_size = grid_size
        self._grid = [
            [TileData(tile_type=TileType.EMPTY) for _ in range(grid_size[1])]
            for _ in range(grid_size[0])
        ]
        self._food_positions = FoodPositions()

    # To award kills, a player needs to be designated as the killer,
    # there may be multiple players at any given position.
    def tile_occupied_by_other(
        self, player_id: PlayerID, position: Position
    ) -> PlayerID | None:
        players = self._grid[position[0]][position[1]].player_ids
        for other_player in players:
            if other_player != player_id:
                return other_player
        return None

    def remove_player_at(self, position: Position, player_id: PlayerID, replace_with_food: bool):
        tile = self._grid[position[0]][position[1]]
        tile.remove_player(player_id)
        # Guard required: if another player still occupies the tile, marking it
        # available allows food to be placed on top of a live snake.
        if tile.tile_type == TileType.EMPTY:
            if replace_with_food:
                self.place_food_at(position)
            else:
                self._food_positions.add_available_food_position(position)


    def add_player_at(self, position: Position, player_id: PlayerID):
        if self._grid[position[0]][position[1]].tile_type == TileType.FOOD:
            self.remove_food_at(position)
        self._grid[position[0]][position[1]].add_player(player_id)
        self._food_positions.remove_available_food_position(position)

    def remove_food_at(self, position: Position):
        if self._grid[position[0]][position[1]].tile_type == TileType.FOOD:
            self._food_positions.remove_food(position)
            if len(self._grid[position[0]][position[1]].player_ids) == 0:
                self._grid[position[0]][position[1]].tile_type = TileType.EMPTY
        else:
            logger.warning(
                f"Attempting to remove food from a tile that does not contain food at position {position}!\n"
            )

    def food_at(self, position: Position) -> bool:
        return self._grid[position[0]][position[1]].tile_type == TileType.FOOD

    def has_player_at(self, position: Position) -> bool:
        return len(self._grid[position[0]][position[1]].player_ids) > 0

    def place_food_at(self, position: Position):
        tile = self._grid[position[0]][position[1]]
        if tile.tile_type == TileType.FOOD:
            return
        tile.make_food()
        self._food_positions.place_food(position)

    def get_grid_size(self) -> GridSize:
        return self._grid_size

    def get_tile_data(self, position: Position) -> TileData:
        return self._grid[position[0]][position[1]]

    def fill_available_food_positions(self):
        for x in range(self._grid_size[0]):
            for y in range(self._grid_size[1]):
                if self._grid[x][y].tile_type == TileType.EMPTY:
                    self._food_positions.add_available_food_position((x, y))

    def get_random_available_food_position(self) -> Position | None:
        return self._food_positions.choose_random_food_position()

    def get_num_food(self) -> int:
        return self._food_positions.get_num_food_tiles()

    def get_grid_data(self) -> NDArray[np.uint8]:
        return np.array(
            [[tile.tile_type for tile in row] for row in self._grid], dtype=np.uint8
        )

    def __iter__(self):
        for row in self._grid:
            for tile in row:
                yield tile

    # We would usually want to have the ServerCodec do this but that is alot of
    # meaningless extra work we would be imposing on the server. The ServerCodec
    # is still needed to structure the actual datagram, but it makes more sense to
    # encode the grid as bytes here.
    def viewport_as_bytes(
        self,
        center_position: tuple[int, int],
        distance_from_center: tuple[int, int],
        player_id: str,
    ) -> bytes:
        width = distance_from_center[0] * 2 + 1
        height = distance_from_center[1] * 2 + 1
        start_x = center_position[0] - distance_from_center[0]
        end_x = center_position[0] + distance_from_center[0]
        start_y = center_position[1] - distance_from_center[1]
        end_y = center_position[1] + distance_from_center[1]
        viewport = bytearray(width * height)
        offset = 0
        for x in range(start_x, end_x + 1):
            for y in range(start_y, end_y + 1):
                # if tile is within the bounds
                if 0 <= x < self._grid_size[0] and 0 <= y < self._grid_size[1]:
                    tile = self._grid[x][y]
                    viewport[offset] = tile.tile_type
                    if tile.tile_type == TileType.SNAKE:
                        viewport[offset] = 3 if player_id in tile.player_ids else 4
                else:
                    viewport[offset] = TileType.WALL
                offset += 1
        return bytes(viewport)


class TileType(IntEnum):
    EMPTY = 0
    WALL = 1
    FOOD = 2
    SNAKE = 3


class TileData:
    def __init__(self, tile_type: int = TileType.EMPTY, player_ids: list[str] =None):
        if player_ids is None:
            player_ids = []
        self.tile_type: int = tile_type
        # Multiple players can occupy the same tile temporarily during collisions
        self.player_ids: list[PlayerID] = player_ids

    def add_player(self, player_id: PlayerID):
        if player_id not in self.player_ids:
            self.player_ids.append(player_id)
            self.tile_type = TileType.SNAKE

    def remove_player(self, player_id: PlayerID):
        if player_id in self.player_ids:
            self.player_ids.remove(player_id)
            if len(self.player_ids) == 0:
                self.tile_type = TileType.EMPTY

    def make_food(self):
        self.tile_type = TileType.FOOD
        if len(self.player_ids) > 0:
            logger.critical(
                f"Attempting to place food on a tile that is currently occupied by players: {self.player_ids}!\n"
            )
