import math
import random
import uuid

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from snaikenet_server.game.grid import Grid, TileData
from snaikenet_server.game.player import SnakePlayer
from snaikenet_server.game.scoreboard import Scoreboard
from snaikenet_server.game.types import PlayerID, GridSize, Position, Direction


class GameState:
    def __init__(
        self,
        grid_size: GridSize,
        tick_index: int = 0,
        viewport_distance_from_center: tuple[int, int] = (14, 14),
        scoreboard_save_dir: str | None = "scoreboard_data",
    ):
        self._dead_players: dict[PlayerID, PlayerID | None] = {}
        self._players: dict[PlayerID, SnakePlayer] = {}
        self._spectators: dict[PlayerID, PlayerID | None] = {}
        self._kills: dict[PlayerID, PlayerID] = {}
        self._grid: Grid = Grid(grid_size)
        self._viewport_distance_from_center: tuple[int, int] = (
            viewport_distance_from_center
        )
        self._max_num_food: int = 1
        self._scoreboard = Scoreboard(scoreboard_save_dir)
        self._curr_tick_index = tick_index

    def update_curr_tick_index(self, tick_index: int):
        self._curr_tick_index = tick_index

    def cull_starving_players(self, max_ticks_since_eaten: int):
        for player_id, player in self._players.items():
            player = self._players.get(player_id, None)
            if (
                player is not None
                and self._curr_tick_index - player.last_tick_eaten
                >= max_ticks_since_eaten
            ):
                self.kill_player(player_id)

    def reset_game_state(self):
        self._scoreboard.save_and_reset()
        # Cache state values we want to persist across games
        players = dict(self._players)
        spectators = dict(self._spectators)
        max_num_food = self._max_num_food
        scoreboard = self._scoreboard

        # reset the state
        self.__init__(self.get_grid_size(), 0, self._viewport_distance_from_center)

        # Restore the cached state values
        self._players = players
        self._spectators = spectators
        self._max_num_food = max_num_food
        self._curr_tick_index = 0
        self._scoreboard = scoreboard

    def get_viewport_distance_from_center(self) -> tuple[int, int]:
        return self._viewport_distance_from_center

    def kill_player(self, player_id: PlayerID, killer: PlayerID | None = None):
        player = self._players.get(player_id, None)
        if player is None:
            logger.warning("Tried killing player that doesn't exist", player_id)
            return
        # Dead players remain in self._players (needed for death frames / spectator
        # state), so callers like cull_starving_players and delete_player can reach
        # this with an already-dead player. Re-running _replace_player_with_food on
        # a dead snake re-stamps food at its old body tiles whenever they are EMPTY.
        if player.is_dead():
            return
        if killer is None:  # self kill
            self._kills[player_id] = player_id
        else:
            self._kills[killer] = player_id
            self._scoreboard.increment_kills(killer)
        player.die()
        self._scoreboard.on_player_death(player_id, self._curr_tick_index)
        self._dead_players[player_id] = None
        self._replace_player_with_food(player)

    def _replace_player_with_food(self, player: SnakePlayer):
        player_id = player.get_player_id()

        for position in player:
            self._grid.remove_player_at(
                position, player_id, True
            )  # Mark all tiles occupied by the player as empty on the grid

    # Adds a new player to the game state
    def add_new_player(self, player_id: PlayerID | None = None) -> PlayerID:
        if player_id is None:
            player_id_ = str(uuid.uuid4())
        else:
            player_id_ = player_id
        self._players[player_id_] = SnakePlayer(
            (0, 0),  # Temporary position, will be set properly in initialize_game_state
            player_id_,
        )
        return player_id_

    def add_spectator(self, player_id: PlayerID = None):
        if player_id is None:
            player_id_ = str(uuid.uuid4())
        else:
            player_id_ = player_id
        self._spectators[player_id_] = None

    def set_player_direction(self, player_id: PlayerID, player_direction: Direction):
        player = self._players.get(player_id, None)
        if player is not None:
            player.set_direction(player_direction)

    def get_player(self, player_id: PlayerID) -> SnakePlayer | None:
        return self._players.get(player_id, None)

    def get_dead_players(self) -> set[PlayerID]:
        return set(self._dead_players)

    def get_living_players(self) -> set[PlayerID]:
        return set(self._players.keys() - self._dead_players)

    def delete_player(self, player_id: PlayerID) -> bool:
        self._spectators.pop(player_id, None)

        if player_id not in self._players:
            return False
        self.kill_player(player_id, player_id)
        self._players.pop(player_id, None)
        self._dead_players.pop(player_id, None)

        return True

    def position_outside_grid(self, position: Position) -> bool:
        grid_size_x, grid_size_y = self._grid.get_grid_size()
        return (
            position[0] < 0
            or position[0] >= grid_size_x
            or position[1] < 0
            or position[1] >= grid_size_y
        )

    # Handles player moves and kills the player if they hit a wall
    def handle_player_moves(self):
        for player_id, player in self._players.items():
            if player.is_dead():
                continue
            next_head_position = player.add_head()

            # Check for collisions with walls
            if self.position_outside_grid(next_head_position):
                logger.debug(
                    f"Player {player_id} collided with wall at position {next_head_position} and died.\n"
                )
                player.remove_head()
                self.kill_player(player_id)
                continue

            # There is an edge case where if multiple players move into the same food tile,
            # whichever player's move is processed first will eat the food and grow
            # Obviously, if multiple players are moving into the same tile, there will be a collision
            # but 1 player will still get to eat the food and grow which means that players new
            # tail will be collidable for the same tick
            if self._grid.food_at(next_head_position):
                self._scoreboard.increment_food_eaten(player_id)
                player.last_tick_eaten = self._curr_tick_index
                logger.debug(
                    f"Player {player_id} ate food at position {next_head_position} and grew to length {len(player)}.\n"
                )
            else:
                tail_position = player.remove_tail()
                self._grid.remove_player_at(
                    tail_position, player_id, False
                )  # Mark the old tail position as empty on the grid

            self._grid.add_player_at(
                next_head_position, player_id
            )  # Mark the new head position as occupied by the player on the grid

    def handle_collisions(self):
        # killed -> killer
        killed_players: dict[PlayerID, PlayerID] = {}

        # Detect the killing collisions
        for player_id, player in self._players.items():
            if player.is_dead():
                continue

            if player.collided_with_self():
                killed_players[player_id] = player_id
            else:
                head_position = player.get_head_position()
                other = self._grid.tile_occupied_by_other(player_id, head_position)
                if other is not None:
                    killed_players[player_id] = other

        # Kill based on the detected collisions
        for killed, killer in killed_players.items():
            self.kill_player(killed, killer)

    def handle_food_spawning(self):
        while self._grid.get_num_food() < self._max_num_food:
            food_position = self._grid.get_random_available_food_position()
            if food_position is not None:
                self._grid.place_food_at(food_position)
            else:
                logger.warning("No available positions to place new food!\n")
                break

    def get_grid_size(self) -> GridSize:
        return self._grid.get_grid_size()

    def get_tile_data(self, position: Position) -> TileData:
        return self._grid.get_tile_data(position)

    def get_grid_data(self) -> NDArray[np.uint8]:
        return self._grid.get_grid_data()

    # Run right before the game loop starts to initialize the game state
    # Can't run immediately because the game state is going to depend on
    # the number of players that have joined, and we want to allow players
    # to join before the game starts
    def initialize_game_state(self) -> bool:
        if len(self._players) == 0:
            logger.warning(
                "No players have joined the game, cannot initialize game state.\n"
            )
            return False

        self._initialize_player_positions()
        self._initialize_food_positions()
        return True

    def _initialize_food_positions(self):
        self._grid.fill_available_food_positions()
        grid_size_x, grid_size_y = self.get_grid_size()
        # Maximum possible would be half the grid size filled with food
        self._max_num_food = int(
            min(
                max(
                    1,
                    (
                        (grid_size_x * grid_size_y)
                        * 0.0025
                        * (math.log(len(self._players)) + 1)
                    ),
                ),
                (grid_size_x * grid_size_y) / 2,
            )
        )
        logger.debug(f"Initialized max number of food with {self._max_num_food}")
        self.handle_food_spawning()

    def _initialize_player_positions(self):
        num_players = len(self._players)
        grid_size_x, grid_size_y = self.get_grid_size()
        grid_size_padded = (
            min(grid_size_x, grid_size_y) * 0.65,
            min(grid_size_x, grid_size_y) * 0.65,
        )
        logger.debug(
            f"Initializing player positions for {num_players} players on a grid of size ({grid_size_x}, {grid_size_y}) (padded size: {grid_size_padded})...\n"
        )

        center_x = int((grid_size_x - 1) / 2)
        center_y = int((grid_size_y - 1) / 2)
        logger.debug(f"Grid center: ({center_x}, {center_y})\n")

        num_players_per_layer = 6

        number_of_layers = (
            num_players + num_players_per_layer - 1
        ) / num_players_per_layer
        logger.info(
            f"Number of players: {num_players}, number of layers needed: {number_of_layers}\n"
        )

        distance_from_center = (
            np.sqrt(
                ((grid_size_padded[0] / 2) - 1) ** 2
                + ((grid_size_padded[1] / 2) - 1) ** 2
            )
            / number_of_layers
        )
        distance_stride = distance_from_center
        logger.debug(
            f"Number of layers: {number_of_layers}, initial distance from center: {distance_from_center}\n"
        )

        for i, player_id in enumerate(self._players):
            if i % num_players_per_layer == 0 and i != 0:
                distance_from_center += distance_stride

            angle = ((1 + 2 * i) * np.pi) / num_players_per_layer
            logger.debug(f"Angle for player {i}: {angle:.2f} radians\n")

            x = int(np.round((center_x + distance_from_center * np.cos(angle))))
            y = int(np.round((center_y + distance_from_center * np.sin(angle))))

            # Update the player position in the game state and mark the grid
            self._players[player_id] = SnakePlayer((x, y), player_id)
            self._grid.add_player_at((x, y), player_id)

    def get_player_viewport(self, player: SnakePlayer) -> bytes:
        return self._grid.viewport_as_bytes(
            player.get_head_position(),
            self._viewport_distance_from_center,
            player.get_player_id(),
        )

    def get_player_states(self) -> dict[PlayerID, PlayerView]:
        states: dict[PlayerID, PlayerView] = {}
        living_players = list(self.get_living_players())

        for player_id in living_players:
            player_state = self.create_player_state(player_id)
            if player_state is not None:
                states[player_id] = player_state

        # Always send death frames, even if no one is alive
        self._resolve_spectator_states(
            living_players, self._dead_players, states, death_frame=True
        )

        # Only assign spectators to living players if there are any
        if living_players:
            self._resolve_spectator_states(living_players, self._spectators, states)

        return states

    def _resolve_spectator_states(
        self,
        living_players: list[PlayerID],
        spectator_map: dict[PlayerID, PlayerID | None],
        states: dict[PlayerID, PlayerView],
        *,
        death_frame: bool = False,
    ) -> None:
        for spectator_id, spectatee in spectator_map.items():
            # If the spectator doesn't have a spectatee assigned, or their assigned spectatee is invalid (e.g. they were killed), assign them a new spectatee if possible
            if (
                spectatee is None
                or spectatee not in self._players
                or spectatee in self._dead_players
            ):
                # Death frame: send the spectator their own state so they receive
                # the death notification before switching to spectating.
                if death_frame:
                    player_state = self.create_player_state(
                        spectator_id, is_spectating=False
                    )
                    if player_state is not None:
                        states[spectator_id] = player_state

                # Assign a living player to follow going forward
                if living_players:
                    spectator_map[spectator_id] = random.choice(living_players)

                continue

            # Already assigned a spectatee — send their view
            player_state = self.create_player_state(spectatee, is_spectating=True)
            if player_state is not None:
                states[spectator_id] = player_state

    def create_player_state(
        self, player_id: PlayerID, is_spectating: bool = False
    ) -> PlayerView | None:
        player = self._players.get(player_id)
        if player is None:
            return None

        return PlayerView(
            viewport_size=(
                self._viewport_distance_from_center[0] * 2 + 1,
                self._viewport_distance_from_center[1] * 2 + 1,
            ),
            length=len(player),
            # get kills by counting how many times this player is the killer in the kills dict, excluding self-kills
            kills=sum(
                1
                for killer, killee in self._kills.items()
                if killee != player_id and killer == player_id
            ),
            is_alive=not (is_spectating or player.is_dead()),
            viewport=self.get_player_viewport(player),
            is_spectating=is_spectating,
        )

    def get_all_players(self):
        return set(self._players.keys())

    def get_all_spectators(self):
        return set(self._spectators.keys())

    def all_players_dead(self):
        return len(self._dead_players) == len(self._players)

    def do_tick(self, tick_index: int, max_ticks_since_eaten: int):
        self.update_curr_tick_index(tick_index)

        self.handle_player_moves()
        self.handle_collisions()
        self.handle_food_spawning()
        self.cull_starving_players(max_ticks_since_eaten)


class PlayerView:
    def __init__(
        self,
        viewport_size: tuple[int, int],
        length: int,
        kills: int,
        is_alive: bool,
        viewport: bytes,
        is_spectating: bool,
    ):
        self.viewport_size = viewport_size
        self.length = length
        self.kills = kills
        self.is_alive = is_alive
        self.viewport = viewport
        self.is_spectating = is_spectating
