import asyncio
import collections
import queue
import threading
import time
from collections.abc import Callable

from loguru import logger

from snaikenet_server.game.game_state import GameState, PlayerView
from snaikenet_server.game.pygame_renderer import RendererEvent
from snaikenet_server.game.types import Direction, GridSize, PlayerID
from snaikenet_server.server.server import SnaikenetServer
from snaikenet_server.server.server_event_handler import SnaikenetServerEventHandler


class Game:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        grid_size: GridSize,
        viewport_distance_from_center: tuple[int, int] = (14, 14),
    ):
        self._game_lock: threading.Lock = threading.Lock()
        self._tick_index: int = 0
        self._game_state = GameState(grid_size, viewport_distance_from_center)
        self._grid_size = grid_size
        self._viewport_distance_from_center = viewport_distance_from_center
        self._pending_players: set[PlayerID] = set()
        self._loop = loop
        self._is_being_played = False
        self._stop_event = asyncio.Event()
        self._start_event = asyncio.Event()

    def is_being_played(self) -> bool:
        return self._is_being_played

    def viewport_size(self) -> tuple[int, int]:
        return (
            self._viewport_distance_from_center[0] * 2 + 1,
            self._viewport_distance_from_center[1] * 2 + 1,
        )

    # Returns the tick index
    def tick(self) -> int:
        tick_start_time = time.perf_counter()
        with self._game_lock:
            new_tick = self._tick()
            tick_ms = (time.perf_counter() - tick_start_time) * 1000
            logger.debug("game.tick(): {:.3f}ms", tick_ms)
            return new_tick

    def start_game(self):
        if not self._game_state.initialize_game_state():
            logger.error("Failed to initialize game state, cannot start game loop.")
            return False
        self._is_being_played = True
        return True

    # Threadsafe
    def stop_game(self):
        logger.info("Stop event received, stopping game loop...")
        self.unset_start_event()
        self._threadsafe(self._stop_event.set)

    # Threadsafe
    def set_start_event(self):
        self._threadsafe(self._start_event.set)

    # Threadsafe
    def unset_start_event(self):
        self._threadsafe(self._start_event.clear)

    # Not threadsafe
    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    async def wait_for_game_start(self):
        logger.debug("Waiting for start event...")
        # ensures that players will be able to join while we wait for the start event
        self._is_being_played = False

        while not self._start_event.is_set():
            try:
                await asyncio.wait_for(self._start_event.wait(), 1)
            except asyncio.TimeoutError:
                # if the stop event is set while we are waiting, then we handle that and stop waiting for the game to start
                if self._stop_event.is_set():
                    return False

        # Clear the start event so that if we need to restart the game, we can wait for it to be set again
        self._start_event.clear()
        logger.info("Start signal received, starting game...")
        return self.restart_game()

    def should_restart(self) -> bool:
        with self._game_lock:
            return self.all_players_dead()

    def game_should_continue(self):
        return len(self._get_all_connected_players()) > 0

    def restart_game(self):
        with self._game_lock:
            self._game_state.reset_game_state()

            self._flush_pending_players()
            self._tick_index = -1

            logger.info("Starting new game...")
            return self.start_game()

    def flush_pending_players(self):
        with self._game_lock:
            self._flush_pending_players()

    def get_grid_data(self):
        return self._game_state.get_grid_data()

    def set_player_direction(self, player_id: PlayerID, direction: Direction):
        with self._game_lock:
            self._game_state.set_player_direction(player_id, direction)

    def get_dead_players(self) -> set[PlayerID]:
        return self._game_state.get_dead_players()

    def get_grid_size(self) -> GridSize:
        return self._game_state.get_grid_size()

    def get_player_states(self) -> dict[PlayerID, PlayerView]:
        return self._game_state.get_player_states()

    def all_players_dead(self) -> bool:
        return self._game_state.all_players_dead()

    def delete_player(self, player_id: PlayerID):
        with self._game_lock:
            if player_id in self._pending_players:
                self._pending_players.remove(player_id)
            else:
                self._game_state.delete_player(player_id)
            logger.info(f"Deleted player with ID: {player_id}")

    def add_new_player(self, player_id: PlayerID | None = None) -> PlayerID:
        with self._game_lock:
            return self._add_new_player(player_id)

    def add_spectator(self, player_id: PlayerID):
        with self._game_lock:
            self._add_spectator(player_id)

    def add_pending_player(self, player_id: PlayerID):
        with self._game_lock:
            self._add_pending_player(player_id)

    def _tick(self):
        self._game_state.handle_player_moves()
        self._game_state.handle_collisions()
        self._game_state.handle_food_spawning()
        self._tick_index += 1
        return self._tick_index

    def _flush_pending_players(self):
        for player_id in self._pending_players:
            logger.info(f"Adding pending player {player_id}")
            self._add_new_player(player_id)
        self._pending_players.clear()

    # pending and non-pending players (not including spectators)
    def _get_all_connected_players(self):
        return self._game_state.get_all_players().union(self._pending_players)

    def _add_new_player(self, player_id: PlayerID | None = None) -> PlayerID:
        player_id_ = self._game_state.add_new_player(player_id)
        logger.info(f"Added new player with ID: {player_id_}")
        return player_id_

    def _add_spectator(self, player_id: PlayerID):
        self._game_state.add_spectator(player_id)

    def _add_pending_player(self, player_id: PlayerID):
        self._pending_players.add(player_id)

    # Wrapper
    def _threadsafe(self, fn):
        # noinspection PyTypeChecker
        self._loop.call_soon_threadsafe(fn)


async def handle_sleep(next_tick_time: float):
    sleep_duration = next_tick_time - time.perf_counter()
    if sleep_duration > 0.0:
        await asyncio.sleep(sleep_duration)
    else:
        logger.warning(f"Tick overran by {-sleep_duration * 1000:.2f}ms\n")


def log_tick_rate(tick_times: collections.deque, tick_index: int, tick_interval: float):
    if tick_index % 100 == 0 and len(tick_times) > 0:
        avg_tick_ms = (sum(tick_times) / len(tick_times)) * 1000
        real_tps = 1.0 / (tick_interval + (sum(tick_times) / len(tick_times)))
        logger.debug(
            f"Tick {tick_index} | avg tick time: {avg_tick_ms:.2f}ms | real TPS: {real_tps:.2f}\n"
        )


class _GameEventHandler(SnaikenetServerEventHandler):
    def __init__(self, game: Game):
        self._game = game

    def on_receive_direction(self, client_id: str, direction: Direction):
        # Handle incoming datagram from clients (e.g., player input)
        self._game.set_player_direction(client_id, direction)

    def on_client_disconnect(self, client_id: str):
        self._game.delete_player(client_id)

    def on_new_client_connect(self, client_id: str, spectator: bool = False):
        logger.info(f"Player connected with client ID: {client_id}\n")
        if spectator:
            self._game.add_spectator(client_id)
        elif self._game.is_being_played():
            logger.info(
                f"Game is being played already. Player will be added to the next game"
            )
            self._game.add_pending_player(client_id)
        else:
            self._game.add_new_player(client_id)


async def game_loop(
    game: Game,
    tick_interval: float,
    host: str = "localhost",
    tcp_port: int = 8888,
    udp_port: int = 8888,
    admin_port: int | None = 8889,
    renderer_queue=None,
    clean_idle_clients: bool = True,
    client_timeout_seconds: float = 20,
    headless: bool = True,
    start_game_callback: Callable[[], None] | None = None,
    restart_game_callback: Callable[[], None] | None = None,
):

    server = SnaikenetServer(
        host=host,
        tcp_port=tcp_port,
        udp_port=udp_port,
        admin_port=admin_port,
        event_handler=_GameEventHandler(game),
        client_timeout_seconds=client_timeout_seconds,
    )
    server.set_admin_callbacks(
        start_game=start_game_callback,
        restart_game=restart_game_callback,
    )
    await server.start(clean_idle_clients=clean_idle_clients)

    logger.info("Game thread started, waiting for start signal...")
    while game.is_running():
        # we want to flush the pending players into the game right before we wait for the game start signal
        game.flush_pending_players()
        if not await game.wait_for_game_start():
            continue

        server.broadcast_game_start(game.viewport_size())
        await server.wait_start_game_timer(3)

        next_tick_time = time.perf_counter()
        tick_times = collections.deque(maxlen=100)

        while game.game_should_continue() and game.is_running():
            tick_start_time = time.perf_counter()

            tick_index = game.tick()

            player_states_time = time.perf_counter()
            player_states = game.get_player_states()
            logger.debug(
                f"player_states: {(time.perf_counter() - player_states_time) * 1000:.3f}ms\n"
            )
            await server.broadcast_game_state_frames(player_states, tick_index)
            if not headless:
                renderer_queue.put_nowait(
                    RendererEvent(kind="frame", grid_data=game.get_grid_data())
                )

            tick_times.append(time.perf_counter() - tick_start_time)
            next_tick_time += tick_interval

            await handle_sleep(next_tick_time)

            log_tick_rate(tick_times, tick_index, tick_interval)

            if game.should_restart():
                logger.debug(
                    f"All players are dead at tick {tick_index}, restarting game...\n"
                )
                # wait for players to receive the last frame for this game
                await asyncio.sleep(1)
                server.broadcast_game_restart()
                game.restart_game()
                await server.wait_start_game_timer(1)
                tick_times = collections.deque(maxlen=tick_times.maxlen)
                next_tick_time = time.perf_counter()

        logger.info(
            f"No more players left to play game. Waiting for start signal before starting again..."
        )
    logger.info(f"Stopping network servers...")
    await server.stop()
    if renderer_queue is not None:
        renderer_queue.put_nowait(RendererEvent(kind="shutdown"))
    logger.info("Game task exiting...")