import asyncio
import dataclasses
import queue
import selectors
import threading
import time
from enum import Enum, auto

from loguru import logger
import pygame


from snaikenet_client.client.client import SnaikenetClient
from snaikenet_client.client.client_event_handler import SnaikenetClientEventHandler
from snaikenet_client.client_data import ClientGameStateFrame
from snaikenet_client.types import ClientDirection, ClientTileType

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
SCOREBOARD_HEIGHT = 40

COLORS = {
    ClientTileType.EMPTY: (20, 20, 20),
    ClientTileType.WALL: (100, 100, 100),
    ClientTileType.FOOD: (220, 50, 50),
    ClientTileType.SNAKE: (50, 220, 80),
    ClientTileType.OTHER_SNAKE: (80, 120, 220),
}


class ClientPhase(Enum):
    WAITING = auto()
    COUNTDOWN = auto()
    PLAYING = auto()
    GAME_OVER = auto()


@dataclasses.dataclass
class ClientEvent:
    kind: str  # "frame", "game_start", "countdown", "game_end", "game_restart"
    frame: ClientGameStateFrame | None = None
    viewport_size: tuple[int, int] | None = None
    seconds_until_start: int | None = None


class QueueClientEventHandler(SnaikenetClientEventHandler):
    def __init__(self):
        self.event_queue: queue.Queue[ClientEvent] = queue.Queue()
        self._curr_sequence_number: int = -1
        self._reset_current_sequence_number()

    def on_game_state_update(self, frame: ClientGameStateFrame):
        logger.debug(
            f"Received frame seq={frame.sequence_number}, "
            f"length={frame.player_length}, kills={frame.num_kills}, "
            f"alive={frame.is_alive}, spectating={frame.is_spectating}"
        )

        self.event_queue.put(ClientEvent(kind="frame", frame=frame))

        if (
            self._curr_sequence_number != -1
            and self._curr_sequence_number + 1 != frame.sequence_number
        ):
            logger.warning(
                f"Out of order sequence numbers: got {frame.sequence_number}, "
                f"expected {self._curr_sequence_number + 1}"
            )

        # check to make sure frame is not behind current frame
        if self._curr_sequence_number < frame.sequence_number:
            self._curr_sequence_number = frame.sequence_number

    def on_game_start(self, viewport_size: tuple[int, int]):
        logger.info(f"Game starting with viewport size {viewport_size}")
        self.event_queue.put(
            ClientEvent(kind="game_restart", viewport_size=viewport_size)
        )
        self._reset_current_sequence_number()

    def on_game_about_to_start(self, seconds_until_start: int):
        self.event_queue.put(
            ClientEvent(kind="countdown", seconds_until_start=seconds_until_start)
        )
        self._reset_current_sequence_number()

    def on_game_end(self):
        logger.info("Game ended")
        self.event_queue.put(ClientEvent(kind="game_end"))
        self._reset_current_sequence_number()

    def on_game_restart(self):
        logger.info("Game restarting")
        self.event_queue.put(ClientEvent(kind="game_restart"))
        self._reset_current_sequence_number()

    def _reset_current_sequence_number(self):
        self._curr_sequence_number = -1


async def run_client(
    handler: QueueClientEventHandler,
    direction_queue: queue.Queue[ClientDirection],
    server_host: str = "localhost",
    server_tcp_port: int = 8888,
    client_uuid: str | None = None,
    spectator: bool = False,
):
    client = SnaikenetClient(
        server_host=server_host,
        server_tcp_port=server_tcp_port,
        event_handler=handler,
        is_spectator=spectator,
    )
    await client.start(client_uuid)
    logger.info("Client connected to server")

    # Default direction so the snake has something valid set
    client.set_direction(ClientDirection.NORTH)

    while True:
        await asyncio.sleep(0.01)

        while True:
            try:
                new_dir = direction_queue.get_nowait()
            except queue.Empty:
                break

            logger.debug(f"Setting direction: {new_dir}")
            client.set_direction(new_dir)


def start_network_thread(
    handler: QueueClientEventHandler,
    direction_queue: queue.Queue[ClientDirection],
    server_host: str = "localhost",
    server_tcp_port: int = 8888,
    client_uuid: str | None = None,
    spectator: bool = False,
):
    selector = selectors.SelectSelector()
    loop = asyncio.SelectorEventLoop(selector)
    asyncio.set_event_loop(loop)
    loop.run_until_complete(
        run_client(
            handler=handler,
            direction_queue=direction_queue,
            server_host=server_host,
            server_tcp_port=server_tcp_port,
            client_uuid=client_uuid,
            spectator=spectator,
        )
    )


ACTION_TO_DIRECTION = {
    0: ClientDirection.NORTH,
    1: ClientDirection.SOUTH,
    2: ClientDirection.WEST,
    3: ClientDirection.EAST,
}


class ClientController:
    """
    Sync bridge around the async/networked SnaikeNET client.

    Gym calls this synchronously:
      - reset_episode() -> first usable frame
      - step(action) -> next frame after applying action
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8888,
        client_uuid: str | None = None,
        spectator: bool = False,
        event_timeout_s: float = 10.0,
    ):
        self.host = host
        self.port = port
        self.client_uuid = client_uuid
        self.spectator = spectator
        self.event_timeout_s = event_timeout_s

        self.handler = QueueClientEventHandler()
        self.direction_queue: queue.Queue[ClientDirection] = queue.Queue()

        self.net_thread: threading.Thread | None = None

        self.phase = ClientPhase.WAITING
        self.latest_frame: ClientGameStateFrame | None = None
        self.old_frame: ClientGameStateFrame | None = None
        self.viewport_size: tuple[int, int] | None = None

        # self.viewport_w = 0
        # self.viewport_h = 0
        # self.tile_size = 1
        # self.grid_offset_x = 0
        # self.grid_offset_y = 0
        # self.countdown_seconds = 0
        # self.running = True

        self.game_over = False
        self.started = False

        # for rendering
        # pygame.init()
        # self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        # pygame.display.set_caption("SnaikeNET")
        # self.font = pygame.font.SysFont("consolas", 18)
        # self.big_font = pygame.font.SysFont("consolas", 120)
        # self.clock = pygame.time.Clock()

    def start(self):
        if self.started:
            return

        self.net_thread = threading.Thread(
            target=start_network_thread,
            args=(
                self.handler,
                self.direction_queue,
                self.host,
                self.port,
                self.client_uuid,
                self.spectator,
            ),
            daemon=True,
        )
        self.net_thread.start()
        self.started = True

    def reset_episode(self, seed=None) -> ClientGameStateFrame:
        """
        Wait for a clean playable frame for the next episode.
        Assumes the server is already running and will eventually produce
        game_start / countdown / frame events.
        """
        if not self.started:
            self.start()

        self._drain_events()

        self.phase = ClientPhase.WAITING
        self.latest_frame = None
        self.viewport_size = None
        self.game_over = False

        deadline = time.monotonic() + self.event_timeout_s

        while True:
            timeout_left = deadline - time.monotonic()
            if timeout_left <= 0:
                raise TimeoutError(
                    "Timed out waiting for initial frame in reset_episode()."
                )

            try:
                ev = self.handler.event_queue.get(timeout=timeout_left)
            except queue.Empty:
                break
            # ev = self.handler.event_queue.get(timeout=timeout_left)

            if ev.kind == "game_start":
                self.viewport_size = ev.viewport_size
                self.phase = ClientPhase.WAITING

            elif ev.kind == "countdown":
                self.phase = ClientPhase.COUNTDOWN

            elif ev.kind == "game_restart":
                self.phase = ClientPhase.WAITING
                self.latest_frame = None
                self.game_over = False

            elif ev.kind == "frame":
                self.latest_frame = ev.frame
                self.phase = ClientPhase.PLAYING
                self.game_over = False
                return ev.frame

            elif ev.kind == "game_end":
                self.phase = ClientPhase.GAME_OVER
                self.game_over = True
                # Keep waiting for the next restart / frame rather than returning failure immediately

    def step(self, action: int) -> ClientGameStateFrame:
        """
        Send one action and block until the next newer frame arrives.
        """
        if self.latest_frame is None:
            raise RuntimeError("step() called before reset_episode().")

        if self.game_over:
            raise RuntimeError(
                "step() called after game over. Call reset_episode() first."
            )

        direction = self._map_action(action)
        previous_seq = self.latest_frame.sequence_number

        self.direction_queue.put(direction)

        deadline = time.monotonic() + self.event_timeout_s

        print(phase)
        while True:
            timeout_left = deadline - time.monotonic()
            if timeout_left <= 0:
                raise TimeoutError(
                    f"Timed out waiting for next frame after action {action}."
                )

            ev = self.handler.event_queue.get(timeout=timeout_left)

            if ev.kind == "frame":
                frame = ev.frame
                if frame.sequence_number > previous_seq:
                    self.latest_frame = frame
                    self.phase = ClientPhase.PLAYING
                    return frame
                # Ignore stale/duplicate frames

            elif ev.kind == "game_end":
                self.phase = ClientPhase.GAME_OVER
                self.game_over = True

                # If game ends before another frame arrives, return the last known frame.
                # The Gym env can interpret done=True from this.
                if self.latest_frame is not None:
                    return self.latest_frame

            elif ev.kind == "game_restart":
                self.phase = ClientPhase.WAITING
                self.game_over = False
                self.latest_frame = None
                raise RuntimeError(
                    "Received game_restart during step(); call reset_episode()."
                )

            elif ev.kind == "game_start":
                self.viewport_size = ev.viewport_size
                self.update_grid_layout(*ev.viewport_size)
                self.phase = ClientPhase.WAITING

            elif ev.kind == "countdown":
                self.countdown_seconds = (
                    ev.seconds_until_start if ev.seconds_until_start else -1
                )
                if phase != ClientPhase.PLAYING:
                    phase = ClientPhase.COUNTDOWN

            # Render
            # self.screen.fill((0, 0, 0))

            # # print(phase)
            # if phase == ClientPhase.WAITING:
            #     if self.viewport_w > 0 and self.viewport_h > 0:
            #         self.render_empty_grid(
            #             self.screen,
            #             self.font,
            #             self.viewport_w,
            #             self.viewport_h,
            #             self.tile_size,
            #             self.grid_offset_x,
            #             self.grid_offset_y,
            #         )
            #     else:
            #         self.render_waiting(self.screen, self.font)
            # elif phase == ClientPhase.COUNTDOWN:
            #     if self.viewport_w > 0 and self.viewport_h > 0:
            #         self.render_empty_grid(
            #             self.screen,
            #             self.font,
            #             self.viewport_w,
            #             self.viewport_h,
            #             self.tile_size,
            #             self.grid_offset_x,
            #             self.grid_offset_y,
            #         )
            #     self.render_countdown(self.screen, self.big_font, self.countdown_seconds)
            # elif phase == ClientPhase.PLAYING and self.latest_frame is not None:
            #     # Recompute layout from actual frame grid in case it differs
            #     grid_w = len(self.latest_frame.grid_data)
            #     grid_h = len(self.latest_frame.grid_data[0]) if grid_w else 0
            #     if grid_w != self.viewport_w or grid_h != self.viewport_h:
            #         self.update_grid_layout(grid_w, grid_h)
            #     self.render_frame(
            #         self.screen,
            #         self.current_frame,
            #         self.font,
            #         self.tile_size,
            #         self.grid_offset_x,
            #         self.grid_offset_y,
            #     )
            # elif phase == ClientPhase.GAME_OVER:
            #     logger.info("GAME END")
            # pygame.display.flip()
            # self.clock.tick(60)

    def close(self):
        # The client thread is daemonized, so this is mostly a logical close.
        self.started = False

    def _map_action(self, action: int) -> ClientDirection:
        try:
            return ACTION_TO_DIRECTION[action]
        except KeyError as e:
            raise ValueError(f"Invalid discrete action: {action}") from e

    def _drain_events(self):
        while True:
            try:
                self.handler.event_queue.get_nowait()
            except queue.Empty:
                break

    def compute_tile_size(self, viewport_w: int, viewport_h: int) -> int:
        """Compute the largest square tile size that fits the viewport in the window."""
        available_h = WINDOW_HEIGHT - SCOREBOARD_HEIGHT
        tile_from_w = WINDOW_WIDTH // viewport_w
        tile_from_h = available_h // viewport_h
        return max(1, min(tile_from_w, tile_from_h))

    # def update_grid_layout(self, vw: int, vh: int):
    #     # nonlocal viewport_w, viewport_h, tile_size, grid_offset_x, grid_offset_y
    #     self.viewport_w, viewport_h = vw, vh
    #     self.tile_size = self.compute_tile_size(vw, vh)
    #     self.grid_pixel_w = vw * self.tile_size
    #     self.grid_pixel_h = vh * self.tile_size
    #     self.grid_offset_x = (WINDOW_WIDTH - self.grid_pixel_w) // 2
    #     self.grid_offset_y = (
    #         SCOREBOARD_HEIGHT + (WINDOW_HEIGHT - SCOREBOARD_HEIGHT - self.grid_pixel_h) // 2
    #     )

    def render_frame(
        self,
        screen: pygame.Surface,
        frame: ClientGameStateFrame,
        font: pygame.font.Font,
        tile_size: int,
        grid_offset_x: int,
        grid_offset_y: int,
    ):
        grid = frame.grid_data
        for x, col in enumerate(grid):
            for y, tile in enumerate(col):
                color = COLORS.get(tile, (0, 0, 0))
                rect = pygame.Rect(
                    grid_offset_x + x * tile_size,
                    grid_offset_y + y * tile_size,
                    tile_size,
                    tile_size,
                )
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (40, 40, 40), rect, 1)

        self.render_scoreboard(screen, font, frame)

    def render_empty_grid(
        self,
        screen: pygame.Surface,
        font: pygame.font.Font,
        viewport_w: int,
        viewport_h: int,
        tile_size: int,
        grid_offset_x: int,
        grid_offset_y: int,
    ):
        for x in range(viewport_w):
            for y in range(viewport_h):
                rect = pygame.Rect(
                    grid_offset_x + x * tile_size,
                    grid_offset_y + y * tile_size,
                    tile_size,
                    tile_size,
                )
                pygame.draw.rect(screen, COLORS[ClientTileType.EMPTY], rect)
                pygame.draw.rect(screen, (40, 40, 40), rect, 1)

        self.render_scoreboard(screen, font)

    def render_scoreboard(
        self,
        screen: pygame.Surface,
        font: pygame.font.Font,
        frame: ClientGameStateFrame | None = None,
    ):
        scoreboard_rect = pygame.Rect(0, 0, WINDOW_WIDTH, SCOREBOARD_HEIGHT)
        pygame.draw.rect(screen, (30, 30, 30), scoreboard_rect)

        if frame is not None:
            status = "ALIVE" if frame.is_alive else "DEAD"
            status_color = (80, 255, 120) if frame.is_alive else (255, 60, 60)
            info = (
                f"Tick: {frame.sequence_number}    "
                f"Length: {frame.player_length}    "
                f"Kills: {frame.num_kills}    "
                f"{status}"
            )
            text_surface = font.render(info, True, status_color)
        else:
            text_surface = font.render("Waiting for game...", True, (180, 180, 180))

        screen.blit(
            text_surface, (10, (SCOREBOARD_HEIGHT - text_surface.get_height()) // 2)
        )

    def render_countdown(
        self, screen: pygame.Surface, big_font: pygame.font.Font, seconds: int
    ):
        text = big_font.render(str(seconds), True, (255, 255, 100))
        x = (WINDOW_WIDTH - text.get_width()) // 2
        y = (WINDOW_HEIGHT - text.get_height()) // 2
        screen.blit(text, (x, y))

    def render_waiting(self, screen: pygame.Surface, font: pygame.font.Font):
        text = font.render("Waiting for server...", True, (180, 180, 180))
        x = (WINDOW_WIDTH - text.get_width()) // 2
        y = (WINDOW_HEIGHT - text.get_height()) // 2
        screen.blit(text, (x, y))
