import queue
from dataclasses import dataclass

from snaikenet_client.client.client_event_handler import SnaikenetClientEventHandler
from snaikenet_client.client_data import ClientGameStateFrame


@dataclass
class GameEvent:
    kind: str
    frame: ClientGameStateFrame | None = None
    viewport_size: tuple[int, int] | None = None


class ClientOnlyEventHandler(SnaikenetClientEventHandler):
    def __init__(self):
        self.event_queue: queue.Queue[GameEvent] = queue.Queue()
        self.last_sequence_number = -1

    def reset_sequence(self):
        self.last_sequence_number = -1

    def on_game_start(self, viewport_size: tuple[int, int]):
        self.reset_sequence()
        self.event_queue.put(GameEvent("game_start", viewport_size=viewport_size))

    def on_game_restart(self):
        self.reset_sequence()
        self.event_queue.put(GameEvent("game_restart"))

    def on_game_about_to_start(self, seconds_until_start: int):
        self.reset_sequence()
        self.event_queue.put(GameEvent("game_about_to_start"))

    def on_game_end(self):
        self.event_queue.put(GameEvent("game_end"))

    def on_game_state_update(self, frame: ClientGameStateFrame):
        if frame.is_spectating:
            return

        if frame.sequence_number <= self.last_sequence_number:
            return

        self.last_sequence_number = frame.sequence_number
        self.event_queue.put(GameEvent("frame", frame=frame))
