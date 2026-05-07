import threading
from dataclasses import dataclass

from snaikenet_client.client.client_event_handler import SnaikenetClientEventHandler
from snaikenet_client.client_data import ClientGameStateFrame


@dataclass(frozen=True)
class BufferedFrame:
    frame: ClientGameStateFrame
    game_epoch: int


class GymSnakeEventHandler(SnaikenetClientEventHandler):
    def __init__(self):
        self.latest_frame = None
        self.latest_frame_epoch = 0
        self.viewport_size = None

        self.started = False
        self.ended = False

        self.game_epoch = 0
        self.last_sequence_number = -1

        self.stale_or_duplicate_frames = 0
        self.dropped_frame_gaps = 0

        self._condition = threading.Condition()

    def on_game_start(self, viewport_size):
        with self._condition:
            self.game_epoch += 1
            self.viewport_size = viewport_size
            self.started = True
            self.ended = False
            self.latest_frame = None
            self.latest_frame_epoch = self.game_epoch
            self.last_sequence_number = -1
            self._condition.notify_all()

    def on_game_restart(self):
        with self._condition:
            self.game_epoch += 1
            self.started = False
            self.ended = False
            self.latest_frame = None
            self.latest_frame_epoch = self.game_epoch
            self.last_sequence_number = -1
            self._condition.notify_all()

    def on_game_about_to_start(self, seconds_until_start):
        with self._condition:
            self.game_epoch += 1
            self.started = False
            self.ended = False
            self.latest_frame = None
            self.latest_frame_epoch = self.game_epoch
            self.last_sequence_number = -1
            self._condition.notify_all()

    def on_game_end(self):
        with self._condition:
            self.ended = True
            self._condition.notify_all()

    def on_game_state_update(self, frame):
        with self._condition:
            if frame.sequence_number <= self.last_sequence_number:
                self.stale_or_duplicate_frames += 1
                return

            if frame.sequence_number != self.last_sequence_number + 1:
                self.dropped_frame_gaps += 1

            self.last_sequence_number = frame.sequence_number
            self.latest_frame = frame
            self.latest_frame_epoch = self.game_epoch
            self.started = True

            self._condition.notify_all()

    def wait_for_start(self, timeout=None):
        with self._condition:
            ok = self._condition.wait_for(
                lambda: self.viewport_size is not None and not self.ended,
                timeout=timeout,
            )
            if not ok:
                raise TimeoutError("Timed out waiting for game start.")

            return self.viewport_size

    def wait_for_frame_after(
        self, previous_epoch, previous_sequence_number, timeout=None
    ):
        def has_new_frame():
            if self.latest_frame is None:
                return False

            if self.latest_frame_epoch > previous_epoch:
                return True

            if self.latest_frame_epoch == previous_epoch:
                return self.latest_frame.sequence_number > previous_sequence_number

            return False

        with self._condition:
            ok = self._condition.wait_for(has_new_frame, timeout=timeout)
            if not ok:
                raise TimeoutError("Timed out waiting for a new game state frame.")

            return BufferedFrame(
                frame=self.latest_frame,
                game_epoch=self.latest_frame_epoch,
            )

    def wait_for_any_frame(self, timeout=None):
        with self._condition:
            ok = self._condition.wait_for(
                lambda: self.latest_frame is not None,
                timeout=timeout,
            )
            if not ok:
                raise TimeoutError("Timed out waiting for a game state frame.")

            return BufferedFrame(
                frame=self.latest_frame,
                game_epoch=self.latest_frame_epoch,
            )

    def wait_for_frame_in_new_epoch(self, previous_epoch, timeout=None):
        def has_new_epoch_frame():
            return (
                self.latest_frame is not None
                and self.latest_frame_epoch > previous_epoch
            )

        with self._condition:
            ok = self._condition.wait_for(has_new_epoch_frame, timeout=timeout)
            if not ok:
                raise TimeoutError("Timed out waiting for a new game epoch frame.")

            return BufferedFrame(
                frame=self.latest_frame,
                game_epoch=self.latest_frame_epoch,
            )
