# When the snake moves, we should only need to remove the tail and add a new head in the direction of movement.
from typing import Iterator
from snaikenet_server.game.types import Direction, PlayerID, Position


class SnakeBodySegment:
    def __init__(
        self,
        position: Position,
        prev_segment: SnakeBodySegment | None = None,
        next_segment: SnakeBodySegment | None = None,
    ):
        self.position = position
        self.prev_segment = prev_segment
        self.next_segment = next_segment

        if prev_segment is not None:
            prev_segment.next_segment = self

        if next_segment is not None:
            next_segment.prev_segment = self

    def is_head(self) -> bool:
        return self.next_segment is None

    def is_tail(self) -> bool:
        return self.prev_segment is None

    def next(self) -> SnakeBodySegment:
        if self.next_segment is None:
            raise ValueError("This segment has no next segment")
        return self.next_segment

    def prev(self) -> SnakeBodySegment:
        if self.prev_segment is None:
            raise ValueError("This segment has no previous segment")
        return self.prev_segment

    def add_next(self, position: Position) -> SnakeBodySegment:
        self.next_segment = SnakeBodySegment(position, prev_segment=self)
        return self.next_segment


class SnakePlayer:
    _head: SnakeBodySegment
    _tail: SnakeBodySegment
    _length: int
    _prev_direction: Direction
    _next_direction: Direction
    _player_id: PlayerID
    _is_alive: bool = True
    last_tick_eaten: int = 0

    def __init__(self, initial_position: Position, player_id: PlayerID):
        # Begins at length 1, so the head and tail are the same tile.
        snake_segment = SnakeBodySegment(initial_position)
        self._head: SnakeBodySegment = snake_segment
        self._tail: SnakeBodySegment = snake_segment
        self._length = 1
        self._prev_direction = Direction.WEST
        self._next_direction = Direction.WEST
        self._player_id = player_id
        self.last_tick_eaten = 0

    def get_head_position(self) -> Position:
        return self._head.position

    def add_head(self) -> Position:
        next_head_position = self.get_next_head_position()
        # Prevent moving backwards directly into the body. The greater than 1 check allows the snake to move backwards when it has length 1, which is necessary for the initial move.
        self._head = self._head.add_next(next_head_position)
        self._prev_direction = self._next_direction
        self._length += 1
        return self._head.position

    def remove_tail(self) -> Position:
        if self._length == 0:
            raise ValueError("Cannot remove tail from an empty snake")
        old_tail_position = self._tail.position
        self._tail = self._tail.next()
        self._tail.prev_segment = None
        self._length -= 1
        return old_tail_position

    def remove_head(self) -> Position:
        if self._length == 0:
            raise ValueError("Cannot remove head from an empty snake")
        old_head_position = self._head.position
        self._head = self._head.prev()
        if self._head is not None:
            self._head.next_segment = None
        self._length -= 1
        return old_head_position

    def get_tail_position(self) -> Position:
        return self._tail.position

    def set_direction(self, direction: Direction):
        if Direction.opposite(direction) == self._prev_direction and self._length > 1:
            return
        self._next_direction = direction

    def get_direction(self) -> Direction:
        return self._next_direction

    def get_next_head_position(self) -> Position:
        return (
            self._head.position[0] + self._next_direction.value[0],
            self._head.position[1] + self._next_direction.value[1],
        )

    def die(self):
        self._is_alive = False

    def is_dead(self) -> bool:
        return not self._is_alive

    def collided_with_self(self) -> bool:
        if len(self) == 1:
            return False

        segment = self._head.prev_segment
        while segment is not None:
            if segment.position == self._head.position:
                return True
            segment = segment.prev_segment
        return False

    def get_player_id(self) -> PlayerID:
        return self._player_id

    def __iter__(self) -> Iterator[Position]:
        current = self._tail
        while current is not None:
            yield current.position
            current = current.next_segment

    def __len__(self) -> int:
        return self._length
