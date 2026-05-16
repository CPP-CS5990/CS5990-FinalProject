import argparse
import asyncio
import random
import sys
from collections import deque

import numpy as np
import torch
from loguru import logger

from snaikenet_rl_david.model import QNet
from snaikenet_client.client.client import SnaikenetClient
from snaikenet_client.client.client_event_handler import SnaikenetClientEventHandler
from snaikenet_client.client_data import ClientGameStateFrame
from snaikenet_client.types import ClientDirection, ClientTileType

OPPOSITE_ACTION = {
    0: 1,
    1: 0,
    2: 3,
    3: 2,
}

ACTION_TO_DIRECTION = {
    0: ClientDirection.NORTH,
    1: ClientDirection.SOUTH,
    2: ClientDirection.EAST,
    3: ClientDirection.WEST,
}

NUM_TILE_TYPES = 5


def frame_to_model_obs(frame: ClientGameStateFrame) -> dict:
    grid = np.asarray(frame.grid_data, dtype=np.uint8)

    one_hot = np.zeros((NUM_TILE_TYPES, *grid.shape), dtype=np.float32)
    for tile_type in range(NUM_TILE_TYPES):
        one_hot[tile_type] = (grid == tile_type).astype(np.float32)

    dy_to_food, dx_to_food = closest_food_offset(grid)

    stats = np.array(
        [
            float(frame.player_length) / 100.0,
            float(frame.num_kills) / 10.0,
            dy_to_food,
            dx_to_food,
        ],
        dtype=np.float32,
    )

    return {
        "grid": one_hot,
        "stats": stats,
    }


def closest_food_offset(grid: np.ndarray) -> tuple[float, float]:
    h, w = grid.shape
    cy, cx = h // 2, w // 2

    food_positions = np.argwhere(grid == ClientTileType.FOOD)

    if len(food_positions) == 0:
        return 0.0, 0.0

    deltas = food_positions - np.array([cy, cx])
    distances = np.abs(deltas).sum(axis=1)

    dy, dx = deltas[np.argmin(distances)]

    return float(dy) / h, float(dx) / w


class DQNClientHandler(SnaikenetClientEventHandler):
    def __init__(
        self,
        q_net: QNet,
        exploration_rate: float = 0.0,
    ):
        self.q_net = q_net
        self.exploration_rate = exploration_rate

        self.client: SnaikenetClient | None = None
        self.last_sequence_number = -1
        self.current_direction = ClientDirection.NORTH
        self.recent_actions = deque(maxlen=6)

    def on_game_start(self, viewport_size: tuple[int, int]):
        print(f"Game started. viewport_size={viewport_size}")
        self.last_sequence_number = -1

    def on_game_restart(self):
        print("Game restarted.")
        self.last_sequence_number = -1

    def on_game_about_to_start(self, seconds_until_start: int):
        print(f"Game starting in {seconds_until_start}s.")
        self.last_sequence_number = -1

    def on_game_end(self):
        print("Game ended.")

    def on_game_state_update(self, frame: ClientGameStateFrame):
        if frame.sequence_number <= self.last_sequence_number:
            return

        self.last_sequence_number = frame.sequence_number

        if self.client is None:
            return

        if not frame.is_alive or frame.is_spectating:
            return

        action = self.choose_action(frame)

        self.recent_actions.append(action)

        if len(self.recent_actions) == 6 and list(self.recent_actions)[0:2] * 3 == list(
            self.recent_actions
        ):
            invalid_reverse = OPPOSITE_ACTION[int(self.current_direction)]
            valid_actions = [a for a in range(4) if a != invalid_reverse]
            action = random.choice(valid_actions)
            self.recent_actions.clear()
        direction = ACTION_TO_DIRECTION[action]

        if direction != self.current_direction:
            self.current_direction = direction
            self.client.set_direction(direction)

    def choose_action(self, frame: ClientGameStateFrame) -> int:
        obs = frame_to_model_obs(frame)

        with torch.no_grad():
            grid = torch.as_tensor(obs["grid"], dtype=torch.float32).unsqueeze(0)
            stats = torch.as_tensor(obs["stats"], dtype=torch.float32).unsqueeze(0)

            q_values = self.q_net(grid, stats).squeeze(0)

        invalid_reverse = OPPOSITE_ACTION[int(self.current_direction)]
        q_values[invalid_reverse] = -float("inf")

        if random.random() < self.exploration_rate:
            valid_actions = [a for a in range(4) if a != invalid_reverse]
            return random.choice(valid_actions)

        return int(torch.argmax(q_values).item())


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--exploration-rate", type=float, default=0.0)
    args = parser.parse_args()

    q_net = QNet(action_dim=4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    q_net.load_state_dict(checkpoint["model_state_dict"])
    q_net.eval()

    handler = DQNClientHandler(
        q_net=q_net,
        exploration_rate=args.exploration_rate,
    )

    client = SnaikenetClient(
        server_host=args.host,
        server_tcp_port=args.port,
        event_handler=handler,
    )

    handler.client = client

    await client.start()
    client.set_direction(ClientDirection.NORTH)

    print(f"Connected as {client.get_client_id()}")

    try:
        await asyncio.Event().wait()
    finally:
        await client.stop()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    asyncio.run(main())
