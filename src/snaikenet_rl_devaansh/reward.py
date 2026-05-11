from snaikenet_client.client_data import ClientGameStateFrame
from snaikenet_client.types import ClientTileType

REWARD_FOOD          =  50.0
REWARD_KILL          =  20.0
REWARD_DEATH         = -100.0
REWARD_SURVIVAL      =  0.05   # small bonus for staying alive each step
REWARD_STEP          = -0.2    # small penalty each step to encourage urgency
REWARD_CLOSER_FOOD   =  2.0    # dense reward for moving toward nearest food
REWARD_FARTHER_FOOD  = -0.5    # small penalty for moving away from food
WALL_PROXIMITY_SCALE = -1.0    # penalty per step scaled by how close the head is to the nearest wall


def _dist_to_nearest_wall(frame: ClientGameStateFrame) -> int:
    """Manhattan distance from viewport center (snake head) to the nearest wall tile."""
    grid = frame.grid_data
    H = len(grid)
    W = len(grid[0])
    cx, cy = H // 2, W // 2
    best = H + W  # upper bound
    for r in range(H):
        for c in range(W):
            if grid[r][c] == ClientTileType.WALL:
                d = abs(r - cx) + abs(c - cy)
                if d < best:
                    best = d
    return best


def _dist_to_food(frame: ClientGameStateFrame) -> float | None:
    """Manhattan distance from viewport center (snake head) to nearest food tile."""
    grid = frame.grid_data
    H = len(grid)
    W = len(grid[0])
    cx, cy = H // 2, W // 2  # head is always at viewport center
    best = None
    for r in range(H):
        for c in range(W):
            if grid[r][c] == ClientTileType.FOOD:
                d = abs(r - cx) + abs(c - cy)
                if best is None or d < best:
                    best = d
    return best


def compute_reward(prev: ClientGameStateFrame,
                   curr: ClientGameStateFrame) -> float:
    """
    Compare two consecutive frames to detect game events and returns a reward.

    Events are detected by diffing scalar fields between frames:
        - player_length increased --> ate food
        - num_kills increased     --> killed an enemy
        - is_alive went False     --> died this step
        - distance to food decreased/increased --> dense shaping signal
    """

    if not curr.is_alive and prev.is_alive:
        return REWARD_DEATH

    wall_dist = _dist_to_nearest_wall(curr)
    wall_penalty = WALL_PROXIMITY_SCALE / max(wall_dist, 1)
    reward = REWARD_STEP + REWARD_SURVIVAL + wall_penalty

    if curr.player_length > prev.player_length:
        reward += REWARD_FOOD

    if curr.num_kills > prev.num_kills:
        reward += REWARD_KILL

    just_ate = curr.player_length > prev.player_length
    if not just_ate:
        prev_dist = _dist_to_food(prev)
        curr_dist = _dist_to_food(curr)
        if prev_dist is not None and curr_dist is not None:
            if curr_dist < prev_dist:
                reward += REWARD_CLOSER_FOOD
            elif curr_dist > prev_dist:
                reward += REWARD_FARTHER_FOOD

    return reward
