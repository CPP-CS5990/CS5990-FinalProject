from snaikenet_client.client_data import ClientGameStateFrame
from snaikenet_client.types import ClientTileType


def closest_food_distance(frame: ClientGameStateFrame) -> float:
    grid = frame.grid_data
    width = len(grid)
    if width == 0:
        return float("inf")
    height = len(grid[0])
    cx, cy = width // 2, height // 2

    closest = float("inf")
    for x in range(width):
        for y in range(height):
            if grid[x][y] == ClientTileType.FOOD:
                d = abs(x - cx) + abs(y - cy)
                if d < closest:
                    closest = d
    return closest


def compute_reward(
    prev: ClientGameStateFrame, curr: ClientGameStateFrame
) -> tuple[float, bool]:
    reward = 0.0
    done = False

    if curr.player_length > prev.player_length:
        reward += 1.0
    if curr.num_kills > prev.num_kills:
        reward += 5.0
    if prev.is_alive and not curr.is_alive:
        reward -= 10.0
        done = True
    elif curr.is_alive:
        reward += 0.01

    prev_d = closest_food_distance(prev)
    curr_d = closest_food_distance(curr)
    if prev_d != float("inf") and curr_d != float("inf"):
        reward += 0.05 * (prev_d - curr_d)

    return reward, done
