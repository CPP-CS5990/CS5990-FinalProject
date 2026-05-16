import torch

from snaikenet_client.client_data import ClientGameStateFrame
from snaikenet_client.types import ClientTileType
from snaikenet_rl_devaansh.preprocessing import (
    FrameStacker,
    NUM_TILE_TYPES,
    frame_to_tensor,
)
from snaikenet_rl_devaansh.reward import (
    REWARD_CLOSER_FOOD,
    REWARD_DEATH,
    REWARD_FARTHER_FOOD,
    REWARD_FOOD,
    REWARD_KILL,
    REWARD_STEP,
    REWARD_SURVIVAL,
    WALL_PROXIMITY_SCALE,
    _dist_to_food,
    _dist_to_nearest_wall,
    compute_reward,
)
from snaikenet_rl_devaansh.rollout_buffer import RolloutBuffer

W, H = 11, 11


def _make_frame(
    length: int = 1,
    kills: int = 0,
    alive: bool = True,
    food_at: tuple[int, int] | None = None,
    wall_at: list[tuple[int, int]] | None = None,
    seq: int = 0,
) -> ClientGameStateFrame:
    grid = [[ClientTileType.EMPTY] * H for _ in range(W)]
    grid[W // 2][H // 2] = ClientTileType.SNAKE
    if food_at is not None:
        grid[food_at[0]][food_at[1]] = ClientTileType.FOOD
    if wall_at is not None:
        for pos in wall_at:
            grid[pos[0]][pos[1]] = ClientTileType.WALL
    return ClientGameStateFrame(seq, length, kills, alive, False, grid)


def _wall_penalty(frame: ClientGameStateFrame) -> float:
    return WALL_PROXIMITY_SCALE / max(_dist_to_nearest_wall(frame), 1)


# -------------------
# reward.py
# -------------------


def test_devaansh__death_returns_flat_penalty():
    prev = _make_frame(alive=True)
    curr = _make_frame(alive=False)
    assert compute_reward(prev, curr) == REWARD_DEATH


def test_devaansh__food_eaten_adds_food_reward():
    prev = _make_frame(length=1, food_at=(W // 2 + 3, H // 2))
    curr = _make_frame(length=2, food_at=(W // 2 + 3, H // 2))
    reward = compute_reward(prev, curr)
    expected = REWARD_STEP + REWARD_SURVIVAL + _wall_penalty(curr) + REWARD_FOOD
    assert abs(reward - expected) < 1e-6


def test_devaansh__kill_adds_kill_reward():
    prev = _make_frame(kills=0)
    curr = _make_frame(kills=1)
    reward = compute_reward(prev, curr)
    # Both frames have no food → off-screen exploration bonus also fires
    expected = (
        REWARD_STEP
        + REWARD_SURVIVAL
        + _wall_penalty(curr)
        + REWARD_KILL
        + REWARD_CLOSER_FOOD * 0.1
    )
    assert abs(reward - expected) < 1e-6


def test_devaansh__moving_closer_to_food_gives_positive_shaping():
    cx, cy = W // 2, H // 2
    prev = _make_frame(food_at=(cx + 4, cy))  # distance 4
    curr = _make_frame(food_at=(cx + 3, cy))  # distance 3
    reward = compute_reward(prev, curr)
    expected = REWARD_STEP + REWARD_SURVIVAL + _wall_penalty(curr) + REWARD_CLOSER_FOOD
    assert abs(reward - expected) < 1e-6


def test_devaansh__moving_farther_from_food_gives_penalty():
    cx, cy = W // 2, H // 2
    prev = _make_frame(food_at=(cx + 2, cy))  # distance 2
    curr = _make_frame(food_at=(cx + 3, cy))  # distance 3
    reward = compute_reward(prev, curr)
    expected = REWARD_STEP + REWARD_SURVIVAL + _wall_penalty(curr) + REWARD_FARTHER_FOOD
    assert abs(reward - expected) < 1e-6


def test_devaansh__just_ate_skips_distance_shaping():
    cx, cy = W // 2, H // 2
    # length increased (ate food) AND food appears farther — shaping should NOT fire
    prev = _make_frame(length=1, food_at=(cx + 1, cy))
    curr = _make_frame(length=2, food_at=(cx + 5, cy))
    reward = compute_reward(prev, curr)
    expected = REWARD_STEP + REWARD_SURVIVAL + _wall_penalty(curr) + REWARD_FOOD
    assert abs(reward - expected) < 1e-6


def test_devaansh__wall_proximity_penalty_increases_near_wall():
    cx, cy = W // 2, H // 2
    far_frame = _make_frame(wall_at=[(cx + 4, cy)])  # wall 4 tiles away
    near_frame = _make_frame(wall_at=[(cx + 1, cy)])  # wall 1 tile away
    assert _wall_penalty(near_frame) < _wall_penalty(far_frame)


def test_devaansh__dist_to_food_returns_none_when_no_food():
    frame = _make_frame(food_at=None)
    assert _dist_to_food(frame) is None


def test_devaansh__dist_to_food_manhattan_from_center():
    cx, cy = W // 2, H // 2
    frame = _make_frame(food_at=(cx + 3, cy - 1))
    assert _dist_to_food(frame) == 4


def test_devaansh__dist_to_nearest_wall_no_wall_returns_upper_bound():
    frame = _make_frame()
    assert _dist_to_nearest_wall(frame) == W + H


def test_devaansh__dist_to_nearest_wall_finds_closest():
    cx, cy = W // 2, H // 2
    frame = _make_frame(wall_at=[(cx + 2, cy), (cx + 5, cy)])
    assert _dist_to_nearest_wall(frame) == 2


# -------------------
# preprocessing.py
# -------------------


def test_devaansh__frame_to_tensor_shape_and_dtype():
    frame = _make_frame(food_at=(2, 3))
    t = frame_to_tensor(frame)
    assert t.shape == (NUM_TILE_TYPES, W, H)
    assert t.dtype == torch.float32


def test_devaansh__frame_to_tensor_is_one_hot():
    frame = _make_frame(food_at=(1, 2))
    t = frame_to_tensor(frame)
    # Every spatial position sums to exactly 1 across channels
    assert torch.all(t.sum(dim=0) == 1.0)


def test_devaansh__frame_to_tensor_food_channel_correct():
    frame = _make_frame(food_at=(1, 2))
    t = frame_to_tensor(frame)
    assert t[int(ClientTileType.FOOD), 1, 2].item() == 1.0
    assert t[int(ClientTileType.EMPTY), 1, 2].item() == 0.0


def test_devaansh__frame_to_tensor_snake_at_center():
    frame = _make_frame()
    t = frame_to_tensor(frame)
    assert t[int(ClientTileType.SNAKE), W // 2, H // 2].item() == 1.0


def test_devaansh__frame_stacker_reset_shape():
    stacker = FrameStacker(n_frames=2)
    frame = _make_frame()
    out = stacker.reset(frame)
    assert out.shape == (NUM_TILE_TYPES * 2, W, H)


def test_devaansh__frame_stacker_reset_zero_pads_older_frames():
    stacker = FrameStacker(n_frames=2)
    frame = _make_frame()
    out = stacker.reset(frame)
    # Older slot (first half of channels) should be all zeros
    older = out[:NUM_TILE_TYPES]
    assert torch.equal(older, torch.zeros_like(older))
    # Most-recent slot (second half) should match the actual frame
    newest = out[NUM_TILE_TYPES:]
    assert not torch.equal(newest, torch.zeros_like(newest))


def test_devaansh__frame_stacker_step_updates_correctly():
    stacker = FrameStacker(n_frames=2)
    frame1 = _make_frame()
    frame2 = _make_frame(food_at=(1, 1))
    stacker.reset(frame1)
    out = stacker.step(frame2)
    assert out.shape == (NUM_TILE_TYPES * 2, W, H)
    # Second half should match frame2
    assert torch.equal(out[NUM_TILE_TYPES:], frame_to_tensor(frame2))


# -------------------
# rollout_buffer.py
# -------------------


def test_devaansh__rollout_buffer_is_full_triggers_at_capacity():
    buf = RolloutBuffer(capacity=4)
    state = torch.zeros(NUM_TILE_TYPES, W, H)
    assert not buf.is_full()
    for _ in range(4):
        buf.add(state, 0, 0.0, 0.0, 0.5, False)
    assert buf.is_full()


def test_devaansh__rollout_buffer_clear_resets():
    buf = RolloutBuffer(capacity=4)
    state = torch.zeros(NUM_TILE_TYPES, W, H)
    for _ in range(4):
        buf.add(state, 0, 0.0, 0.0, 0.5, False)
    buf.clear()
    assert not buf.is_full()
    assert len(buf.states) == 0


def test_devaansh__rollout_buffer_compute_advantages_shapes():
    buf = RolloutBuffer(capacity=8)
    state = torch.zeros(NUM_TILE_TYPES, W, H)
    for i in range(8):
        buf.add(state, 0, 0.0, 1.0, 0.5, i == 7)
    advantages, _ = buf.compute_advantages(last_value=0.0)
    assert advantages.shape == (8,)


def test_devaansh__rollout_buffer_terminal_zeroes_bootstrap():
    buf = RolloutBuffer(capacity=2, gamma=0.99, gae_lambda=0.95)
    state = torch.zeros(NUM_TILE_TYPES, W, H)
    buf.add(state, 0, 0.0, -100.0, 0.5, True)  # terminal step
    buf.add(state, 0, 0.0, 1.0, 0.5, False)
    advantages, _ = buf.compute_advantages(last_value=1.0)
    # At the terminal step (t=0), next value should be zeroed out
    # delta = reward + gamma * 0 - value = -100 - 0.5 = -100.5
    assert advantages[0].item() < 0
