import tempfile
from pathlib import Path

import torch

from snaikenet_client.client_data import ClientGameStateFrame
from snaikenet_client.types import ClientTileType, ClientDirection
from snaikenet_rl_jack.agent import DoubleDQNAgent
from snaikenet_rl_jack.network import QNetwork, NUM_ACTIONS
from snaikenet_rl_jack.replay_buffer import ReplayBuffer
from snaikenet_rl_jack.reward import closest_food_distance, compute_reward
from snaikenet_rl_jack.state import NUM_TILE_TYPES, VIEWPORT_SIZE, encode_frame


W, H = VIEWPORT_SIZE
DEVICE = torch.device("cpu")


def _make_frame(
    length: int = 1,
    kills: int = 0,
    alive: bool = True,
    food_at: tuple[int, int] | None = None,
    seq: int = 0,
) -> ClientGameStateFrame:
    grid = [[ClientTileType.EMPTY for _ in range(H)] for _ in range(W)]
    grid[W // 2][H // 2] = ClientTileType.SNAKE
    if food_at is not None:
        grid[food_at[0]][food_at[1]] = ClientTileType.FOOD
    return ClientGameStateFrame(seq, length, kills, alive, False, grid)


def test_jack__encode_frame_shape_and_one_hot():
    frame = _make_frame(food_at=(1, 2))
    state = encode_frame(frame, DEVICE)
    assert state.shape == (NUM_TILE_TYPES, W, H)
    assert state.dtype == torch.float32
    # One-hot: every spatial cell sums to exactly 1 across the channel dim
    assert torch.all(state.sum(dim=0) == 1.0)
    # FOOD channel hot at the food location
    assert state[int(ClientTileType.FOOD), 1, 2].item() == 1.0
    # SNAKE channel hot at the head (viewport center)
    assert state[int(ClientTileType.SNAKE), W // 2, H // 2].item() == 1.0


def test_jack__closest_food_distance_uses_viewport_center():
    cx, cy = W // 2, H // 2
    frame = _make_frame(food_at=(cx + 3, cy - 1))
    assert closest_food_distance(frame) == 4
    no_food = _make_frame(food_at=None)
    assert closest_food_distance(no_food) == float("inf")


def test_jack__compute_reward_food_event():
    cx, cy = W // 2, H // 2
    prev = _make_frame(length=1, food_at=(cx + 5, cy))
    curr = _make_frame(length=2, food_at=(cx + 5, cy))
    reward, done = compute_reward(prev, curr)
    # +1.0 ate food, +0.01 survived, no distance change
    assert reward == 1.01
    assert done is False


def test_jack__compute_reward_kill_event():
    prev = _make_frame(kills=0)
    curr = _make_frame(kills=1)
    reward, done = compute_reward(prev, curr)
    # +5.0 kill, +0.01 survived
    assert reward == 5.01
    assert done is False


def test_jack__compute_reward_death_is_terminal():
    prev = _make_frame(alive=True)
    curr = _make_frame(alive=False)
    reward, done = compute_reward(prev, curr)
    assert reward == -10.0
    assert done is True


def test_jack__compute_reward_distance_shaping():
    cx, cy = W // 2, H // 2
    prev = _make_frame(food_at=(cx + 5, cy))  # distance 5
    curr = _make_frame(food_at=(cx + 3, cy))  # distance 3
    reward, _ = compute_reward(prev, curr)
    # +0.01 survive, +0.05 * (5 - 3) = +0.10 shaping
    assert abs(reward - 0.11) < 1e-6


def test_jack__replay_buffer_push_and_sample():
    buffer = ReplayBuffer(capacity=10)
    state = torch.zeros(NUM_TILE_TYPES, W, H)
    for i in range(8):
        buffer.push(state, i % NUM_ACTIONS, float(i), state, i == 7)
    assert len(buffer) == 8
    states, actions, rewards, next_states, dones = buffer.sample(4)
    assert states.shape == (4, NUM_TILE_TYPES, W, H)
    assert actions.shape == (4,) and actions.dtype == torch.long
    assert rewards.shape == (4,) and rewards.dtype == torch.float32
    assert next_states.shape == (4, NUM_TILE_TYPES, W, H)
    assert dones.shape == (4,) and dones.dtype == torch.float32


def test_jack__qnetwork_forward_handles_batched_and_unbatched():
    net = QNetwork(in_channels=NUM_TILE_TYPES, height=W, width=H)
    unbatched = torch.zeros(NUM_TILE_TYPES, W, H)
    batched = torch.zeros(3, NUM_TILE_TYPES, W, H)
    assert net(unbatched).shape == (1, NUM_ACTIONS)
    assert net(batched).shape == (3, NUM_ACTIONS)


def test_jack__agent_select_action_in_valid_range():
    agent = DoubleDQNAgent(NUM_TILE_TYPES, W, H, DEVICE)
    state = torch.zeros(NUM_TILE_TYPES, W, H)
    for _ in range(50):
        a = agent.select_action(state)
        assert 0 <= a < NUM_ACTIONS
        assert ClientDirection(a) in ClientDirection


def test_jack__agent_optimize_step_skips_when_buffer_empty():
    agent = DoubleDQNAgent(NUM_TILE_TYPES, W, H, DEVICE, min_buffer=4, batch_size=4)
    assert agent.optimize_step() is None


def test_jack__agent_target_sync_and_save_load_round_trip():
    agent = DoubleDQNAgent(
        NUM_TILE_TYPES,
        W,
        H,
        DEVICE,
        min_buffer=2,
        batch_size=2,
        target_sync_every=3,
        eps_decay_steps=10,
    )
    state = torch.zeros(NUM_TILE_TYPES, W, H)
    for i in range(6):
        agent.push(state, i % NUM_ACTIONS, 1.0, state, False)
        agent.increment_step()
    loss = agent.optimize_step()
    assert loss is not None and loss >= 0.0
    assert agent.step_count() == 6
    # ε should have decayed below the start value
    assert agent.epsilon() < 1.0

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        agent.save(path)
        fresh = DoubleDQNAgent(NUM_TILE_TYPES, W, H, DEVICE)
        fresh.load(path)
        assert fresh.step_count() == 6
        with torch.no_grad():
            assert torch.equal(agent._online(state), fresh._online(state))
