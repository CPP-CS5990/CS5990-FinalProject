import csv
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from snaikenet_rl_david.client_env import SnaikeNetClientEnv
from snaikenet_rl_david.model import QNet
from snaikenet_rl_david.obs_wrapper import SnaikeNetDQNObsWrapper


def run_validation_episode(env, q_net, max_steps=1000):
    obs, info = env.reset()

    total_reward = 0.0
    episode_length = 0
    max_snake_length = info["player_length"]
    final_snake_length = info["player_length"]
    previous_length = info["player_length"]
    food_collected = 0

    q_net.eval()

    with torch.no_grad():
        for _ in range(max_steps):
            grid = torch.as_tensor(obs["grid"], dtype=torch.float32).unsqueeze(0)
            stats = torch.as_tensor(obs["stats"], dtype=torch.float32).unsqueeze(0)
            q_values = q_net(grid, stats)
            action = int(torch.argmax(q_values, dim=1).item())

            obs, reward, terminated, truncated, info = env.step(action)
            current_length = info["player_length"]
            food_collected += max(0, current_length - previous_length)
            previous_length = current_length
            total_reward += reward
            episode_length += 1
            final_snake_length = info["player_length"]

            max_snake_length = max(max_snake_length, final_snake_length)

            if terminated or truncated:
                break

    q_net.train()

    return total_reward, episode_length, final_snake_length, max_snake_length, food_collected


def main():
    results_path = Path("results/dqn_returns.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    csv_file = results_path.open("w", newline="")
    fieldnames = [
        "phase",
        "episode",
        "timestep",
        "return",
        "episode_length",
        "food_collected",
        "total_food_collected",
        "final_snake_length",
        "max_snake_length",
        "epsilon",
    ]

    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    episode = 0
    episode_return = 0.0
    episode_length = 0
    episode_max_length = 0

    # env = SnaikeNETObsWrapper(SnaikenetGymEnv(step_timeout=30.0))
    env = SnaikeNetDQNObsWrapper(
        SnaikeNetClientEnv(
            server_host="localhost",
            server_tcp_port=8888,
            frame_timeout=30.0,
        )
    )

    action_dim = env.action_space.n

    q_net = QNet(action_dim)
    target_net = QNet(action_dim)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)

    buffer = deque(maxlen=10000)

    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = 80_000
    epsilon = epsilon_start

    learning_starts = 5000
    batch_size = 64
    gamma = 0.99
    target_update_interval = 2000
    validation_interval = 10000

    obs, info = env.reset()

    episode_food_collected = 0
    total_food_collected = 0
    previous_length = info["player_length"]

    for step in range(100000):
        epsilon = max(
            epsilon_end,
            epsilon_start - step / epsilon_decay_steps * (epsilon_start - epsilon_end),
        )
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                grid = torch.as_tensor(obs["grid"], dtype=torch.float32).unsqueeze(0)
                stats = torch.as_tensor(obs["stats"], dtype=torch.float32).unsqueeze(0)
                q_vals = q_net(grid, stats)
                action = int(torch.argmax(q_vals, dim=1).item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        episode_length += 1

        done = terminated or truncated
        buffer.append((obs, action, reward, next_obs, done))
        current_length = info["player_length"]
        episode_max_length = max(episode_max_length, current_length)

        food_delta = max(0, current_length - previous_length)

        episode_food_collected += food_delta
        total_food_collected += food_delta

        previous_length = current_length

        obs = next_obs

        if terminated or truncated:
            writer.writerow({
                "phase": "train",
                "episode": episode,
                "timestep": step,
                "return": episode_return,
                "episode_length": episode_length,
                "final_snake_length": info["player_length"],
                "max_snake_length": episode_max_length,
                "food_collected": episode_food_collected,
                "total_food_collected": total_food_collected,
                "epsilon": epsilon,
            })
            csv_file.flush()

            episode += 1
            episode_return = 0.0
            episode_length = 0
            episode_max_length = 0

            episode_food_collected = 0
            obs, info = env.reset()
            previous_length = info["player_length"]

        # Train
        if step > learning_starts and len(buffer) >= batch_size:
            batch = random.sample(buffer, batch_size)

            states, actions, rewards, next_states, dones = zip(*batch)

            grids = torch.as_tensor(
                np.stack([s["grid"] for s in states]),
                dtype=torch.float32,
            )
            stats = torch.as_tensor(
                np.stack([s["stats"] for s in states]),
                dtype=torch.float32,
            )
            next_grids = torch.as_tensor(
                np.stack([s["grid"] for s in next_states]),
                dtype=torch.float32,
            )
            next_stats = torch.as_tensor(
                np.stack([s["stats"] for s in next_states]),
                dtype=torch.float32,
            )

            actions = torch.as_tensor(actions, dtype=torch.long)
            rewards = torch.as_tensor(rewards, dtype=torch.float32)
            dones = torch.as_tensor(dones, dtype=torch.float32)

            q_all = q_net(grids, stats)
            q_selected = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_max = target_net(next_grids, next_stats).max(dim=1).values
                target = rewards + gamma * next_q_max * (1.0 - dones)

            loss = nn.functional.mse_loss(q_selected, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update target
        if step % target_update_interval == 0:
            target_net.load_state_dict(q_net.state_dict())

        if step % validation_interval == 0 and step > 0:
            if step % validation_interval == 0 and step > 0:
                torch.save({
                    "step": step,
                    "model_state_dict": q_net.state_dict(),
                    "target_state_dict": target_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epsilon": epsilon,
                }, checkpoint_dir / f"dqn_step_{step}.pt")
            val_return, val_length, val_final_length, val_max_length, val_food = run_validation_episode(env, q_net)
            writer.writerow({
                "phase": "validation",
                "episode": episode,
                "timestep": step,
                "return": val_return,
                "episode_length": val_length,
                "food_collected": val_food,
                "total_food_collected": total_food_collected,
                "final_snake_length": val_final_length,
                "max_snake_length": val_max_length,
                "epsilon": 0.0,
            })
            csv_file.flush()

    csv_file.close()
    env.close()


if __name__ == "__main__":
    main()
