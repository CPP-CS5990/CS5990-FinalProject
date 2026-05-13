import random

import torch

from snaikenet_agent.old.obs_wrapper import SnaikeNETObsWrapper
from snaikenet_agent.train_dqn import QNet
from snaikenet_agent.old.gym_env import SnaikenetGymEnv


def main():
    checkpoint_path = "checkpoints/dqn_step_90000.pt"

    env = SnaikeNETObsWrapper(SnaikenetGymEnv(step_timeout=30.0))
    action_dim = env.action_space.n

    q_net = QNet(action_dim)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    q_net.load_state_dict(checkpoint["model_state_dict"])
    q_net.eval()

    obs, info = env.reset()

    try:
        recent_actions = []
        while True:
            with torch.no_grad():
                grid = torch.as_tensor(obs["grid"], dtype=torch.float32).unsqueeze(0)
                stats = torch.as_tensor(obs["stats"], dtype=torch.float32).unsqueeze(0)

                q_values = q_net(grid, stats)
                if random.random() < 0.2:
                    action = env.action_space.sample()
                else:
                    action = int(torch.argmax(q_values, dim=1).item())

            recent_actions.append(action)
            recent_actions = recent_actions[-6:]

            if len(recent_actions) == 6 and recent_actions[:2] * 3 == recent_actions:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                obs, info = env.reset()

    finally:
        env.close()


if __name__ == "__main__":
    main()
