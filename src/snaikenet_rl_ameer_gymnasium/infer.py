import argparse
import asyncio
import sys
import time

import torch
from loguru import logger

from snaikenet_rl_ameer_gymnasium.agent import DQNAgent, obs_to_tensors
from snaikenet_rl_ameer_gymnasium.env import SnaikeNetEnv


class InferenceAgent:
    """Wraps a trained DQN policy for greedy action selection (no exploration, no training)."""

    def __init__(self, model_path: str, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model from {model_path} (device: {self.device})")

        # Reuse DQNAgent's network construction + load logic, then freeze for inference.
        self._agent = DQNAgent(device=self.device)
        self._agent.load(model_path)
        self._agent.policy_net.eval()
        self._agent.target_net.eval()

    def select_action(self, obs: dict) -> int:
        with torch.no_grad():
            grids, scalars = obs_to_tensors(obs, self.device)
            q_values = self._agent.policy_net(grids, scalars)
            return q_values.argmax(dim=1).item()


def infer(
    model_path: str,
    server_host: str = "localhost",
    server_tcp_port: int = 8888,
    num_episodes: int = 10,
    max_steps_per_episode: int = 5000,
):
    env = SnaikeNetEnv(server_host=server_host, server_tcp_port=server_tcp_port)
    agent = InferenceAgent(model_path)

    logger.info(f"Running inference for {num_episodes} episodes")

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        episode_reward = 0.0
        response_latency_total = 0.0
        response_latency_count = 0
        food_visible_steps = 0
        start_time = time.time()
        step = 0

        for step in range(max_steps_per_episode):
            select_start = time.perf_counter()
            action = agent.select_action(obs)
            select_elapsed = time.perf_counter() - select_start
            next_obs, reward, terminated, truncated, info = env.step(action)
            response_latency_total += select_elapsed + info.get("step_processing_time", 0.0)
            response_latency_count += 1

            dy, dx = obs["closest_food"]
            if dy != 0 or dx != 0:
                food_visible_steps += 1
                logger.debug(f"closest_food offset: dy={dy:+.0f}, dx={dx:+.0f}")

            episode_reward += reward
            obs = next_obs

            if terminated or truncated:
                break

        elapsed = time.time() - start_time
        avg_latency_ms = (
            (response_latency_total / response_latency_count) * 1000.0
            if response_latency_count > 0
            else 0.0
        )
        food_visible_pct = (food_visible_steps / (step + 1)) * 100.0 if step + 1 > 0 else 0.0

        logger.info(
            f"Episode {episode}/{num_episodes} | "
            f"Steps: {step + 1} | "
            f"Reward: {episode_reward:.1f} | "
            f"Food visible: {food_visible_pct:.0f}% | "
            f"Avg Latency: {avg_latency_ms:.1f}ms | "
            f"Time: {elapsed:.1f}s"
        )

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained DQN agent on SnaikeNET (inference only)")
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint (.pt file) to load")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8888, help="Server TCP port")
    parser.add_argument("--episodes", type=int, default=10, help="Number of inference episodes")
    parser.add_argument("--max-steps", type=int, default=5000, help="Max steps per episode")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--num-agents", type=int, default=1, help="Number of agents")
    return parser.parse_args()

async def run_agents(args):
    logger.info(f"Starting {args.num_agents} agents")
    tasks = [
        asyncio.to_thread(
            infer,
            model_path=args.model,
            server_host=args.host,
            server_tcp_port=args.port,
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
        )
        for _ in range(args.num_agents)
    ]
    await asyncio.gather(*tasks)

def main():
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")

    asyncio.run(run_agents(args))



if __name__ == "__main__":
    main()