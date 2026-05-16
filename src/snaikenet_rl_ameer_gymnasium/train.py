import argparse
import os
import sys
import time

from loguru import logger

from snaikenet_rl_ameer_gymnasium.agent import DQNAgent
from snaikenet_rl_ameer_gymnasium.env import SnaikeNetEnv


def _find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Find the latest checkpoint in the directory, preferring latest.pt."""
    latest = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.exists(latest):
        return latest
    # Fall back to highest-numbered episode checkpoint
    candidates = []
    for f in os.listdir(checkpoint_dir) if os.path.isdir(checkpoint_dir) else []:
        if f.startswith("episode_") and f.endswith(".pt"):
            try:
                num = int(f[len("episode_") : -len(".pt")])
                candidates.append((num, os.path.join(checkpoint_dir, f)))
            except ValueError:
                continue
    if candidates:
        candidates.sort()
        return candidates[-1][1]
    return None


def train(
    server_host: str = "localhost",
    server_tcp_port: int = 8888,
    num_episodes: int = 1000,
    max_steps_per_episode: int = 5000,
    save_interval: int = 50,
    checkpoint_dir: str = "checkpoints",
    resume: str | None = None,
    fresh: bool = False,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = SnaikeNetEnv(server_host=server_host, server_tcp_port=server_tcp_port)
    agent = DQNAgent()

    if not fresh:
        checkpoint = resume or _find_latest_checkpoint(checkpoint_dir)
        if checkpoint:
            logger.info(f"Resuming from checkpoint: {checkpoint}")
            agent.load(checkpoint)
    else:
        logger.info("Starting fresh (ignoring existing checkpoints)")

    logger.info(f"Training on device: {agent.device}")
    logger.info(f"Starting training for {num_episodes} episodes")

    total_steps = 0

    # Load existing best reward so we don't overwrite a better checkpoint
    best_reward = float("-inf")
    best_path = os.path.join(checkpoint_dir, "best.pt")
    if os.path.exists(best_path) and not fresh:
        import torch

        saved = torch.load(best_path, map_location="cpu", weights_only=True)
        if "best_reward" in saved:
            best_reward = saved["best_reward"]
            logger.info(f"Existing best reward: {best_reward:.1f}")

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_loss = 0.0
        loss_count = 0
        response_latency_total = 0.0
        response_latency_count = 0
        start_time = time.time()

        for step in range(max_steps_per_episode):
            select_start = time.perf_counter()
            action = agent.select_action(obs)
            select_elapsed = time.perf_counter() - select_start
            next_obs, reward, terminated, truncated, info = env.step(action)
            response_latency_total += select_elapsed + info.get(
                "step_processing_time", 0.0
            )
            response_latency_count += 1
            done = terminated or truncated

            agent.store_transition(obs, action, reward, next_obs, done)
            loss = agent.train_step()

            if loss is not None:
                episode_loss += loss
                loss_count += 1

            episode_reward += reward
            total_steps += 1
            obs = next_obs

            if done:
                break

        elapsed = time.time() - start_time
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0.0
        avg_latency_ms = (
            (response_latency_total / response_latency_count) * 1000.0
            if response_latency_count > 0
            else 0.0
        )

        logger.info(
            f"Episode {episode}/{num_episodes} | "
            f"Steps: {step + 1} | "
            f"Reward: {episode_reward:.1f} | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Avg Latency: {avg_latency_ms:.1f}ms | "
            f"Epsilon: {agent.epsilon:.3f} | "
            f"Buffer: {len(agent.replay_buffer)} | "
            f"Time: {elapsed:.1f}s"
        )

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(best_path, extra={"best_reward": best_reward})
            logger.info(f"New best reward: {best_reward:.1f}")

        # Always save latest for auto-resume
        agent.save(os.path.join(checkpoint_dir, "latest.pt"))

        if episode % save_interval == 0:
            path = os.path.join(checkpoint_dir, f"episode_{episode}.pt")
            agent.save(path)
            logger.info(f"Checkpoint saved: {path}")

    agent.save(os.path.join(checkpoint_dir, "final.pt"))
    logger.info("Training complete")
    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent on SnaikeNET")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8888, help="Server TCP port")
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of training episodes"
    )
    parser.add_argument(
        "--max-steps", type=int, default=5000, help="Max steps per episode"
    )
    parser.add_argument(
        "--save-interval", type=int, default=50, help="Save checkpoint every N episodes"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a specific checkpoint path",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Start training from scratch (ignore existing checkpoints)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")
    train(
        server_host=args.host,
        server_tcp_port=args.port,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        fresh=args.new,
    )
