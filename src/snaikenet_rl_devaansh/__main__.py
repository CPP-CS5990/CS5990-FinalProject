"""
Entry point for the PPO RL agent:

Threading model (mirrors pygame_client_demo):
    - Network thread: runs asyncio event loop, receives game frames, sends direction
    - Main thread: owns the PPOAgentEventHandler and the training loop

Checkpointing:
    - Model weights are saved to --checkpoint path every SAVE_EVERY updates
    - On startup, the checkpoint is loaded if it exists
    - Pass --checkpoint to give each agent instance its own file when running multiple agents
"""

import argparse
import asyncio
import selectors
import sys
import threading
from pathlib import Path

import torch
from loguru import logger

from snaikenet_client.client.client import SnaikenetClient
from snaikenet_rl_devaansh.agent import PPOAgentEventHandler
from snaikenet_rl_devaansh.plots import save_session_plot

DEFAULT_CHECKPOINT = Path("checkpoint/ppo_agent.ppo")
SAVE_EVERY = 10     # save checkpoint every N PPO updates

def setup_logger(verbose: bool):
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level)
    logger.add("logs/rl.agent.log", rotation = "1 MB", level = level)

def save_checkpoint(handler: PPOAgentEventHandler, update_count: int, path: Path):
    path.parent.mkdir(parents = True, exist_ok = True)
    torch.save(
        {
            "network": handler.network.state_dict(),
            "optimizer": handler.ppo.optimizer.state_dict(),
            "update_count": update_count,
        }, path
    )
    logger.info(f"Checkpoint saved at update {update_count} -> {path}")

def load_checkpoint(handler: PPOAgentEventHandler, path: Path) -> int:
    if not path.exists():
        return 0
    data = torch.load(path)
    handler.network.load_state_dict(data["network"])
    if "optimizer" in data and handler.ppo is not None:
        handler.ppo.optimizer.load_state_dict(data["optimizer"])
    if data["update_count"] >= 200 and handler.ppo is not None:
        handler.ppo.ent_coef = 0.01
    logger.info(f"Checkpoint loaded from {path} at update {data['update_count']}")
    return data["update_count"]

async def run_client(
    handler: PPOAgentEventHandler,
    server_host: str,
    server_tcp_port: int,
    client_uuid: str | None,
    checkpoint_path: Path,
):
    client = SnaikenetClient(
        server_host = server_host,
        server_tcp_port = server_tcp_port,
        event_handler = handler,
    )
    await client.start(client_uuid)
    logger.info("RL agent connected to server")

    # wire up the direction sender so the handler can call it
    handler.send_direction = client.set_direction

    # Wait until on_game_start has created the network, then load checkpoint
    while handler.network is None:
        await asyncio.sleep(0.05)
    handler.update_count = load_checkpoint(handler, checkpoint_path)
    handler.session_start_update = handler.update_count

    # Save after every SAVE_EVERY PPO updates; decay ent_coef after early exploration phase
    def on_ppo_update(update_count: int):
        if update_count == 200:
            handler.ppo.ent_coef = 0.01
            logger.info("ent_coef decayed to 0.01 for stable convergence")
        if update_count % SAVE_EVERY == 0:
            save_checkpoint(handler, update_count, checkpoint_path)

    handler.on_ppo_update = on_ppo_update

    while True:
        await asyncio.sleep(1)

def start_network_thread(
        handler: PPOAgentEventHandler,
        server_host: str,
        server_tcp_port: int,
        client_uuid: str | None,
        checkpoint_path: Path,
):
    selector = selectors.SelectSelector()
    loop = asyncio.SelectorEventLoop(selector)
    asyncio.set_event_loop(loop)
    loop.run_until_complete(
        run_client(handler, server_host, server_tcp_port, client_uuid, checkpoint_path)
    )

def main():
    parser = argparse.ArgumentParser(description="SnaikeNET PPO RL Agent")
    parser.add_argument("--host", "-H", type=str, default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", "-p", type=int, default=8888, help="Server TCP port (default: 8888)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--reconnect-uuid", "-r", type=str, default=None, help="UUID to reconnect with")
    parser.add_argument("--spectator", "-s", action="store_true", default=False, help="Connect as spectator")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to checkpoint file (default: {DEFAULT_CHECKPOINT})",
    )
    args = parser.parse_args()
    setup_logger(verbose = args.verbose)
    checkpoint_path = args.checkpoint

    handler = PPOAgentEventHandler()

    net_thread = threading.Thread(
        target = start_network_thread,
        args = (handler, args.host, args.port, args.reconnect_uuid, checkpoint_path),
        daemon = True,
    )
    net_thread.start()

    logger.info(f"RL agent running. Checkpoint: {checkpoint_path}. Press Ctrl+C to stop.")
    try:
        net_thread.join()
    except KeyboardInterrupt:
        logger.info("Stopping RL agent...")
        if handler.network is not None:
            save_checkpoint(handler, handler.update_count, checkpoint_path)
            plot_path = save_session_plot(
                handler.metrics,
                handler.session_start_update,
                handler.update_count,
                checkpoint_path.stem,
            )
            if plot_path:
                logger.info(f"Session plot saved -> {plot_path}")

if __name__ == "__main__":
    main()

