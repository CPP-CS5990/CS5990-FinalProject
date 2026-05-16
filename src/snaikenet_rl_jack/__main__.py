import argparse
import asyncio
import selectors
import sys
from pathlib import Path

import torch
from loguru import logger

from snaikenet_client.client.client import SnaikenetClient
from snaikenet_rl_jack.client import JackAgentEventHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jack's RL agent for SnaikeNET")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints/jack"))
    parser.add_argument("--load-checkpoint", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--optimize-every", type=int, default=4)
    parser.add_argument(
        "--eval", action="store_true", help="Greedy actions, no learning"
    )
    parser.add_argument("--reconnect-uuid", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def setup_logger(verbose: bool):
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level)
    logger.add("logs/jack.log", rotation="1 MB", level=level)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


async def run(args: argparse.Namespace):
    device = resolve_device(args.device)
    handler = JackAgentEventHandler(
        device=device,
        save_dir=args.save_dir,
        save_every=args.save_every,
        optimize_every=args.optimize_every,
        eval_only=args.eval,
        load_checkpoint=args.load_checkpoint,
    )
    client = SnaikenetClient(
        server_host=args.host,
        server_tcp_port=args.port,
        event_handler=handler,
    )
    handler.bind_set_direction(client.set_direction)
    await client.start(args.reconnect_uuid)
    logger.info(f"Connected; client UUID={client.get_client_id()}")

    try:
        while True:
            await asyncio.sleep(60)
    except asyncio.CancelledError:
        pass
    finally:
        await client.stop()


def main():
    args = parse_args()
    setup_logger(args.verbose)
    selector = selectors.SelectSelector()
    loop = asyncio.SelectorEventLoop(selector)
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run(args))
    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down")


if __name__ == "__main__":
    main()
