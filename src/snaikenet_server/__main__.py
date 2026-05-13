# Game Server Initialization:
# Parse program arguments
# Create a Game State object
# Create a UDP server to listen for player connections and messages (mock for now)
#   - operate on game state object
# Create a command interface for server administration and debugging (allows us to start/stop the game, add/remove players, etc.)
# - (Later) Create an HTTP server to expose the command interface via a REST API (for remote administration and integration with other services)
import asyncio
import queue
import threading
# Events:
# When new player makes a connection, they are assigned an ID and are added to the game state
# When the connection is closed, the player is removed
# When the game starts, the server sends the initial game state to all players
# - If a player is killed
#   - The method of death is determined
#   - Whoever caused the death is credited with a kill (if applicable)
#   - The server sends a message to the player that they have been killed and removes them from the game state
#   - Each segment of the snake is removed from the board and becomes an available food location

# Game start initialization:
# - Place players at random starting positions on the board (everyone starts with a length of 1)
# - Place initial food items at random locations on the board
# - If headless is disabled, render the initial game state with pygame

# On game reset:
# - Clear player data and reset game state
# - Do not reset player connections, just reset the game state and send the new state to all players

# Game Loop (runs at a fixed tick rate):
# - Check for collisions
# - Handle collisions
#   - If player collides with food, grow from the head and add new food to the board
#   - If player collides with wall, die
#   - If player collides with another player, die
#   - If player collides with itself, die
# - Move players
#   - Check that direction is valid (not directly opposite of current direction)
#       - If invalid, ignore the direction change and keep moving in the current direction
# - Calculate viewport for all players
# - Send viewport to all players
# - If headless is disabled, render game state with pygame

# On the UDP Server, we will listen for messages from players:
#
import sys
from snaikenet_server.game.game import Game, game_loop
from snaikenet_server.parse_args import parse_args
from loguru import logger

from snaikenet_server.server_commands import (
    GameServerCommandInterface,
    console_loop,
)


def setup_logger(verbose: bool):
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level)
    logger.add(
        "logs/server.log", rotation="1 MB", level=level
    )  # Log to file as well, with rotation


async def async_main(args, renderer_queue):
    loop = asyncio.get_running_loop()

    tick_interval = 1 / args.tick_rate
    game = Game(
        loop,
        args.grid_size,
        viewport_distance_from_center=args.viewport_distance,
        max_ticks_since_eaten=args.max_ticks_since_eaten,
        scoreboard_save_dir=args.scoreboard_save_dir,
    )

    def stop_server():
        game.stop_game()
        command_interface.set_stop_signal()

    command_interface = GameServerCommandInterface(
        start_game=game.set_start_event,
        stop_server=stop_server,
        restart_game=game.restart_game,
    )

    # Create thread instances
    await asyncio.gather(
        game_loop(
            game,
            tick_interval,
            host=args.host,
            tcp_port=args.tcp_port,
            udp_port=args.udp_port,
            renderer_queue=renderer_queue,
            clean_idle_clients=args.clean_idle_clients,
            client_timeout_seconds=args.client_timeout,
            headless=args.headless,
            preset_player_ids=args.preset_player_ids,
        ),
        asyncio.to_thread(console_loop, command_interface),
    )


def main():
    args = parse_args()
    setup_logger(args.verbose)

    renderer_queue = queue.Queue()

    if args.headless:
        # No pygame, so we can just run asyncio on the main thread
        asyncio.run(async_main(args, renderer_queue))
    else:
        logger.info(
            "Detected not running in headless: Starting pygame renderer thread..."
        )
        # pygame must run on the main thread (required on macOS), so push
        # the asyncio event loop into a background thread instead.
        async_thread = threading.Thread(
            target=lambda: asyncio.run(async_main(args, renderer_queue)),
            daemon=True,
        )
        async_thread.start()

        from snaikenet_server.game import pygame_renderer

        pygame_renderer.render_loop(renderer_queue)


def run_main():
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Server shutting down due to keyboard interrupt")


if __name__ == "__main__":
    run_main()
