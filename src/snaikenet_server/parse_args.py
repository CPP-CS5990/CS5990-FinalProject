import argparse


class ArgNamespace(argparse.Namespace):
    headless: bool
    verbose: bool
    tcp_port: int
    udp_port: int
    host: str
    grid_size: tuple[int, int]
    viewport_distance: tuple[int, int]
    tick_rate: int
    clean_idle_clients: bool
    client_timeout: float
    max_ticks_since_eaten: int
    preset_player_ids: list[str]
    scoreboard_save_dir: str | None


def parse_args() -> ArgNamespace:
    parser = argparse.ArgumentParser(description="Multiplayer Snake Game Server")
    parser.add_argument(
        "--headless",
        "-H",
        action="store_true",
        help="Run the server in headless mode. No graphical interface will be displayed.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging for debugging purposes.",
    )
    parser.add_argument(
        "--tcp-port",
        type=int,
        default=8888,
        help="TCP port for client registration (default: 8888)",
    )
    parser.add_argument(
        "--udp-port",
        type=int,
        default=8888,
        help="UDP port for game data communication (default: 8888)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host/IP address to bind the server to (default: localhost)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        nargs=2,
        default=(64, 64),
        metavar=("W", "H"),
        help="Grid dimensions as W H (default: 64 64)",
    )
    parser.add_argument(
        "--viewport-distance",
        type=int,
        nargs=2,
        default=(20, 20),
        metavar=("X", "Y"),
        help="Viewport distance from center as X Y (default: 20 20)",
    )
    parser.add_argument(
        "--tick-rate",
        type=int,
        default=6,
        help="Game tick rate in ticks per second (default: 6)",
    )
    parser.add_argument(
        "--client-timeout",
        type=float,
        default=20,
        help="Seconds before an idle client is cleaned up (default: 20)",
    )
    parser.add_argument(
        "--no-clean-idle-clients",
        action="store_false",
        dest="clean_idle_clients",
        default=True,
        help="Disable automatic cleanup of idle clients (enabled by default)",
    )
    parser.add_argument(
        "--max-ticks-since-eaten",
        type=int,
        default=250,
        help="Kill any snake that hasn't eaten in this many ticks (default: 250)",
    )
    parser.add_argument(
        "--preset-player-ids",
        type=str,
        nargs="*",
        default=[],
        metavar="ID",
        help="Preset player IDs assigned to new clients in order before falling back to random UUIDs (default: none)",
    )
    parser.add_argument(
        "--scoreboard-save-dir",
        type=str,
        default="scoreboard_data",
        help="Directory to write per-player scoreboard CSVs (pass empty string to disable saving; default: scoreboard_data)",
    )
    args = parser.parse_args(namespace=ArgNamespace())
    if args.scoreboard_save_dir == "":
        args.scoreboard_save_dir = None
    args.grid_size = tuple(args.grid_size)
    args.viewport_distance = tuple(args.viewport_distance)
    return args
