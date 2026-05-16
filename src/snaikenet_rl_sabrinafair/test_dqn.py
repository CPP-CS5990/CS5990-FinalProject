import argparse
import sys
import time

from loguru import logger
from stable_baselines3 import DQN

from snaikenet_rl_sabrinafair.client_env import ClientEnv
from snaikenet_rl_sabrinafair.game_controller import ClientController


def main():
    parser = argparse.ArgumentParser(description="Test a trained DQN agent against the SnaikeNET server.")
    parser.add_argument(
        "--model-path",
        default="dqn_final",
        help="Path to the training data that DQN.load pulls from.",
    )
    parser.add_argument("--host", default="localhost", help="Server host.")
    parser.add_argument("--port", type=int, default=8888, help="Server port.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable pygame rendering.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")

    controller = ClientController(host=args.host, port=args.port)
    controller.reset_episode()
    env = ClientEnv(controller=controller, headless=args.headless)

    model = DQN.load(args.model_path)

    obs, info = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        logger.debug("obs: {}", obs[:3])
        logger.debug("action: {}", action)
        logger.debug("reward: {}", reward)
        logger.debug("done: {}", terminated)
        logger.debug("------")
        time.sleep(1 / 240)
        if terminated or truncated:
            obs, info = env.reset()


if __name__ == "__main__":
    main()
