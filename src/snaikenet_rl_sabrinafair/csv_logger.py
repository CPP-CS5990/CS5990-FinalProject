import csv
import os
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


class CsvTrainLogger(BaseCallback):
    def __init__(self, csv_path: str, log_freq: int = 1000, reward_window: int = 100):
        super().__init__()
        self.csv_path = csv_path
        self.log_freq = log_freq
        self.reward_window = reward_window
        self.episode_rewards = deque(maxlen=reward_window)
        self.episode_lengths = deque(maxlen=reward_window)

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timesteps",
                "episodes_seen",
                "mean_episode_reward",
                "mean_episode_length",
                "last_episode_reward",
                "last_episode_length",
                "exploration_rate",
                "loss",
            ])

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if done and "episode" in info:
                ep = info["episode"]
                self.episode_rewards.append(float(ep["r"]))
                self.episode_lengths.append(float(ep["l"]))

        if self.n_calls % self.log_freq == 0:
            mean_reward = (
                sum(self.episode_rewards) / len(self.episode_rewards)
                if self.episode_rewards else float("nan")
            )
            mean_length = (
                sum(self.episode_lengths) / len(self.episode_lengths)
                if self.episode_lengths else float("nan")
            )

            # DQN usually has exploration_rate as an attribute
            exploration_rate = getattr(self.model, "exploration_rate", float("nan"))

            # Logger values are not guaranteed to exist at every step
            loss = float("nan")
            try:
                logger_dict = self.model.logger.name_to_value
                loss = logger_dict.get("train/loss", float("nan"))
            except Exception:
                pass

            last_reward = self.episode_rewards[-1] if self.episode_rewards else float("nan")
            last_length = self.episode_lengths[-1] if self.episode_lengths else float("nan")

            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.num_timesteps,
                    len(self.episode_rewards),
                    mean_reward,
                    mean_length,
                    last_reward,
                    last_length,
                    exploration_rate,
                    loss,
                ])

        return True