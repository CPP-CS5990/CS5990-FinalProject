# plot_training.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

LOG_PATH = "./logs/train_metrics.csv"
OUT_DIR = Path("./logs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(LOG_PATH)

# Optional smoothing
window = 5
df["reward_smooth"] = df["mean_episode_reward"].rolling(window, min_periods=1).mean()
df["length_smooth"] = df["mean_episode_length"].rolling(window, min_periods=1).mean()
df["loss_smooth"] = df["loss"].rolling(window, min_periods=1).mean()


def save_plot(x, y, title, ylabel, filename, y_smooth=None):
    plt.figure(figsize=(10, 5))
    plt.plot(df[x], df[y], alpha=0.35, label=y)

    if y_smooth is not None:
        plt.plot(df[x], df[y_smooth], label=f"{y} smoothed")

    plt.xlabel("Timesteps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename)
    plt.close()


# 1. Most important: mean reward
save_plot(
    "timesteps",
    "mean_episode_reward",
    "Mean Episode Reward vs Timesteps",
    "Mean Episode Reward",
    "mean_reward.png",
    "reward_smooth",
)

# 2. Episode length
save_plot(
    "timesteps",
    "mean_episode_length",
    "Mean Episode Length vs Timesteps",
    "Mean Episode Length",
    "mean_episode_length.png",
    "length_smooth",
)

# 3. Loss
save_plot(
    "timesteps",
    "loss",
    "DQN Loss vs Timesteps",
    "Loss",
    "loss.png",
    "loss_smooth",
)

# 4. Exploration rate
save_plot(
    "timesteps",
    "exploration_rate",
    "Exploration Rate vs Timesteps",
    "Epsilon",
    "exploration_rate.png",
)

# 5. Last episode reward, noisier but useful
save_plot(
    "timesteps",
    "last_episode_reward",
    "Last Episode Reward vs Timesteps",
    "Last Episode Reward",
    "last_episode_reward.png",
)

# 6. Combined reward + exploration
plt.figure(figsize=(10, 5))
plt.plot(df["timesteps"], df["reward_smooth"], label="Mean reward smoothed")
plt.xlabel("Timesteps")
plt.ylabel("Mean Reward")
plt.twinx()
plt.plot(df["timesteps"], df["exploration_rate"], label="Exploration rate", linestyle="--")
plt.ylabel("Exploration Rate")
plt.title("Reward and Exploration vs Timesteps")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "reward_vs_exploration.png")
plt.close()

print(f"Saved plots to {OUT_DIR}")