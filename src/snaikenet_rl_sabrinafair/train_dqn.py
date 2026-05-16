import inspect
import os

from snaikenet_rl_sabrinafair.client_env import ClientEnv
from snaikenet_rl_sabrinafair.csv_logger import CsvTrainLogger
from snaikenet_rl_sabrinafair.game_controller import ClientController
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

controller = ClientController(host="localhost", port=8888)

frame = controller.reset_episode()
os.makedirs("logs", exist_ok=True)

print("Initial frame seq:", frame.sequence_number)


env = ClientEnv(controller=controller)
env = Monitor(env, filename="logs/monitor.csv")
csv_logger = CsvTrainLogger("logs/train_metrics.csv", log_freq=1000)

check_env(env)


model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=5_000,
    learning_starts=200,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=5_00,
    exploration_fraction=0.7,  # increase exploration time, not enough time to learn env
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    verbose=1,
    # tensorboard_log="./tensorboard/"
)

print("Monitor file:", os.path.abspath("logs/monitor1.csv"))
print("Train metrics file:", os.path.abspath("logs/train_metrics1.csv"))

model.learn(
    total_timesteps=100_000,
    callback=csv_logger,
)

model.save("dqn_client1")
