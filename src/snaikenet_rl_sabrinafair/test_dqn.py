import time
from stable_baselines3 import DQN
from snaikenet_rl_sabrinafair.client_env import ClientEnv
from snaikenet_rl_sabrinafair.game_controller import ClientController


controller = ClientController(host="localhost", port=8888)

frame = controller.reset_episode()
print("Initial frame seq:", frame.sequence_number)
env = ClientEnv(controller=controller)

model = DQN.load("dqn_client")

obs, info = env.reset()

while True:

    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)
    print("obs:", obs[:3])
    print("action:", action)
    print("reward:", reward)
    print("done:", terminated)
    print("------")
    time.sleep(1/240)
    if terminated or truncated:
        obs, info = env.reset()