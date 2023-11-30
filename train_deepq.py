import env_2048
from stable_baselines3.common.env_checker import check_env
import numpy as np

if __name__ == "__main__":
    env = env_2048.GameEnv()
    check_env(env)  # Outputs warning if environment does not follow Gym interface that is supported by Stable Baselines3.