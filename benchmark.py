from train import DQN
import torch
import time
import env_2048
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

if __name__ == "__main__":
    model = DQN((12, 4, 4), 4)
    model.load_state_dict(torch.load('./model_1000.pt'))
    model.eval()

    env = env_2048.GameEnv()

    n_games = 1000

    scores = []
    highest_tile = []

    start_time = time.time()
    for i_game in range(n_games):
        print('Playing {} game.'.format(i_game))
        observation, _ = env.reset()
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        done = False

        while not done:
            action = model(observation).max(1).indices.view(1, 1).item()
            observation, _, done, _, _ = env.step(action)
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

            if done:
                scores.append(env.score)
                highest_tile.append(env.grid.max())

    highest_tile_bins = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    highest_tile_labels = ['0', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1024', '2048']
    highest_tile_hist = np.histogram(highest_tile, bins=highest_tile_bins)[0]/len(highest_tile)
    
    plt.figure()
    plt.title('Maximum tile at game end distribution')
    plt.xlabel('Maximum tile at game end')
    plt.ylabel('Percentage of ocurrences')
    plt.bar(x=highest_tile_labels, height=highest_tile_hist, color='tab:green')
    # plt.hist(np.array(highest_tile), bins=[0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], weights=np.ones(len(highest_tile)) / len(highest_tile), color='tab:green', rwidth=0.7)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.savefig('./max_dist.png')

    plt.figure()
    plt.title('Score distribution')
    plt.xlabel('Score at game end')
    plt.ylabel('Percentage of ocurrences')
    plt.hist(np.array(scores), weights=np.ones(len(scores)) / len(scores), color='tab:orange', rwidth=0.7)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.savefig('./score_dist.png')

    print('Benchmark completed in: ', time.time() - start_time)
