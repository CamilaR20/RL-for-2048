import time
import env_2048
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

if __name__ == "__main__":
    env = env_2048.GameEnv()

    n_games = 1000

    scores = []
    highest_tile = []
    action = 0

    start_time = time.time()
    for i_game in range(n_games):
        print('Playing {} game.'.format(i_game))
        observation, _ = env.reset()
        done = False

        while not done:
            action = 0 if action==3 else 3
            # action = np.random.choice([0, 1, 2, 3])
            _, _, done, _, _ = env.step(action)

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
    plt.savefig('./max_dist_ld.png')

    plt.figure()
    plt.title('Score distribution')
    plt.xlabel('Score at game end')
    plt.ylabel('Percentage of ocurrences')
    plt.hist(np.array(scores), weights=np.ones(len(scores)) / len(scores), color='tab:orange', rwidth=0.7)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.savefig('./score_dist_ld.png')

    print('Average score: ', np.average(np.array(scores)))
    print('Benchmark completed in: ', time.time() - start_time)
