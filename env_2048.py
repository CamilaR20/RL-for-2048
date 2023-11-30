import numpy as np
import gymnasium as gym

N_ACTIONS = 4
ACTIONS = {'LEFT': 0, 'UP':1, 'RIGHT':2, 'DOWN':3}

class GameEnv(gym.Env):
    # Implements 2048 board in a 4 by 4 numpy array

    def __init__(self, score=0, grid=None):
        # Define action and observation space: https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.observation_space = gym.spaces.Box(low=0, high=2048, shape=(1, 4, 4), dtype=np.uint8)

        # Can initialize a game with a previous state and score
        self.score = score
        self.grid = grid
        self.done = False
        self.win = False

        if grid is None:
            self.grid = np.zeros((4, 4), dtype=int)
            # Add 2 random tiles to start
            self.add_new_tile(value=2)
            self.add_new_tile(value=2)

    def step(self, action):
        # Action must be an integer
        # To simulate a gym environment, return: observation, reward, terminated, truncated, info
        grid_copy = np.rot90(np.copy(self.grid), k=action)
        reward, grid_copy = self.slide_left(grid_copy)
        self.grid = np.rot90(np.copy(grid_copy), k=-action)
        self.score += max(0, reward)  # Score is not affected if reward is -1

        self.add_new_tile()
        self.game_over()

        return np.copy(self.grid), reward, self.done, False, None
    
    def reset(self, seed=None, options=None):
        # Initialize environment from scratch
        np.random.seed(seed)

        self.score = 0
        self.grid = None
        self.done = False
        self.win = False

        if self.grid is None:
            self.grid = np.zeros((4, 4), dtype=int)
            # Add 2 random tiles to start
            self.add_new_tile(value=2)
            self.add_new_tile(value=2)

        observation = 0
        return observation, None
    
    def add_new_tile(self, value=None):
        value = value if value is not None else np.random.choice(np.array([2, 4]), p=[0.9, 0.1])

        empty_cells = np.argwhere(self.grid == 0)
        if empty_cells.size == 0:
            self.done = True
            return

        idx, idy = empty_cells[np.random.choice(empty_cells.shape[0])]
        self.grid[idx, idy] = value

    def slide_left(self, grid_copy):
        reward = 0 
        start_grid = np.copy(grid_copy)
        for row in grid_copy:
            leftmost = 0  # Leftmost free spot
            tmp = 0  # Keeps value of possible combination
            for j, tile in enumerate(row):
                if tile == 0:
                    continue
                row[j] = 0
                if tmp:
                    if tile == tmp:
                        row[leftmost] = 2 * tile
                        leftmost += 1
                        reward += 2 * tile
                        tmp = 0
                    else:
                        row[leftmost] = tmp
                        leftmost += 1
                        tmp = tile
                else:
                    tmp = tile
            if tmp:
                row[leftmost] = tmp
                
        reward = -1 if np.array_equal(start_grid, grid_copy) else reward # If movement is invalid, reward is -1
        return reward, grid_copy

    def game_over(self):
        if self.done:
            return # If another tile could not be added
        elif 2048 in self.grid:
            self.done = True
            self.win = True
        else:
            rewards = np.zeros((N_ACTIONS,))
            for action in range(N_ACTIONS):
                grid_copy = np.rot90(np.copy(self.grid), action)
                rewards[action], _ = self.slide_left(grid_copy)
            over = np.all(rewards == -1)
            self.done = True if over else False
                

if __name__ == "__main__":
    game = GameEnv()
    print("Game Over")
