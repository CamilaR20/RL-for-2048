# RL for 2048
Convolutional DQN Agent vs 2048: code to explore environment and reward variations for the game of 2048 played by a DQN agent.

## **Installation**
You can use Miniconda to setup the environment:
```
conda env create -f environment.yml 
``` 

## Usage
The repository contains the following scripts and files:
- *env_2048.py*: implementation of the environment for the 2048 game. Designed as a custom gymnasium environment.
- *train_conv.py*, *train_fcn.py*, *train_deepq.py*: train different types of Deep-Q agents to play 2048.
- *gui.py*: to play game with keyboard or see the game being played by a trained RL agent. Change the path to the saved model to select agent.
- *style.py*: contains variables that determine the appearance of the GUI.
- *benchmark.py*, *benchmark_random.py*: run benchmarks to compare the performance of different agents and strategies (e.g. random, up-down). The benchmarks returns the results of 1.000 games as plots for the distribution of the maximum tile and the maximum score reached by the agent/strategy.
- *report.pdf*: contains a literature overview on DQN agents for 2048 and the results of experiments run on different environment and reward variations.
- *clip.mov*: example of an agent playing 2048.