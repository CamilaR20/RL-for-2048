# Developed GUI with structure and style from: https://github.com/kiteco/python-youtube-code/blob/master/build-2048-in-python/2048.py

from tkinter import Frame, Label, CENTER
import numpy as np
import env_2048
from style import *
from train import DQN
import torch
import time

class Board(Frame):
    def __init__(self, env, ai=None):
        Frame.__init__(self)

        self.grid()
        self.master.title('2048')
        
        self.main_grid = Frame(self, bg=GRID_COLOR, bd=3, width=400, height=400)
        self.main_grid.grid(pady=(80, 0))

        self.init_GUI()  # cells is a 2d list that holds the information for each cell

        self.master.bind("<Key>", self.key_press)
        self.commands = {"'w'": "UP", "'s'": "DOWN", "'a'": "LEFT", "'d'": "RIGHT"}
        self.ai = ai  # Possible values: HUMAN, DEEPQ
        self.env = env
        self.done = False

        self.update_GUI(self.env.grid, self.env.score)

        self.mainloop()

    def init_GUI(self):
        # Initialize grid
        self.cells = []
        for i in range(4):
            row = []
            for j in range(4):
                cell_frame = Frame(self.main_grid, bg=EMPTY_CELL_COLOR, width=100, height=100)
                cell_frame.grid(row=i, column=j, padx=5, pady=5)
                cell_number = Label(self.main_grid, bg=EMPTY_CELL_COLOR)
                cell_number.grid(row=i, column=j)
                cell_data = {"frame": cell_frame, "number": cell_number}
                row.append(cell_data)
            self.cells.append(row)

        # Initialize score label
        score_frame = Frame(self)
        score_frame.place(relx=0.5, y=40, anchor="center")
        Label(score_frame, text="Score", font=SCORE_LABEL_FONT).grid(row=0)
        self.score_label = Label(score_frame, text="0", font=SCORE_FONT)
        self.score_label.grid(row=1)

    def update_GUI(self, grid, score):
        for i in range(CELL_COUNT):
            for j in range(CELL_COUNT):
                cell_value = grid[i, j]
                if cell_value == 0:
                    self.cells[i][j]["frame"].configure(bg=EMPTY_CELL_COLOR)
                    self.cells[i][j]["number"].configure(bg=EMPTY_CELL_COLOR, text="")
                else:
                    self.cells[i][j]["frame"].configure(
                        bg=CELL_COLORS[cell_value])
                    self.cells[i][j]["number"].configure(
                        bg=CELL_COLORS[cell_value],
                        fg=CELL_NUMBER_COLORS[cell_value],
                        font=CELL_NUMBER_FONTS[cell_value],
                        text=str(cell_value))
        self.score_label.configure(text=score)
        self.update_idletasks()

    def game_over(self, win=True):
        if win:
            game_over_frame = Frame(self.main_grid, borderwidth=2)
            game_over_frame.place(relx=0.5, rely=0.5, anchor="center")
            Label(game_over_frame, text="You win!", bg=WINNER_BG, fg=GAME_OVER_FONT_COLOR, font=GAME_OVER_FONT).pack()
        else:
            game_over_frame = Frame(self.main_grid, borderwidth=2)
            game_over_frame.place(relx=0.5, rely=0.5, anchor="center")
            Label(game_over_frame, text="Game over!", bg=LOSER_BG, fg=GAME_OVER_FONT_COLOR, font=GAME_OVER_FONT).pack()

    def key_press(self, event):
        global model
        key = repr(event.char)

        if self.done:
            return
        
        if self.ai == "HUMAN":
            if key in self.commands:
                action = env_2048.ACTIONS[self.commands[key]]
                _, _, self.done, _, _ = self.env.step(action)
                self.update_GUI(self.env.grid, self.env.score)
                if self.done:
                    self.game_over(self.env.win)
            else:
                print("Key not supported (try 'a', 's', 'd', 'w').")
        else:
            # AI is playing
            observation = torch.tensor(self.env.encode_grid(np.copy(self.env.grid)), dtype=torch.float32).unsqueeze(0)
            action = model(observation).max(1).indices.view(1, 1).item()
            _, _, self.done, _, _ = self.env.step(action)
            self.update_GUI(self.env.grid, self.env.score)
            if self.done:
                self.game_over(self.env.win)


if __name__ == "__main__":
    # ai can be the path to a trained model that takes in the grid (observation) as input and outputs an action
    ai = "AI" # If AI is HUMAN, there is a human player and key presses are received

    # If an AI is playing, the model must be loaded before
    model = DQN((12, 4, 4), 4)
    model.load_state_dict(torch.load('./model_1000.pt'))
    model.eval()

    env = env_2048.GameEnv()
    board = Board(env, ai)

    print("Game Over")