from ga3c.Config import Config
from ga3c.GameManager import GameManager
import numpy as np

class CartPoleEnvironment():
    def __init__(self):
        self.game = GameManager(Config.GAME, display=Config.PLAY_MODE)
        self.previous_state = None
        self.current_state = None
        self.reset()

    def get_num_actions(self):
        return self.game.env.action_space.n

    def reset(self):
        self.previous_state = None
        self.current_state = None
        self.game.reset()

    def step(self, action):
        observation, reward, done, _ = self.game.step(action)
        observation = np.reshape(observation, [1,4,1])
        self.previous_state = self.current_state
        self.current_state = observation
        return reward, done
