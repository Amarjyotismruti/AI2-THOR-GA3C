import tensorflow as tf
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
from collections import deque
import sys
if sys.version_info[0] == 2:
    from env import Environment
else:
    from a3c.env.env import Environment

class AtariEnvironment(Environment):
    """
    Small wrapper for gym atari environments.
    Responsible for preprocessing screens and holding on to a screen buffer 
    of size agent_history_length from which environment state
    is constructed.
    """
    def __init__(self, gym_env, resized_width, resized_height, agent_history_length):
        self.env = gym_env
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.agent_history_length = agent_history_length
        self.gym_actions = range(gym_env.action_space.n)

        # Screen buffer of size AGENT_HISTORY_LENGTH to be able
        # to build state arrays of size [1, AGENT_HISTORY_LENGTH, width, height]
        self.state_buffer = deque()

    def reset(self):
        """
        Resets the atari game, clears the state buffer
        """
        # Clear the state buffer
        self.state_buffer = deque()

        x_t = self.env.reset()
        x_t = self.get_preprocessed_state(x_t)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis = 0)
        
        for i in range(self.agent_history_length-1):
            self.state_buffer.append(x_t)
        return s_t

    def get_preprocessed_state(self, observation):
        """
        See Methods->Preprocessing in Mnih et al.
        1) Get image grayscale
        2) Rescale image
        """
        return resize(rgb2gray(observation), (self.resized_width, self.resized_height), mode = 'constant')

    def step(self, action_index):
        """
        Excecutes an action in the gym environment.
        Builds current state (concatenation of agent_history_length-1 previous frames and current one).
        Pops oldest frame, adds current frame to the state buffer.
        Returns current state.
        """

        x_t1, r_t, terminal, info = self.env.step(self.gym_actions[action_index])
        x_t1 = self.get_preprocessed_state(x_t1)

        previous_frames = np.array(self.state_buffer)
        s_t1 = np.empty((self.agent_history_length, self.resized_height, self.resized_width))
        s_t1[:self.agent_history_length-1, ...] = previous_frames
        s_t1[self.agent_history_length-1] = x_t1

        # Pop the oldest frame, add the current frame to the queue
        self.state_buffer.popleft()
        self.state_buffer.append(x_t1)

        return s_t1, r_t, terminal, info

    def render(self):
        self.env.render()
