from ga3c.Config import Config
from ga3c.GameManager import GameManager
from ga3c.env.Environment import Environment
import numpy as np
import robosims.server
from PIL import Image

class THOREnvironment(Environment):
    def __init__(self, floor_name='FloorPlan223'):
        #Change the linux_build path to the thor data file path.
        self.env_thor = robosims.server.Controller(
                                player_screen_width=800,
                                player_screen_height=800,
                                darwin_build='./thor-cmu-201703101557-OSXIntel64.app/Contents/MacOS/thor-cmu-201703101557-OSXIntel64',
                                linux_build='./thor-cmu-201703101558-Linux64',
                                x_display="0.0")
        self.actions = ['MoveAhead', 'MoveBack', 'MoveRight', 'MoveLeft', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown']
        self.floor_name=floor_name
        self.start_unity=True
        self.observation_space = (224,224,3)
        self.nA=len(self.actions)

        self.previous_state = None
        self.current_state = None

        self.reset()
        
    def start(self):
        self.env_thor.start(start_unity=self.start_unity)

    def get_num_actions(self):
        return self.nA

    def reset(self):
        event=self.env_thor.reset(self.floor_name)
        obs=np.array(event.frame)
        obs=Image.fromarray(np.uint8(obs))
        obs=obs.resize((224,224))
        #obs=np.expand_dims(np.array(obs), axis=0)
        return np.float32(obs)/255

    def step(self, action):
        event=self.env_thor.step(dict(action=self.actions[action]))
        reward,terminal=0,False
        obs=np.array(event.frame)
        obs=Image.fromarray(np.uint8(obs))
        obs=np.float32(obs.resize((224,224)))/255
        
        #obs=np.expand_dims(np.array(obs), axis=0)
        # print event.metadata['objects'][4]
        if event.metadata['objects'][4]['visible']:
            reward=10
            terminal=True

        self.previous_state = self.current_state
        self.current_state = obs

        return reward,terminal
