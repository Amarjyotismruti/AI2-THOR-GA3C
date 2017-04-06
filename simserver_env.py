from gym import Env
import robosims.server
import numpy as np
from PIL import Image



# env = robosims.server.Controller(
#         player_screen_width=800,
#         player_screen_height=800,
#         linux_build='/home/amar/RL-assignment/DRL-project/thor-cmu-201703101558-Linux64',
#         x_display="0.0")

# env.start()

# while True:
# 	event=env.step(dict(action=actions[0]))
# 	key=raw_input('Press WSAD to move, JL to rotate.')

# 	if key=='w':
# 		event=env.step(dict(action=actions[0]))
# 	elif key=='s':
# 		event=env.step(dict(action=actions[1]))
# 	elif key=='a':
# 		event=env.step(dict(action=actions[3]))
# 	elif key=='d':
# 		event=env.step(dict(action=actions[2]))
# 	elif key=='j':
# 		event=env.step(dict(action=actions[4]))
# 	elif key=='l':
# 		event=env.step(dict(action=actions[5]))
# 	elif key=='q':
# 		env.stop()
# 	elif key=='r':
# 		env.reset()
	
# 	print "Last action success", event.metadata["lastActionSuccess"]

class THORenv(Env):

	def __init__(self, floor_name='FloorPlan225'):

		#Change the linux_build path to the thor data file path.
		self.env = robosims.server.Controller(
        player_screen_width=800,
        player_screen_height=800,
        linux_build='/home/amar/RL-assignment/DRL-project/thor-cmu-201703101558-Linux64',
        x_display="0.0")
		self.actions = ['MoveAhead', 'MoveBack', 'MoveRight', 'MoveLeft', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown']
		self.floor_name=floor_name
		self.start_unity=False
	
	def start(self):

		self.env.start(start_unity=self.start_unity)


 	def reset(self):
		self.env.reset(self.floor_name)


	def step(self, action):

		event=self.env.step(dict(action=self.actions[action]))
		reward,terminal=-1,False
		obs=np.array(event.frame)
		obs=Image.fromarray(np.uint8(obs))
		obs=obs.resize((224,224))

		if event.metadata['objects'][4]['visible']:
			reward=10
			terminal=True

		return (obs,reward,terminal)

	def stop(self):

		self.env.stop()

	def render(self):

		self.start_unity=True







  


