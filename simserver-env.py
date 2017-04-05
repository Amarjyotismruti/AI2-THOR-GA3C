from gym import env
import robosims.server



env = robosims.server.Controller(
        player_screen_width=800,
        player_screen_height=800,
        linux_build='/home/amar/RL-assignment/DRL-project/thor-cmu-201703101558-Linux64',
        x_display="0.0")

env.start()
actions = ['MoveAhead', 'MoveBack', 'MoveRight', 'MoveLeft', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown']

while True:

	key=raw_input('Press WSAD to move, JL to rotate.')

	if key=='w':
		event=env.step(dict(action=actions[0]))
	elif key=='s':
		event=env.step(dict(action=actions[1]))
	elif key=='a':
		event=env.step(dict(action=actions[3]))
	elif key=='d':
		event=env.step(dict(action=actions[2]))
	elif key=='j':
		event=env.step(dict(action=actions[4]))
	elif key=='l':
		event=env.step(dict(action=actions[5]))
	elif key=='q':
		env.stop()

class THOR-env(env):

	def __init__(self, floor_name='FloorPlan225'):

		#Change the linux_build path to the thor data file path.
		self.env = robosims.server.Controller(
        player_screen_width=800,
        player_screen_height=800,
        linux_build='/home/amar/RL-assignment/DRL-project/thor-cmu-201703101558-Linux64',
        x_display="0.0")

        self.floor_name=floor_name


    def reset(self):

    	self.env.reset(self.floor_name)

    def 


