import tensorflow as tf
import keras.backend as K
from a3c import A3CAgent
import gym
from atari_environment import AtariEnvironment

flags = tf.app.flags
flags.DEFINE_string('env', 'Breakout-v0', 'Name of the environment')
flags.DEFINE_boolean('testing', False, 'If true, run gym evaluation')
flags.DEFINE_integer('num_concurrent', 8, 'Number of concurrent threads during training.')
flags.DEFINE_integer('agent_history_length', 4, 'History window length.')
flags.DEFINE_integer('resized_width', 84, 'Scale screen to this width.')
flags.DEFINE_integer('resized_height', 84, 'Scale screen to this height.')
FLAGS = flags.FLAGS

def main(_):
	if FLAGS.testing:
		FLAGS.num_concurrent = 1
		
	# Set up game environments (one per thread)
	envs = [gym.make(FLAGS.env) for i in range(FLAGS.num_concurrent)]
	envs = [AtariEnvironment(gym_env=env, \
							 resized_width=FLAGS.resized_width, \
							 resized_height=FLAGS.resized_height, \
							 agent_history_length=FLAGS.agent_history_length)
							 for env in envs]

	with tf.Graph().as_default(), tf.Session() as session:
		K.set_session(session)

		a3cAgent = A3CAgent()
		graph_ops = a3cAgent.compile()
		saver = tf.train.Saver()

		if FLAGS.testing:
			a3cAgent.evaluation(envs[0], session, graph_ops, saver)
		else:
			a3cAgent.train(envs, session, graph_ops, saver)

if __name__ == "__main__":
	tf.app.run()