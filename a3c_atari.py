import tensorflow as tf
import keras.backend as K
from a3c import A3CAgent
import gym
from atari_environment import AtariEnvironment
from utils import get_output_folder

flags = tf.app.flags
flags.DEFINE_string('output', 'atari-v0', 'Name of the output folder')
flags.DEFINE_string('env', 'Breakout-v0', 'Name of the environment')
flags.DEFINE_string('model', 'deep_Q_network', 'type of network')
flags.DEFINE_boolean('testing', False, 'If true, run gym evaluation')
flags.DEFINE_integer('checkpoint_interval', 5000, 'Checkpoint the model every n iterations')
flags.DEFINE_integer('summary_interval', 5, 'Save training summary to file every n iterations')
flags.DEFINE_boolean('show_training', True, 'If true, then render evironments during training')
flags.DEFINE_integer('num_concurrent', 8, 'Number of concurrent threads during training.')
flags.DEFINE_integer('agent_history_length', 4, 'History window length.')
flags.DEFINE_integer('resized_width', 84, 'Scale screen to this width.')
flags.DEFINE_integer('resized_height', 84, 'Scale screen to this height.')
flags.DEFINE_float('gamma', 0.99, 'Reward discount rate.')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate.')
flags.DEFINE_integer('num_iterations', 25000000, 'Number of training timesteps.')
flags.DEFINE_integer('batch_size', 32, 'Size of batch to update network')
FLAGS = flags.FLAGS

def main(_):
	output_dir = get_output_folder(FLAGS.output, FLAGS.env)

	if FLAGS.testing:
		FLAGS.num_concurrent = 1
		
	# Set up game environments (one per thread)
	envs = [gym.make(FLAGS.env) for _ in range(FLAGS.num_concurrent)]
	envs = [AtariEnvironment(gym_env=env, \
							 resized_width=FLAGS.resized_width, \
							 resized_height=FLAGS.resized_height, \
							 agent_history_length=FLAGS.agent_history_length)
							 for env in envs]
	num_actions = envs[0].env.action_space.n
	input_shape = (FLAGS.resized_width, FLAGS.resized_height)

	with tf.Graph().as_default(), tf.Session() as session:
		K.set_session(session)

		a3cAgent = A3CAgent(FLAGS.model, 
							FLAGS.checkpoint_interval, 
							FLAGS.summary_interval, 
							FLAGS.show_training, 
							FLAGS.num_concurrent, 
							FLAGS.agent_history_length, 
							input_shape,
							FLAGS.gamma, 
							FLAGS.learning_rate, 
							FLAGS.num_iterations, 
							FLAGS.batch_size, 
							num_actions, 
							output_dir)
		graph_ops = a3cAgent.compile()
		saver = tf.train.Saver()

		if FLAGS.testing:
			a3cAgent.evaluation(envs[0], session, graph_ops, saver)
		else:
			a3cAgent.train(envs, session, graph_ops, saver)

if __name__ == "__main__":
	tf.app.run()