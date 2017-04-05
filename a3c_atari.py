import tensorflow as tf
import keras.backend as K
from a3c import A3CAgent

flags = tf.app.flags
flags.DEFINE_boolean('testing', False, 'If true, run gym evaluation')
FLAGS = flags.FLAGS

def main(_):
	with tf.Graph().as_default(), tf.Session() as session:
		K.set_session(session)

		a3cAgent = A3CAgent()
		graph_ops = a3cAgent.compile()
		saver = tf.train.Saver()

		if FLAGS.testing:
			a3cAgent.evaluation(session, graph_ops, saver)
		else:
			a3cAgent.train(session, graph_ops, saver)

if __name__ == "__main__":
	tf.app.run()