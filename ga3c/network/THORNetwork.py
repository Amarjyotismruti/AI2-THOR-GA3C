import os
import re
import numpy as np
import tensorflow as tf

from keras.layers import Flatten, Dense, Convolution2D, Input
from keras.models import Model

from ga3c.Config import Config
from ga3c.network.Network import Network
from ga3c.utils import mean_huber_loss

class THORNetwork(Network):
    def __init__(self, device, model_name, num_actions):
        self.device = device
        self.model_name = model_name
        self.num_actions = num_actions

        self.img_width = Config.IMAGE_WIDTH
        self.img_height = Config.IMAGE_HEIGHT
        self.img_channels = Config.STACKED_FRAMES

        self.learning_rate = Config.LEARNING_RATE_START
        self.beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(self.device):
                self._create_graph()

                self.sess = tf.Session(
                    graph=self.graph,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)))
                self.sess.run(tf.global_variables_initializer())

                if Config.TENSORBOARD: self._create_tensor_board()
                if Config.LOAD_CHECKPOINT or Config.SAVE_MODELS:
                    vars = tf.global_variables()
                    self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)
                

    def _create_graph(self):
        self.x = tf.placeholder(
            tf.float32, [None, self.img_width, self.img_height, 3], name='X')
        self.y_r = tf.placeholder(tf.float32, [None], name='Yr')

        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])

        self.global_step = tf.Variable(0, trainable=False, name='step')

        self.action_index = tf.placeholder(tf.float32, [None, self.num_actions])

        # As implemented in A3C paper
        inputs = Input(shape=(self.img_width, self.img_height, 3))
        shared = Convolution2D(name="conv1", nb_filter=16, nb_row=8, nb_col=8, subsample=(4,4), activation='relu', border_mode='same')(inputs)
        shared = Convolution2D(name="conv2", nb_filter=32, nb_row=4, nb_col=4, subsample=(2,2), activation='relu', border_mode='same')(shared)
        shared = Flatten()(shared)
        shared = Dense(name="h1", output_dim=256, activation='relu')(shared)

        action_probs = Dense(name="p", output_dim=self.num_actions, activation='softmax')(shared)
        
        state_value = Dense(name="v", output_dim=1, activation='linear')(shared)

        self.softmax_p = Model(input=inputs, output=action_probs)(self.x)
        self.logits_v = Model(input=inputs, output=state_value)(self.x)

        optimizer = tf.train.AdamOptimizer(self.var_learning_rate)

        action_mask = tf.placeholder('float', [None, self.num_actions])

        logits_v_flat = tf.reshape(self.logits_v, shape=[-1]);
        logits_p_masked = tf.reduce_sum(tf.multiply(self.softmax_p, self.action_index), reduction_indices=1)
        self.cost_p = mean_huber_loss(logits_p_masked, self.y_r - logits_v_flat, max_grad=np.inf)
        self.cost_v = tf.reduce_mean(tf.square(self.y_r - self.logits_v)) / 2

        total_loss = self.cost_p + self.cost_v
        self.train_op = optimizer.minimize(total_loss, global_step=self.global_step)

    def _create_tensor_board(self):
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        max_p = tf.reduce_mean(tf.reduce_max(self.softmax_p, reduction_indices=[1]))
        mean_v = tf.reduce_mean(self.logits_v)
        summaries.append(tf.summary.scalar("Max Policy", max_p))
        summaries.append(tf.summary.scalar("Value", mean_v))
        summaries.append(tf.summary.scalar("Pcost", self.cost_p))
        summaries.append(tf.summary.scalar("Vcost", self.cost_v))
        summaries.append(tf.summary.scalar("LearningRate", self.var_learning_rate))
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram("weights_%s" % var.name, var))

        self.summary_op = tf.summary.merge(summaries)
        self.log_writer = tf.summary.FileWriter("./logs/%s" % self.model_name, self.sess.graph)

    def __get_base_feed_dict(self):
        return {self.var_learning_rate: self.learning_rate}

    def get_global_step(self):
        step = self.sess.run(self.global_step)
        return step

    def predict_single(self, x):
        return self.predict_p(x[None, :])[0]

    def predict_v(self, x):
        prediction = self.sess.run(self.logits_v, feed_dict={self.x: x})
        return prediction

    def predict_p(self, x):
        prediction = self.sess.run(self.softmax_p, feed_dict={self.x: x})
        return prediction
    
    def predict_p_and_v(self, x):
        return self.sess.run([self.softmax_p, self.logits_v], feed_dict={self.x: x})
    
    def train(self, x, y_r, a, trainer_id):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x, self.y_r: y_r, self.action_index: a})
        self.sess.run(self.train_op, feed_dict=feed_dict)

    def log(self, x, y_r, a):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x, self.y_r: y_r, self.action_index: a})
        step, summary = self.sess.run([self.global_step, self.summary_op], feed_dict=feed_dict)
        self.log_writer.add_summary(summary, step)

    def _checkpoint_filename(self, episode):
        return 'checkpoints/%s_%08d' % (self.model_name, episode)
    
    def _get_episode_from_filename(self, filename):
        # TODO: hacky way of getting the episode. ideally episode should be stored as a TF variable
        return int(re.split('/|_|\.', filename)[2])

    def save(self, episode):
        self.saver.save(self.sess, self._checkpoint_filename(episode))

    def load(self):
        filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename(episode=0)))
        if Config.LOAD_EPISODE > 0:
            filename = self._checkpoint_filename(Config.LOAD_EPISODE)
        self.saver.restore(self.sess, filename)
        return self._get_episode_from_filename(filename)
       
    def get_variables_names(self):
        return [var.name for var in self.graph.get_collection('trainable_variables')]

    def get_variable_value(self, name):
        return self.sess.run(self.graph.get_tensor_by_name(name))
