import os
import re
import numpy as np
import tensorflow as tf

from ga3c.Config import Config
from ga3c.network.Model import create_p_and_v_models
from ga3c.Utility import mean_huber_loss

class Network:
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

        self.log_path = 'ga3c_output/logs/%s' % self.model_name
        self.checkpoint_path = 'ga3c_output/checkpoints/%s' % self.model_name

        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

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
        self.x = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.img_channels], name='X')
        self.y_r = tf.placeholder(tf.float32, [None], name='Yr')
        self.action_index = tf.placeholder(tf.float32, [None, self.num_actions])

        p_model, v_model = create_p_and_v_models(self.model_name, 
                                                 self.num_actions, 
                                                 self.img_height, 
                                                 self.img_width, 
                                                 self.img_channels)
        self.softmax_p = p_model(self.x)
        self.logits_v = v_model(self.x)

        logits_v_flat = tf.reshape(self.logits_v, shape=[-1]);
        logits_p_masked = tf.reduce_sum(tf.multiply(self.softmax_p, self.action_index), reduction_indices=1)
        self.cost_p = mean_huber_loss(logits_p_masked, self.y_r - logits_v_flat, max_grad=Config.GRAD_CLIP_NORM)
        self.cost_v = tf.reduce_mean(tf.square(self.y_r - self.logits_v)) / 2

        total_loss = self.cost_p + self.cost_v

        self.global_step = tf.Variable(0, trainable=False, name='step')
        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])
        optimizer = tf.train.AdamOptimizer(self.var_learning_rate)
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

        self.summary_op = tf.summary.merge(summaries)
        self.log_writer = tf.summary.FileWriter(self.log_path, self.sess.graph)

    def __get_base_feed_dict(self):
        return {self.var_learning_rate: self.learning_rate}
    
    def predict_p_and_v(self, x):
        return self.sess.run([self.softmax_p, self.logits_v], feed_dict={self.x: x})
    
    def train(self, x, y_r, a, trainer_id):
        feed_dict = self.__get_base_feed_dict()
        if y_r.ndim == 2 and y_r.shape[0] > 1 and y_r.shape[1] == 1:
            y_r = np.squeeze(y_r)
        elif y_r.ndim == 2 and y_r.shape == (1,1):
            y_r = y_r[0]
        feed_dict.update({self.x: x, self.y_r: y_r, self.action_index: a})
        self.sess.run(self.train_op, feed_dict=feed_dict)

    def log(self, x, y_r, a):
        feed_dict = self.__get_base_feed_dict()
        if y_r.ndim == 2 and y_r.shape[0] > 1 and y_r.shape[1] == 1:
            y_r = np.squeeze(y_r)
        elif y_r.ndim == 2 and y_r.shape == (1,1):
            y_r = y_r[0]
        feed_dict.update({self.x: x, self.y_r: y_r, self.action_index: a})
        step, summary = self.sess.run([self.global_step, self.summary_op], feed_dict=feed_dict)
        self.log_writer.add_summary(summary, step)

    def _checkpoint_filename(self, episode):
        return os.path.join(self.checkpoint_path, '%s_%08d' % (self.model_name, episode))
    
    def _get_episode_from_filename(self, filename):
        # TODO: hacky way of getting the episode. ideally episode should be stored as a TF variable
        return int(re.split('/|_|\.', filename)[-1])

    def save(self, episode):
        self.saver.save(self.sess, self._checkpoint_filename(episode))

    def load(self):
        filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename(episode=0)))
        if Config.LOAD_EPISODE > 0:
            filename = self._checkpoint_filename(Config.LOAD_EPISODE)
        self.saver.restore(self.sess, filename)
        return self._get_episode_from_filename(filename)
