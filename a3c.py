#!/usr/bin/env python
from skimage.transform import resize
from skimage.color import rgb2gray
import threading
import tensorflow as tf
import sys
import random
import numpy as np
import time
import gym
from keras import backend as K
from keras.layers import Convolution2D, Flatten, Dense
from collections import deque
from a3c_model import build_policy_and_value_networks
from keras import backend as K
from atari_environment import AtariEnvironment

# TRAINING = False
TRAINING = True

class A3CAgent:
    # Path params
    EXPERIMENT_NAME = "breakout_a3c"
    SUMMARY_SAVE_PATH = "./logs/%s" % EXPERIMENT_NAME
    CHECKPOINT_SAVE_PATH = "./models/%s.ckpt" % EXPERIMENT_NAME
    CHECKPOINT_NAME = "./models/%s.ckpt-5" % EXPERIMENT_NAME
    CHECKPOINT_INTERVAL=5000
    SUMMARY_INTERVAL=5

    SHOW_TRAINING = True
    # SHOW_TRAINING = False

    # Experiment params
    GAME = "Breakout-v0"
    ACTIONS = 3
    NUM_CONCURRENT = 8
    NUM_EPISODES = 20000

    AGENT_HISTORY_LENGTH = 4
    RESIZED_WIDTH = 84
    RESIZED_HEIGHT = 84

    # DQN Params
    GAMMA = 0.99

    # Optimization Params
    LEARNING_RATE = 0.00001

    #Shared global parameters
    T = 0
    TMAX = 80000000
    t_max = 32

    def __init__(self):
        pass

    def sample_policy_action(self, num_actions, probs):
        """
        Sample an action from an action probability distribution output by
        the policy network.
        """
        # Subtract a tiny value from probabilities in order to avoid
        # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
        probs = probs - np.finfo(np.float32).epsneg

        histogram = np.random.multinomial(1, probs)
        action_index = int(np.nonzero(histogram)[0])
        return action_index

    def actor_learner_thread(self, num, env, session, graph_ops, summary_ops, saver):
        # Unpack graph ops
        s, a, R, minimize, p_network, v_network = graph_ops

        # Unpack tensorboard summary stuff
        r_summary_placeholder, \
        update_ep_reward, \
        val_summary_placeholder, \
        update_ep_val, \
        summary_op = summary_ops

        # Wrap env with AtariEnvironment helper class
        env = AtariEnvironment(gym_env=env, \
                               resized_width=self.RESIZED_WIDTH, \
                               resized_height=self.RESIZED_HEIGHT, \
                               agent_history_length=self.AGENT_HISTORY_LENGTH)

        time.sleep(5*num)

        # Set up per-episode counters
        ep_reward = 0
        ep_avg_v = 0
        v_steps = 0
        ep_t = 0

        probs_summary_t = 0

        s_t = env.get_initial_state()
        terminal = False

        while self.T < self.TMAX:
            s_batch = []
            past_rewards = []
            a_batch = []

            t = 0
            t_start = t

            while not (terminal or ((t - t_start)  == self.t_max)):
                # Perform action a_t according to policy pi(a_t | s_t)
                probs = session.run(p_network, feed_dict={s: [s_t]})[0]
                action_index = self.sample_policy_action(self.ACTIONS, probs)
                a_t = np.zeros([self.ACTIONS])
                a_t[action_index] = 1

                if probs_summary_t % 100 == 0:
                    print("P, ", np.max(probs), "V ", session.run(v_network, feed_dict={s: [s_t]})[0][0])

                s_batch.append(s_t)
                a_batch.append(a_t)

                s_t1, r_t, terminal, info = env.step(action_index)
                ep_reward += r_t

                r_t = np.clip(r_t, -1, 1)
                past_rewards.append(r_t)

                t += 1
                self.T += 1
                ep_t += 1
                probs_summary_t += 1
                
                s_t = s_t1

            if terminal:
                R_t = 0
            else:
                R_t = session.run(v_network, feed_dict={s: [s_t]})[0][0] # Bootstrap from last state

            R_batch = np.zeros(t)
            for i in reversed(range(t_start, t)):
                R_t = past_rewards[i] + self.GAMMA * R_t
                R_batch[i] = R_t

            session.run(minimize, feed_dict={R : R_batch,
                                             a : a_batch,
                                             s : s_batch})
            
            # Save progress every 5000 iterations
            if self.T % self.CHECKPOINT_INTERVAL == 0:
                saver.save(session, CHECKPOINT_SAVE_PATH, global_step = T)

            if terminal:
                # Episode ended, collect stats and reset game
                session.run(update_ep_reward, feed_dict={r_summary_placeholder: ep_reward})
                print("THREAD:", num, "/ TIME", self.T, "/ REWARD", ep_reward)
                s_t = env.get_initial_state()
                terminal = False
                # Reset per-episode counters
                ep_reward = 0
                ep_t = 0

    def compile(self):
        # Create shared global policy and value networks
        s, \
        p_network, \
        v_network, \
        p_params, \
        v_params = build_policy_and_value_networks(num_actions=self.ACTIONS, \
                                                   agent_history_length=self.AGENT_HISTORY_LENGTH, \
                                                   resized_width=self.RESIZED_WIDTH, \
                                                   resized_height=self.RESIZED_HEIGHT)

        # Shared global optimizer
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)

        # Op for applying remote gradients
        R_t = tf.placeholder("float", [None])
        a_t = tf.placeholder("float", [None, self.ACTIONS])
        log_prob = tf.log(tf.reduce_sum(tf.multiply(p_network, a_t), reduction_indices=1))
        p_loss = -log_prob * (R_t - v_network)
        v_loss = tf.reduce_mean(tf.square(R_t - v_network))

        total_loss = p_loss + (0.5 * v_loss)

        minimize = optimizer.minimize(total_loss)
        return s, a_t, R_t, minimize, p_network, v_network

    # Set up some episode summary ops to visualize on tensorboard.
    def setup_summaries(self):
        episode_reward = tf.Variable(0.)
        tf.summary.scalar("Episode Reward", episode_reward)
        r_summary_placeholder = tf.placeholder("float")
        update_ep_reward = episode_reward.assign(r_summary_placeholder)
        ep_avg_v = tf.Variable(0.)
        tf.summary.scalar("Episode Value", ep_avg_v)
        val_summary_placeholder = tf.placeholder("float")
        update_ep_val = ep_avg_v.assign(val_summary_placeholder)
        summary_op = tf.summary.merge_all()
        return r_summary_placeholder, \
               update_ep_reward, \
               val_summary_placeholder, \
               update_ep_val, \
               summary_op

    def train(self, session, graph_ops, saver):
        # Set up game environments (one per thread)
        envs = [gym.make(self.GAME) for i in range(self.NUM_CONCURRENT)]
        
        summary_ops = self.setup_summaries()
        summary_op = summary_ops[-1]

        # Initialize variables
        session.run(tf.initialize_all_variables())
        writer = tf.summary.FileWriter(self.SUMMARY_SAVE_PATH, session.graph)

        # Start NUM_CONCURRENT training threads
        actor_learner_threads = [threading.Thread(target=self.actor_learner_thread, \
                                                  args=(thread_id, \
                                                        envs[thread_id], \
                                                        session, \
                                                        graph_ops, \
                                                        summary_ops, \
                                                        saver)) for thread_id in range(self.NUM_CONCURRENT)]
        for t in actor_learner_threads:
            t.start()

        # Show the agents training and write summary statistics
        last_summary_time = 0
        while True:
            if self.SHOW_TRAINING:
                for env in envs:
                    env.render()
            now = time.time()
            if now - last_summary_time > self.SUMMARY_INTERVAL:
                summary_str = session.run(summary_op)
                writer.add_summary(summary_str, float(self.T))
                last_summary_time = now
        for t in actor_learner_threads:
            t.join()

    def evaluation(self, session, graph_ops, saver):
        saver.restore(session, CHECKPOINT_NAME)
        print("Restored model weights from ", CHECKPOINT_NAME)
        monitor_env = gym.make(self.GAME)
        monitor_env.monitor.start('/tmp/'+EXPERIMENT_NAME+"/eval")

        # Unpack graph ops
        s, a_t, R_t, minimize, p_network, v_network = graph_ops

        # Wrap env with AtariEnvironment helper class
        env = AtariEnvironment(gym_env=monitor_env, \
                               resized_width=self.RESIZED_WIDTH, \
                               resized_height=self.RESIZED_HEIGHT, \
                               agent_history_length=self.AGENT_HISTORY_LENGTH)

        for i_episode in xrange(100):
            s_t = env.get_initial_state()
            ep_reward = 0
            terminal = False
            while not terminal:
                monitor_env.render()
                # Forward the deep q network, get Q(s,a) values
                probs = p_network.eval(session = session, feed_dict = {s : [s_t]})[0]
                action_index = self.sample_policy_action(self.ACTIONS, probs)
                s_t1, r_t, terminal, info = env.step(action_index)
                s_t = s_t1
                ep_reward += r_t
            print(ep_reward)
        monitor_env.monitor.close()

def main(_):
  with tf.Graph().as_default(), tf.Session() as session:
    K.set_session(session)

    a3cAgent = A3CAgent()
    graph_ops = a3cAgent.compile()
    saver = tf.train.Saver()
    
    if TRAINING:
        a3cAgent.train(session, graph_ops, saver)
    else:
        a3cAgent.evaluation(session, graph_ops, saver)

if __name__ == "__main__":
  tf.app.run()
