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
import os

class A3CAgent:
    # Path params
    summary_save_path = "./%s/logs"
    model_save_path = "./%s/models"
    checkpoint_name = "%s.ckpt"
    video_save_path = "./%s/video"

    iteration = 0
    
    def __init__(self, 
                 model_name, 
                 checkpoint_interval, 
                 summary_interval, 
                 show_training, 
                 num_concurrent, 
                 agent_history_length, 
                 input_shape,  
                 gamma, 
                 learning_rate, 
                 num_iterations, 
                 async_update,
                 num_actions, 
                 output_dir,
                 max_grad):
        self.model_name = model_name
        self.checkpoint_interval = checkpoint_interval
        self.summary_interval = summary_interval
        self.show_training = show_training
        self.num_concurrent = num_concurrent
        self.agent_history_length = agent_history_length
        self.input_shape = input_shape
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.async_update = async_update
        self.num_actions = num_actions
        self.output_dir = output_dir
        self.max_grad = max_grad

        self.summary_save_path = self.summary_save_path % self.output_dir
        self.model_save_path = self.model_save_path % self.output_dir
        self.checkpoint_name = self.checkpoint_name % self.model_name
        self.video_save_path = self.video_save_path % self.output_dir

        self.checkpoint_save_path = os.path.join(self.model_save_path, self.checkpoint_name)

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        if not os.path.exists(self.video_save_path):
            os.makedirs(self.video_save_path)

    def sample_policy_action(self, num_actions, probs):
        """
        Sample an action from an action probability distribution output by
        the policy network.
        """
        # Subtract a tiny value from probabilities in order to avoid
        # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
        probs = probs - np.finfo(np.float32).epsneg

        histogram = np.random.multinomial(1, probs)
        action = int(np.nonzero(histogram)[0])
        return action

    def actor_learner_thread(self, num, env, session, graph_ops, summary_ops, saver):
        # Unpack graph ops
        state_input, action_input, target_input, minimize, p_network, v_network = graph_ops

        # Unpack tensorboard summary stuff
        r_summary_placeholder, \
        update_ep_reward, \
        val_summary_placeholder, \
        update_ep_val, \
        pol_summary_placeholder, \
        update_ep_pol, \
        summary_op = summary_ops

        time.sleep(5*num)

        # Set up per-episode counters
        ep_reward = 0
        ep_avg_v = 0
        ep_max_p = 0
        v_steps = 0
        ep_t = 0

        state = env.get_initial_state()
        terminal = False

        while self.iteration < self.num_iterations:
            state_batch = []
            past_rewards = []
            action_batch = []

            t = 0
            t_start = t

            while not (terminal or ((t - t_start)  == self.async_update)):
                # Perform action according to policy pi(action | state)
                probs = session.run(p_network, feed_dict={state_input: [state]})[0]
                action = self.sample_policy_action(self.num_actions, probs)
                action_mask = np.zeros([self.num_actions])
                action_mask[action] = 1

                state_batch.append(state)
                action_batch.append(action_mask)

                next_state, r_t, terminal, info = env.step(action)
                ep_reward += r_t

                r_t = np.clip(r_t, -1, 1)
                past_rewards.append(r_t)

                max_p = np.max(probs)
                ep_max_p = ep_max_p + max_p

                t += 1
                self.iteration += 1
                ep_t += 1
                
                state = next_state

            if terminal:
                target = 0
            else:
                target = session.run(v_network, feed_dict={state_input: [state]})[0][0] # Bootstrap from last state

            ep_avg_v = ep_avg_v + target
            v_steps = v_steps + 1

            target_batch = np.zeros(t)
            for i in reversed(range(t_start, t)):
                target_batch[i] = past_rewards[i] + self.gamma * target

            session.run(minimize, feed_dict={target_input: target_batch,
                                             action_input: action_batch,
                                             state_input: state_batch})
            
            if self.iteration % self.checkpoint_interval == 0:
                saver.save(session, self.checkpoint_save_path)

            if terminal:
                # Episode ended, collect stats and reset game
                if v_steps > 0:
                    session.run(update_ep_val, feed_dict={val_summary_placeholder: ep_avg_v/v_steps})
                if ep_t > 0:
                    session.run(update_ep_pol, feed_dict={pol_summary_placeholder: ep_max_p/ep_t})
                session.run(update_ep_reward, feed_dict={r_summary_placeholder: ep_reward})
                print("THREAD:", num, "/ TIME", self.iteration, "/ REWARD", ep_reward)
                state = env.get_initial_state()
                terminal = False
                # Reset per-episode counters
                ep_reward = 0
                ep_t = 0
                ep_avg_v = 0
                v_steps = 0
                ep_max_p = 0

    def compile(self, loss_func):
        # Create shared global policy and value networks
        state, \
        p_network, \
        v_network, \
        p_params, \
        v_params = build_policy_and_value_networks(model_name=self.model_name, \
                                                   num_actions=self.num_actions, \
                                                   input_shape=self.input_shape, \
                                                   window=self.agent_history_length)

        # Shared global optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # Op for applying remote gradients
        target = tf.placeholder("float", [None])
        action_mask = tf.placeholder("float", [None, self.num_actions])

        v_network_flat = tf.reshape(v_network, shape=[-1]);
        p_network_masked = tf.reduce_sum(tf.multiply(p_network, action_mask), reduction_indices=1)
        p_loss = loss_func(p_network_masked, target - v_network_flat, max_grad=self.max_grad)
        v_loss = tf.reduce_mean(tf.square(target - v_network)) / 2

        total_loss = p_loss + v_loss
        minimize = optimizer.minimize(total_loss)
        
        return state, action_mask, target, minimize, p_network, v_network

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

        ep_max_p = tf.Variable(0.)
        tf.summary.scalar("Episode Max Policy", ep_max_p)
        pol_summary_placeholder = tf.placeholder("float")
        update_ep_pol = ep_max_p.assign(pol_summary_placeholder)

        summary_op = tf.summary.merge_all()
        return r_summary_placeholder, \
               update_ep_reward, \
               val_summary_placeholder, \
               update_ep_val, \
               pol_summary_placeholder, \
               update_ep_pol, \
               summary_op

    def train(self, envs, session, graph_ops, saver):        
        summary_ops = self.setup_summaries()
        summary_op = summary_ops[-1]

        # Initialize variables
        session.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(self.summary_save_path, session.graph)

        # Start num_concurrent training threads
        actor_learner_threads = [threading.Thread(target=self.actor_learner_thread, \
                                                  args=(thread_id, \
                                                        envs[thread_id], \
                                                        session, \
                                                        graph_ops, \
                                                        summary_ops, \
                                                        saver)) for thread_id in range(self.num_concurrent)]
        for t in actor_learner_threads:
            t.start()

        # Show the agents training and write summary statistics
        last_summary_time = 0
        while True:
            if self.show_training:
                for env in envs:
                    env.render()
            now = time.time()
            if now - last_summary_time > self.summary_interval:
                summary_str = session.run(summary_op)
                writer.add_summary(summary_str, float(self.iteration))
                last_summary_time = now
        for t in actor_learner_threads:
            t.join()

    def evaluation(self, monitor_env, session, graph_ops, saver):
        saver.restore(session, self.checkpoint_save_path)
        print("Restored model weights from ", self.checkpoint_save_path)
        monitor_env.monitor.start(self.video_save_path)

        # Unpack graph ops
        state_input, action_mask, target, minimize, p_network, v_network = graph_ops

        for i_episode in range(100):
            state = env.get_initial_state()
            ep_reward = 0
            terminal = False
            while not terminal:
                monitor_env.render()
                # Forward the deep q network, get Q(s,a) values
                probs = p_network.eval(session = session, feed_dict = {state_input: [state]})[0]
                action = self.sample_policy_action(self.num_actions, probs)
                next_state, r_t, terminal, info = env.step(action)
                state = next_state
                ep_reward += r_t
            print(ep_reward)
        monitor_env.monitor.close()
