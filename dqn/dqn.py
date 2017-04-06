"""Main DQN agent."""

import os
import numpy as np
import tensorflow as tf
from keras.layers import Input, Lambda
from keras.models import Model, model_from_config
import keras.backend as K
import sys
if sys.version_info[0] == 2:
    from preprocessors import HistoryPreprocessor
    from utils import create_directory, log_tb_value
    from constants import model_path, model_file, log_path, video_capture_path
else:
    from dqn.preprocessors import HistoryPreprocessor
    from dqn.utils import create_directory, log_tb_value
    from dqn.constants import model_path, model_file, log_path, video_capture_path
import time
from gym import wrappers
from PIL import Image

class DQNAgent:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: dqn.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: dqn.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """
    def __init__(self,
                 model_name,
                 q_network,
                 preprocessor,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size,
                 num_actions,
                 window,
                 save_network_freq,
                 video_capture_points,
                 eval_train_freq,
                 eval_train_num_ep,
                 phase,
                 target_fixing=True,
                 render=False,
                 print_summary=False,
                 max_grad=1.,
                 double_dqn=False,
                 use_history=True):
        self.model_name = model_name
        self.q_network = q_network
        self.preprocessor = preprocessor
        self.memory = memory
        self.policy = policy
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.window = window
        self.save_network_freq = save_network_freq
        self.video_capture_points = video_capture_points
        self.eval_train_freq = eval_train_freq
        self.eval_train_num_ep = eval_train_num_ep
        self.phase = phase
        self.target_fixing = target_fixing
        self.render = render
        self.print_summary = print_summary
        self.max_grad = max_grad
        self.double_dqn = double_dqn
        self.use_history = use_history

    def compile(self, optimizer, loss_func, output='.'):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """

        # Create folders for the log, models, and videos
        if self.phase == 'train':
            global model_path, model_file, log_path, video_capture_path
            model_path = os.path.join(output, model_path)
            model_file = os.path.join(model_path, model_file)
            log_path = os.path.join(output, log_path)
            create_directory(model_path)
            create_directory(log_path)
        elif self.phase == 'video':
            video_capture_path = os.path.join(output, video_capture_path)
            create_directory(video_capture_path)
        
        # Initialize target network
        with tf.name_scope('Target'):
            if self.target_fixing:
                config = {
                    'class_name': self.q_network.__class__.__name__,
                    'config': self.q_network.get_config(),
                }
                self.target_network = model_from_config(config)
                self.target_network.set_weights(self.q_network.get_weights())
            else:
                self.target_network = self.q_network

        # Calculate individual Huber loss (Keras calculates the mean)
        with tf.name_scope('Lambda'):
            target = Input(shape=(self.num_actions,))
            action_mask = Input(shape=(self.num_actions,))
            error = lambda x: K.sum(loss_func(x[0] * x[1], x[2], self.max_grad), axis=-1)
            output = Lambda(error, output_shape=(self.num_actions,))([self.q_network.output, action_mask, target])

        self.extended_q_network = Model(input=[self.q_network.input, target, action_mask], output=output)

        # Compile all networks
        with tf.name_scope('Loss'):
            self.q_network.compile(optimizer=optimizer, loss=loss_func)
            self.target_network.compile(optimizer=optimizer, loss=loss_func)
            self.extended_q_network.compile(optimizer=optimizer, loss=lambda y_true, y_pred: y_pred)

    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        state = self.preprocessor.process_batch(state)
        return self.q_network.predict(state)

    def process_batch(self, batch):
        """Given a batch, separate in to separate batches.

        Return
        ------
        Batch of current states
                 actions
                 rewards
                 next states
                 terminal boolean
        """
        batch_size = len(batch)
        assert(batch_size > 0)

        state_shape = batch[0].state.shape

        state_batch = np.zeros((batch_size,)+state_shape)
        action_batch = np.zeros(batch_size, dtype=np.int)
        reward_batch = np.zeros(batch_size)
        next_state_batch = np.zeros((batch_size,)+state_shape)
        is_terminal_batch = np.zeros(batch_size)

        for idx in range(batch_size):
            sample = batch[idx]
            state_batch[idx] = sample.state
            action_batch[idx] = sample.action
            reward_batch[idx] = sample.reward
            next_state_batch[idx] = sample.next_state
            is_terminal_batch[idx] = 1 if sample.is_terminal else 0

        return state_batch, action_batch, reward_batch, next_state_batch, is_terminal_batch

    def select_action(self, state, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
        q_values = self.calc_q_values(state)
        return self.policy.select_action(q_values), q_values

    def update_policy(self):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """

        # Get minibatch from memory
        minibatch = self.memory.sample(self.batch_size)

        # Separate minibatch into separate batches
        state_batch, \
        action_batch, \
        reward_batch, \
        next_state_batch, \
        is_terminal_batch = self.process_batch(minibatch)

        state_batch = self.preprocessor.process_batch(state_batch)
        next_state_batch = self.preprocessor.process_batch(next_state_batch)

        if not self.use_history:
            state_batch = np.squeeze(state_batch, axis=-1)
            next_state_batch = np.squeeze(next_state_batch, axis=-1)

        # Calculate target Q values (depending on double_dqn boolean)
        q_target_values_batch = self.target_network.predict_on_batch(next_state_batch)
        if self.double_dqn:
            target_q_values = self.calc_q_values(next_state_batch)
            target_actions = np.argmax(target_q_values, axis=-1)
            q_target_values_batch = q_target_values_batch[range(self.batch_size), target_actions]
        else:
            q_target_values_batch = np.max(q_target_values_batch, axis=-1)
        
        # Calculate target values
        target_batch = reward_batch + self.gamma \
                                      * (1 - is_terminal_batch) \
                                      * q_target_values_batch

        # Format target and action mask to calculate loss
        targets = np.zeros((self.batch_size, self.num_actions))
        action_mask = np.zeros((self.batch_size, self.num_actions))        
        for idx in range(self.batch_size):
            action = action_batch[idx]
            target = target_batch[idx]
            targets[idx, action] = target
            action_mask[idx, action] = 1

        return self.extended_q_network.train_on_batch([state_batch, targets, action_mask], target_batch)

    def fit(self, env, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """

        episode = 0
        total_loss = 0
        total_max_q = 0
        total_mean_q = 0
        discounted_reward = 0
        total_reward = 0
        loss_steps = 0
        steps = 0
        discount = 1
        evaluate = False

        # Initialize writer for tensorboard
        sess = K.get_session()
        writer = tf.summary.FileWriter(log_path, sess.graph)

        start_time = time.time()
        
        # Initialize env and state
        observation = env.reset()
        observation = self.preprocessor.process_state_for_memory(observation)

        if self.use_history:
            hp = HistoryPreprocessor(observation.shape, self.window)
            state = hp.process_state_for_network(observation)
        else:
            state = observation

        # Update the deep Q network
        for iteration in range(num_iterations+1):
            # Given state, select an action
            state = np.array([state])
            action, q_values = self.select_action(state)
            action = action[0]

            # Perform action on env
            next_observation, reward, is_terminal, info = env.step(action)

            if self.render:
                env.render()

            # Process reward and state then append to memory
            reward = self.preprocessor.process_reward(reward)
            self.memory.append(observation, action, reward, is_terminal)

            observation = next_observation
            observation = self.preprocessor.process_state_for_memory(observation)
            
            if self.use_history:
                state = hp.process_state_for_network(observation)
            else:
                state = observation

            # Train network
            if iteration % self.train_freq == 0 and iteration > self.num_burn_in:
                loss = self.update_policy()
                total_loss = total_loss + loss
                loss_steps = loss_steps + 1

            # Video capture checkpoint
            if iteration in self.video_capture_points:
                self.extended_q_network.save_weights(
                    model_file % (self.model_name, str(iteration)))

            # Save network weights
            if iteration % self.save_network_freq == 0 and iteration > self.num_burn_in:
                self.extended_q_network.save_weights(
                    model_file % (self.model_name, ''))

            # Update target network
            if self.target_fixing \
               and iteration % self.target_update_freq == 0 \
               and iteration > self.num_burn_in:
                self.target_network.set_weights(self.q_network.get_weights())

            # Set boolean to true to evaluate for performance plot once episode ends
            if iteration % self.eval_train_freq == 0:
                evaluate = True

            # Update statistics
            discounted_reward = discounted_reward + discount * reward
            total_reward = total_reward + reward
            discount = self.gamma * discount
            total_max_q = total_max_q + np.max(q_values, axis=-1)[0]
            total_mean_q = total_mean_q + np.mean(q_values, axis=-1)[0]
            steps = steps + 1

            if is_terminal or (max_episode_length != None \
                               and max_episode_length > 0 \
                               and iteration % max_episode_length == 0):

                # Record statistics in tensorboard
                if loss_steps > 0 and iteration < num_iterations:
                    writer.add_summary(log_tb_value('Average loss', total_loss/loss_steps), episode)
                    if self.print_summary:
                        print('Average loss in episode %d is %f' % (episode, total_loss/loss_steps))

                if steps > 0 and iteration < num_iterations:
                    writer.add_summary(log_tb_value('Avergae max Q value', total_max_q/steps), episode)
                    writer.add_summary(log_tb_value('Average mean Q value', total_mean_q/steps), episode)
                    writer.add_summary(log_tb_value('Average reward', total_reward/steps), episode)

                if iteration < num_iterations:
                    writer.add_summary(log_tb_value('Discounted reward', discounted_reward), episode)
                    writer.add_summary(log_tb_value('Total reward', total_reward), episode)

                    if self.print_summary:
                        print('Discounted reward in episode %d is %f' % (episode, discounted_reward))
                        print('Total reward in episode %d is %f' % (episode, total_reward))
                        print('--- %s seconds ---' % (time.time() - start_time))
                        print('--- %d iterations ---' % iteration)

                # Evaluate model and record performance
                if evaluate:
                    average_reward = self.evaluate(env, self.eval_train_num_ep)
                    writer.add_summary(log_tb_value('Performance plot', average_reward), iteration)
                    evaluate = False
                    
                # Reset statistics and state
                episode = episode + 1
                total_loss = 0
                total_max_q = 0
                total_mean_q = 0
                discounted_reward = 0
                total_reward = 0
                loss_steps = 0
                steps = 0
                discount = 1

                self.memory.append(observation, 0, 0, True)

                observation = env.reset()
                observation = self.preprocessor.process_state_for_memory(observation)
                
                if self.use_history:
                    hp.reset()
                    state = hp.process_state_for_network(observation)
                else:
                    state = observation

    def evaluate(self, env, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        if self.print_summary:
            print('Evaluating %d episodes...' % num_episodes)
        
        # Initialize state
        observation = env.reset()
        observation = self.preprocessor.process_state_for_memory(observation)

        if self.use_history:
            hp = HistoryPreprocessor(observation.shape, self.window)
            state = hp.process_state_for_network(observation)
        else:
            state = observation

        episode = 0
        total_reward = 0
        episode_length = 0
        total_rewards = []
        episode_lengths = []
        while episode < num_episodes:
            # Given state, select an action
            state = np.array([state])
            action, q_values = self.select_action(state)
            action = action[0]

            # Perform action on env
            observation, reward, is_terminal, info = env.step(action)
            
            if self.render:
                env.render()

            observation = self.preprocessor.process_state_for_memory(observation)
            
            if self.use_history:
                state = hp.process_state_for_network(observation)
            else:
                state = observation

            # Update statistics
            episode_length = episode_length + 1
            total_reward = total_reward + reward

            if is_terminal:
                # Reset statistics and state
                observation = env.reset()
                observation = self.preprocessor.process_state_for_memory(observation)
                
                if self.use_history:
                    hp.reset()
                    state = hp.process_state_for_network(observation)
                else:
                    state = observation
                    
                episode_lengths.append(episode_length)
                total_rewards.append(total_reward)

                episode = episode + 1
                total_reward = 0
                episode_length = 0

        if self.print_summary:
            print(u'Average total reward is %f +/- %F' 
                % (np.mean(total_rewards), np.std(total_rewards)))
            print(u'Average episode length is %f +/- %f' 
                % (np.mean(episode_lengths), np.std(episode_lengths)))

        return np.mean(total_rewards)

    def capture_episode_video(self, env, video_name=''):
        """Record the performance of your agent.

        You should load the weights of your agent before
        running here.

        Pass in the video name for a unique folder name.
        """
        video_file = os.path.join(video_capture_path, video_name)
        env = wrappers.Monitor(env, video_file)
        self.evaluate(env, 1)