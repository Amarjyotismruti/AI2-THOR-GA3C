import random
import numpy as np
import sys
if sys.version_info[0] == 2:
	from core import ReplayMemory, Sample, Observation
	from ringbuffer import RingBuffer
else:
	from dqn.core import ReplayMemory, Sample, Observation
	from dqn.ringbuffer import RingBuffer
class BasicMemory(ReplayMemory):
	"""Memory to record and sample past experience.

	Uses a single ring buffer to record all previous
	observations, actions, rewards, and terminals
	where each index stores a tuple.
	"""
	def __init__(self, max_size, window_length):
		self.max_size = max_size
		self.window_length = window_length
		self.memory = RingBuffer(max_size)

	def append(self, state, action, reward, is_terminal):
		obs = Observation(state, action, reward, is_terminal)
		self.memory.append(obs)

	def sample(self, batch_size, indexes=None):
		# Get random indexes if not given
		if indexes is None:
			try:
				all_indexes = xrange(self.memory.size)
			except:
				all_indexes = range(self.memory.size)
			indexes = random.sample(all_indexes, batch_size)

		assert(self.memory.size > self.window_length)
		assert(batch_size == len(indexes))

		state_shape = np.shape(self.memory.buffer[0].state)

		samples = []
		for index in indexes:
			index = index % self.memory.size
			state_shape = np.shape(self.memory.buffer[0].state)

			# Sample index again if index is at the end of an episode
			# or index is at the ring buffer boundary
			while self.is_episode_end(index) or index == self.memory.index - 1:
				try:
					all_indexes = xrange(self.memory.size)
				except:
					all_indexes = range(self.memory.size)
				index = random.sample(all_indexes, 1)[0]

			states = np.zeros(state_shape+(self.window_length,))
			next_states = np.zeros(state_shape+(self.window_length,))

			# Get the most recent frames without crossing episode boundary
			for frame in range(self.window_length):
				frame_index = (index - frame) % self.memory.size
				if self.invalid_frame_index(frame_index, index, frame):
					break
				state = self.memory.buffer[frame_index].state
				states[..., self.window_length - frame - 1] = state

			# Get the next frames
			next_state = self.memory.buffer[(index + 1) % self.memory.size].state
			next_states[..., :-1] = states[..., 1:]
			next_states[..., -1] = next_state

			action = self.memory.buffer[index].action
			reward = self.memory.buffer[index].reward
			is_terminal = self.memory.buffer[index].is_terminal

			sample = Sample(states, action, reward, next_states, is_terminal)
			samples.append(sample)

		return samples

	# Checks if index is at the end of an episode or index is at the ring buffer boundary
	def invalid_frame_index(self, frame_index, index, frame):
		return (self.memory.size == self.max_size and frame_index == self.memory.index) \
			or (self.memory.size < self.max_size and frame_index == self.memory.index - 1) \
			or (self.memory.buffer[frame_index % self.memory.size].is_terminal and frame > 0)
	
	# Checks if index is at the end of an episode
	def is_episode_end(self, index):
		return self.memory.buffer[(index - 1) % self.memory.size].is_terminal \
			   and self.memory.buffer[index  % self.memory.size].is_terminal

	def clear(self):
		self.memory = RingBuffer(self.max_size)
		

class NaiveMemory(BasicMemory):
	"""Naive memory to sample latest experiences.

	Get the latest batch_size number of experience from memory.
	"""
	def sample(self, batch_size, indexes=None):
		indexes = []

		for i in range(batch_size):
			indexes.insert(0, (self.memory.index - i - 2) % self.max_size)

		return BasicMemory.sample(self, batch_size, indexes=indexes)