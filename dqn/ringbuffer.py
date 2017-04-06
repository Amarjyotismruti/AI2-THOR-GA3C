class RingBuffer:
	def __init__(self, max_size):
		self.max_size = max_size
		self.buffer = max_size * [None]
		self.index = 0
		self.size = 0
		
	def append(self, item):
		self.buffer[self.index] = item
		
		self.index += 1
		self.index %= self.max_size

		if self.size < self.max_size:
			self.size += 1