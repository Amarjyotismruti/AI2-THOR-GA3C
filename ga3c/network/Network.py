class Network:
    def __init__(self):
        NotImplementedError

    def train(self, x, y_r, a, trainer_id):
        NotImplementedError

    def log(self, x, y_r, a):
        NotImplementedError

    def save(self, episode):
        NotImplementedError

    def load(self):
        NotImplementedError