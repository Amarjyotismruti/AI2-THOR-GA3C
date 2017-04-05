class CartPoleEnvironment(Environment):
    def __init__(self, gym_env):
        self.env = gym_env
        self.gym_actions = range(gym_env.action_space.n)

    def get_initial_state(self):
        state = self.env.reset()
        return np.array([state])

    def step(self, action_index):
        state, reward, is_terminal, info = self.env.step(self.gym_actions[action_index])
        return np.array([state]), reward, is_terminal, info

    def render(self):
        pass