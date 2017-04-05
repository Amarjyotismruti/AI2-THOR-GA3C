class Environment(object):
    """
    Small wrapper for environments.
    Responsible for preprocessing screens and holding on to a screen buffer 
    of size agent_history_length from which environment state is constructed.
    """
    def __init__(self):
        NotImplementedError

    def get_initial_state(self):
        NotImplementedError

    def get_preprocessed_frame(self):
        NotImplementedError

    def step(self):
        NotImplementedError