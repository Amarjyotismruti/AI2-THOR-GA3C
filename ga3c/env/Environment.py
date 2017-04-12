from ga3c.Config import Config
from ga3c.env.AtariEnvironment import AtariEnvironment
from ga3c.env.CartPoleEnvironment import CartPoleEnvironment

def Environment():
    if 'atari' in Config.NETWORK_NAME:
        return AtariEnvironment()
    elif 'cartpole' in Config.NETWORK_NAME:
        return CartPoleEnvironment()
    else:
        raise('Env does not exist.')