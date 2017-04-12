from ga3c.Config import Config
from ga3c.env.AtariEnvironment import AtariEnvironment
from ga3c.env.CartPoleEnvironment import CartPoleEnvironment
from ga3c.env.THOREnvironment import THOREnvironment

def Environment():
	if 'atari' in Config.NETWORK_NAME:
		return AtariEnvironment()
	elif 'cartpole' in Config.NETWORK_NAME:
		return CartPoleEnvironment()
	elif 'thor' in Config.NETWORK_NAME:
		return THOREnvironment()
	else:
		raise('Env does not exist.')