import tensorflow as tf
from keras.layers import Input, Dense, Flatten, Convolution2D, Activation, Lambda
from keras.models import Model, Sequential
import keras.backend as K

def create_model(window, input_shape, num_actions, model_name='q_network'):
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance (you can also
    use the SequentialModel class).

    We highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understnad your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """
    if 'dueling_deep_Q_network' in model_name:
    	model = create_dueling_deep_Q_network(window, input_shape, num_actions)
    elif 'deep_Q_network' in model_name:
        model = create_deep_Q_network(window, input_shape, num_actions)
    elif 'linear_Q_network' in model_name:
        model = create_linear_Q_network(window, input_shape, num_actions)
    elif 'cartpole' in model_name:
    	model = create_cartpole_model(input_shape, num_actions)
    else:
        raise Exception('Model %s is not valid.' % model_name)
    
    print(model.summary())

    return model

def create_dueling_deep_Q_network(window, input_shape, num_actions):
	with tf.name_scope('Input'):
		input = Input(shape=input_shape+(window,))
	with tf.name_scope('Conv1'):
		x = Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu')(input)
	with tf.name_scope('Conv2'):
		x = Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu')(x)
	with tf.name_scope('Conv3'):
		x = Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu')(x)
	with tf.name_scope('Flatten'):
		x = Flatten()(x)
	with tf.name_scope('FC_A'):
		x_A = Dense(512, activation='relu')(x)
	with tf.name_scope('FC_V'):
		x_V = Dense(512, activation='relu')(x)
	with tf.name_scope('A'):
		A = Dense(num_actions, activation='linear')(x_A)
	with tf.name_scope('Lambda_A'):
		A = Lambda(lambda x: x - K.max(x, axis=-1, keepdims=True))(A)
	with tf.name_scope('V'):
		V = Dense(1, activation='linear')(x_V)
	with tf.name_scope('Sum'):
		output = Lambda(lambda x: x[0] + x[1])([A, V])
		
	model = Model(input=input, output=output)

	return model

def create_linear_Q_network(window, input_shape, num_actions):
	with tf.name_scope('Input'):
		input = Input(shape=input_shape+(window,))
	with tf.name_scope('Flatten'):
		flatten = Flatten()(input)
	with tf.name_scope('FC'):
		hidden = Dense(36, activation='relu')(flatten)
	with tf.name_scope('Output'):
		output = Dense(num_actions, activation='linear')(hidden)

	model = Model(input=input, output=output)

	return model

def create_deep_Q_network(window, input_shape, num_actions):
	with tf.name_scope('Input'):
		input = Input(shape=input_shape+(window,))
	with tf.name_scope('Conv1'):
		x = Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu')(input)
	with tf.name_scope('Conv2'):
		x = Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu')(x)
	with tf.name_scope('Conv3'):
		x = Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu')(x)
	with tf.name_scope('Flatten'):
		x = Flatten()(x)
	with tf.name_scope('FC'):
		x = Dense(512, activation='relu')(x)
	with tf.name_scope('Output'):
		output = Dense(num_actions, activation='linear')(x)
	
	model = Model(input=input, output=output)

	return model

def create_cartpole_model(input_shape, num_actions):
	input = Input(shape=input_shape+(1,))
	x = Flatten()(input)
	x = Dense(32, activation='relu')(x)
	x = Dense(32, activation='relu')(x)
	x = Dense(32, activation='relu')(x)
	output = Dense(num_actions, activation='linear')(x)

	model = Model(input=input, output=output)

	return model