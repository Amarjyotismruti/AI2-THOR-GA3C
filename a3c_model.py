import tensorflow as tf
from keras import backend as K
from keras.layers import Convolution2D, Flatten, Dense, Input
from keras.models import Model

def build_policy_and_value_networks(model_name, num_actions, input_shape, window):
    if 'a3c_networks' in model_name:
        return build_a3c_networks(num_actions, window, input_shape)
    elif 'cartpole' in model_name:
        return build_cartpole_networks(num_actions, input_shape)
    else:
        raise('Model does not exist.')

def build_a3c_networks(num_actions, window, input_shape):
    with tf.device("/cpu:0"):
        state = tf.placeholder("float", [None, window] + list(input_shape))
        
        inputs = Input(shape=(window,)+input_shape)
        shared = Convolution2D(name="conv1", nb_filter=16, nb_row=8, nb_col=8, subsample=(4,4), activation='relu', border_mode='same')(inputs)
        shared = Convolution2D(name="conv2", nb_filter=32, nb_row=4, nb_col=4, subsample=(2,2), activation='relu', border_mode='same')(shared)
        shared = Flatten()(shared)
        shared = Dense(name="h1", output_dim=256, activation='relu')(shared)

        action_probs = Dense(name="p", output_dim=num_actions, activation='softmax')(shared)
        
        state_value = Dense(name="v", output_dim=1, activation='linear')(shared)

        policy_network = Model(input=inputs, output=action_probs)
        value_network = Model(input=inputs, output=state_value)

        p_params = policy_network.trainable_weights
        v_params = value_network.trainable_weights

        p_out = policy_network(state)
        v_out = value_network(state)

    return state, p_out, v_out, p_params, v_params

def build_cartpole_networks(num_actions, input_shape):
    with tf.device("/cpu:0"):
        state = tf.placeholder("float", [None, 1] + list(input_shape))

        inputs = Input(shape=(1,)+input_shape)
        shared = Flatten()(inputs)
        shared = Dense(32, activation="relu")(shared)
        shared = Dense(32, activation="relu")(shared)
        shared = Dense(32, activation="relu")(shared)

        action_probs = Dense(name="p", output_dim=num_actions, activation='softmax')(shared)
        
        state_value = Dense(name="v", output_dim=1, activation='linear')(shared)

        policy_network = Model(input=inputs, output=action_probs)
        value_network = Model(input=inputs, output=state_value)

        p_params = policy_network.trainable_weights
        v_params = value_network.trainable_weights

        p_out = policy_network(state)
        v_out = value_network(state)

    return state, p_out, v_out, p_params, v_params